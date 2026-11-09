"""Standalone Stage-4B anchored reference-DPO trainer.

Unlike ``train_stage4_scale_wise.py``, this entry point never constructs DMD
real/fake score models or a critic optimizer.  The only trainable module is the
causal policy.  The causal upsampler remains frozen and can optionally restore
the exact component embedded in the Stage-4A policy checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
)
from ltx_distillation.causal_window_upsampler import (
    build_causal_upsampler_from_checkpoint,
)
from ltx_distillation.models.text_encoder_wrapper import (
    create_text_encoder_wrapper,
)
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_distillation.stage4_reference_dpo import Stage4ReferenceDPO
from ltx_distillation.stage35_data import build_stage35_dataloader
from ltx_distillation.stage35_media import (
    decode_video_tiled,
    stitch_side_by_side,
    write_video_pyav,
)
from ltx_distillation.stage35_refiner import (
    build_causal_wrapper_from_checkpoint,
    build_official_refiner,
)
from ltx_distillation.train_distillation import compute_latent_shapes
from ltx_distillation.util import (
    barrier,
    cycle,
    fsdp_state_dict,
    fsdp_wrap,
    init_logging_folder,
    launch_distributed_job,
    set_seed,
)


def load_embedded_causal_upsampler(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
) -> int:
    """Strictly restore a Stage4A/4B embedded causal-upscaler component."""

    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Stage4 checkpoint must be a mapping: {path}")
    state = checkpoint.get("causal_upsampler")
    if not isinstance(state, Mapping):
        raise KeyError(f"Checkpoint has no causal_upsampler state: {path}")
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "strict causal upsampler restore returned "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    step = int(checkpoint.get("step", 0))
    del state, checkpoint
    return step


def validate_stage4b_config(
    config: DictConfig,
    *,
    check_assets: bool = True,
) -> None:
    """Validate topology before allocating any 19B model."""

    if str(config.get("training_mode", "")).lower() != "stage4_reference_dpo":
        raise ValueError(
            "Stage4ReferenceDPOTrainer requires "
            "training_mode=stage4_reference_dpo"
        )
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("Stage4B reference DPO requires batch_size=1")
    if int(config.get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError(
            "Stage4B gradient_accumulation_steps must be >= 1"
        )
    if float(config.get("weight_decay", 0.0)) != 0.0:
        raise ValueError(
            "Stage4B reference-DPO-only uses weight_decay=0.0 to preserve "
            "the Stage4A anchor outside the measured preference gradient"
        )
    schedule = tuple(
        int(value)
        for value in config.get(
            "denoising_step_list", [1000, 960, 889, 727, 0]
        )
    )
    if schedule != Stage4ReferenceDPO.EXPECTED_SCHEDULE:
        raise ValueError(
            "Stage4B requires schedule [1000, 960, 889, 727, 0], got "
            f"{list(schedule)}"
        )
    if bool(config.get("train_causal_upsampler", False)):
        raise ValueError(
            "The reference-DPO-only baseline freezes the causal upsampler; "
            "set train_causal_upsampler=false"
        )
    required = (
        "checkpoint_path",
        "policy_init_checkpoint",
        "causal_upsampler_checkpoint_path",
        "spatial_upsampler_path",
        "refiner_lora_path",
        "text_encoder_checkpoint",
        "data_config_path",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"missing required Stage4B config keys: {missing}")
    if check_assets:
        asset_keys = (
            "checkpoint_path",
            "policy_init_checkpoint",
            "causal_upsampler_checkpoint_path",
            "spatial_upsampler_path",
            "refiner_lora_path",
            "text_encoder_checkpoint",
            "data_config_path",
        )
        if config.get("reference_checkpoint"):
            asset_keys = (*asset_keys, "reference_checkpoint")
        if config.get("causal_upsampler_state_checkpoint"):
            asset_keys = (*asset_keys, "causal_upsampler_state_checkpoint")
        missing_assets = [
            f"{key}={config.get(key)}"
            for key in asset_keys
            if not Path(str(config.get(key))).exists()
        ]
        if missing_assets:
            raise FileNotFoundError(
                "Stage4B asset paths do not exist: "
                + ", ".join(missing_assets)
            )
    if float(config.get("dpo_beta", 1.0)) <= 0:
        raise ValueError("dpo_beta must be > 0")
    if float(config.get("dpo_loss_weight", 1.0)) < 0:
        raise ValueError("dpo_loss_weight must be >= 0")
    for key in (
        "negative_video_anchor_weight",
        "negative_audio_anchor_weight",
    ):
        if float(config.get(key, 0.0)) < 0:
            raise ValueError(f"{key} must be >= 0")
    positive_video_mix = float(config.get("positive_video_mix", 1.0))
    if not 0.0 < positive_video_mix <= 1.0:
        raise ValueError("positive_video_mix must be in (0, 1]")
    positive_audio_source = str(
        config.get("positive_audio_source", "refined")
    ).strip().lower()
    if positive_audio_source not in {"negative", "refined"}:
        raise ValueError(
            "positive_audio_source must be negative or refined"
        )
    pair_source = str(
        config.get("dpo_pair_source", "online_policy")
    ).strip().lower()
    if pair_source not in {"online_policy", "frozen_reference"}:
        raise ValueError(
            "dpo_pair_source must be online_policy or frozen_reference"
        )
    candidates = tuple(
        int(value)
        for value in config.get("dpo_timestep_candidates", [889, 727])
    )
    if not candidates or any(value not in (889, 727) for value in candidates):
        raise ValueError(
            "dpo_timestep_candidates must be a non-empty subset of [889, 727]"
        )
    if not bool(config.get("variable_length_rollout", False)):
        num_frames = int(config.get("num_frames", 121))
        if (num_frames - 1) % 8 != 0:
            raise ValueError("fixed Stage4B num_frames must be 8n+1")
    if bool(config.get("stability_guard_enabled", False)):
        if not 0.0 <= float(
            config.get("stability_guard_ema_decay", 0.9)
        ) < 1.0:
            raise ValueError("stability_guard_ema_decay must be in [0, 1)")
        if int(config.get("stability_guard_patience", 3)) < 1:
            raise ValueError("stability_guard_patience must be >= 1")
    prompt_source = str(
        config.get("stage35_prompt_source", "metadata")
    ).strip().lower()
    if prompt_source not in {"local_files", "metadata"}:
        raise ValueError(
            "stage35_prompt_source must be local_files or metadata"
        )
    if prompt_source == "local_files":
        prompt_files = [
            Path(str(value))
            for value in config.get("stage35_local_prompt_files", [])
        ]
        prompt_globs = [
            str(value)
            for value in config.get("stage35_local_prompt_globs", [])
        ]
        matched = [path for pattern in prompt_globs for path in glob(pattern)]
        if not any(path.is_file() for path in prompt_files) and not matched:
            raise FileNotFoundError(
                "stage35_prompt_source=local_files but no prompt files matched"
            )


class Stage4ReferenceDPOTrainer:
    def __init__(self, config: DictConfig) -> None:
        validate_stage4b_config(config)
        self.config = config
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        rank, world_size, _ = launch_distributed_job()
        self.global_rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = (
            torch.bfloat16
            if bool(config.get("mixed_precision", True))
            else torch.float32
        )
        self.is_main_process = self.global_rank == 0

        seed = int(config.get("seed", 42))
        if seed == 0:
            sampled = torch.randint(0, 10_000_000, (1,), device=self.device)
            if dist.is_initialized():
                dist.broadcast(sampled, src=0)
            seed = int(sampled.item())
        self.seed = seed
        set_seed(seed + self.global_rank)

        self.output_path = str(config.output_path)
        if self.is_main_process:
            self.output_path, _ = init_logging_folder(config)
        else:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.output_path = self._broadcast_string(self.output_path)
        barrier()

        self.num_frame_per_block = int(
            config.get("num_frame_per_block", 3)
        )
        self.first_block_video_frames = int(
            config.get("first_block_video_frames", 4)
        )
        self.variable_length_rollout = bool(
            config.get("variable_length_rollout", False)
        )
        self.gradient_accumulation_steps = int(
            config.get("gradient_accumulation_steps", 1)
        )
        self.max_steps = int(config.get("max_steps", 2000))
        self.step = 0
        self._stability_ema: Dict[str, float] = {}
        self._stability_violation_count = 0

        self._init_models()
        self._wrap_models()
        self._init_objective()
        self._init_optimizer()
        self._update_learning_rate()

        self.dataloader = cycle(build_stage35_dataloader(config))
        self.local_visualization_iters = int(
            config.get("local_visualization_iters", 25)
        )
        self.local_video_fps = int(config.get("local_video_fps", 24))
        self.benchmark_iters = int(config.get("benchmark_iters", 400))
        self.checkpoint_iters = int(config.get("checkpoint_iters", 400))
        self.previous_time: Optional[float] = None
        self._init_benchmark_prompts()
        if self.is_main_process:
            effective_batch = (
                int(config.get("batch_size", 1))
                * self.gradient_accumulation_steps
                * self.world_size
            )
            print(
                "[Stage4B] gradient accumulation: "
                f"{self.gradient_accumulation_steps}; effective batch = "
                f"{int(config.get('batch_size', 1))} x "
                f"{self.gradient_accumulation_steps} x {self.world_size} = "
                f"{effective_batch}",
                flush=True,
            )
            pair_source = str(
                config.get("dpo_pair_source", "online_policy")
            ).strip().lower()
            print(
                "[Stage4B] preference pair source: "
                f"{pair_source}; positive_video_mix="
                f"{float(config.get('positive_video_mix', 1.0))}; "
                f"positive_audio_source="
                f"{config.get('positive_audio_source', 'refined')}",
                flush=True,
            )

    def _broadcast_string(self, value: str) -> str:
        if not dist.is_initialized() or self.world_size == 1:
            return value
        max_bytes = 2048
        if self.is_main_process:
            encoded = value.encode("utf-8")
            if len(encoded) >= max_bytes:
                raise ValueError("distributed string is too long")
            payload = torch.tensor(
                list(encoded) + [0] * (max_bytes - len(encoded)),
                device=self.device,
                dtype=torch.uint8,
            )
        else:
            payload = torch.zeros(
                max_bytes, device=self.device, dtype=torch.uint8
            )
        dist.broadcast(payload, src=0)
        result = bytes(payload.cpu().tolist()).rstrip(b"\x00").decode("utf-8")
        Path(result).mkdir(parents=True, exist_ok=True)
        return result

    def _init_models(self) -> None:
        config = self.config
        resume = config.get("resume_checkpoint")
        policy_source = str(resume or config.policy_init_checkpoint)
        reference_source = str(
            config.get("reference_checkpoint")
            or config.policy_init_checkpoint
        )

        if self.is_main_process:
            print(
                f"[Stage4B] loading trainable online policy: {policy_source}",
                flush=True,
            )
        self.policy = build_causal_wrapper_from_checkpoint(
            config=config,
            checkpoint_path=policy_source,
            device=self.device,
            dtype=self.dtype,
            trainable=True,
            gradient_checkpointing=bool(
                config.get("gradient_checkpointing", True)
            ),
            use_ema=bool(config.get("policy_use_ema", False)),
        )

        if self.is_main_process:
            print(
                f"[Stage4B] loading frozen reference policy: "
                f"{reference_source}",
                flush=True,
            )
        self.reference = build_causal_wrapper_from_checkpoint(
            config=config,
            checkpoint_path=reference_source,
            device=self.device,
            dtype=self.dtype,
            trainable=False,
            gradient_checkpointing=False,
            use_ema=bool(config.get("reference_use_ema", False)),
        )

        if self.is_main_process:
            print(
                "[Stage4B] loading frozen official bidirectional refiner...",
                flush=True,
            )
        self.official_refiner = build_official_refiner(
            base_checkpoint=str(config.checkpoint_path),
            lora_path=str(config.refiner_lora_path),
            lora_strength=float(config.get("refiner_lora_strength", 1.0)),
            device=self.device,
            dtype=self.dtype,
            high_height=int(config.video_height) * 2,
            high_width=int(config.video_width) * 2,
        )

        upsampler_state_source = str(
            resume
            or config.get("causal_upsampler_state_checkpoint")
            or ""
        )
        if self.is_main_process:
            print(
                "[Stage4B] building frozen causal upsampler from: "
                f"{config.causal_upsampler_checkpoint_path}",
                flush=True,
            )
        self.causal_upsampler = build_causal_upsampler_from_checkpoint(
            checkpoint_path=str(config.checkpoint_path),
            spatial_upsampler_path=str(config.spatial_upsampler_path),
            causal_upsampler_checkpoint_path=str(
                config.causal_upsampler_checkpoint_path
            ),
            device=self.device,
            dtype=self.dtype,
            history_frames=int(
                config.get("causal_upsampler_history_frames", 18)
            ),
            gradient_checkpointing=False,
        )
        if upsampler_state_source:
            restored_step = load_embedded_causal_upsampler(
                self.causal_upsampler,
                upsampler_state_source,
            )
            if self.is_main_process:
                print(
                    "[Stage4B] restored embedded causal upsampler from "
                    f"{upsampler_state_source} (checkpoint step "
                    f"{restored_step})",
                    flush=True,
                )
        self.causal_upsampler.eval().requires_grad_(False)

        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=str(config.checkpoint_path),
            gemma_path=str(config.text_encoder_checkpoint),
            device=self.device,
            dtype=self.dtype,
        )
        self.text_encoder.eval().requires_grad_(False)

        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=str(config.checkpoint_path),
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        self.video_vae.eval().requires_grad_(False)
        self.audio_vae.eval().requires_grad_(False)

        if resume:
            try:
                resumed = torch.load(
                    str(resume),
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
            except TypeError:
                resumed = torch.load(str(resume), map_location="cpu")
            self.step = int(resumed.get("step", 0))
            del resumed
            if self.is_main_process:
                print(
                    f"[Stage4B] resumed online weights at step {self.step}; "
                    "optimizer is intentionally reinitialized",
                    flush=True,
                )

    def _wrap_models(self) -> None:
        config = self.config

        def wrap(module, key: str):
            return fsdp_wrap(
                module,
                sharding_strategy=str(
                    config.get("sharding_strategy", "hybrid_full")
                ),
                mixed_precision=bool(config.get("mixed_precision", True)),
                wrap_strategy=str(config.get(key, "size")),
            )

        self.policy = wrap(self.policy, "policy_fsdp_wrap_strategy")
        self.reference = wrap(
            self.reference, "reference_fsdp_wrap_strategy"
        )
        self.official_refiner = wrap(
            self.official_refiner, "official_refiner_fsdp_wrap_strategy"
        )
        self.causal_upsampler = wrap(
            self.causal_upsampler,
            "causal_upsampler_fsdp_wrap_strategy",
        )
        self.text_encoder = wrap(
            self.text_encoder, "text_encoder_fsdp_wrap_strategy"
        )
        self.policy.eval().requires_grad_(True)
        self.reference.eval().requires_grad_(False)
        self.official_refiner.eval().requires_grad_(False)
        self.causal_upsampler.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    def _init_objective(self) -> None:
        config = self.config
        self.objective = Stage4ReferenceDPO(
            policy=self.policy,
            reference=self.reference,
            official_refiner=self.official_refiner,
            causal_upsampler=self.causal_upsampler,
            device=self.device,
            dtype=self.dtype,
            num_frame_per_block=self.num_frame_per_block,
            first_block_video_frames=self.first_block_video_frames,
            enable_chunk_recycling=bool(
                config.get("enable_chunk_recycling", False)
            ),
            dpo_beta=float(config.get("dpo_beta", 1.0)),
            dpo_loss_weight=float(config.get("dpo_loss_weight", 1.0)),
            dpo_video_weight=float(config.get("dpo_video_weight", 1.0)),
            dpo_audio_weight=float(config.get("dpo_audio_weight", 0.0)),
            negative_video_anchor_weight=float(
                config.get("negative_video_anchor_weight", 0.0)
            ),
            negative_audio_anchor_weight=float(
                config.get("negative_audio_anchor_weight", 0.0)
            ),
            dpo_pair_source=str(
                config.get("dpo_pair_source", "online_policy")
            ),
            positive_video_mix=float(
                config.get("positive_video_mix", 1.0)
            ),
            positive_audio_source=str(
                config.get("positive_audio_source", "refined")
            ),
            dpo_timestep_candidates=tuple(
                config.get("dpo_timestep_candidates", [889, 727])
            ),
            refiner_sigmas=tuple(
                config.get(
                    "refiner_sigmas",
                    [0.909375, 0.725, 0.421875, 0.0],
                )
            ),
        )

    def _init_optimizer(self) -> None:
        config = self.config
        parameters = [
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("Stage4B online policy has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.get("generator_lr", 2e-6)),
            betas=(
                float(config.get("beta1", 0.9)),
                float(config.get("beta2", 0.999)),
            ),
            eps=float(config.get("adam_eps", 1e-8)),
            weight_decay=0.0,
        )
        self.base_lr = float(self.optimizer.param_groups[0]["lr"])

    def _lr_factor(self, step: int) -> float:
        warmup = int(self.config.get("warmup_steps", 50))
        min_lr = float(self.config.get("min_lr", 1e-7))
        min_factor = min(min_lr / max(self.base_lr, 1e-20), 1.0)
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / float(
            max(1, self.max_steps - warmup)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_factor, cosine)

    def _update_learning_rate(self) -> None:
        self.optimizer.param_groups[0]["lr"] = (
            self.base_lr * self._lr_factor(self.step)
        )

    @staticmethod
    def _snap_ltx_frames(value: int) -> int:
        value = max(int(value), 1)
        return value - ((value - 1) % 8)

    def _batch_shapes(self, batch: Dict[str, Any]):
        if self.variable_length_rollout:
            real_num_frames = self._snap_ltx_frames(
                int(batch["num_frames"])
            )
        else:
            real_num_frames = int(self.config.num_frames)
            if (real_num_frames - 1) % 8 != 0:
                raise ValueError("fixed num_frames must be 8n+1")

        rollout_num_frames = real_num_frames
        if (
            self.variable_length_rollout
            and dist.is_initialized()
            and self.world_size > 1
        ):
            maximum = torch.tensor(
                [real_num_frames], device=self.device, dtype=torch.long
            )
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            rollout_num_frames = int(maximum.item())

        height = int(batch["height"])
        width = int(batch["width"])
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=rollout_num_frames,
            video_height=height,
            video_width=width,
            batch_size=1,
        )
        audio_shape[1] = int(
            compute_aligned_audio_frames(
                total_video_latent_frames=int(video_shape[1]),
                num_frame_per_block=self.num_frame_per_block,
                first_block_video_frames=self.first_block_video_frames,
            )
        )
        real_video_frames = 1 + (real_num_frames - 1) // 8
        real_audio_frames = int(
            compute_aligned_audio_frames(
                total_video_latent_frames=real_video_frames,
                num_frame_per_block=self.num_frame_per_block,
                first_block_video_frames=self.first_block_video_frames,
            )
        )
        return (
            video_shape,
            audio_shape,
            real_num_frames,
            rollout_num_frames,
            real_video_frames,
            real_audio_frames,
            height,
            width,
        )

    def _reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory(self, phase: str) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        gib = float(1024**3)
        values = torch.tensor(
            [
                torch.cuda.max_memory_allocated(self.device) / gib,
                torch.cuda.max_memory_reserved(self.device) / gib,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return {
            f"memory/{phase}/peak_allocated_gib": float(values[0].item()),
            f"memory/{phase}/peak_reserved_gib": float(values[1].item()),
        }

    @staticmethod
    def _detach_conditional(values: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in values.items()
        }

    def _reduce_metrics(
        self, metrics: Dict[str, torch.Tensor | float]
    ) -> Dict[str, float]:
        keys = sorted(metrics)
        values = torch.tensor(
            [
                float(value.detach().float().mean().item())
                if torch.is_tensor(value)
                else float(value)
                for value in (metrics[key] for key in keys)
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= float(self.world_size)
        return {
            key: float(value)
            for key, value in zip(
                keys, values.cpu().tolist(), strict=True
            )
        }

    def _update_stability_guard(
        self,
        metrics: Dict[str, float],
    ) -> tuple[bool, str]:
        """Track detached collapse signals and request a synchronized stop."""

        if not bool(self.config.get("stability_guard_enabled", False)):
            return False, ""

        denominator = max(
            metrics["dpo/reference_negative_video"], 1.0e-12
        )
        current = {
            "negative_reference_ratio": (
                metrics["dpo/online_negative_video"] / denominator
            ),
            "preference_probability": metrics[
                "dpo/preference_probability"
            ],
            "grad_norm": metrics["grad/policy"],
        }
        decay = float(self.config.get("stability_guard_ema_decay", 0.9))
        for key, value in current.items():
            previous = self._stability_ema.get(key)
            self._stability_ema[key] = (
                value if previous is None else decay * previous + (1.0 - decay) * value
            )
            metrics[f"stability/ema_{key}"] = self._stability_ema[key]

        violations: list[str] = []
        if self.step >= int(
            self.config.get("stability_guard_min_steps", 25)
        ):
            checks = (
                (
                    "negative_reference_ratio",
                    float(
                        self.config.get(
                            "stability_guard_max_negative_reference_ratio",
                            2.0,
                        )
                    ),
                ),
                (
                    "preference_probability",
                    float(
                        self.config.get(
                            "stability_guard_max_preference_probability",
                            0.8,
                        )
                    ),
                ),
                (
                    "grad_norm",
                    float(
                        self.config.get(
                            "stability_guard_max_grad_norm", 5.0
                        )
                    ),
                ),
            )
            violations = [
                f"{key}={self._stability_ema[key]:.4g}>{threshold:.4g}"
                for key, threshold in checks
                if self._stability_ema[key] > threshold
            ]

        if violations:
            self._stability_violation_count += 1
        else:
            self._stability_violation_count = 0
        metrics["stability/violation_count"] = float(
            self._stability_violation_count
        )
        should_stop = self._stability_violation_count >= int(
            self.config.get("stability_guard_patience", 3)
        )
        metrics["stability/emergency_stop"] = float(should_stop)
        return should_stop, ", ".join(violations)

    @staticmethod
    def _accumulate_microbatch_metrics(
        accumulated: Dict[str, float],
        metrics: Dict[str, torch.Tensor | float],
        *,
        mean_weight: float,
    ) -> None:
        """Merge one micro-batch without retaining device tensors.

        DPO and rollout metrics represent a sample mean and are therefore
        averaged across micro-batches.  Peak-memory metrics retain the maximum
        observed micro-batch peak.
        """

        for key, value in metrics.items():
            scalar = (
                float(value.detach().float().mean().item())
                if torch.is_tensor(value)
                else float(value)
            )
            if key.startswith("memory/"):
                accumulated[key] = max(
                    accumulated.get(key, float("-inf")), scalar
                )
            else:
                accumulated[key] = (
                    accumulated.get(key, 0.0) + mean_weight * scalar
                )

    def train_one_step(
        self,
        *,
        keep_visualization_context: bool,
    ) -> tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        self.policy.eval()
        self.reference.eval()
        self.official_refiner.eval()
        self.causal_upsampler.eval()
        self.text_encoder.eval()
        self.optimizer.zero_grad(set_to_none=True)

        accumulation_steps = self.gradient_accumulation_steps
        mean_weight = 1.0 / float(accumulation_steps)
        accumulated_metrics: Dict[str, float] = {}
        visualization_context = None
        timestep: Optional[int] = None
        replay_tolerance = float(
            self.config.get("dpo_replay_error_tolerance", 5e-4)
        )

        for micro_index in range(accumulation_steps):
            batch = next(self.dataloader)
            prompts = list(batch["prompts"])
            (
                video_shape,
                audio_shape,
                real_num_frames,
                rollout_num_frames,
                real_video_frames,
                real_audio_frames,
                height,
                width,
            ) = self._batch_shapes(batch)

            with torch.no_grad():
                conditional_dict = self._detach_conditional(
                    self.text_encoder(text_prompts=prompts)
                )

            self._reset_peak_memory()
            pair = self.objective.build_preference_pair(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )
            memory_metrics = self._peak_memory("pair_construction")
            torch.cuda.empty_cache()

            # Sampling after the first pair preserves the accumulation=1 RNG
            # order.  All later micro-batches reuse this synchronized node so
            # one optimizer update has the same timestep semantics as a
            # physical global batch.
            if timestep is None:
                timestep = self.objective.sample_synced_dpo_timestep()

            self._reset_peak_memory()
            dpo_metrics = self.objective.dpo_backward(
                pair=pair,
                conditional_dict=conditional_dict,
                timestep=timestep,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
                gradient_scale=mean_weight,
            )
            memory_metrics.update(
                self._peak_memory("dpo_score_and_replay")
            )

            replay_errors = torch.stack(
                [
                    dpo_metrics[
                        "dpo/replay_positive_error"
                    ].detach().float(),
                    dpo_metrics[
                        "dpo/replay_negative_error"
                    ].detach().float(),
                ]
            )
            if dist.is_initialized():
                # Every rank must make the same stop/continue decision;
                # otherwise one bad rank could raise while peers hang in FSDP
                # grad clipping.
                dist.all_reduce(replay_errors, op=dist.ReduceOp.MAX)
            maximum_replay_error = float(replay_errors.max().item())
            if maximum_replay_error > replay_tolerance:
                raise RuntimeError(
                    "Stage4B DPO replay error "
                    f"{maximum_replay_error:.6g} exceeds tolerance "
                    f"{replay_tolerance:.6g} at accumulation micro-batch "
                    f"{micro_index + 1}/{accumulation_steps}; no-grad "
                    "coefficient and gradient replay are not value-aligned"
                )

            raw_microbatch_metrics: Dict[str, torch.Tensor | float] = {
                **dpo_metrics,
                **memory_metrics,
                "rollout/real_num_frames": float(real_num_frames),
                "rollout/num_frames": float(rollout_num_frames),
                "rollout/lr_height": float(height),
                "rollout/lr_width": float(width),
                "rollout/hr_height": float(height * 2),
                "rollout/hr_width": float(width * 2),
            }
            self._accumulate_microbatch_metrics(
                accumulated_metrics,
                raw_microbatch_metrics,
                mean_weight=mean_weight,
            )

            if (
                keep_visualization_context
                and self.is_main_process
                and micro_index == accumulation_steps - 1
            ):
                visualization_context = {
                    "negative_video": pair.negative_video[
                        :, :real_video_frames
                    ].detach().cpu(),
                    "positive_video": pair.positive_video[
                        :, :real_video_frames
                    ].detach().cpu(),
                    "prompt": prompts[0],
                    "timestep": timestep,
                    "seed": self.seed + self.step,
                }

            del pair, conditional_dict, dpo_metrics
            if micro_index < accumulation_steps - 1:
                torch.cuda.empty_cache()

        max_grad_norm = float(self.config.get("max_grad_norm", 10.0))
        if hasattr(self.policy, "clip_grad_norm_"):
            grad_norm = self.policy.clip_grad_norm_(max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        accumulated_metrics.update(
            {
                "grad/policy": grad_norm.detach(),
                "lr/policy": float(self.optimizer.param_groups[0]["lr"]),
                "optimization/gradient_accumulation_steps": float(
                    accumulation_steps
                ),
                "optimization/effective_global_batch_size": float(
                    int(self.config.get("batch_size", 1))
                    * accumulation_steps
                    * self.world_size
                ),
            }
        )
        metrics = self._reduce_metrics(accumulated_metrics)

        torch.cuda.empty_cache()
        return metrics, visualization_context

    @torch.no_grad()
    def _run_local_visualization(self, context: Dict[str, Any]) -> None:
        failure = ""
        if self.is_main_process:
            try:
                self.video_vae.to(device=self.device, dtype=self.dtype)
                negative = decode_video_tiled(
                    video_vae=self.video_vae,
                    latent_bfchw=context["negative_video"].to(
                        device=self.device, dtype=self.dtype
                    ),
                    seed=int(context["seed"]),
                )
                positive = decode_video_tiled(
                    video_vae=self.video_vae,
                    latent_bfchw=context["positive_video"].to(
                        device=self.device, dtype=self.dtype
                    ),
                    seed=int(context["seed"]),
                )
                comparison = stitch_side_by_side(negative, positive)
                output_path = (
                    Path(self.output_path)
                    / "local_logs"
                    / (
                        f"step_{self.step:06d}_"
                        "negative_vs_official_positive.mp4"
                    )
                )
                write_video_pyav(
                    str(output_path),
                    comparison,
                    fps=self.local_video_fps,
                )
                output_path.with_suffix(".txt").write_text(
                    "layout: online_2lr2hr_negative | "
                    "official_refiner_positive\n"
                    f"step: {self.step}\n"
                    f"dpo_timestep: {context['timestep']}\n"
                    f"prompt: {context['prompt']}\n",
                    encoding="utf-8",
                )
                print(
                    f"[Stage4B][local-log] wrote {output_path}",
                    flush=True,
                )
                del negative, positive, comparison
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
            finally:
                self.video_vae.to(device="cpu")
                torch.cuda.empty_cache()

        failed = torch.tensor(
            [1 if failure else 0],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.broadcast(failed, src=0)
        if int(failed.item()) != 0:
            if self.is_main_process:
                raise RuntimeError(
                    f"Stage4B local visualization failed: {failure}"
                )
            raise RuntimeError("Stage4B local visualization failed on rank 0")
        barrier()

    @staticmethod
    def _read_prompt_lines(path: str) -> list[str]:
        prompts: list[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                        prompt = str(payload.get("prompt", "")).strip()
                        if prompt:
                            prompts.append(prompt)
                            continue
                    except json.JSONDecodeError:
                        pass
                prompts.append(line)
        return prompts

    def _init_benchmark_prompts(self) -> None:
        self.benchmark_enabled = bool(
            self.config.get("benchmark_enabled", True)
        )
        self.benchmark_seed = int(
            self.config.get("benchmark_seed", 12345)
        )
        self.benchmark_video_fps = int(
            self.config.get("benchmark_video_fps", 24)
        )
        self.benchmark_audio_sample_rate = int(
            self.config.get("benchmark_audio_sample_rate", 24000)
        )
        self.benchmark_num_prompts = int(
            self.config.get("benchmark_num_prompts", 48)
        )
        self.benchmark_prompts: list[str] = []
        self.benchmark_groups: list[dict[str, Any]] = []
        if not self.benchmark_enabled or self.benchmark_iters <= 0:
            self.benchmark_enabled = False
            return

        prompt_files = self.config.get("benchmark_prompt_files", [])
        prompt_files = (
            [str(prompt_files)]
            if isinstance(prompt_files, str)
            else [str(path) for path in prompt_files]
        )
        limits = self.config.get("benchmark_prompts_per_file")
        if limits is None:
            limits = [0] * len(prompt_files)
        elif isinstance(limits, int):
            limits = [int(limits)] * len(prompt_files)
        else:
            limits = [int(value) for value in limits]
        if len(limits) != len(prompt_files):
            raise ValueError(
                "benchmark_prompts_per_file must match prompt files"
            )
        for path, limit in zip(prompt_files, limits, strict=True):
            current = self._read_prompt_lines(path)
            if limit > 0:
                current = current[:limit]
            self.benchmark_prompts.extend(current)
        self.benchmark_prompts = self.benchmark_prompts[
            : self.benchmark_num_prompts
        ]
        if not self.benchmark_prompts:
            self.benchmark_enabled = False
            return

        for group in self.config.get("benchmark_t2av_groups", []):
            start = max(
                0, min(int(group.get("start", 0)), len(self.benchmark_prompts))
            )
            end = max(
                start,
                min(
                    int(group.get("end", len(self.benchmark_prompts))),
                    len(self.benchmark_prompts),
                ),
            )
            if end > start:
                self.benchmark_groups.append(
                    {
                        "name": str(group.get("name", "t2av")),
                        "start": start,
                        "end": end,
                    }
                )
        if not self.benchmark_groups:
            self.benchmark_groups = [
                {
                    "name": "t2av",
                    "start": 0,
                    "end": len(self.benchmark_prompts),
                }
            ]
        if self.is_main_process:
            print(
                f"[Stage4B][benchmark] loaded "
                f"{len(self.benchmark_prompts)} prompts",
                flush=True,
            )

    def _benchmark_group(self, index: int) -> str:
        for group in self.benchmark_groups:
            if int(group["start"]) <= index < int(group["end"]):
                return str(group["name"])
        return "t2av"

    @torch.no_grad()
    def _run_benchmark(self) -> Dict[str, float]:
        """Generate the same 48-case fixed 2LR+2HR online benchmark."""

        num_cases = len(self.benchmark_prompts)
        rounds = math.ceil(num_cases / self.world_size)
        num_frames = int(
            self.config.get(
                "benchmark_num_frames", self.config.get("num_frames", 121)
            )
        )
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=int(self.config.video_height),
            video_width=int(self.config.video_width),
            batch_size=1,
        )
        audio_shape[1] = int(
            compute_aligned_audio_frames(
                total_video_latent_frames=int(video_shape[1]),
                num_frame_per_block=self.num_frame_per_block,
                first_block_video_frames=self.first_block_video_frames,
            )
        )
        output_root = (
            Path(self.output_path)
            / "benchmark"
            / f"step_{self.step:07d}"
            / "2lr_2hr"
        )
        if self.is_main_process:
            print(
                f"[Stage4B][benchmark] {num_cases} cases; "
                f"LR={self.config.video_height}x{self.config.video_width}, "
                f"HR={int(self.config.video_height) * 2}x"
                f"{int(self.config.video_width) * 2}",
                flush=True,
            )
        barrier()

        self.video_vae.to(device=self.device, dtype=self.dtype)
        self.audio_vae.to(device=self.device, dtype=self.dtype)
        wall_start = time.perf_counter()
        local_generate_seconds = 0.0
        try:
            for round_index in range(rounds):
                position = round_index * self.world_size + self.global_rank
                has_case = position < num_cases
                case_index = position if has_case else 0
                prompt = [self.benchmark_prompts[case_index]]
                conditional_dict = self._detach_conditional(
                    self.text_encoder(text_prompts=prompt)
                )
                seed = self.benchmark_seed + position
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    started = time.perf_counter()
                    rollout = self.objective.sample_scale_wise(
                        video_shape=video_shape,
                        audio_shape=audio_shape,
                        conditional_dict=conditional_dict,
                        model=self.policy,
                    )
                    local_generate_seconds += (
                        time.perf_counter() - started
                    )
                if has_case:
                    group = self._benchmark_group(case_index)
                    output_path = (
                        output_root / group / f"sample_{case_index}.mp4"
                    )
                    frames = decode_video_tiled(
                        video_vae=self.video_vae,
                        latent_bfchw=rollout.hr_video,
                        seed=seed,
                    )
                    waveform = self.audio_vae.decode_to_waveform(
                        rollout.hr_audio
                    )
                    write_video_pyav(
                        str(output_path),
                        frames,
                        fps=self.benchmark_video_fps,
                        audio_array=waveform[0].detach().cpu().float(),
                        audio_fps=self.benchmark_audio_sample_rate,
                        audio_codec="aac",
                    )
                    del frames, waveform
                del rollout, conditional_dict
                torch.cuda.empty_cache()
                barrier()
        finally:
            self.video_vae.to(device="cpu")
            self.audio_vae.to(device="cpu")
            torch.cuda.empty_cache()

        generate_seconds = torch.tensor(
            [local_generate_seconds],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(generate_seconds, op=dist.ReduceOp.SUM)
        wall_seconds = time.perf_counter() - wall_start
        barrier()
        if self.is_main_process:
            print(
                f"[Stage4B][benchmark] wrote {output_root} in "
                f"{wall_seconds:.1f}s",
                flush=True,
            )
        return {
            "benchmark/2lr_2hr/wall_seconds": wall_seconds,
            "benchmark/2lr_2hr/generate_seconds": float(
                generate_seconds.item()
            ),
            "benchmark/2lr_2hr/num_cases": float(num_cases),
        }

    def save(self) -> None:
        """Save deployable online policy plus the frozen upsampler component."""

        if self.is_main_process:
            print("[Stage4B] gathering checkpoint states...", flush=True)
        generator_state = fsdp_state_dict(self.policy)
        upsampler_state = None
        if bool(self.config.get("save_causal_upsampler", True)):
            upsampler_state = fsdp_state_dict(self.causal_upsampler)

        if self.is_main_process:
            checkpoint_dir = (
                Path(self.output_path) / f"checkpoint_{self.step:06d}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state: Dict[str, Any] = {
                "generator": generator_state,
                "step": self.step,
                "training_mode": "stage4_reference_dpo",
                "policy_init_checkpoint": str(
                    self.config.policy_init_checkpoint
                ),
                "reference_checkpoint": str(
                    self.config.get("reference_checkpoint")
                    or self.config.policy_init_checkpoint
                ),
                "causal_upsampler_source": str(
                    self.config.get("causal_upsampler_state_checkpoint")
                    or self.config.causal_upsampler_checkpoint_path
                ),
                "dpo_beta": float(self.config.get("dpo_beta", 1.0)),
                "dpo_loss_weight": float(
                    self.config.get("dpo_loss_weight", 1.0)
                ),
                "dpo_pair_source": str(
                    self.config.get("dpo_pair_source", "online_policy")
                ),
                "positive_video_mix": float(
                    self.config.get("positive_video_mix", 1.0)
                ),
                "positive_audio_source": str(
                    self.config.get("positive_audio_source", "refined")
                ),
                "negative_video_anchor_weight": float(
                    self.config.get("negative_video_anchor_weight", 0.0)
                ),
                "negative_audio_anchor_weight": float(
                    self.config.get("negative_audio_anchor_weight", 0.0)
                ),
            }
            if upsampler_state is not None:
                state["causal_upsampler"] = upsampler_state
            path = checkpoint_dir / "model.pt"
            torch.save(state, path)
            print(f"[Stage4B] checkpoint saved to {path}", flush=True)
        barrier()

    def train(self) -> None:
        if bool(
            self.step == 0
            and self.config.get("benchmark_at_start", False)
            and not self.config.get("no_visualize", False)
            and self.benchmark_enabled
        ):
            baseline_metrics = self._run_benchmark()
            if self.is_main_process:
                wandb.log(baseline_metrics, step=0)

        while self.step < self.max_steps:
            next_step = self.step + 1
            visualize = bool(
                not self.config.get("no_visualize", False)
                and self.local_visualization_iters > 0
                and next_step % self.local_visualization_iters == 0
            )
            started = time.perf_counter()
            metrics, visualization_context = self.train_one_step(
                keep_visualization_context=visualize
            )
            self.step = next_step
            self._update_learning_rate()
            metrics["time/iteration_seconds"] = (
                time.perf_counter() - started
            )
            emergency_stop, emergency_reason = (
                self._update_stability_guard(metrics)
            )

            if emergency_stop:
                if visualization_context is not None:
                    del visualization_context
                if self.is_main_process:
                    wandb.log(metrics, step=self.step)
                    print(
                        f"[Stage4B][step {self.step}] stability guard "
                        f"triggered: {emergency_reason}",
                        flush=True,
                    )
                if not bool(self.config.get("no_save", False)):
                    self.save()
                barrier()
                if self.is_main_process:
                    wandb.finish()
                return

            if visualize:
                # Every rank enters because the policy is FSDP-wrapped around
                # this point in the loop.  Only rank 0 owns/decodes the saved
                # CPU tensors.
                self._run_local_visualization(
                    visualization_context or {}
                )
                del visualization_context
                torch.cuda.empty_cache()

            run_benchmark = bool(
                not self.config.get("no_visualize", False)
                and self.benchmark_enabled
                and self.benchmark_iters > 0
                and self.step % self.benchmark_iters == 0
            )
            if run_benchmark:
                metrics.update(self._run_benchmark())

            if self.is_main_process:
                wandb.log(metrics, step=self.step)
                print(
                    f"[Stage4B][step {self.step}] "
                    f"loss={metrics['dpo/loss']:.6f} "
                    f"adv={metrics['dpo/advantage']:.6f} "
                    f"p={metrics['dpo/preference_probability']:.4f} "
                    f"t={int(metrics['dpo/timestep'])} "
                    f"grad={metrics['grad/policy']:.4f} "
                    f"time={metrics['time/iteration_seconds']:.2f}s",
                    flush=True,
                )

            if (
                not bool(self.config.get("no_save", False))
                and self.checkpoint_iters > 0
                and self.step % self.checkpoint_iters == 0
            ):
                self.save()
                torch.cuda.empty_cache()
            barrier()

        if (
            not bool(self.config.get("no_save", False))
            and (
                self.checkpoint_iters <= 0
                or self.step % self.checkpoint_iters != 0
            )
        ):
            self.save()
        barrier()
        if self.is_main_process:
            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate Stage4B config/imports without allocating models.",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True
    if args.no_visualize:
        config.no_visualize = True
    validate_stage4b_config(config)
    if args.dry_run:
        print(
            f"[Stage4B][dry-run] config validated: {args.config_path}",
            flush=True,
        )
        return
    trainer = Stage4ReferenceDPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
