"""
Stage 2 trainer: Teacher Forcing + DMD distillation.

Pipeline position:
    Stage 0 (TF + Causal Self-Flow, multi-step) → trained AR teacher
    Stage 1 (LTX-2 bidirectional DMD)            → independent track
    Stage 2 (THIS FILE: TF + DMD, 8-step)        → distil AR teacher to few-step
    Stage 3 (Self-Forcing DMD, 4-step)           → close train-inference gap

Design:
    - Generator: causal AR (sourced from Stage 0 ckpt), forward via the
      same TF doubled-sequence path used by ``ar_teacher.LTX2ARTeacher``
      (clean context as TF prefix, noisy 5s clip as the prediction target).
    - real_score / fake_score: bidirectional LTX-2 from the base ckpt,
      handled by ``DMDStage2`` (CausVid-style hybrid).
    - Loss: standard DMD distribution-matching on the generator's single-step
      pred_x0, plus the fake_score critic learning to denoise generator
      outputs. ``dfake_gen_update_ratio`` controls generator/critic
      alternation (default 5).

Notes:
    - Per-batch one sigma sampled from ``denoising_step_list[:-1]``, broadcast
      to all video and audio frames (entire 5s clip uses the same sigma per
      batch item). This matches the few-step inference path's per-block
      shared timestep.
    - Generator pred_x0 is re-computed under ``torch.no_grad()`` for the
      critic step, mirroring Stage 3's dfake_gen_update_ratio discipline.
    - Optional SVI-online augments the TF context and generator input with
      current-student single-step residuals from a CPU replay bank.
"""

import os
import sys
import time
import gc
import copy
import functools
import argparse
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from omegaconf import OmegaConf, DictConfig
import wandb

from ltx_distillation.ar_teacher.ar_teacher import LTX2ARTeacher, ARTeacherConfig
from ltx_distillation.dmd_stage2 import DMDStage2
from ltx_distillation.util import (
    fsdp_state_dict,
    init_logging_folder,
    cycle,
    barrier,
)


# ============================================================================
# Distributed Training Utilities (mirrors stage 0 trainer)
# ============================================================================

def launch_distributed_job():
    timeout_sec = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SEC", "3600"))
    pg_timeout = timedelta(seconds=timeout_sec)
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=pg_timeout)
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    elif torch.cuda.is_available():
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:29500',
            world_size=1,
            rank=0,
            timeout=pg_timeout,
        )
        torch.cuda.set_device(0)
    else:
        raise RuntimeError("CUDA is required for training")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_video_compat(
    filename: str,
    video_array: torch.Tensor,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
    audio_codec: Optional[str] = None,
):
    """Write an mp4 across torchvision versions.

    torchvision 0.27 no longer exposes torchvision.io.write_video. Use it when
    present, otherwise fall back to PyAV with optional AAC audio muxing.
    """
    try:
        from torchvision.io import write_video

        kwargs = {}
        if audio_array is not None:
            kwargs.update(
                audio_array=audio_array,
                audio_fps=audio_fps,
                audio_codec=audio_codec,
            )
        return write_video(filename, video_array, fps=fps, **kwargs)
    except ImportError:
        pass

    import av
    import numpy as np

    frames = video_array.detach().cpu()
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
    frames_np = np.ascontiguousarray(frames.numpy())

    if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
        raise ValueError(f"expected video tensor [T, H, W, 3], got {tuple(frames_np.shape)}")

    height, width = frames_np.shape[1], frames_np.shape[2]
    container = av.open(filename, mode="w")
    try:
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        audio_stream = None
        audio_np = None
        if audio_array is not None:
            if audio_fps is None:
                raise ValueError("audio_fps is required when audio_array is provided")
            audio = audio_array.detach().cpu().float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
                audio = audio.transpose(0, 1)
            elif audio.ndim != 2:
                raise ValueError(
                    f"expected audio tensor [T], [C, T], or [T, C], got {tuple(audio.shape)}"
                )
            if audio.shape[0] > 2:
                audio = audio[:2]
            audio_np = np.ascontiguousarray(audio.clamp(-1, 1).numpy().astype(np.float32))
            layout = "mono" if audio_np.shape[0] == 1 else "stereo"
            audio_stream = container.add_stream(audio_codec or "aac", rate=int(audio_fps))
            audio_stream.layout = layout

        for frame_np in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)

        if audio_stream is not None and audio_np is not None:
            chunk_samples = 1024
            for start in range(0, audio_np.shape[1], chunk_samples):
                audio_chunk = audio_np[:, start : start + chunk_samples]
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_chunk, format="fltp", layout=audio_stream.layout.name,
                )
                audio_frame.sample_rate = int(audio_fps)
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)
    finally:
        container.close()


def get_sharding_strategy(name: str) -> ShardingStrategy:
    strategies = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategies.get(name, ShardingStrategy.FULL_SHARD)


def fsdp_wrap(
    module: torch.nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
    wrap_strategy: str = "size",
    min_num_params: int = 1e8,
    transformer_module: Optional[tuple] = None,
) -> FSDP:
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )

    auto_wrap_policy = None
    if wrap_strategy == "transformer" and transformer_module is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_module),
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(min_num_params),
        )

    return FSDP(
        module,
        sharding_strategy=get_sharding_strategy(sharding_strategy),
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )


# ============================================================================
# Trainer
# ============================================================================

class SVIErrorBank:
    """CPU replay bank for Stage 2 SVI-online residuals.

    The bank stores per-sample residuals keyed by nearest few-step timestep and
    latent sample shape. Tensors live in fp16 on CPU; a small feature vector is
    kept for cheap diversity-preserving replacement.
    """

    def __init__(
        self,
        denoising_step_list: List[int],
        capacity: int = 64,
        min_per_bucket: int = 8,
        replace: str = "l2",
    ):
        self.denoising_steps = [int(t) for t in denoising_step_list]
        self.capacity = int(capacity)
        self.min_per_bucket = int(min_per_bucket)
        self.replace = str(replace)
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[Dict[str, torch.Tensor]]] = {}

    def _nearest_timestep(self, sigma_or_timestep: float) -> int:
        t = float(sigma_or_timestep)
        if t <= 1.0:
            t *= 1000.0
        return min(self.denoising_steps, key=lambda x: abs(x - t))

    @staticmethod
    def _feature(sample: torch.Tensor) -> torch.Tensor:
        sample_det = sample.detach()
        if sample_det.dim() == 4:
            # Video sample [F, C, H, W] -> channel-wise residual signature.
            return sample_det.mean(dim=(0, 2, 3), dtype=torch.float32).cpu()
        if sample_det.dim() == 2:
            # Audio sample [F, C] -> channel-wise residual signature.
            return sample_det.mean(dim=0, dtype=torch.float32).cpu()
        return sample_det.mean(dtype=torch.float32).view(1).cpu()

    def _push_one(self, key: Tuple[int, Tuple[int, ...]], sample: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        bucket = self.buckets.setdefault(key, [])
        entry = {
            "tensor": sample.detach().to(device="cpu", dtype=torch.float16).contiguous(),
            "feature": self._feature(sample),
        }
        if len(bucket) < self.capacity:
            bucket.append(entry)
            return

        if self.replace == "random":
            idx = int(torch.randint(0, len(bucket), (1,)).item())
        else:
            feats = torch.stack([e["feature"].float() for e in bucket])
            dist2 = (feats - entry["feature"].float().view(1, -1)).pow(2).mean(dim=1)
            # Replace the most similar entry to preserve bank diversity.
            idx = int(torch.argmin(dist2).item())
        bucket[idx] = entry

    def push(self, residual: torch.Tensor, sigma_batch: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        residual_det = residual.detach()
        if sigma_batch.dim() == 0:
            sigma_batch = sigma_batch.view(1).expand(residual_det.shape[0])
        sigma_cpu = sigma_batch.detach().float().cpu()
        for i in range(residual_det.shape[0]):
            timestep = self._nearest_timestep(float(sigma_cpu[i].item()))
            sample = residual_det[i]
            key = (timestep, tuple(int(x) for x in sample.shape))
            self._push_one(key, sample)

    def _candidate_keys(
        self,
        timestep: int,
        shape: Tuple[int, ...],
        require_ready: bool,
    ) -> List[Tuple[int, Tuple[int, ...]]]:
        min_size = self.min_per_bucket if require_ready else 1
        exact = (timestep, shape)
        if len(self.buckets.get(exact, [])) >= min_size:
            return [exact]

        same_shape = [
            key for key, bucket in self.buckets.items()
            if key[1] == shape and len(bucket) >= min_size
        ]
        same_shape.sort(key=lambda key: abs(key[0] - timestep))
        return same_shape

    def sample_like(
        self,
        target: torch.Tensor,
        sigma_batch: torch.Tensor,
        require_ready: bool = True,
    ) -> Optional[torch.Tensor]:
        samples = []
        sigma_cpu = sigma_batch.detach().float().cpu()
        for i in range(target.shape[0]):
            timestep = self._nearest_timestep(float(sigma_cpu[i].item()))
            shape = tuple(int(x) for x in target[i].shape)
            keys = self._candidate_keys(timestep, shape, require_ready=require_ready)
            if not keys:
                return None
            bucket = self.buckets[keys[0]]
            idx = int(torch.randint(0, len(bucket), (1,)).item())
            samples.append(bucket[idx]["tensor"])
        return torch.stack(samples).to(device=target.device, dtype=target.dtype)

    def ready_for(self, target: torch.Tensor, sigma_batch: torch.Tensor) -> bool:
        return self.sample_like(target, sigma_batch, require_ready=True) is not None

    def total_size(self) -> int:
        return sum(len(bucket) for bucket in self.buckets.values())

    def ready_bucket_count(self) -> int:
        return sum(1 for bucket in self.buckets.values() if len(bucket) >= self.min_per_bucket)


class Stage2Trainer:
    """Stage 2 TF + DMD trainer.

    Compared to ``ar_teacher.train_ar_teacher_CausalSF.ARTeacherTrainer``:
    - Adds ``DMDStage2`` module (real_score + fake_score, both bidirectional
      from LTX-2 base) and a critic optimizer.
    - Per-step alternation: every step trains the critic; every Nth step
      (N=``dfake_gen_update_ratio``) also trains the generator with the DMD
      loss.
    - No Causal Self-Flow / EMA teacher / projector heads.
    - Generator init checkpoint must come from a Stage 0 CausalSF run.
    """

    # -------------------------------------------------------------- #
    # Initialisation                                                  #
    # -------------------------------------------------------------- #
    def __init__(self, config: DictConfig):
        self.config = config

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0

        # Seed.
        seed = config.get("seed", 42)
        if seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            seed = random_seed.item()
        set_seed(seed + self.global_rank)

        # Logging folder (rank 0 creates, broadcast to others).
        self.output_path = config.output_path
        if self.is_main_process:
            self.output_path, _ = init_logging_folder(config)
        else:
            os.makedirs(self.output_path, exist_ok=True)
        if dist.is_initialized() and self.world_size > 1:
            if self.is_main_process:
                path_bytes = self.output_path.encode('utf-8')
                path_tensor = torch.tensor(
                    list(path_bytes) + [0] * (512 - len(path_bytes)),
                    dtype=torch.uint8, device=self.device,
                )
            else:
                path_tensor = torch.zeros(512, dtype=torch.uint8, device=self.device)
            dist.broadcast(path_tensor, src=0)
            if not self.is_main_process:
                self.output_path = bytes(
                    path_tensor.cpu().tolist()
                ).rstrip(b'\x00').decode('utf-8')
                os.makedirs(self.output_path, exist_ok=True)
        barrier()

        # DMD update ratio: every step trains critic; every Nth step also
        # trains the generator. Same semantics as Stage 3.
        self.dfake_gen_update_ratio = int(config.get("dfake_gen_update_ratio", 5))
        self.svi_enabled = bool(config.get("svi_enabled", False))
        self.svi_clean_prob = float(config.get("svi_clean_prob", 0.3))
        self.svi_p_img = float(config.get("svi_p_img", 0.9))
        self.svi_p_vid = float(config.get("svi_p_vid", 0.9))
        self.svi_img_scale = float(config.get("svi_img_scale", 1.0))
        self.svi_bank = SVIErrorBank(
            denoising_step_list=list(config.get("denoising_step_list", [1000, 0])),
            capacity=int(config.get("svi_bank_capacity", 64)),
            min_per_bucket=int(config.get("svi_bank_min_per_grid", 8)),
            replace=str(config.get("svi_bank_replace", "l2")),
        )
        self.svi_audio_bank = SVIErrorBank(
            denoising_step_list=list(config.get("denoising_step_list", [1000, 0])),
            capacity=int(config.get("svi_bank_capacity", 64)),
            min_per_bucket=int(config.get("svi_bank_min_per_grid", 8)),
            replace=str(config.get("svi_bank_replace", "l2")),
        )

        # ------------------------------------------------------------------ #
        # Generator (causal AR) — same construction path as Stage 0          #
        # ------------------------------------------------------------------ #
        ar_config = ARTeacherConfig(
            checkpoint_path=config.get("checkpoint_path", ""),
            bidirectional_model_checkpoint=config.get("bidirectional_model_checkpoint"),
            causal_model_checkpoint=config.get("causal_model_checkpoint"),
            text_encoder_checkpoint=config.get(
                "text_encoder_checkpoint", config.get("gemma_path", "")
            ),
            num_frame_per_block=config.get("num_frame_per_block", 3),
            first_block_video_frames=config.get("first_block_video_frames", 4),
            gradient_checkpointing=config.gradient_checkpointing,
            mixed_precision=config.mixed_precision,
            num_train_timestep=config.get("num_train_timestep", 1000),
            sigma_shift=config.get("sigma_shift", 1.0),
            sigma_min=config.get("sigma_min", 0.0),
            sigma_max=config.get("sigma_max", 1.0),
            timestep_sampler_type=config.get("timestep_sampler_type", "uniform_shifted"),
            timestep_sampler_std=config.get("timestep_sampler_std", 1.0),
            timestep_sampler_uniform_prob=config.get("timestep_sampler_uniform_prob", 0.1),
            timestep_sampler_eps=config.get("timestep_sampler_eps", 1e-3),
            use_gaussian_weight=config.get("use_gaussian_weight", False),
            noise_augmentation_max_sigma=config.get("noise_augmentation_max_sigma", 0.0),
            video_loss_weight=config.get("video_loss_weight", 1.0),
            audio_loss_weight=config.get("audio_loss_weight", 0.16),
            num_audio_sink_tokens=config.get("num_audio_sink_tokens", 0),
            enable_causal_log_rescale=config.get("enable_causal_log_rescale", False),
            enable_chunk_recycling=config.get("enable_chunk_recycling", False),
            first_frame_conditioning_p=config.get("first_frame_conditioning_p", 0.0),
        )
        self.ar_teacher = LTX2ARTeacher(ar_config, device=self.device)
        self.ar_teacher._load_models()

        # ------------------------------------------------------------------ #
        # Stage 0 generator checkpoint — load BEFORE FSDP wrap                #
        # ------------------------------------------------------------------ #
        # Loading before wrap is the simplest path: plain `.load_state_dict`
        # on the raw module (no FSDP unshard contortions). FSDP wrap then
        # shards the loaded weights and releases the full GPU copy.
        # Order overall:
        #   ar_teacher build  → generator + text_encoder on GPU (full copies)
        #   stage 0 ckpt load → overwrites generator weights, also via GPU
        #   FSDP wrap         → frees full copies, only shards remain
        #   real/fake_score build → wrap → free, one at a time
        generator_ckpt = config.get("generator_ckpt", None)
        skip_generator_ckpt = bool(config.get("debug_skip_stage0_generator_load", False))
        if not generator_ckpt and not skip_generator_ckpt:
            raise ValueError(
                "Stage 2 requires generator_ckpt pointing to a trained "
                "Stage 0 (CausalSF) checkpoint."
            )
        if skip_generator_ckpt:
            if self.is_main_process:
                print("[Stage2][Debug] Skipping Stage 0 generator checkpoint load.")
        else:
            self._load_generator_from_stage0(generator_ckpt)

        # ------------------------------------------------------------------ #
        # FSDP wrap generator + text encoder IMMEDIATELY                      #
        # ------------------------------------------------------------------ #
        # Memory ordering matters: every full-precision 19B model that sits
        # on the GPU before being sharded costs ~38GB. We FSDP-wrap as soon
        # as the model is materialised (and the Stage 0 ckpt is loaded) so
        # the full copy is released and only 1/N shard remains, then move
        # on to the next model.
        sharding_strategy = config.get("sharding_strategy", "hybrid_full")
        gen_wrap_strategy = config.get("generator_fsdp_wrap_strategy", "size")
        gen_transformer_module = None
        if gen_wrap_strategy == "transformer":
            from ltx_causal.transformer.causal_block import CausalAVTransformerBlock
            gen_transformer_module = (CausalAVTransformerBlock,)
        fsdp_gen_kwargs = dict(
            sharding_strategy=sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=gen_wrap_strategy,
            transformer_module=gen_transformer_module,
        )

        if self.is_main_process:
            print("[Stage2] FSDP wrapping generator...")
        self.ar_teacher._generator = fsdp_wrap(
            self.ar_teacher._generator, **fsdp_gen_kwargs
        )
        torch.cuda.empty_cache()
        if self.is_main_process:
            print("[Stage2] Generator FSDP wrapped.")

        if self.is_main_process:
            print("[Stage2] FSDP wrapping text encoder...")
        self.ar_teacher._text_encoder = fsdp_wrap(
            self.ar_teacher._text_encoder,
            sharding_strategy=sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.get("text_encoder_fsdp_wrap_strategy", "size"),
        )
        torch.cuda.empty_cache()
        if self.is_main_process:
            print("[Stage2] Text encoder FSDP wrapped.")

        # ------------------------------------------------------------------ #
        # VAE encoders for online encoding (same as Stage 0). CPU-resident.   #
        # ------------------------------------------------------------------ #
        self._build_vae_encoders()

        # ------------------------------------------------------------------ #
        # DMD module — constructs schedule buffers, no model build yet        #
        # ------------------------------------------------------------------ #
        if self.is_main_process:
            print("[Stage2] Initialising DMDStage2 module...")
        self.dmd = DMDStage2(config, device=self.device)
        if self.is_main_process:
            print("[Stage2] DMDStage2 module initialised.")

        # ------------------------------------------------------------------ #
        # Build score models with independent registries. SingleGPUModelBuilder
        # loads meta modules with assign=True; when the first model is FSDP
        # sharded, reusing the same cached tensors for the second model can
        # leave the second model with zero-sized parameter storage.
        # ------------------------------------------------------------------ #
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        skip_score_weight_load = bool(config.get("debug_skip_score_weight_load", False))
        if self.is_main_process and skip_score_weight_load:
            print("[Stage2][Debug] Building real/fake score models without loading transformer weights.")
        real_score_registry = StateDictRegistry()
        fake_score_registry = StateDictRegistry()

        # ------------------------------------------------------------------ #
        # Build real_score → FSDP wrap → assign to self.dmd                   #
        # ------------------------------------------------------------------ #
        if self.is_main_process:
            print("[Stage2] Building real_score (frozen teacher)...")
        real_score = create_ltx2_wrapper(
            checkpoint_path=config.checkpoint_path,
            gemma_path=config.gemma_path,
            device=self.device,
            dtype=self.dtype,
            video_height=config.get("video_height", 512),
            video_width=config.get("video_width", 768),
            registry=real_score_registry,
            skip_weight_load=skip_score_weight_load,
        )
        real_grad = config.get("real_score_grad", None)
        if real_grad is not None:
            real_score.set_module_grad(real_grad)
        else:
            real_score.requires_grad_(False)

        if self.is_main_process:
            print("[Stage2] FSDP wrapping real_score...")
        self.dmd.real_score = fsdp_wrap(
            real_score,
            sharding_strategy=sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.get("real_score_fsdp_wrap_strategy", "size"),
        )
        del real_score
        real_score_registry.clear()
        torch.cuda.empty_cache()
        if self.is_main_process:
            print("[Stage2] real_score FSDP wrapped.")

        # ------------------------------------------------------------------ #
        # Build fake_score → FSDP wrap → assign to self.dmd                   #
        # ------------------------------------------------------------------ #
        if self.is_main_process:
            print("[Stage2] Building fake_score (trainable critic)...")
        fake_score = create_ltx2_wrapper(
            checkpoint_path=config.checkpoint_path,
            gemma_path=config.gemma_path,
            device=self.device,
            dtype=self.dtype,
            video_height=config.get("video_height", 512),
            video_width=config.get("video_width", 768),
            registry=fake_score_registry,
            skip_weight_load=skip_score_weight_load,
        )
        fake_grad = config.get("fake_score_grad", None)
        if fake_grad is not None:
            fake_score.set_module_grad(fake_grad)
        else:
            fake_score.requires_grad_(True)
        if config.get("gradient_checkpointing", False):
            fake_score.enable_gradient_checkpointing()

        if self.is_main_process:
            print("[Stage2] FSDP wrapping fake_score...")
        self.dmd.fake_score = fsdp_wrap(
            fake_score,
            sharding_strategy=sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.get("fake_score_fsdp_wrap_strategy", "size"),
        )
        del fake_score
        fake_score_registry.clear()
        torch.cuda.empty_cache()
        if self.is_main_process:
            print("[Stage2] fake_score FSDP wrapped.")

        # ------------------------------------------------------------------ #
        # Optimizers                                                          #
        # ------------------------------------------------------------------ #
        gen_lr = float(config.get("generator_lr", config.get("lr", 2e-5)))
        critic_lr = float(config.get("critic_lr", config.get("lr", 2e-5)))

        gen_params = [p for p in self.ar_teacher._generator.parameters() if p.requires_grad]
        self.generator_optimizer = torch.optim.AdamW(
            gen_params,
            lr=gen_lr,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            weight_decay=config.get("weight_decay", 0.01),
        )
        critic_params = [p for p in self.dmd.fake_score.parameters() if p.requires_grad]
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=critic_lr,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            weight_decay=config.get("weight_decay", 0.01),
        )

        self.gen_lr = gen_lr
        self.critic_lr = critic_lr
        self.warmup_steps = int(config.get("warmup_steps", 0))

        # ------------------------------------------------------------------ #
        # Dataloader                                                          #
        # ------------------------------------------------------------------ #
        if self.is_main_process:
            print("[Stage2] Initialising dataloader...")
        self._init_dataloader()
        if self.is_main_process:
            print("[Stage2] Dataloader initialised.")

        # ------------------------------------------------------------------ #
        # Unconditional caching (built lazily on first batch)                 #
        # ------------------------------------------------------------------ #
        self._unconditional_dict_cache: Optional[Dict[str, Any]] = None
        self._negative_prompt = str(config.get(
            "negative_prompt",
            "blurry, out of focus, low quality, artifacts.",
        ))

        # ------------------------------------------------------------------ #
        # Training state                                                      #
        # ------------------------------------------------------------------ #
        self.step = 0
        self.max_grad_norm = float(config.get("max_grad_norm", 10.0))
        self.previous_time = None

        if self.is_main_process:
            effective_batch = config.batch_size * self.world_size
            print(f"[Stage2] Effective batch size = {config.batch_size} x "
                  f"{self.world_size} = {effective_batch}")
            print(f"[Stage2] dfake_gen_update_ratio = {self.dfake_gen_update_ratio}")
            print(f"[Stage2] denoising_step_list = {list(config.denoising_step_list)}")
            print(
                "[Stage2] score sigma samplers: "
                f"DMD={self.dmd.dmd_timestep_sampler_type}"
                f"(shift={self.dmd.dmd_sigma_shift:g}, "
                f"actual_support={self.dmd.score_sigma_support()}), "
                f"fake={self.dmd.fake_score_timestep_sampler_type}"
                f"(shift={self.dmd.fake_score_sigma_shift:g}, "
                f"actual_support={self.dmd.score_sigma_support(fake_score=True)})"
            )
            if self.svi_enabled:
                print(
                    "[Stage2][SVI] enabled "
                    f"clean_prob={self.svi_clean_prob} "
                    f"p_img={self.svi_p_img} p_vid={self.svi_p_vid} "
                    f"scale={self.svi_img_scale} "
                    f"capacity={self.svi_bank.capacity} "
                    f"min_per_bucket={self.svi_bank.min_per_bucket} "
                    f"replace={self.svi_bank.replace}"
                )

        # Benchmark (configured multi-schedule KV-cache T2AV inference, no CFG).
        self._init_benchmark()
        self._init_dmd_local_visualization()

        # Resume support (overrides Stage 0 ckpt loading).
        resume_path = config.get("resume_checkpoint", None)
        if resume_path:
            self._resume_from_checkpoint(resume_path)

    # -------------------------------------------------------------- #
    # Stage 0 generator checkpoint loading                            #
    # -------------------------------------------------------------- #
    def _load_generator_from_stage0(self, ckpt_path: str) -> None:
        if self.is_main_process:
            print(f"[Stage2] Loading Stage 0 generator from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        gen_sd = ckpt.get("generator", ckpt)
        missing, unexpected = self.ar_teacher._generator.load_state_dict(
            gen_sd, strict=False
        )
        real_missing = [k for k in missing if "mask_builder" not in k]
        if self.is_main_process:
            if real_missing:
                print(f"[Stage2]   generator missing ({len(real_missing)}): {real_missing[:10]}")
            if unexpected:
                print(f"[Stage2]   generator unexpected ({len(unexpected)}): {unexpected[:10]}")
        if self.is_main_process:
            print("[Stage2] Stage 0 generator checkpoint loaded.")

    # -------------------------------------------------------------- #
    # VAE encoders (mirrors Stage 0)                                  #
    # -------------------------------------------------------------- #
    def _build_vae_encoders(self):
        from ltx_pipelines.utils.model_ledger import ModelLedger
        from ltx_core.model.audio_vae import (
            AudioEncoderConfigurator,
            AudioProcessor,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder

        config = self.config
        if self.is_main_process:
            print("[Stage2] Building VAE encoders...")

        registry = StateDictRegistry()
        ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=config.checkpoint_path,
            registry=registry,
        )
        self.video_encoder = ledger.video_encoder()
        self.video_encoder.eval()

        audio_enc_builder = Builder(
            model_path=config.checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.audio_encoder = audio_enc_builder.build(
            device=torch.device("cpu"), dtype=self.dtype
        ).eval()
        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins,
            mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
        )
        if self.is_main_process:
            print(f"[Stage2] VAE encoders built "
                  f"(audio sr={self.audio_encoder.sample_rate}, "
                  f"mel_bins={self.audio_encoder.mel_bins})")

    def _encoders_to_device(self):
        self.video_encoder = self.video_encoder.to(device=self.device, dtype=self.dtype)
        self.audio_encoder = self.audio_encoder.to(device=self.device, dtype=self.dtype)
        self.audio_processor = self.audio_processor.to(device=self.device)

    def _encoders_to_cpu(self):
        self.video_encoder = self.video_encoder.to(device="cpu")
        self.audio_encoder = self.audio_encoder.to(device="cpu")
        self.audio_processor = self.audio_processor.to(device="cpu")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_batch(self, batch: dict):
        self._encoders_to_device()

        video = batch["video"].to(device=self.device, dtype=self.dtype)
        audio_wav = batch["audio_waveform"].to(device=self.device, dtype=self.dtype)
        text_prompts = batch["captions"]

        video_latent = self.video_encoder(video)
        video_latent = video_latent.permute(0, 2, 1, 3, 4).contiguous()

        mel = self.audio_processor.waveform_to_mel(
            audio_wav, waveform_sample_rate=self.audio_encoder.sample_rate
        )
        audio_latent = self.audio_encoder(mel)
        B, C_z, T_a, F_mel = audio_latent.shape
        audio_latent = audio_latent.permute(0, 2, 1, 3).reshape(B, T_a, C_z * F_mel)

        self._encoders_to_cpu()
        return video_latent, audio_latent, text_prompts

    def _align_audio(self, clean_video, clean_audio):
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        F_v = clean_video.shape[1]
        expected_a = compute_aligned_audio_frames(
            F_v,
            self.config.get("num_frame_per_block", 3),
            self.config.get("first_block_video_frames", 4),
        )
        actual_a = clean_audio.shape[1]
        if actual_a > expected_a:
            clean_audio = clean_audio[:, :expected_a, :]
        elif actual_a < expected_a:
            pad = expected_a - actual_a
            clean_audio = torch.nn.functional.pad(clean_audio, (0, 0, 0, pad))
        return clean_audio

    # -------------------------------------------------------------- #
    # Dataloader (mirrors Stage 0)                                    #
    # -------------------------------------------------------------- #
    def _init_dataloader(self):
        sys.path.insert(0, str(self._dataset_package_path()))
        from video_audio_dataset import VideoAudioDataset, collate_real_data
        from frame_range_dataset import FrameRangeVideoAudioDataset

        config = self.config

        dataset_type = str(config.get("dataset_type", "frame_range"))
        if dataset_type == "frame_range":
            if int(config.batch_size) != 1:
                raise ValueError(
                    "frame_range uses variable T/spatial shapes; keep batch_size=1."
                )
            default_meta = os.environ.get("DATASET_META_PREFIX", "s3://your-bucket/metadata")
            dataset = FrameRangeVideoAudioDataset(
                data_config_path=config.data_config_path,
                oss_conf_path=getattr(config, "oss_conf_path", None),
                video_height=config.video_height,
                video_width=config.video_width,
                target_fps=getattr(config, "target_fps", 24),
                audio_sample_rate=getattr(config, "audio_sample_rate", 16000),
                require_audio=getattr(config, "require_audio", True),
                require_audio_flag=getattr(config, "require_audio_flag", True),
                caption_priority=config.get("caption_priority", None),
                part_key_priority=config.get("part_key_priority", None),
                prompt_filter_remove_entity_tags=bool(
                    config.get("prompt_filter_remove_entity_tags", False)
                ),
                prompt_filter_remove_speaker_span_tags=bool(
                    config.get("prompt_filter_remove_speaker_span_tags", False)
                ),
                prompt_filter_cleanup_spacing=bool(
                    config.get("prompt_filter_cleanup_spacing", True)
                ),
                expand_tasks=getattr(config, "expand_tasks", False),
                max_entries_per_group=getattr(config, "max_entries_per_group", 0),
                epoch_size=getattr(config, "epoch_size", 0),
                min_frames=getattr(config, "min_frames", 113),
                max_frames=getattr(config, "max_frames", 193),
                timeout_sec=getattr(config, "data_timeout_sec", 120),
                preserve_aspect_ratio=getattr(config, "preserve_aspect_ratio", True),
                spatial_align=getattr(config, "spatial_align", 32),
                min_side=getattr(config, "min_side", 128),
                meta_prefix=getattr(config, "meta_prefix", default_meta),
                seed=getattr(config, "seed", 7182),
                audio_range_mode=getattr(config, "audio_range_mode", "segment"),
            )
        elif dataset_type == "video_audio":
            data_list_paths = config.data_list_paths
            if isinstance(data_list_paths, str):
                data_list_paths = [data_list_paths]

            dataset = VideoAudioDataset(
                data_list_paths=list(data_list_paths),
                oss_conf_path=getattr(config, "oss_conf_path", None),
                num_frames=config.num_frames,
                video_height=config.video_height,
                video_width=config.video_width,
                target_fps=getattr(config, "target_fps", 24),
                audio_sample_rate=getattr(config, "audio_sample_rate", 16000),
                require_audio=getattr(config, "require_audio", True),
                max_entries_per_list=getattr(config, "max_entries_per_list", 0),
                timeout_sec=getattr(config, "data_timeout_sec", 120),
                preserve_aspect_ratio=getattr(config, "preserve_aspect_ratio", False),
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_real_data,
            num_workers=getattr(config, "num_workers", 2),
            pin_memory=True,
        )
        self.dataloader = cycle(dataloader)

    @staticmethod
    def _dataset_package_path():
        return Path(__file__).resolve().parents[3] / "dataset"

    # -------------------------------------------------------------- #
    # LR warmup                                                       #
    # -------------------------------------------------------------- #
    def _lr_scale(self) -> float:
        if self.warmup_steps <= 0 or self.step >= self.warmup_steps:
            return 1.0
        return max(1e-3, self.step / self.warmup_steps)

    def _apply_lr(self) -> None:
        scale = self._lr_scale()
        for pg in self.generator_optimizer.param_groups:
            pg["lr"] = self.gen_lr * scale
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.critic_lr * scale

    # -------------------------------------------------------------- #
    # Conditional / unconditional text embeddings                      #
    # -------------------------------------------------------------- #
    def _get_unconditional_dict(self, batch_size: int) -> Dict[str, Any]:
        """Cached uncond embeddings (same negative prompt, repeated to B)."""
        if (
            self._unconditional_dict_cache is None
            or next(iter(self._unconditional_dict_cache.values())).shape[0] != batch_size
        ):
            with torch.no_grad():
                uncond = self.ar_teacher.text_encoder(
                    text_prompts=[self._negative_prompt] * batch_size,
                )
            self._unconditional_dict_cache = {k: v.detach() for k, v in uncond.items()}
        return self._unconditional_dict_cache

    # -------------------------------------------------------------- #
    # SVI-online helpers                                             #
    # -------------------------------------------------------------- #
    def _svi_make_iteration_inputs(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        sigma_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build optional SVI-corrupted TF context and generator inputs."""
        context_video = clean_video
        context_audio = clean_audio
        generator_video = clean_video
        generator_audio = clean_audio
        log = {
            "enabled": float(self.svi_enabled),
            "force_clean": 1.0,
            "img_injected": 0.0,
            "vid_injected": 0.0,
            "video_bank_size": float(self.svi_bank.total_size()),
            "audio_bank_size": float(getattr(self, "svi_audio_bank", self.svi_bank).total_size())
                if hasattr(self, "svi_audio_bank") else 0.0,
            "video_ready_buckets": float(self.svi_bank.ready_bucket_count()),
            "audio_ready_buckets": float(getattr(self, "svi_audio_bank", self.svi_bank).ready_bucket_count())
                if hasattr(self, "svi_audio_bank") else 0.0,
        }
        if not self.svi_enabled:
            return context_video, context_audio, generator_video, generator_audio, log

        if not hasattr(self, "svi_audio_bank"):
            self.svi_audio_bank = SVIErrorBank(
                denoising_step_list=list(self.config.get("denoising_step_list", [1000, 0])),
                capacity=int(self.config.get("svi_bank_capacity", 64)),
                min_per_bucket=int(self.config.get("svi_bank_min_per_grid", 8)),
                replace=str(self.config.get("svi_bank_replace", "l2")),
            )

        force_clean = bool(torch.rand((), device=clean_video.device).item() < self.svi_clean_prob)
        if dist.is_initialized() and self.world_size > 1:
            flag = torch.tensor([1 if force_clean else 0], device=clean_video.device, dtype=torch.int64)
            dist.broadcast(flag, src=0)
            force_clean = bool(flag.item())
        log["force_clean"] = 1.0 if force_clean else 0.0
        if force_clean:
            return context_video, context_audio, generator_video, generator_audio, log

        do_img = bool(torch.rand((), device=clean_video.device).item() < self.svi_p_img)
        do_vid = bool(torch.rand((), device=clean_video.device).item() < self.svi_p_vid)
        if dist.is_initialized() and self.world_size > 1:
            flags = torch.tensor(
                [1 if do_img else 0, 1 if do_vid else 0],
                device=clean_video.device,
                dtype=torch.int64,
            )
            dist.broadcast(flags, src=0)
            do_img = bool(flags[0].item())
            do_vid = bool(flags[1].item())

        if do_img:
            err_v = self.svi_bank.sample_like(clean_video, sigma_batch, require_ready=True)
            err_a = self.svi_audio_bank.sample_like(clean_audio, sigma_batch, require_ready=True)
            if err_v is not None and err_a is not None:
                context_video = clean_video + self.svi_img_scale * err_v
                context_audio = clean_audio + self.svi_img_scale * err_a
                log["img_injected"] = 1.0

        if do_vid:
            err_v = self.svi_bank.sample_like(clean_video, sigma_batch, require_ready=True)
            err_a = self.svi_audio_bank.sample_like(clean_audio, sigma_batch, require_ready=True)
            if err_v is not None and err_a is not None:
                generator_video = clean_video + self.svi_img_scale * err_v
                generator_audio = clean_audio + self.svi_img_scale * err_a
                log["vid_injected"] = 1.0

        log["video_bank_size"] = float(self.svi_bank.total_size())
        log["audio_bank_size"] = float(self.svi_audio_bank.total_size())
        log["video_ready_buckets"] = float(self.svi_bank.ready_bucket_count())
        log["audio_ready_buckets"] = float(self.svi_audio_bank.ready_bucket_count())
        return context_video, context_audio, generator_video, generator_audio, log

    def _svi_update_bank(
        self,
        pred_x0_video: torch.Tensor,
        pred_x0_audio: torch.Tensor,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        sigma_batch: torch.Tensor,
    ) -> None:
        if not self.svi_enabled:
            return
        if not hasattr(self, "svi_audio_bank"):
            return
        self.svi_bank.push(pred_x0_video.detach() - clean_video.detach(), sigma_batch)
        self.svi_audio_bank.push(pred_x0_audio.detach() - clean_audio.detach(), sigma_batch)

    # -------------------------------------------------------------- #
    # Generator TF forward (returns pred_x0)                          #
    # -------------------------------------------------------------- #
    def _generator_tf_forward(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        context_video: Optional[torch.Tensor] = None,
        context_audio: Optional[torch.Tensor] = None,
        generator_video: Optional[torch.Tensor] = None,
        generator_audio: Optional[torch.Tensor] = None,
    ):
        """Run the TF doubled-sequence forward for the generator.

        Mirrors ``LTX2ARTeacher.generator_loss`` forward portion:
        - context_video / context_audio used as the TF prefix
        - generator_video / generator_audio are noised as the prediction target
        - context sigma fixed at 0 because SVI residuals are treated as clean
        - ``use_causal_timestep=False`` keeps the doubled-seq TF semantics
        """
        B, F_v = clean_video.shape[:2]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        if context_video is None:
            context_video = clean_video
        if context_audio is None:
            context_audio = clean_audio
        if generator_video is None:
            generator_video = clean_video
        if generator_audio is None:
            generator_audio = clean_audio

        noisy_video = self.ar_teacher._add_noise(generator_video, video_sigma)
        noisy_audio = self.ar_teacher._add_noise(generator_audio, audio_sigma)

        clean_video_sigma = torch.zeros(B, F_v, device=device, dtype=video_sigma.dtype)
        clean_audio_sigma = torch.zeros(B, F_a, device=device, dtype=audio_sigma.dtype)

        pred_x0_video, pred_x0_audio = self.ar_teacher.generator(
            _tf_kwargs=dict(
                noisy_video=noisy_video,
                clean_video=context_video,
                conditional_dict=conditional_dict,
                noisy_timestep=video_sigma.to(self.dtype),
                clean_timestep=clean_video_sigma.to(self.dtype),
                noisy_audio=noisy_audio,
                clean_audio=context_audio,
                noisy_audio_timestep=audio_sigma.to(self.dtype),
                clean_audio_timestep=clean_audio_sigma.to(self.dtype),
                use_causal_timestep=False,
            )
        )
        return pred_x0_video, pred_x0_audio

    # -------------------------------------------------------------- #
    # Training step                                                    #
    # -------------------------------------------------------------- #
    def train_one_step(self):
        self.ar_teacher._generator.train()
        self.dmd.fake_score.train()
        self.dmd.real_score.eval()
        self._apply_lr()
        self.dmd.current_step = self.step

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        train_generator_this_step = (self.step % self.dfake_gen_update_ratio == 0)
        run_local_visualization = self._should_run_dmd_local_visualization(
            train_generator=train_generator_this_step
        )

        # ------------------------------------------------------------------ #
        # Data preparation                                                    #
        # ------------------------------------------------------------------ #
        batch = next(self.dataloader)
        clean_video, clean_audio, text_prompts = self._encode_batch(batch)
        clean_audio = self._align_audio(clean_video, clean_audio)

        with torch.no_grad():
            conditional_dict = self.ar_teacher.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self._get_unconditional_dict(batch_size=len(text_prompts))

        # ------------------------------------------------------------------ #
        # Sigma for generator forward                                         #
        # Per-batch single sigma sampled from denoising_step_list[:-1],       #
        # broadcast to all video & audio frames (whole 5s shares the sigma).  #
        # Rank-0 sample + broadcast for FSDP collective sync.                 #
        # ------------------------------------------------------------------ #
        sigma_batch = self.dmd.sample_few_step_sigma(batch_size=clean_video.shape[0])
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(sigma_batch, src=0)
        video_sigma, audio_sigma = self.dmd.expand_sigma_to_frames(
            sigma_batch,
            num_video_frames=clean_video.shape[1],
            num_audio_frames=clean_audio.shape[1],
        )
        (
            context_video,
            context_audio,
            generator_video,
            generator_audio,
            svi_log,
        ) = self._svi_make_iteration_inputs(
            clean_video=clean_video,
            clean_audio=clean_audio,
            sigma_batch=sigma_batch,
        )

        gen_log: Dict[str, Any] = {}
        critic_log: Dict[str, Any] = {}
        gen_loss_value = 0.0
        gen_grad_norm = 0.0
        local_visualization_context: Optional[Dict[str, Any]] = None

        # ------------------------------------------------------------------ #
        # Generator step (every Nth step)                                    #
        # ------------------------------------------------------------------ #
        if train_generator_this_step:
            # SenseFlow-style requires_grad toggle (see senseflow
            # trainer_flux_senseflow.py L891): turn OFF fake_score grad
            # during the generator step so FSDP treats fake_score as fully
            # frozen — no backward hooks, no unshard state-machine issues
            # when its forward is invoked under torch.no_grad() inside
            # _compute_kl_grad. real_score is already frozen; toggling it
            # again here is a no-op but kept explicit for clarity.
            self.dmd.fake_score.requires_grad_(False)
            self.dmd.real_score.requires_grad_(False)

            self.generator_optimizer.zero_grad()

            pred_x0_video, pred_x0_audio = self._generator_tf_forward(
                clean_video=clean_video,
                clean_audio=clean_audio,
                conditional_dict=conditional_dict,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                context_video=context_video,
                context_audio=context_audio,
                generator_video=generator_video,
                generator_audio=generator_audio,
            )

            gen_loss, gen_log = self.dmd.compute_distribution_matching_loss(
                pred_x0_video=pred_x0_video,
                pred_x0_audio=pred_x0_audio,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=None,
                audio_loss_mask=None,
                return_video_predictions=(
                    run_local_visualization and self.is_main_process
                ),
            )
            if run_local_visualization and self.is_main_process:
                local_visualization_context = (
                    self._capture_dmd_local_visualization(
                        generator_video=pred_x0_video,
                        generator_log=gen_log,
                        generator_sigma_mean=float(
                            sigma_batch.float().mean().item()
                        ),
                        prompt=text_prompts[0],
                        step=self.step,
                    )
                )
            gen_loss.backward()

            if hasattr(self.ar_teacher._generator, "clip_grad_norm_"):
                gen_grad_norm = self.ar_teacher._generator.clip_grad_norm_(self.max_grad_norm)
            else:
                gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ar_teacher._generator.parameters(), self.max_grad_norm
                )
            self.generator_optimizer.step()
            # Defensive: clear any spurious grad accumulation on critic
            # before re-enabling its grad below.
            self.critic_optimizer.zero_grad()
            self._svi_update_bank(
                pred_x0_video=pred_x0_video,
                pred_x0_audio=pred_x0_audio,
                clean_video=clean_video,
                clean_audio=clean_audio,
                sigma_batch=sigma_batch,
            )
            gen_loss_value = gen_loss.item()
            del pred_x0_video, pred_x0_audio, gen_loss
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------ #
        # Critic step (every step)                                           #
        # Re-runs generator under no_grad to get fresh detached pred_x0.     #
        # ------------------------------------------------------------------ #
        # Restore fake_score grad so the critic backward populates it.
        # real_score stays frozen.
        self.dmd.fake_score.requires_grad_(True)
        self.dmd.real_score.requires_grad_(False)

        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            pred_x0_video_det, pred_x0_audio_det = self._generator_tf_forward(
                clean_video=clean_video,
                clean_audio=clean_audio,
                conditional_dict=conditional_dict,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                context_video=context_video,
                context_audio=context_audio,
                generator_video=generator_video,
                generator_audio=generator_audio,
            )
            pred_x0_video_det = pred_x0_video_det.detach()
            pred_x0_audio_det = pred_x0_audio_det.detach()

        critic_loss, critic_log = self.dmd.critic_loss(
            pred_x0_video_det=pred_x0_video_det,
            pred_x0_audio_det=pred_x0_audio_det,
            conditional_dict=conditional_dict,
            video_loss_mask=None,
            audio_loss_mask=None,
        )
        critic_loss.backward()

        if hasattr(self.dmd.fake_score, "clip_grad_norm_"):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(self.max_grad_norm)
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        critic_loss_value = critic_loss.item()
        del pred_x0_video_det, pred_x0_audio_det, critic_loss

        # Decode only after both optimizer phases. Visualization tensors were
        # detached to CPU before generator backward, so no training graph or
        # peak-memory score tensor is retained until this point.
        if run_local_visualization:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self._run_dmd_local_visualization(local_visualization_context)
            del local_visualization_context
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------ #
        # Logging                                                             #
        # ------------------------------------------------------------------ #
        if self.is_main_process:
            log_data = {
                "train/critic_loss": critic_loss_value,
                "train/critic_video_loss": critic_log.get("critic_video_loss", 0.0),
                "train/critic_audio_loss": critic_log.get("critic_audio_loss", 0.0),
                "train/critic_sigma_mean": critic_log.get("critic_sigma_mean", 0.0),
                "train/critic_grad_norm": float(critic_grad_norm)
                    if not isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm.item(),
                "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                "train/sigma_mean": float(sigma_batch.float().mean().item()),
                "train/generator_sigma_mean": float(
                    sigma_batch.float().mean().item()
                ),
                "train/svi_enabled": svi_log["enabled"],
                "train/svi_force_clean": svi_log["force_clean"],
                "train/svi_img_injected": svi_log["img_injected"],
                "train/svi_vid_injected": svi_log["vid_injected"],
                "train/svi_video_bank_size": float(self.svi_bank.total_size()),
                "train/svi_audio_bank_size": float(self.svi_audio_bank.total_size()),
                "train/svi_video_ready_buckets": float(self.svi_bank.ready_bucket_count()),
                "train/svi_audio_ready_buckets": float(self.svi_audio_bank.ready_bucket_count()),
            }
            if train_generator_this_step:
                log_data.update({
                    "train/gen_loss": gen_loss_value,
                    "train/gen_video_dmd_loss": float(gen_log.get("video_dmd_loss", torch.tensor(0.0)).item()
                        if isinstance(gen_log.get("video_dmd_loss"), torch.Tensor) else gen_log.get("video_dmd_loss", 0.0)),
                    "train/gen_audio_dmd_loss": float(gen_log.get("audio_dmd_loss", torch.tensor(0.0)).item()
                        if isinstance(gen_log.get("audio_dmd_loss"), torch.Tensor) else gen_log.get("audio_dmd_loss", 0.0)),
                    "train/gen_grad_norm": float(gen_grad_norm)
                        if not isinstance(gen_grad_norm, torch.Tensor) else gen_grad_norm.item(),
                    "train/gen_lr": self.generator_optimizer.param_groups[0]["lr"],
                    "train/dmdtrain_gradient_norm_video": gen_log.get("dmdtrain_gradient_norm_video", 0.0),
                    "train/dmdtrain_gradient_norm_audio": gen_log.get("dmdtrain_gradient_norm_audio", 0.0),
                    # Standardized aliases shared by the AR-fake ablation
                    # trainers, so all three rows can be plotted together.
                    "train/generator_loss": gen_loss_value,
                    "train/generator_video_dmd_loss": float(
                        gen_log.get("video_dmd_loss", torch.tensor(0.0)).item()
                        if isinstance(gen_log.get("video_dmd_loss"), torch.Tensor)
                        else gen_log.get("video_dmd_loss", 0.0)
                    ),
                    "train/generator_audio_dmd_loss": float(
                        gen_log.get("audio_dmd_loss", torch.tensor(0.0)).item()
                        if isinstance(gen_log.get("audio_dmd_loss"), torch.Tensor)
                        else gen_log.get("audio_dmd_loss", 0.0)
                    ),
                    "train/generator_grad_norm": float(gen_grad_norm)
                        if not isinstance(gen_grad_norm, torch.Tensor)
                        else gen_grad_norm.item(),
                    "train/generator_lr": self.generator_optimizer.param_groups[0]["lr"],
                    "train/dmd_gradient_norm_video": gen_log.get(
                        "dmdtrain_gradient_norm_video", 0.0
                    ),
                    "train/dmd_gradient_norm_audio": gen_log.get(
                        "dmdtrain_gradient_norm_audio", 0.0
                    ),
                    "train/dmd_sigma_mean": gen_log.get(
                        "alignment/video_sigma_mean", 0.0
                    ),
                    "train/real_video_cfg": gen_log.get(
                        "real_video_guidance_scale",
                        self.dmd.real_video_guidance_scale,
                    ),
                })
            wandb.log(log_data, step=self.step)

            if self.step < 10 or self.step % 50 == 0:
                if train_generator_this_step:
                    print(
                        f"[Step {self.step}] G+C | "
                        f"gen={gen_loss_value:.4f} "
                        f"critic={critic_loss_value:.4f} "
                        f"sigma={float(sigma_batch.float().mean().item()):.3f} "
                        f"svi=img{int(svi_log['img_injected'])}/vid{int(svi_log['vid_injected'])} "
                        f"bank={self.svi_bank.total_size()}/{self.svi_audio_bank.total_size()}"
                    )
                else:
                    print(
                        f"[Step {self.step}] C   | "
                        f"critic={critic_loss_value:.4f} "
                        f"sigma={float(sigma_batch.float().mean().item()):.3f} "
                        f"svi=img{int(svi_log['img_injected'])}/vid{int(svi_log['vid_injected'])} "
                        f"bank={self.svi_bank.total_size()}/{self.svi_audio_bank.total_size()}"
                    )

        del clean_video, clean_audio, conditional_dict
        del context_video, context_audio, generator_video, generator_audio
        torch.cuda.empty_cache()

    # -------------------------------------------------------------- #
    # Benchmark (multi-schedule KV-cache T2AV inference, no CFG)      #
    # -------------------------------------------------------------- #
    def _init_benchmark(self):
        config = self.config
        self.benchmark_enabled = bool(config.get("benchmark_enabled", False))
        self.benchmark_sampler = str(
            config.get("benchmark_sampler", "euler")
        ).lower()
        if self.benchmark_sampler not in {"euler", "consistency"}:
            raise ValueError(
                "benchmark_sampler must be 'euler' or 'consistency', got "
                f"{self.benchmark_sampler!r}"
            )
        from ltx_distillation.tf_dcm_utils import validate_benchmark_schedules

        raw_benchmark_schedules = config.get("benchmark_denoising_step_lists", None)
        if raw_benchmark_schedules is None:
            raw_benchmark_schedules = [config.get("denoising_step_list")]
        self.benchmark_denoising_step_lists = validate_benchmark_schedules(
            raw_benchmark_schedules,
            num_train_timestep=int(config.get("num_train_timestep", 1000)),
        )
        self.benchmark_iters = int(config.get("benchmark_iters", config.get("log_iters", 1000)))
        self.benchmark_seed = int(config.get("benchmark_seed", 12345))
        self.benchmark_num_prompts = int(config.get("benchmark_num_prompts", 16))
        self.benchmark_video_fps = int(config.get("benchmark_video_fps", 24))
        self.benchmark_audio_sample_rate = int(config.get("benchmark_audio_sample_rate", 24000))
        self.benchmark_clear_cuda_cache_per_round = bool(
            config.get("benchmark_clear_cuda_cache_per_round", False)
        )
        self.benchmark_prompts: list = []
        self.benchmark_t2av_groups = []

        if self.benchmark_iters <= 0:
            self.benchmark_enabled = False
        if not self.benchmark_enabled:
            if self.is_main_process:
                print("[Benchmark] Disabled.")
            return

        prompt_files = config.get("benchmark_prompt_files", None)
        if prompt_files is None:
            prompt_file = config.get("benchmark_prompt_file")
            prompt_files = [prompt_file] if prompt_file else []
        elif isinstance(prompt_files, str):
            prompt_files = [prompt_files]

        if prompt_files:
            try:
                all_prompts = []
                prompts_per_file = config.get("benchmark_prompts_per_file", None)
                if prompts_per_file is not None:
                    if isinstance(prompts_per_file, int):
                        prompts_per_file = [prompts_per_file] * len(prompt_files)
                    else:
                        prompts_per_file = list(prompts_per_file)
                    if len(prompts_per_file) != len(prompt_files):
                        raise ValueError(
                            "benchmark_prompts_per_file must have the same length "
                            "as benchmark_prompt_files"
                        )

                file_counts = []
                for file_idx, prompt_file in enumerate(prompt_files):
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        file_prompts = [l.strip() for l in f if l.strip()]
                    if prompts_per_file is not None:
                        limit = int(prompts_per_file[file_idx])
                        if limit > 0:
                            file_prompts = file_prompts[:limit]
                    file_counts.append((prompt_file, len(file_prompts)))
                    all_prompts.extend(file_prompts)
                self.benchmark_prompts = all_prompts[: self.benchmark_num_prompts]
                if self.is_main_process:
                    print(
                        f"[Benchmark] Loaded {len(self.benchmark_prompts)} T2AV prompts "
                        f"from {list(prompt_files)}"
                    )
                    for prompt_file, count in file_counts:
                        print(f"  {prompt_file}: {count} prompt(s)")
                    for i, p in enumerate(self.benchmark_prompts):
                        print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")
            except Exception as e:
                if self.is_main_process:
                    print(f"[Benchmark] Failed to load T2AV prompts: {e}")

        if not self.benchmark_prompts:
            if self.is_main_process:
                print("[Benchmark] No prompts loaded, benchmark disabled.")
            self.benchmark_enabled = False
            return

        t2av_groups = config.get("benchmark_t2av_groups", [])
        if t2av_groups:
            for group in t2av_groups:
                name = str(group.get("name", "t2av"))
                start = int(group.get("start", 0))
                end = int(group.get("end", len(self.benchmark_prompts)))
                start = max(0, min(start, len(self.benchmark_prompts)))
                end = max(start, min(end, len(self.benchmark_prompts)))
                if end > start:
                    self.benchmark_t2av_groups.append({
                        "name": name,
                        "start": start,
                        "end": end,
                    })
        else:
            self.benchmark_t2av_groups = [{
                "name": "t2av",
                "start": 0,
                "end": len(self.benchmark_prompts),
            }]

        if self.is_main_process:
            print("[Benchmark] T2AV groups:")
            for group in self.benchmark_t2av_groups:
                print(f"  {group['name']}: [{group['start']}, {group['end']})")
            print(
                "[Benchmark] schedules="
                f"{[len(schedule) - 1 for schedule in self.benchmark_denoising_step_lists]} "
                f"sampler={self.benchmark_sampler}"
            )

        self._build_vae_decoders()

    def _benchmark_t2av_group_name(self, prompt_idx: int) -> str:
        for group in self.benchmark_t2av_groups:
            if group["start"] <= prompt_idx < group["end"]:
                return group["name"]
        return "t2av"

    def _benchmark_num_frames_for_prompt(self, prompt: str) -> int:
        default_frames = int(self.config.get("benchmark_num_frames", self.config.get("num_frames", 121)))
        if not bool(self.config.get("benchmark_duration_from_prompt_timestamps", True)):
            return default_frames

        timestamp_values = [
            float(value)
            for pair in re.findall(r"\[(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s\]", prompt)
            for value in pair
        ]
        if not timestamp_values:
            return default_frames

        fps = float(self.config.get("benchmark_video_fps", 24))
        raw_frames = max(1, int(math.ceil(max(timestamp_values) * fps)))
        num_frames = raw_frames + ((8 - ((raw_frames - 1) % 8)) % 8)

        min_frames = int(self.config.get("benchmark_min_frames", default_frames))
        max_frames = self.config.get("benchmark_max_frames", None)
        num_frames = max(num_frames, min_frames)
        if max_frames is not None:
            num_frames = min(num_frames, int(max_frames))
            num_frames = num_frames - ((num_frames - 1) % 8)
        return int(num_frames)

    def _build_vae_decoders(self):
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        if self.is_main_process:
            print("[Benchmark] Building VAE decoders...")
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=self.config.checkpoint_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)
        if self.is_main_process:
            print("[Benchmark] VAE decoders built (on CPU).")

    def _vae_decoders_to_device(self):
        if hasattr(self, "video_vae"):
            self.video_vae = self.video_vae.to(device=self.device)
        if hasattr(self, "audio_vae"):
            self.audio_vae = self.audio_vae.to(device=self.device)

    def _vae_decoders_to_cpu(self):
        if hasattr(self, "video_vae"):
            self.video_vae = self.video_vae.to(device="cpu")
        if hasattr(self, "audio_vae"):
            self.audio_vae = self.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    # -------------------------------------------------------------- #
    # Local DMD visualization (generator x0 | real x0 | fake x0)     #
    # -------------------------------------------------------------- #
    def _init_dmd_local_visualization(self) -> None:
        configured_mode = str(
            self.config.get("training_mode", "causal_tf_dmd")
        )
        if configured_mode not in {"causal_tf_dmd", "hybrid_tf_dmd"}:
            self.local_visualization_iters = 0
            return
        self.local_visualization_iters = int(
            self.config.get("local_visualization_iters", 0)
        )
        self.local_video_fps = int(
            self.config.get(
                "local_video_fps",
                self.config.get("benchmark_video_fps", 24),
            )
        )
        if (
            self.local_visualization_iters > 0
            and self.is_main_process
            and not hasattr(self, "video_vae")
        ):
            self._build_vae_decoders()
        if self.is_main_process and self.local_visualization_iters > 0:
            print(
                "[Stage2 DMD][local-log] enabled | "
                f"every={self.local_visualization_iters} "
                "layout=generator_x0|cfg_real_score_x0|fake_score_x0",
                flush=True,
            )

    def _should_run_dmd_local_visualization(
        self,
        *,
        train_generator: bool,
    ) -> bool:
        return bool(
            train_generator
            and getattr(self, "local_visualization_iters", 0) > 0
            and self.step > 0
            and self.step % self.local_visualization_iters == 0
            and not bool(self.config.get("no_visualize", False))
        )

    @staticmethod
    def _capture_dmd_local_visualization(
        *,
        generator_video: torch.Tensor,
        generator_log: Dict[str, Any],
        generator_sigma_mean: float,
        prompt: str,
        step: int,
    ) -> Dict[str, Any]:
        pred_real_video = generator_log.pop("_pred_real_video_x0")
        pred_fake_video = generator_log.pop("_pred_fake_video_x0")
        return {
            "generator_video": generator_video[:1].detach().cpu(),
            "real_score_video_x0": pred_real_video[:1].detach().cpu(),
            "fake_score_video_x0": pred_fake_video[:1].detach().cpu(),
            "generator_sigma_mean": float(generator_sigma_mean),
            "dmd_sigma_mean": float(generator_log["alignment/video_sigma_mean"]),
            "real_video_guidance_scale": float(
                generator_log["real_video_guidance_scale"]
            ),
            "prompt": str(prompt),
            "step": int(step),
        }

    @torch.no_grad()
    def _run_dmd_local_visualization(
        self,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Decode and save generator | CFG real-score | fake-score with PyAV."""
        from ltx_distillation.stage35_media import (
            decode_video_tiled,
            write_video_pyav,
        )

        failure = ""
        if self.is_main_process:
            if context is None:
                failure = "rank 0 did not capture DMD visualization tensors"
            elif not hasattr(self, "video_vae"):
                failure = "DMD local visualization has no video VAE"
            else:
                try:
                    self.video_vae.to(device=self.device, dtype=self.dtype)
                    seed = int(self.config.get("seed", 42)) + int(context["step"])
                    decoded = []
                    for key in (
                        "generator_video",
                        "real_score_video_x0",
                        "fake_score_video_x0",
                    ):
                        latent = context[key].to(
                            device=self.device,
                            dtype=self.dtype,
                        )
                        decoded.append(
                            decode_video_tiled(
                                video_vae=self.video_vae,
                                latent_bfchw=latent,
                                seed=seed,
                            )
                        )
                        del latent
                        torch.cuda.empty_cache()

                    frames = min(int(item.shape[0]) for item in decoded)
                    height = min(int(item.shape[1]) for item in decoded)
                    comparison = torch.cat(
                        [item[:frames, :height] for item in decoded],
                        dim=2,
                    )
                    log_dir = Path(self.output_path) / "local_logs"
                    output_path = log_dir / (
                        f"step_{int(context['step']):06d}_generator_real_fake.mp4"
                    )
                    write_video_pyav(
                        str(output_path),
                        comparison,
                        fps=self.local_video_fps,
                    )
                    metadata_path = output_path.with_suffix(".txt")
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_text(
                        "layout: clean_tf_generator_x0 | "
                        "cfg_real_score_pred_x0 | fake_score_pred_x0\n"
                        f"score_ablation_setting: "
                        f"{self.config.get('score_ablation_setting', 'unknown')}\n"
                        f"step: {int(context['step'])}\n"
                        f"generator_sigma_mean: "
                        f"{float(context['generator_sigma_mean']):.8f}\n"
                        f"dmd_sigma_mean: "
                        f"{float(context['dmd_sigma_mean']):.8f}\n"
                        f"real_video_guidance_scale: "
                        f"{float(context['real_video_guidance_scale']):.8f}\n"
                        f"prompt: {context['prompt']}\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[Stage2 DMD][local-log] wrote {output_path}",
                        flush=True,
                    )
                    del decoded, comparison
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
                    f"Stage-2 DMD local visualization failed: {failure}"
                )
            raise RuntimeError(
                "Stage-2 DMD local visualization failed on rank 0"
            )

    @staticmethod
    def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    def _init_kv_caches_for_conditional(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        batch_size: int,
        total_video_frames: int,
        total_audio_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        video_frame_seqlen: Optional[int] = None,
    ):
        wrapper = self._unwrap_module(self.ar_teacher._generator)
        init_fn = getattr(wrapper, "init_av_kv_caches", None)
        if init_fn is None:
            raise RuntimeError(
                "Generator does not expose init_av_kv_caches for KV-cache benchmark."
            )
        text_ctx = conditional_dict.get("video_context")
        if text_ctx is None:
            text_ctx = conditional_dict.get("context")
        if text_ctx is None:
            raise ValueError("conditional_dict must provide video_context or context.")
        text_seq_len = int(text_ctx.shape[1])
        return init_fn(
            batch_size=batch_size,
            num_video_frames=total_video_frames,
            num_audio_frames=total_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
            video_frame_seqlen=video_frame_seqlen,
        )

    def _refresh_kv_cache_block(
        self,
        current_video: torch.Tensor,
        current_audio: Optional[torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor],
        kv_caches,
        block,
    ) -> None:
        """Refresh per-block KV cache with the just-denoised clean block (sigma=0).

        Stage 2 has no `context_noise` knob — the benchmark always refreshes
        with the clean predicted x0, matching how Stage 2 trained the
        generator under perfectly-clean TF context.
        """
        B = current_video.shape[0]
        video_ctx_sigma = torch.zeros(
            (B, current_video.shape[1]),
            device=current_video.device,
            dtype=self.dtype,
        )
        audio_ctx_sigma = None
        if current_audio is not None:
            audio_ctx_sigma = torch.zeros(
                (B, current_audio.shape[1]),
                device=current_audio.device,
                dtype=self.dtype,
            )

        self.ar_teacher._generator(
            _kv_cache_kwargs=dict(
                noisy_video_block=current_video.detach(),
                noisy_audio_block=current_audio.detach() if current_audio is not None else None,
                conditional_dict=conditional_dict,
                timestep_block=video_ctx_sigma,
                audio_timestep_block=audio_ctx_sigma,
                kv_caches=kv_caches,
                current_video_start_frame=block.video_start,
                current_audio_start_frame=block.audio_start,
                ref_video_block=None,
            )
        )

    @torch.no_grad()
    def _generate_causal_kv_cache(
        self,
        pos_conditional_dict: Dict[str, torch.Tensor],
        video_shape: tuple,
        audio_shape: tuple,
        seed: int,
        denoising_sigmas: torch.Tensor,
        pad_to_video_shape: Optional[tuple] = None,
        pad_to_audio_shape: Optional[tuple] = None,
    ):
        """Configured-schedule KV-cache blockwise inference, no CFG.

        Each block uses the supplied benchmark sigmas with the configured
        Euler or consistency update and a shared KV cache; cache is refreshed
        with the clean predicted x0 at the end of each block.
        """
        from ltx_causal.attention.mask_builder import compute_av_blocks

        device = self.device
        dtype = torch.bfloat16
        B = video_shape[0]
        F_v = video_shape[1]
        rollout_video_shape = pad_to_video_shape or video_shape
        rollout_audio_shape = pad_to_audio_shape or audio_shape
        rollout_F_v = rollout_video_shape[1]

        sigmas = denoising_sigmas.to(device=device, dtype=torch.float32)

        blocks = compute_av_blocks(
            total_video_latent_frames=F_v,
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            first_block_video_frames=self.config.get("first_block_video_frames", 4),
        )
        rollout_blocks = compute_av_blocks(
            total_video_latent_frames=rollout_F_v,
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            first_block_video_frames=self.config.get("first_block_video_frames", 4),
        )

        total_audio_frames = rollout_audio_shape[1] if rollout_audio_shape is not None else 0
        H_lat, W_lat = rollout_video_shape[-2], rollout_video_shape[-1]
        frame_seqlen = int(H_lat) * int(W_lat)
        kv_pos = self._init_kv_caches_for_conditional(
            conditional_dict=pos_conditional_dict,
            batch_size=B,
            total_video_frames=rollout_F_v,
            total_audio_frames=total_audio_frames,
            device=device,
            dtype=dtype,
            video_frame_seqlen=frame_seqlen,
        )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = torch.zeros(audio_shape, device=device, dtype=dtype) if audio_shape else None

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            for block_idx, rollout_block in enumerate(rollout_blocks):
                has_real_block = block_idx < len(blocks)
                block = blocks[block_idx] if has_real_block else rollout_block
                cur_video = torch.randn(
                    (B, block.video_frames, *video_shape[2:]),
                    device=device, dtype=dtype,
                )
                cur_audio = None
                if audio is not None and block.audio_frames > 0:
                    cur_audio = torch.randn(
                        (B, block.audio_frames, audio_shape[2]),
                        device=device, dtype=dtype,
                    )

                # Denoise over sigmas[0..-2]; sigmas[-1] is the clean endpoint.
                # Consistency sampling re-noises each x0 prediction with fresh
                # Gaussian noise at the next sigma.
                for step_idx, sigma in enumerate(sigmas[:-1]):
                    video_sigma = sigma.expand(B, cur_video.shape[1]).to(self.dtype)
                    audio_sigma = (
                        sigma.expand(B, cur_audio.shape[1]).to(self.dtype)
                        if cur_audio is not None else None
                    )

                    x0_v, x0_a = self.ar_teacher._generator(
                        _kv_cache_kwargs=dict(
                            noisy_video_block=cur_video,
                            noisy_audio_block=cur_audio,
                            conditional_dict=pos_conditional_dict,
                            timestep_block=video_sigma,
                            audio_timestep_block=audio_sigma,
                            kv_caches=kv_pos,
                            current_video_start_frame=block.video_start,
                            current_audio_start_frame=block.audio_start,
                            ref_video_block=None,
                        )
                    )

                    sigma_next = sigmas[step_idx + 1]
                    if sigma_next > 0:
                        if self.benchmark_sampler == "consistency":
                            next_video_sigma = sigma_next.expand(
                                B, x0_v.shape[1]
                            ).to(self.dtype)
                            cur_video = self.ar_teacher._add_noise(
                                x0_v,
                                next_video_sigma,
                                noise=torch.randn_like(x0_v),
                            )
                            if cur_audio is not None and x0_a is not None:
                                next_audio_sigma = sigma_next.expand(
                                    B, x0_a.shape[1]
                                ).to(self.dtype)
                                cur_audio = self.ar_teacher._add_noise(
                                    x0_a,
                                    next_audio_sigma,
                                    noise=torch.randn_like(x0_a),
                                )
                        else:
                            dt = sigma_next - sigma
                            cur_video = (
                                cur_video.float()
                                + (cur_video.float() - x0_v.float())
                                / sigma.float()
                                * dt.float()
                            ).to(dtype)
                            if cur_audio is not None and x0_a is not None:
                                cur_audio = (
                                    cur_audio.float()
                                    + (cur_audio.float() - x0_a.float())
                                    / sigma.float()
                                    * dt.float()
                                ).to(dtype)
                    else:
                        cur_video = x0_v
                        if cur_audio is not None and x0_a is not None:
                            cur_audio = x0_a
                    del x0_v, x0_a

                if has_real_block:
                    video[:, block.video_start:block.video_end] = cur_video
                if has_real_block and audio is not None and cur_audio is not None:
                    audio[:, block.audio_start:block.audio_end] = cur_audio

                # Refresh KV cache with the clean predicted x0 for this block.
                self._refresh_kv_cache_block(
                    current_video=cur_video,
                    current_audio=cur_audio,
                    conditional_dict=pos_conditional_dict,
                    kv_caches=kv_pos,
                    block=block,
                )

                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

        return video, audio

    def _decode_and_save_sample(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor],
        prompt_idx: int,
        step_dir: str,
        prompt_text: Optional[str] = None,
    ):
        """Decode latents via VAE and save as mp4 (video + audio if available)."""
        video_pixel = self.video_vae.decode_to_pixel(video_latent)

        audio_waveform = None
        if audio_latent is not None:
            try:
                audio_waveform = self.audio_vae.decode_to_waveform(audio_latent)
            except Exception as e:
                print(f"[Benchmark][Rank {self.global_rank}] Audio decode failed: {e}")

        vid = video_pixel[0]  # [C, F, H, W]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)
        vid = vid.permute(0, 2, 3, 1)  # [F, H, W, C]
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")
        written_ok = False
        if audio_waveform is not None:
            try:
                wav = audio_waveform[0].cpu().float()
                _write_video_compat(
                    sample_path, vid, fps=self.benchmark_video_fps,
                    audio_array=wav, audio_fps=self.benchmark_audio_sample_rate,
                    audio_codec="aac",
                )
                written_ok = True
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] Audio mux failed for "
                    f"{sample_path}; falling back to video-only mp4: {e}",
                    flush=True,
                )

        if not written_ok:
            try:
                _write_video_compat(sample_path, vid, fps=self.benchmark_video_fps)
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video failed: {e}"
                )
                return

        if prompt_text is not None:
            txt_path = os.path.splitext(sample_path)[0] + ".txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"index: {prompt_idx}\n")
                    f.write(f"group: {os.path.basename(step_dir)}\n")
                    f.write(f"sampler: {self.benchmark_sampler}\n")
                    f.write(f"prompt: {prompt_text}\n")
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] Failed to save prompt "
                    f"sidecar for {sample_path}: {e}",
                    flush=True,
                )

        del video_pixel, audio_waveform
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_benchmark_and_log(self):
        """Run configured benchmark schedules sequentially."""
        num_train_timestep = int(self.config.get("num_train_timestep", 1000))
        schedules = []
        for raw_schedule in self.benchmark_denoising_step_lists:
            step_count = len(raw_schedule) - 1
            sigmas = torch.tensor(
                [value / float(num_train_timestep) for value in raw_schedule],
                device=self.device,
                dtype=torch.float32,
            )
            schedules.append((f"{step_count}_steps", sigmas))
        if self.is_main_process:
            print(
                "[Benchmark] schedules="
                f"{[tag for tag, _ in schedules]} sampler={self.benchmark_sampler} "
                "(run in this order)",
                flush=True,
            )
        for schedule_tag, sigmas in schedules:
            self._run_benchmark_schedule(
                schedule_tag=schedule_tag,
                denoising_sigmas=sigmas,
                namespace_outputs=len(schedules) > 1,
            )

    @torch.no_grad()
    def _run_benchmark_schedule(
        self,
        *,
        schedule_tag: str,
        denoising_sigmas: torch.Tensor,
        namespace_outputs: bool,
    ):
        """Run one T2AV benchmark schedule across ranks; rank 0 logs to wandb.

        All ranks participate (FSDP collectives are required). Work is
        distributed across ranks in ceil(N/W) rounds; ranks with no real
        prompt this round still run inference on a dummy prompt (the first
        one) but their outputs are not saved.
        """
        import math as _math

        config = self.config
        torch.cuda.empty_cache()

        num_t2av = len(self.benchmark_prompts)
        if num_t2av == 0:
            return

        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: {num_t2av} T2AV samples "
                f"({schedule_tag} {self.benchmark_sampler} KV-cache, no CFG) across "
                f"{self.world_size} ranks..."
            )

        step_dir = os.path.join(self.output_path, "benchmark", f"step_{self.step:07d}")
        if namespace_outputs:
            step_dir = os.path.join(step_dir, schedule_tag)
        t2av_group_dirs = {
            group["name"]: os.path.join(step_dir, group["name"])
            for group in self.benchmark_t2av_groups
        }
        if not t2av_group_dirs:
            t2av_group_dirs = {"t2av": os.path.join(step_dir, "t2av")}
        for group_dir in t2av_group_dirs.values():
            os.makedirs(group_dir, exist_ok=True)

        from ltx_distillation.train_distillation import compute_latent_shapes
        prompt_num_frames = [
            self._benchmark_num_frames_for_prompt(prompt)
            for prompt in self.benchmark_prompts
        ]
        max_prompt_num_frames = max(prompt_num_frames)
        max_video_shape, max_audio_shape = compute_latent_shapes(
            num_frames=max_prompt_num_frames,
            video_height=config.get("video_height", 512),
            video_width=config.get("video_width", 768),
            batch_size=1,
        )
        if self.is_main_process:
            print(
                f"[Benchmark] T2AV max rollout frames={max_prompt_num_frames} "
                f"latent_frames={max_video_shape[1]}",
                flush=True,
            )

        was_training = self.ar_teacher._generator.training
        self.ar_teacher._generator.eval()
        try:
            self._vae_decoders_to_device()
            benchmark_start = time.time()

            num_rounds = _math.ceil(num_t2av / self.world_size)
            for round_idx in range(num_rounds):
                prompt_idx = round_idx * self.world_size + self.global_rank
                has_real = prompt_idx < num_t2av
                prompt_text = (
                    self.benchmark_prompts[prompt_idx] if has_real
                    else self.benchmark_prompts[0]
                )
                actual_prompt_idx = prompt_idx if has_real else 0
                video_shape, audio_shape = compute_latent_shapes(
                    num_frames=prompt_num_frames[actual_prompt_idx],
                    video_height=config.get("video_height", 512),
                    video_width=config.get("video_width", 768),
                    batch_size=1,
                )

                with torch.no_grad():
                    pos_cond = self.ar_teacher.text_encoder(text_prompts=[prompt_text])

                seed = self.benchmark_seed + prompt_idx
                video_latent, audio_latent = self._generate_causal_kv_cache(
                    pos_conditional_dict=pos_cond,
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    seed=seed,
                    denoising_sigmas=denoising_sigmas,
                    pad_to_video_shape=tuple(max_video_shape),
                    pad_to_audio_shape=tuple(max_audio_shape),
                )

                if has_real:
                    group_name = self._benchmark_t2av_group_name(prompt_idx)
                    self._decode_and_save_sample(
                        video_latent=video_latent,
                        audio_latent=audio_latent,
                        prompt_idx=prompt_idx,
                        step_dir=t2av_group_dirs.get(
                            group_name, next(iter(t2av_group_dirs.values())),
                        ),
                        prompt_text=prompt_text,
                    )

                del video_latent, audio_latent, pos_cond
                torch.cuda.empty_cache()
                barrier()

            elapsed = time.time() - benchmark_start
            if self.is_main_process:
                print(
                    f"[Benchmark][{schedule_tag}] Step {self.step}: "
                    f"T2AV done in {elapsed:.1f}s, "
                    f"samples saved under {step_dir}"
                )
                # Log mp4s to wandb.
                wandb_prefix = (
                    f"benchmark/{schedule_tag}" if namespace_outputs else "benchmark"
                )
                wandb_dict = {}
                t2av_rows_by_group = {name: [] for name in t2av_group_dirs}
                for group_name, group_dir in t2av_group_dirs.items():
                    for fname in sorted(os.listdir(group_dir)):
                        if not fname.endswith(".mp4"):
                            continue
                        fpath = os.path.join(group_dir, fname)
                        try:
                            prompt_idx = int(os.path.splitext(fname)[0].split("_")[-1])
                        except Exception:
                            prompt_idx = -1
                        try:
                            wandb_dict[f"{wandb_prefix}/{group_name}/sample_{prompt_idx}"] = wandb.Video(
                                fpath, fps=self.benchmark_video_fps, format="mp4",
                            )
                        except Exception as e:
                            print(f"[Benchmark] wandb.Video failed for {fname}: {e}")
                        if 0 <= prompt_idx < len(self.benchmark_prompts):
                            t2av_rows_by_group.setdefault(group_name, []).append([
                                prompt_idx,
                                self.benchmark_prompts[prompt_idx],
                                fpath,
                            ])
                for group_name, rows in t2av_rows_by_group.items():
                    if rows:
                        rows.sort(key=lambda row: row[0])
                        wandb_dict[f"{wandb_prefix}/{group_name}/prompt_table"] = wandb.Table(
                            columns=["index", "prompt", "local_path"],
                            data=rows,
                        )
                if wandb_dict:
                    wandb.log(wandb_dict, step=self.step, commit=False)
        finally:
            self._vae_decoders_to_cpu()
            if was_training:
                self.ar_teacher._generator.train()

    # -------------------------------------------------------------- #
    # Checkpointing                                                    #
    # -------------------------------------------------------------- #
    def save(self):
        if self.is_main_process:
            print("[Stage2] Gathering distributed model states...")

        generator_state = fsdp_state_dict(self.ar_teacher._generator)
        fake_score_state = fsdp_state_dict(self.dmd.fake_score)

        state_dict = {
            "generator": generator_state,
            "fake_score": fake_score_state,
            "step": self.step,
        }

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path,
                f"checkpoint_{self.step:06d}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"[Stage2] Model saved to {checkpoint_path}")

        del generator_state, fake_score_state, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def _resume_from_checkpoint(self, resume_path: str):
        if self.is_main_process:
            print(f"[Stage2] Resume from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        if "generator" in ckpt:
            self.ar_teacher._generator.load_state_dict(ckpt["generator"], strict=False)
        if "fake_score" in ckpt:
            self.dmd.fake_score.load_state_dict(ckpt["fake_score"], strict=False)
        if "step" in ckpt:
            self.step = int(ckpt["step"])
        if self.is_main_process:
            print(f"[Stage2] Resumed at step {self.step}")

    # -------------------------------------------------------------- #
    # Main loop                                                        #
    # -------------------------------------------------------------- #
    def train(self):
        max_steps = self.config.get("max_steps", float("inf"))
        save_iters = int(self.config.get("log_iters", 1000))

        while self.step < max_steps:
            self.train_one_step()

            if (
                not self.config.get("no_save", False)
                and self.step > 0
                and self.step % save_iters == 0
            ):
                self.save()
                torch.cuda.empty_cache()
                self.ar_teacher._generator.train()
                self.dmd.fake_score.train()

            # Multi-schedule T2AV benchmark sampling.
            if (
                self.benchmark_enabled
                and len(self.benchmark_prompts) > 0
                and self.step > 0
                and self.step % self.benchmark_iters == 0
            ):
                self._run_benchmark_and_log()
                self.ar_teacher._generator.train()
                self.dmd.fake_score.train()

            barrier()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None:
                    wandb.log(
                        {"per_iteration_time": current_time - self.previous_time},
                        step=self.step,
                    )
                self.previous_time = current_time

            self.step += 1

        if not self.config.get("no_save", False):
            self.save()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: TF + DMD distillation")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True

    trainer = Stage2Trainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
