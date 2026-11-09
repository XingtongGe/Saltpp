"""Trainer entry point for Stage-4A 2-step-LR + 2-step-HR DMD."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import wandb

from ltx_causal.attention.mask_builder import compute_aligned_audio_frames
from ltx_distillation.stage35_data import build_stage35_dataloader
from ltx_distillation.stage35_media import (
    decode_video_tiled,
    write_video_pyav,
)
from ltx_distillation.train_distillation import (
    Trainer,
    _load_config_with_base,
    _write_video_compat,
    compute_latent_shapes,
)
from ltx_distillation.util import barrier, cycle, fsdp_state_dict, fsdp_wrap


class Stage4ScaleWiseTrainer(Trainer):
    """Reuse stable Stage-3 setup while isolating Stage-4 data and objectives."""

    def __init__(self, config):
        if str(config.get("distillation_loss", "")).lower() != (
            "stage4_scale_wise_dmd"
        ):
            raise ValueError(
                "Stage4ScaleWiseTrainer requires "
                "distillation_loss=stage4_scale_wise_dmd"
            )
        if str(config.get("training_mode", "")).lower() != (
            "causal_self_forcing"
        ):
            raise ValueError(
                "Stage4ScaleWiseTrainer requires "
                "training_mode=causal_self_forcing"
            )
        if int(config.get("batch_size", 1)) != 1:
            raise ValueError("Stage4A training requires batch_size=1")
        if int(config.get("gradient_accumulation_steps", 1)) != 1:
            raise ValueError(
                "Stage4A V1 does not implement gradient accumulation; "
                "set gradient_accumulation_steps=1"
            )
        super().__init__(config)
        self.upsampler_optimizer = None
        if self.dmd.stage4_train_causal_upsampler:
            upsampler_parameters = [
                parameter
                for parameter in self.dmd.causal_upsampler.parameters()
                if parameter.requires_grad
            ]
            if not upsampler_parameters:
                raise RuntimeError(
                    "stage4_train_causal_upsampler=true but the wrapped "
                    "upsampler exposes no trainable parameters"
                )
            # Deliberately reuse the generator optimizer hyperparameters so
            # enabling the upsampler introduces only one config switch.
            self.upsampler_optimizer = torch.optim.AdamW(
                upsampler_parameters,
                lr=float(self.generator_optimizer.param_groups[0]["lr"]),
                betas=(float(config.beta1), float(config.beta2)),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )

        resume_checkpoint = config.get("resume_checkpoint")
        if resume_checkpoint:
            try:
                resumed = torch.load(
                    str(resume_checkpoint),
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
            except TypeError:
                resumed = torch.load(
                    str(resume_checkpoint),
                    map_location="cpu",
                )
            if "causal_upsampler" in resumed:
                self.dmd.causal_upsampler.load_state_dict(
                    resumed["causal_upsampler"]
                )
                if self.is_main_process:
                    print(
                        "[Stage4A][resume] restored causal upsampler from "
                        f"{resume_checkpoint}",
                        flush=True,
                    )
            elif (
                self.dmd.stage4_train_causal_upsampler
                and self.is_main_process
            ):
                print(
                    "[Stage4A][resume] checkpoint has no causal_upsampler; "
                    "keeping causal_upsampler_checkpoint_path initialization",
                    flush=True,
                )
            del resumed
        self._phase_memory: Dict[str, float] = {}
        self.local_visualization_iters = int(
            config.get("local_visualization_iters", 0)
        )
        self.local_video_fps = int(
            config.get(
                "local_video_fps",
                config.get("benchmark_video_fps", 24),
            )
        )

    def _wrap_with_fsdp(self):
        super()._wrap_with_fsdp()
        self.dmd.causal_upsampler = fsdp_wrap(
            self.dmd.causal_upsampler,
            sharding_strategy=self.config.sharding_strategy,
            mixed_precision=self.config.mixed_precision,
            wrap_strategy=str(
                self.config.get(
                    "causal_upsampler_fsdp_wrap_strategy", "size"
                )
            ),
        )
        self.dmd.causal_upsampler.eval().requires_grad_(
            self.dmd.stage4_train_causal_upsampler
        )

    def _init_dataloader(self):
        # Stage4A only needs prompt, duration and spatial metadata.  No media is
        # downloaded or VAE-encoded during training.
        self.backward_simulation = True
        self.variable_length_rollout = bool(
            self.config.get("variable_length_rollout", False)
        )
        if not self.variable_length_rollout:
            fixed_num_frames = int(self.config.num_frames)
            snapped_num_frames = self._snap_ltx_frames(fixed_num_frames)
            if fixed_num_frames != snapped_num_frames:
                raise ValueError(
                    "Fixed Stage4A num_frames must be an LTX-valid 8n+1 "
                    f"value, got {fixed_num_frames} (nearest lower valid "
                    f"value: {snapped_num_frames})"
                )
        self.use_real_data = False
        self.dataloader = cycle(build_stage35_dataloader(self.config))

    def _reset_phase_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _record_phase_memory(self, phase: str) -> None:
        if not torch.cuda.is_available():
            return
        gib = float(1024**3)
        self._phase_memory[f"memory/{phase}/allocated_gib"] = (
            torch.cuda.memory_allocated(self.device) / gib
        )
        self._phase_memory[f"memory/{phase}/reserved_gib"] = (
            torch.cuda.memory_reserved(self.device) / gib
        )
        self._phase_memory[f"memory/{phase}/peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(self.device) / gib
        )
        self._phase_memory[f"memory/{phase}/peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved(self.device) / gib
        )
        torch.cuda.reset_peak_memory_stats(self.device)

    def _reduce_memory_metrics(self) -> Dict[str, float]:
        if not self._phase_memory:
            return {}
        keys = sorted(self._phase_memory)
        values = torch.tensor(
            [self._phase_memory[key] for key in keys],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return {
            key: float(value)
            for key, value in zip(keys, values.cpu().tolist())
        }

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().float().mean().item())
        return value

    def _batch_shapes(self, batch: Dict[str, Any]):
        if self.variable_length_rollout:
            real_num_frames = self._snap_ltx_frames(
                int(batch["num_frames"])
            )
        else:
            # Metadata still supplies the prompt, aspect ratio and source
            # duration, but fixed-length training must not let that duration
            # override config.num_frames.
            real_num_frames = int(self.config.num_frames)
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
        audio_shape[1] = compute_aligned_audio_frames(
            total_video_latent_frames=int(video_shape[1]),
            num_frame_per_block=int(self.config.num_frame_per_block),
            first_block_video_frames=int(
                self.config.first_block_video_frames
            ),
        )
        real_video_frames = 1 + (real_num_frames - 1) // 8
        real_audio_frames = compute_aligned_audio_frames(
            total_video_latent_frames=real_video_frames,
            num_frame_per_block=int(self.config.num_frame_per_block),
            first_block_video_frames=int(
                self.config.first_block_video_frames
            ),
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

    def train_one_step(self):
        self.dmd.eval()
        self.dmd.generator.train()
        self.dmd.fake_score.train()
        self.dmd.real_score.eval()
        # The released upsampler has no train/eval-dependent normalization.
        # Keeping eval mode also guarantees that the no-grad rollout and the
        # gradient replay evaluate the same deterministic function.
        self.dmd.causal_upsampler.eval()
        self.dmd.current_step = self.step

        config = self.config
        train_generator = self.step % int(config.dfake_gen_update_ratio) == 0
        run_local_visualization = bool(
            train_generator
            and self.local_visualization_iters > 0
            and self.step > 0
            and self.step % self.local_visualization_iters == 0
            and not bool(getattr(config, "no_visualize", False))
        )
        self.dmd.set_hr_dmd_visualization_capture(
            run_local_visualization and self.is_main_process
        )
        log_layerwise = (
            self.step % self.layerwise_grad_log_interval == 0
        )
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

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
            conditional_dict = self.dmd.text_encoder(text_prompts=prompts)
            unconditional_dict = self.dmd.text_encoder(
                text_prompts=[str(config.negative_prompt)]
            )
            unconditional_dict = {
                key: value.detach()
                for key, value in unconditional_dict.items()
            }

        self._phase_memory = {}
        self._reset_phase_memory()
        self.dmd.set_memory_hook(self._record_phase_memory)

        generator_logs = {}
        generator_loss_value = None
        generator_grad_norm_value = None
        upsampler_grad_norm_value = None
        local_visualization_context = None
        generator_layerwise = {}
        if train_generator:
            self.generator_optimizer.zero_grad(set_to_none=True)
            if self.upsampler_optimizer is not None:
                self.upsampler_optimizer.zero_grad(set_to_none=True)
                # Follow the generator's existing warmup/cosine schedule
                # without adding a separate LR scheduler or LR hyperparameter.
                generator_lr = float(
                    self.generator_optimizer.param_groups[0]["lr"]
                )
                for group in self.upsampler_optimizer.param_groups:
                    group["lr"] = generator_lr
            generator_loss, generator_logs = self.dmd.generator_loss(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )
            local_visualization_context = (
                self.dmd.pop_hr_dmd_visualization_context()
            )
            if local_visualization_context is not None:
                local_visualization_context.update(
                    {
                        "prompt": prompts[0],
                        "step": int(self.step),
                        "hr_rollout_steps": int(
                            generator_logs[
                                "stage4/generator_hr_rollout_steps"
                            ]
                        ),
                    }
                )
            if generator_loss.requires_grad:
                raise RuntimeError(
                    "Stage4A generator_loss must perform its own blockwise backward"
                )
            generator_layerwise = (
                self._compute_layerwise_grad_norms(
                    self.dmd.generator, "generator"
                )
                if log_layerwise
                else {}
            )
            if hasattr(self.dmd.generator, "clip_grad_norm_"):
                generator_grad_norm = self.dmd.generator.clip_grad_norm_(
                    self.max_grad_norm
                )
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.dmd.generator.parameters(), self.max_grad_norm
                )
            upsampler_dmd_active = bool(
                generator_logs.get(
                    "stage4/upsampler_hr_dmd_active", False
                )
            )
            if self.upsampler_optimizer is not None and upsampler_dmd_active:
                if hasattr(self.dmd.causal_upsampler, "clip_grad_norm_"):
                    upsampler_grad_norm = (
                        self.dmd.causal_upsampler.clip_grad_norm_(
                            self.max_grad_norm
                        )
                    )
                else:
                    upsampler_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.dmd.causal_upsampler.parameters(),
                        self.max_grad_norm,
                    )
                upsampler_grad_norm_value = float(
                    upsampler_grad_norm.item()
                )
            self.generator_optimizer.step()
            if self.upsampler_optimizer is not None and upsampler_dmd_active:
                self.upsampler_optimizer.step()
                # Release the large gradient shards before critic training.
                self.upsampler_optimizer.zero_grad(set_to_none=True)
            generator_loss_value = float(generator_loss.item())
            generator_grad_norm_value = float(generator_grad_norm.item())
            del generator_loss, generator_grad_norm
            if upsampler_grad_norm_value is not None:
                del upsampler_grad_norm
            torch.cuda.empty_cache()
            self._record_phase_memory("generator/optimizer_step")

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss, critic_logs = self.dmd.critic_loss(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
        )
        if critic_loss.requires_grad:
            raise RuntimeError(
                "Stage4A critic_loss must perform its own scale-wise backward"
            )
        critic_layerwise = (
            self._compute_layerwise_grad_norms(
                self.dmd.fake_score, "critic"
            )
            if log_layerwise
            else {}
        )
        if hasattr(self.dmd.fake_score, "clip_grad_norm_"):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(
                self.max_grad_norm
            )
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        critic_loss_value = float(critic_loss.item())
        critic_grad_norm_value = float(critic_grad_norm.item())
        del critic_loss, critic_grad_norm
        torch.cuda.empty_cache()
        self._record_phase_memory("critic/optimizer_step")
        self.dmd.set_memory_hook(None)

        if run_local_visualization:
            self._run_local_visualization(local_visualization_context)
            del local_visualization_context
            torch.cuda.empty_cache()

        run_benchmark = (
            self.benchmark_enabled
            and self.benchmark_prompts
            and self.benchmark_iters > 0
            and self.step > 0
            and self.step % self.benchmark_iters == 0
            and not bool(getattr(config, "no_visualize", False))
        )
        if run_benchmark:
            self._run_benchmark_and_log()

        memory_logs = self._reduce_memory_metrics()
        if self.is_main_process:
            logs = {
                "train/critic_loss": critic_loss_value,
                "train/critic_grad_norm": critic_grad_norm_value,
                "train/lr_generator": self.generator_optimizer.param_groups[0][
                    "lr"
                ],
                "train/lr_critic": self.critic_optimizer.param_groups[0]["lr"],
                "train/real_num_frames": real_num_frames,
                "train/rollout_num_frames": rollout_num_frames,
                "train/lr_height": height,
                "train/lr_width": width,
                "train/hr_height": height * 2,
                "train/hr_width": width * 2,
            }
            if self.upsampler_optimizer is not None:
                logs["train/lr_upsampler"] = (
                    self.upsampler_optimizer.param_groups[0]["lr"]
                )
            logs.update(
                {
                    f"train/{key}": self._scalar(value)
                    for key, value in critic_logs.items()
                }
            )
            logs.update(critic_layerwise)
            logs.update(
                {
                    f"train/{key}": value
                    for key, value in memory_logs.items()
                }
            )
            if train_generator:
                logs["train/generator_loss"] = generator_loss_value
                logs["train/generator_grad_norm"] = (
                    generator_grad_norm_value
                )
                if upsampler_grad_norm_value is not None:
                    logs["train/upsampler_grad_norm"] = (
                        upsampler_grad_norm_value
                    )
                logs.update(
                    {
                        f"train/{key}": self._scalar(value)
                        for key, value in generator_logs.items()
                    }
                )
                logs.update(generator_layerwise)
            wandb.log(logs, step=self.step)

        del conditional_dict, unconditional_dict
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_local_visualization(
        self,
        context: Dict[str, Any] | None,
    ) -> None:
        """Decode and save generator | real-score x0 | fake-score x0."""

        failure = ""
        if self.is_main_process:
            if context is None:
                failure = (
                    "rank 0 did not capture the requested HR DMD "
                    "visualization tensors"
                )
            else:
                video_vae = self.dmd.video_vae
                try:
                    video_vae.to(device=self.device, dtype=self.dtype)
                    seed = int(self.config.seed) + int(context["step"])
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
                                video_vae=video_vae,
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
                    output_path = (
                        log_dir
                        / (
                            f"step_{int(context['step']):06d}_"
                            "generator_real_fake.mp4"
                        )
                    )
                    write_video_pyav(
                        str(output_path),
                        comparison,
                        fps=self.local_video_fps,
                    )
                    metadata_path = output_path.with_suffix(".txt")
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_text(
                        "layout: generator_hr | real_score_pred_x0"
                        " | fake_score_pred_x0\n"
                        f"step: {int(context['step'])}\n"
                        f"hr_rollout_steps: "
                        f"{int(context['hr_rollout_steps'])}\n"
                        f"dmd_video_sigma_mean: "
                        f"{float(context['video_sigma_mean']):.8f}\n"
                        f"prompt: {context['prompt']}\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[Stage4A][local-log] wrote {output_path}",
                        flush=True,
                    )
                    del decoded, comparison
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                finally:
                    video_vae.to(device="cpu")
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
                    f"Stage4 local visualization failed: {failure}"
                )
            raise RuntimeError("Stage4 local visualization failed on rank 0")

    def save(self):
        """Save Stage4 states, including a jointly trained causal upsampler."""

        print("Gathering distributed Stage4 model states...")
        generator_state_dict = fsdp_state_dict(self.dmd.generator)
        critic_state_dict = fsdp_state_dict(self.dmd.fake_score)
        causal_upsampler_state_dict = None
        if self.dmd.stage4_train_causal_upsampler:
            causal_upsampler_state_dict = fsdp_state_dict(
                self.dmd.causal_upsampler
            )

        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict,
            "step": self.step,
        }
        if causal_upsampler_state_dict is not None:
            state_dict["causal_upsampler"] = causal_upsampler_state_dict

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path,
                f"checkpoint_{self.step:06d}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, save_path)
            print(f"Checkpoint saved to {save_path}")

    @torch.no_grad()
    def _run_benchmark_and_log(self):
        """Distributed fixed-case benchmark using tiled HR VAE decoding."""

        config = self.config
        num_prompts = len(self.benchmark_prompts)
        rounds = math.ceil(num_prompts / self.world_size)
        prompt_frames = [
            self._benchmark_num_frames_for_prompt(prompt)
            for prompt in self.benchmark_prompts
        ]
        max_frames = max(prompt_frames)
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=max_frames,
            video_height=int(config.video_height),
            video_width=int(config.video_width),
            batch_size=1,
        )
        audio_shape[1] = compute_aligned_audio_frames(
            total_video_latent_frames=int(video_shape[1]),
            num_frame_per_block=int(config.num_frame_per_block),
            first_block_video_frames=int(config.first_block_video_frames),
        )

        output_root = (
            Path(self.output_path)
            / "benchmark"
            / f"step_{self.step:07d}"
            / "2lr_2hr"
        )
        if self.is_main_process:
            print(
                f"[Stage4A][benchmark] {num_prompts} cases, "
                f"{rounds} distributed rounds, LR={config.video_height}x"
                f"{config.video_width}, HR={int(config.video_height) * 2}x"
                f"{int(config.video_width) * 2}",
                flush=True,
            )
        barrier()

        generator_was_training = self.dmd.generator.training
        self.dmd.generator.eval()
        self.dmd.causal_upsampler.eval()
        self._vae_to_device()
        wall_start = time.perf_counter()
        local_generate_seconds = 0.0

        try:
            for round_index in range(rounds):
                position = round_index * self.world_size + self.global_rank
                has_case = position < num_prompts
                case_index = position if has_case else 0
                prompt = [self.benchmark_prompts[case_index]]
                conditional_dict = self.dmd.text_encoder(
                    text_prompts=prompt
                )
                seed = int(self.benchmark_seed) + position
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    start = time.perf_counter()
                    hr_video, hr_audio, _ = self.dmd.sample_scale_wise(
                        video_shape=video_shape,
                        audio_shape=audio_shape,
                        conditional_dict=conditional_dict,
                    )
                    local_generate_seconds += time.perf_counter() - start

                if has_case:
                    real_video_frames = 1 + (
                        int(prompt_frames[case_index]) - 1
                    ) // 8
                    real_audio_frames = compute_aligned_audio_frames(
                        total_video_latent_frames=real_video_frames,
                        num_frame_per_block=int(config.num_frame_per_block),
                        first_block_video_frames=int(
                            config.first_block_video_frames
                        ),
                    )
                    group = self._benchmark_t2av_group_name(case_index)
                    output_dir = output_root / group
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"sample_{case_index}.mp4"

                    frames = decode_video_tiled(
                        video_vae=self.dmd.video_vae,
                        latent_bfchw=hr_video[:, :real_video_frames],
                        seed=seed,
                    )
                    audio_waveform = self.dmd.audio_vae.decode_to_waveform(
                        hr_audio[:, :real_audio_frames]
                    )
                    _write_video_compat(
                        str(output_path),
                        frames,
                        fps=int(self.benchmark_video_fps),
                        audio_array=audio_waveform[0].detach().cpu().float(),
                        audio_fps=int(self.benchmark_audio_sample_rate),
                        audio_codec="aac",
                    )
                    del frames, audio_waveform

                del hr_video, hr_audio, conditional_dict
                torch.cuda.empty_cache()
                barrier()
        finally:
            self._vae_to_cpu()
            if generator_was_training:
                self.dmd.generator.train()

        total_generate = torch.tensor(
            [local_generate_seconds],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(total_generate, op=dist.ReduceOp.SUM)
        wall_seconds = time.perf_counter() - wall_start
        barrier()

        if self.is_main_process:
            benchmark_logs = {
                "benchmark/2lr_2hr/wall_seconds": wall_seconds,
                "benchmark/2lr_2hr/generate_seconds": float(
                    total_generate.item()
                ),
            }
            for case_index in range(num_prompts):
                group = self._benchmark_t2av_group_name(case_index)
                output_path = (
                    output_root / group / f"sample_{case_index}.mp4"
                )
                if output_path.is_file():
                    benchmark_logs[
                        f"benchmark/2lr_2hr/{group}/sample_{case_index}"
                    ] = wandb.Video(
                        str(output_path),
                        fps=int(self.benchmark_video_fps),
                        format="mp4",
                    )
            wandb.log(benchmark_logs, step=self.step, commit=False)
            print(
                f"[Stage4A][benchmark] wrote {output_root} in "
                f"{wall_seconds:.1f}s",
                flush=True,
            )
        barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    args = parser.parse_args()

    # Keep Stage-4 launch behavior aligned with the Stage-3 entry point.  In
    # particular, lightweight experiment configs may inherit the established
    # Stage-4A recipe through a relative ``base_config_path`` and override only
    # their initialization/output fields.
    config = _load_config_with_base(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    trainer = Stage4ScaleWiseTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
