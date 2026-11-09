"""Stage 2 teacher-forcing discrete consistency distillation.

This is the clean TF-dCM bridge between the Stage-0 causal teacher and the
Stage-3 self-forcing DMD refinement.  It deliberately owns no bidirectional
real/fake score models and does not use the Stage-2 SVI replay bank.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from ltx_distillation.ar_teacher.ar_teacher import ARTeacherConfig, LTX2ARTeacher
from ltx_distillation.tf_dcm_utils import (
    add_noise,
    classifier_free_guidance,
    dcm_sigma_path,
    expand_sigma,
    teacher_euler_step,
    validate_benchmark_schedules,
)
from ltx_distillation.train_stage2 import (
    Stage2Trainer,
    _write_video_compat,
    barrier,
    fsdp_state_dict,
    fsdp_wrap,
    init_logging_folder,
    launch_distributed_job,
    set_seed,
)


class Stage2TFDCMTrainer(Stage2Trainer):
    """Pure teacher-forcing dCM trainer with a frozen causal teacher."""

    training_mode = "causal_tf_dcm"

    def __init__(self, config: DictConfig):
        self.config = config
        if str(config.get("training_mode", self.training_mode)) != self.training_mode:
            raise ValueError(
                f"{type(self).__name__} requires training_mode={self.training_mode}"
            )
        if bool(config.get("svi_enabled", False)):
            raise ValueError("Pure TF-dCM Stage 2 requires svi_enabled=false")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0

        seed = int(config.get("seed", 42))
        if seed == 0:
            random_seed = torch.randint(0, 10_000_000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            seed = int(random_seed.item())
        set_seed(seed + self.global_rank)

        self.output_path = config.output_path
        if self.is_main_process:
            self.output_path, _ = init_logging_folder(config)
        else:
            os.makedirs(self.output_path, exist_ok=True)
        if dist.is_initialized() and self.world_size > 1:
            self.output_path = self._broadcast_output_path(self.output_path)
        barrier()

        self.dcm_total_steps = int(config.get("dcm_total_steps", 48))
        self.dcm_skipping_interval_steps = int(
            config.get("dcm_skipping_interval_steps", 1)
        )
        if self.dcm_total_steps <= 0:
            raise ValueError("dcm_total_steps must be positive")
        if not 1 <= self.dcm_skipping_interval_steps <= self.dcm_total_steps:
            raise ValueError(
                "dcm_skipping_interval_steps must be in [1, dcm_total_steps]"
            )
        self.sigma_shift = float(config.get("sigma_shift", 8.0))
        self.video_loss_weight = float(config.get("video_loss_weight", 1.0))
        self.audio_loss_weight = float(config.get("audio_loss_weight", 0.16))
        self.cm_loss_scale = float(config.get("cm_loss_scale", 100.0))
        self.teacher_video_guidance = float(
            config.get("causal_teacher_video_guidance_scale", 4.0)
        )
        self.teacher_audio_guidance = float(
            config.get("causal_teacher_audio_guidance_scale", 4.0)
        )
        if self.teacher_video_guidance <= 1.0 or self.teacher_audio_guidance <= 1.0:
            raise ValueError(
                "The requested TF-dCM recipe requires causal teacher CFG to be enabled "
                "for both video and audio (guidance scale > 1)."
            )

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
            sigma_shift=self.sigma_shift,
            sigma_min=config.get("sigma_min", 0.0),
            sigma_max=config.get("sigma_max", 1.0),
            video_loss_weight=self.video_loss_weight,
            audio_loss_weight=self.audio_loss_weight,
            num_audio_sink_tokens=config.get("num_audio_sink_tokens", 0),
            enable_causal_log_rescale=config.get("enable_causal_log_rescale", False),
            enable_chunk_recycling=config.get("enable_chunk_recycling", False),
            first_frame_conditioning_p=config.get("first_frame_conditioning_p", 0.0),
        )
        self.ar_teacher = LTX2ARTeacher(ar_config, device=self.device)
        self.ar_teacher._load_models()

        student_ckpt = config.get("generator_ckpt")
        teacher_ckpt = config.get("causal_teacher_ckpt", student_ckpt)
        if not student_ckpt:
            raise ValueError(
                "generator_ckpt must point to the Stage-0 student initialization"
            )
        if not teacher_ckpt:
            raise ValueError(
                "causal_teacher_ckpt must point to a frozen causal teacher"
            )

        self._load_generator_from_stage0(str(student_ckpt))

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
            print("[Stage2 TF-dCM] FSDP wrapping student...")
        self.ar_teacher._generator = fsdp_wrap(
            self.ar_teacher._generator, **fsdp_gen_kwargs
        )
        torch.cuda.empty_cache()

        if self.is_main_process:
            print("[Stage2 TF-dCM] FSDP wrapping text encoder...")
        self.ar_teacher._text_encoder = fsdp_wrap(
            self.ar_teacher._text_encoder,
            sharding_strategy=sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.get("text_encoder_fsdp_wrap_strategy", "size"),
        )
        torch.cuda.empty_cache()

        # Build the second causal model only after the student has been sharded.
        # The Stage-0 checkpoint is a complete generator state dict, so this path
        # does not need a second text encoder or a bidirectional base model.
        self.causal_teacher = self._build_frozen_causal_teacher(
            checkpoint_path=str(teacher_ckpt),
        )
        if self.is_main_process:
            print("[Stage2 TF-dCM] FSDP wrapping frozen causal teacher...")
        self.causal_teacher = fsdp_wrap(self.causal_teacher, **fsdp_gen_kwargs)
        self.causal_teacher.requires_grad_(False)
        self.causal_teacher.eval()
        self._assert_teacher_frozen(check_grads=False)
        torch.cuda.empty_cache()

        self._build_vae_encoders()

        self.gen_lr = float(config.get("generator_lr", config.get("lr", 2e-6)))
        self.warmup_steps = int(config.get("warmup_steps", 0))
        trainable_params = [
            parameter
            for parameter in self.ar_teacher._generator.parameters()
            if parameter.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("TF-dCM student has no trainable parameters")
        self.generator_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.gen_lr,
            betas=(float(config.get("beta1", 0.0)), float(config.get("beta2", 0.999))),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )

        self._init_dataloader()
        self._unconditional_dict_cache: Optional[Dict[str, Any]] = None
        self._negative_prompt = str(
            config.get(
                "negative_prompt", "blurry, out of focus, low quality, artifacts."
            )
        )

        self.step = 0
        self.max_grad_norm = float(config.get("max_grad_norm", 10.0))
        self.previous_time = None
        self.train_vis_iters = int(config.get("train_vis_iters", 50))

        raw_benchmark_schedules = config.get("benchmark_denoising_step_lists", None)
        if raw_benchmark_schedules is None:
            raw_benchmark_schedules = [config.get("denoising_step_list")]
        self.benchmark_denoising_step_lists = validate_benchmark_schedules(
            raw_benchmark_schedules,
            num_train_timestep=int(config.get("num_train_timestep", 1000)),
        )
        self._init_benchmark()

        resume_path = config.get("resume_checkpoint", None)
        if resume_path:
            self._resume_from_checkpoint(str(resume_path))

        if self.is_main_process and self.training_mode == "causal_tf_dcm":
            print(
                "[Stage2 TF-dCM] ready | "
                f"dense_steps={self.dcm_total_steps} "
                f"skip={self.dcm_skipping_interval_steps} "
                f"shift={self.sigma_shift} "
                f"teacher_cfg(video/audio)={self.teacher_video_guidance}/"
                f"{self.teacher_audio_guidance} "
                f"loss_scale={self.cm_loss_scale}"
            )
            print(
                "[Stage2 TF-dCM] benchmark consistency schedules="
                f"{[len(schedule) - 1 for schedule in self.benchmark_denoising_step_lists]}"
            )

    def _broadcast_output_path(self, output_path: str) -> str:
        max_bytes = 2048
        if self.is_main_process:
            encoded = output_path.encode("utf-8")
            if len(encoded) >= max_bytes:
                raise ValueError("output_path is too long to broadcast")
            path_tensor = torch.tensor(
                list(encoded) + [0] * (max_bytes - len(encoded)),
                dtype=torch.uint8,
                device=self.device,
            )
        else:
            path_tensor = torch.zeros(max_bytes, dtype=torch.uint8, device=self.device)
        dist.broadcast(path_tensor, src=0)
        decoded = bytes(path_tensor.cpu().tolist()).rstrip(b"\x00").decode("utf-8")
        if not self.is_main_process:
            os.makedirs(decoded, exist_ok=True)
        return decoded

    def _build_frozen_causal_teacher(
        self,
        *,
        checkpoint_path: str,
    ) -> torch.nn.Module:
        from ltx_causal import CausalLTXModel

        if self.is_main_process:
            print(f"[Stage2 TF-dCM] Building causal teacher from {checkpoint_path}")
        model = CausalLTXModel(self.ar_teacher._make_model_config())
        teacher = self.ar_teacher._make_wrapper(model).to(self.device, self.dtype)
        state_dict = LTX2ARTeacher._load_checkpoint_state_dict(checkpoint_path)
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
        real_missing = [key for key in missing if "mask_builder" not in key]
        if real_missing:
            raise RuntimeError(
                "Causal teacher checkpoint is incomplete; missing keys include "
                f"{real_missing[:10]}"
            )
        if self.is_main_process and unexpected:
            print(
                f"[Stage2 TF-dCM] teacher unexpected keys ({len(unexpected)}): "
                f"{unexpected[:10]}"
            )
        teacher.requires_grad_(False)
        teacher.eval()
        # Gradient checkpointing is intentionally not enabled for a no-grad teacher.
        return teacher

    def _assert_teacher_frozen(self, *, check_grads: bool) -> None:
        if any(
            parameter.requires_grad for parameter in self.causal_teacher.parameters()
        ):
            raise RuntimeError("causal teacher unexpectedly has trainable parameters")
        if check_grads and any(
            parameter.grad is not None for parameter in self.causal_teacher.parameters()
        ):
            raise RuntimeError("causal teacher unexpectedly received gradients")

    def _lr_scale(self) -> float:
        if self.warmup_steps <= 0 or self.step >= self.warmup_steps:
            return 1.0
        return max(1e-3, self.step / self.warmup_steps)

    def _apply_lr(self) -> None:
        scale = self._lr_scale()
        for param_group in self.generator_optimizer.param_groups:
            param_group["lr"] = self.gen_lr * scale

    def _sample_sigma_path(self, batch_size: int) -> torch.Tensor:
        if self.global_rank == 0:
            uniform = torch.rand(batch_size, 1, device=self.device, dtype=torch.float32)
        else:
            uniform = torch.zeros(
                batch_size, 1, device=self.device, dtype=torch.float32
            )
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(uniform, src=0)
        return dcm_sigma_path(
            uniform,
            total_steps=self.dcm_total_steps,
            skipping_interval_steps=self.dcm_skipping_interval_steps,
            sigma_shift=self.sigma_shift,
        )

    def _tf_predict_noisy(
        self,
        *,
        model: torch.nn.Module,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        conditional_dict: Dict[str, Any],
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = noisy_video.shape[0]
        video_sigma = expand_sigma(sigma, noisy_video.shape[1]).to(self.dtype)
        audio_sigma = expand_sigma(sigma, noisy_audio.shape[1]).to(self.dtype)
        clean_video_sigma = torch.zeros(
            batch_size,
            clean_video_context.shape[1],
            device=self.device,
            dtype=self.dtype,
        )
        clean_audio_sigma = torch.zeros(
            batch_size,
            clean_audio_context.shape[1],
            device=self.device,
            dtype=self.dtype,
        )
        return model(
            _tf_kwargs=dict(
                noisy_video=noisy_video,
                clean_video=clean_video_context,
                conditional_dict=conditional_dict,
                noisy_timestep=video_sigma,
                clean_timestep=clean_video_sigma,
                noisy_audio=noisy_audio,
                clean_audio=clean_audio_context,
                noisy_audio_timestep=audio_sigma,
                clean_audio_timestep=clean_audio_sigma,
                use_causal_timestep=False,
            )
        )

    @torch.no_grad()
    def _teacher_bridge_and_student_target(
        self,
        *,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        sigma_path: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the complete teacher bridge and stop-gradient student target."""
        current_video = noisy_video
        current_audio = noisy_audio
        for path_index in range(self.dcm_skipping_interval_steps):
            sigma_from = sigma_path[:, path_index : path_index + 1]
            sigma_to = sigma_path[:, path_index + 1 : path_index + 2]

            teacher_video_cond, teacher_audio_cond = self._tf_predict_noisy(
                model=self.causal_teacher,
                noisy_video=current_video,
                noisy_audio=current_audio,
                clean_video_context=clean_video,
                clean_audio_context=clean_audio,
                conditional_dict=conditional_dict,
                sigma=sigma_from,
            )
            teacher_video_uncond, teacher_audio_uncond = self._tf_predict_noisy(
                model=self.causal_teacher,
                noisy_video=current_video,
                noisy_audio=current_audio,
                clean_video_context=clean_video,
                clean_audio_context=clean_audio,
                conditional_dict=unconditional_dict,
                sigma=sigma_from,
            )
            teacher_video = classifier_free_guidance(
                teacher_video_cond,
                teacher_video_uncond,
                self.teacher_video_guidance,
            )
            teacher_audio = classifier_free_guidance(
                teacher_audio_cond,
                teacher_audio_uncond,
                self.teacher_audio_guidance,
            )
            current_video = teacher_euler_step(
                current_video, teacher_video, sigma_from, sigma_to
            )
            current_audio = teacher_euler_step(
                current_audio, teacher_audio, sigma_from, sigma_to
            )

        sigma_target = sigma_path[:, -1:]
        target_video, target_audio = self._tf_predict_noisy(
            model=self.ar_teacher._generator,
            noisy_video=current_video,
            noisy_audio=current_audio,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=conditional_dict,
            sigma=sigma_target,
        )
        return target_video.detach(), target_audio.detach()

    @torch.no_grad()
    def _log_training_samples(
        self,
        *,
        clean_video: torch.Tensor,
        student_x0_video: torch.Tensor,
        dcm_target_x0_video: torch.Tensor,
        sigma_high: torch.Tensor,
        sigma_target: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> None:
        """Save a local ``GT | student x0 | dCM target x0`` comparison mp4.

        This path intentionally has no WandB call. Only rank 0 invokes it;
        the caller synchronizes all ranks before the next FSDP training step.
        """
        if not self.is_main_process:
            return
        if not hasattr(self, "video_vae"):
            self._build_vae_decoders()

        # This local comparison contains video only. Keep the audio decoder on
        # CPU to reduce peak memory while student, teacher, and optimizer state
        # remain resident for training.
        torch.cuda.empty_cache()
        self.video_vae = self.video_vae.to(device=self.device)
        try:
            log_dir = os.path.join(self.output_path, "train_vis")
            os.makedirs(log_dir, exist_ok=True)
            latents = [
                clean_video[:1],
                student_x0_video[:1],
                dcm_target_x0_video[:1],
            ]
            decoded = []
            for latent in latents:
                pixel = self.video_vae.decode_to_pixel(
                    latent.to(device=self.device, dtype=self.dtype)
                )
                video = pixel[0]
                if video.shape[0] == 3:
                    video = video.permute(1, 0, 2, 3)
                video = video.permute(0, 2, 3, 1)
                decoded.append((video.clamp(0, 1) * 255).cpu().to(torch.uint8))

            min_frames = min(video.shape[0] for video in decoded)
            side_by_side = torch.cat([video[:min_frames] for video in decoded], dim=2)
            sigma_hi_value = float(sigma_high[0, 0].float().item())
            sigma_target_value = float(sigma_target[0, 0].float().item())
            filename = (
                f"step{self.step:06d}_sigma{int(sigma_hi_value * 1000):04d}"
                f"_to{int(sigma_target_value * 1000):04d}_t2av.mp4"
            )
            sample_path = os.path.join(log_dir, filename)
            _write_video_compat(
                sample_path,
                side_by_side,
                fps=int(self.config.get("benchmark_video_fps", 24)),
            )

            meta_path = os.path.splitext(sample_path)[0] + ".txt"
            with open(meta_path, "w", encoding="utf-8") as file:
                file.write("layout: GT | student_single_step_x0 | dcm_target_x0\n")
                file.write(f"step: {self.step}\n")
                file.write(f"sigma_high: {sigma_hi_value:.8f}\n")
                file.write(f"sigma_target: {sigma_target_value:.8f}\n")
                if metadata:
                    for key in (
                        "entry",
                        "clip_url",
                        "audio_url",
                        "fps",
                        "f_num",
                        "caption",
                    ):
                        file.write(f"{key}: {metadata.get(key, '')}\n")
            print(
                f"[Stage2 TF-dCM][TrainVis] Step {self.step}: saved {filename} "
                "(GT | student x0 | dCM target x0; local only)"
            )
        except Exception as error:
            print(
                f"[Stage2 TF-dCM][TrainVis] Step {self.step}: "
                f"local visualization failed: {error}"
            )
        finally:
            self.video_vae = self.video_vae.to(device="cpu")
            torch.cuda.empty_cache()

    def train_one_step(self) -> None:
        self.ar_teacher._generator.train()
        self.causal_teacher.eval()
        self._assert_teacher_frozen(check_grads=False)
        self._apply_lr()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        batch = next(self.dataloader)
        clean_video, clean_audio, text_prompts = self._encode_batch(batch)
        clean_audio = self._align_audio(clean_video, clean_audio)
        with torch.no_grad():
            conditional_dict = self.ar_teacher.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self._get_unconditional_dict(batch_size=len(text_prompts))

        sigma_path = self._sample_sigma_path(clean_video.shape[0])
        sigma_high = sigma_path[:, :1]
        video_noise = torch.randn_like(clean_video)
        audio_noise = torch.randn_like(clean_audio)
        noisy_video_high = add_noise(clean_video, video_noise, sigma_high)
        noisy_audio_high = add_noise(clean_audio, audio_noise, sigma_high)

        # Run all frozen work first.  Besides enforcing the intended gradient
        # boundary, this ordering leaves the final student forward in the
        # grad-enabled state expected by FSDP before backward().
        target_video, target_audio = self._teacher_bridge_and_student_target(
            noisy_video=noisy_video_high,
            noisy_audio=noisy_audio_high,
            clean_video=clean_video,
            clean_audio=clean_audio,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            sigma_path=sigma_path,
        )
        if target_video.requires_grad or target_audio.requires_grad:
            raise RuntimeError("TF-dCM target branch must be stop-gradient")

        self.generator_optimizer.zero_grad(set_to_none=True)
        pred_video_high, pred_audio_high = self._tf_predict_noisy(
            model=self.ar_teacher._generator,
            noisy_video=noisy_video_high,
            noisy_audio=noisy_audio_high,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=conditional_dict,
            sigma=sigma_high,
        )
        if not pred_video_high.requires_grad or not pred_audio_high.requires_grad:
            raise RuntimeError("TF-dCM online student branch must carry gradients")

        video_cm_loss = torch.nn.functional.mse_loss(
            pred_video_high.float(), target_video.float()
        )
        audio_cm_loss = torch.nn.functional.mse_loss(
            pred_audio_high.float(), target_audio.float()
        )
        total_loss = self.cm_loss_scale * (
            self.video_loss_weight * video_cm_loss
            + self.audio_loss_weight * audio_cm_loss
        )
        total_loss.backward()
        self._assert_teacher_frozen(check_grads=True)

        if hasattr(self.ar_teacher._generator, "clip_grad_norm_"):
            grad_norm = self.ar_teacher._generator.clip_grad_norm_(self.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ar_teacher._generator.parameters(), self.max_grad_norm
            )
        self.generator_optimizer.step()

        do_vis_step = (
            self.train_vis_iters > 0
            and self.step > 0
            and self.step % self.train_vis_iters == 0
        )
        if do_vis_step and self.is_main_process:

            def _first_value(key: str) -> Any:
                value = batch.get(key, "")
                if isinstance(value, (list, tuple)):
                    return value[0] if value else ""
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    return value.item()
                return value

            self._log_training_samples(
                clean_video=clean_video.detach(),
                student_x0_video=pred_video_high.detach(),
                dcm_target_x0_video=target_video.detach(),
                sigma_high=sigma_high.detach(),
                sigma_target=sigma_path[:, -1:].detach(),
                metadata={
                    "entry": _first_value("entries"),
                    "clip_url": _first_value("clip_urls"),
                    "audio_url": _first_value("audio_urls"),
                    "fps": _first_value("fps"),
                    "f_num": _first_value("f_nums"),
                    "caption": text_prompts[0] if text_prompts else "",
                },
            )
        if do_vis_step:
            # Keep all ranks aligned: non-main ranks must not enter the next
            # FSDP collective while rank 0 is still decoding/writing the mp4.
            barrier()

        if self.is_main_process:
            grad_norm_value = (
                float(grad_norm.item())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm)
            )
            wandb.log(
                {
                    "train/cm_loss": float(total_loss.item()),
                    "train/video_cm_loss": float(video_cm_loss.item()),
                    "train/audio_cm_loss": float(audio_cm_loss.item()),
                    "train/grad_norm": grad_norm_value,
                    "train/lr": self.generator_optimizer.param_groups[0]["lr"],
                    "train/sigma_high_mean": float(sigma_path[:, 0].mean().item()),
                    "train/sigma_target_mean": float(sigma_path[:, -1].mean().item()),
                    "train/teacher_video_cfg": self.teacher_video_guidance,
                    "train/teacher_audio_cfg": self.teacher_audio_guidance,
                },
                step=self.step,
            )
            if self.step < 10 or self.step % 50 == 0:
                print(
                    f"[Stage2 TF-dCM][{self.step}] loss={total_loss.item():.5f} "
                    f"video={video_cm_loss.item():.5f} audio={audio_cm_loss.item():.5f} "
                    f"sigma={sigma_path[:, 0].mean().item():.4f}->"
                    f"{sigma_path[:, -1].mean().item():.4f} grad={grad_norm_value:.4f}"
                )

        del (
            clean_video,
            clean_audio,
            conditional_dict,
            unconditional_dict,
            target_video,
            target_audio,
            pred_video_high,
            pred_audio_high,
            total_loss,
        )
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_benchmark_and_log(self) -> None:
        """Run Stage3's consistency sampler for each configured schedule."""
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
                "[Stage2 TF-dCM][Benchmark] consistency sampler schedules="
                f"{[tag for tag, _ in schedules]} (run in this order)",
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
    ) -> None:
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames
        from ltx_distillation.inference.causal_pipeline import CausalAVInferencePipeline
        from ltx_distillation.train_distillation import compute_latent_shapes

        num_prompts = len(self.benchmark_prompts)
        if num_prompts == 0:
            return
        num_rounds = math.ceil(num_prompts / self.world_size)
        step_dir = os.path.join(self.output_path, "benchmark", f"step_{self.step:07d}")
        if namespace_outputs:
            step_dir = os.path.join(step_dir, schedule_tag)
        group_dirs = {
            group["name"]: os.path.join(step_dir, group["name"])
            for group in self.benchmark_t2av_groups
        }
        for group_dir in group_dirs.values():
            os.makedirs(group_dir, exist_ok=True)

        prompt_num_frames = [
            self._benchmark_num_frames_for_prompt(prompt)
            for prompt in self.benchmark_prompts
        ]
        max_num_frames = max(prompt_num_frames)
        max_video_shape, max_audio_shape = compute_latent_shapes(
            num_frames=max_num_frames,
            video_height=self.config.get("video_height", 512),
            video_width=self.config.get("video_width", 768),
            batch_size=1,
        )
        if self.is_main_process:
            print(
                f"[Stage2 TF-dCM][Benchmark][{schedule_tag}] "
                f"prompts={num_prompts}, max_frames={max_num_frames}, "
                f"sigmas={denoising_sigmas.cpu().tolist()}",
                flush=True,
            )

        was_training = self.ar_teacher._generator.training
        self.ar_teacher._generator.eval()
        self._vae_decoders_to_device()
        start_time = time.perf_counter()
        try:
            pipeline = CausalAVInferencePipeline(
                generator=self.ar_teacher._generator,
                add_noise_fn=add_noise,
                denoising_sigmas=denoising_sigmas,
                num_frame_per_block=int(
                    self.config.get(
                        "benchmark_num_frame_per_block",
                        self.config.get("num_frame_per_block", 3),
                    )
                ),
                first_block_video_frames=int(
                    self.config.get("first_block_video_frames", 4)
                ),
                use_kv_cache=bool(self.config.get("benchmark_use_kv_cache", True)),
                clear_cuda_cache_per_round=self.benchmark_clear_cuda_cache_per_round,
                context_noise=0,
                num_train_timestep=int(self.config.get("num_train_timestep", 1000)),
            )
            for round_index in range(num_rounds):
                prompt_index = round_index * self.world_size + self.global_rank
                has_real_prompt = prompt_index < num_prompts
                actual_index = prompt_index if has_real_prompt else 0
                prompt = self.benchmark_prompts[actual_index]
                conditional_dict = self.ar_teacher.text_encoder(text_prompts=[prompt])

                prompt_seed = self.benchmark_seed + prompt_index
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(prompt_seed)
                    torch.cuda.manual_seed(prompt_seed)
                    video_latent, audio_latent = pipeline.generate(
                        video_shape=tuple(max_video_shape),
                        audio_shape=tuple(max_audio_shape),
                        conditional_dict=conditional_dict,
                    )

                if has_real_prompt:
                    real_video_shape, _ = compute_latent_shapes(
                        num_frames=prompt_num_frames[actual_index],
                        video_height=self.config.get("video_height", 512),
                        video_width=self.config.get("video_width", 768),
                        batch_size=1,
                    )
                    real_video_frames = int(real_video_shape[1])
                    video_out = video_latent[:, :real_video_frames]
                    audio_out = audio_latent
                    if audio_latent is not None:
                        real_audio_frames = compute_aligned_audio_frames(
                            total_video_latent_frames=real_video_frames,
                            num_frame_per_block=int(
                                self.config.get("benchmark_num_frame_per_block", 3)
                            ),
                            first_block_video_frames=int(
                                self.config.get("first_block_video_frames", 4)
                            ),
                        )
                        audio_out = audio_latent[:, :real_audio_frames]
                    group_name = self._benchmark_t2av_group_name(prompt_index)
                    self._decode_and_save_sample(
                        video_latent=video_out,
                        audio_latent=audio_out,
                        prompt_idx=prompt_index,
                        step_dir=group_dirs.get(group_name, step_dir),
                        prompt_text=prompt,
                    )

                del video_latent, audio_latent, conditional_dict
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()
                barrier()
        finally:
            self._vae_decoders_to_cpu()
            if was_training:
                self.ar_teacher._generator.train()

        elapsed = time.perf_counter() - start_time
        barrier()
        if self.is_main_process:
            wandb_prefix = (
                f"benchmark/{schedule_tag}" if namespace_outputs else "benchmark"
            )
            wandb_data: Dict[str, Any] = {
                f"{wandb_prefix}/wall_seconds": elapsed,
                f"{wandb_prefix}/wall_seconds_per_video": elapsed / max(1, num_prompts),
            }
            rows = []
            for prompt_index, prompt in enumerate(self.benchmark_prompts):
                group_name = self._benchmark_t2av_group_name(prompt_index)
                sample_path = os.path.join(
                    group_dirs.get(group_name, step_dir), f"sample_{prompt_index}.mp4"
                )
                if os.path.exists(sample_path):
                    wandb_data[f"{wandb_prefix}/{group_name}/sample_{prompt_index}"] = (
                        wandb.Video(
                            sample_path, fps=self.benchmark_video_fps, format="mp4"
                        )
                    )
                    rows.append(
                        [
                            prompt_index,
                            group_name,
                            prompt_num_frames[prompt_index],
                            prompt,
                            sample_path,
                        ]
                    )
            if rows:
                wandb_data[f"{wandb_prefix}/prompt_table"] = wandb.Table(
                    columns=["index", "group", "num_frames", "prompt", "local_path"],
                    data=rows,
                )
            wandb.log(wandb_data, step=self.step, commit=False)
            print(
                f"[Stage2 TF-dCM][Benchmark][{schedule_tag}] done in {elapsed:.1f}s; "
                f"saved to {step_dir}",
                flush=True,
            )
        barrier()

    def save(self) -> None:
        if self.is_main_process:
            print("[Stage2 TF-dCM] Gathering student state...")
        generator_state = fsdp_state_dict(self.ar_teacher._generator)
        state_dict = {
            "generator": generator_state,
            "step": self.step,
            "training_mode": "causal_tf_dcm",
        }
        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path, f"checkpoint_{self.step:06d}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"[Stage2 TF-dCM] Model saved to {checkpoint_path}")
        del generator_state, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def _resume_from_checkpoint(self, resume_path: str) -> None:
        if self.is_main_process:
            print(f"[Stage2 TF-dCM] Resume from {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        if "generator" not in checkpoint:
            raise ValueError("TF-dCM resume checkpoint has no generator state")
        self.ar_teacher._generator.load_state_dict(
            checkpoint["generator"], strict=False
        )
        self.step = int(checkpoint.get("step", 0))
        if self.is_main_process:
            print(f"[Stage2 TF-dCM] Resumed at step {self.step}")

    def train(self) -> None:
        max_steps = int(self.config.get("max_steps", 100_000))
        save_iters = int(self.config.get("log_iters", 500))
        while self.step < max_steps:
            self.train_one_step()

            if (
                not self.config.get("no_save", False)
                and self.step > 0
                and self.step % save_iters == 0
            ):
                self.save()
                self.ar_teacher._generator.train()

            if (
                self.benchmark_enabled
                and self.benchmark_prompts
                and self.step > 0
                and self.step % self.benchmark_iters == 0
            ):
                self._run_benchmark_and_log()
                self.ar_teacher._generator.train()
                self.causal_teacher.eval()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: causal TF-dCM")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True
    trainer = Stage2TFDCMTrainer(config)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
