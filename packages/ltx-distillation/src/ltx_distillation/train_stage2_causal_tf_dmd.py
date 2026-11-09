"""Isolated Stage-2 causal teacher-forced DMD trainer.

The frozen Stage-0 AR teacher is the real score.  A separate trainable causal
model is the fake score.  Both score paths use identical GT teacher-forcing
context and masks; no CM objective or CM/DMD alternation is present here.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from ltx_distillation.ar_teacher.ar_teacher import LTX2ARTeacher
from ltx_distillation.causal_tf_dmd import CausalTFDMD
from ltx_distillation.tf_dcm_utils import add_noise
from ltx_distillation.train_stage2 import barrier, fsdp_state_dict, fsdp_wrap
from ltx_distillation.train_stage2_tf_dcm import Stage2TFDCMTrainer


class Stage2CausalTFDMDTrainer(Stage2TFDCMTrainer):
    """Pure causal TF-DMD; deliberately separate from the TF-dCM trainer."""

    training_mode = "causal_tf_dmd"

    def __init__(self, config: DictConfig):
        self._deferred_resume_path: Optional[str] = None
        super().__init__(config)

        self.dfake_gen_update_ratio = int(config.get("dfake_gen_update_ratio", 5))
        if self.dfake_gen_update_ratio <= 0:
            raise ValueError("dfake_gen_update_ratio must be positive")

        self.dmd = CausalTFDMD(config, device=self.device)
        self.dmd.real_score = self.causal_teacher

        fake_score_ckpt = str(
            config.get(
                "fake_score_ckpt",
                config.get("causal_teacher_ckpt", config.get("generator_ckpt")),
            )
        )
        fake_score = self._build_causal_fake_score(fake_score_ckpt)
        sharding_strategy = config.get("sharding_strategy", "hybrid_full")
        wrap_strategy = config.get(
            "fake_score_fsdp_wrap_strategy",
            config.get("generator_fsdp_wrap_strategy", "size"),
        )
        transformer_module = None
        if wrap_strategy == "transformer":
            from ltx_causal.transformer.causal_block import CausalAVTransformerBlock

            transformer_module = (CausalAVTransformerBlock,)
        if self.is_main_process:
            print("[Stage2 causal TF-DMD] FSDP wrapping causal fake-score...")
        self.dmd.fake_score = fsdp_wrap(
            fake_score,
            sharding_strategy=sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=wrap_strategy,
            transformer_module=transformer_module,
        )
        del fake_score
        torch.cuda.empty_cache()

        self.critic_lr = float(config.get("critic_lr", config.get("lr", 2e-5)))
        critic_params = [
            parameter
            for parameter in self.dmd.fake_score.parameters()
            if parameter.requires_grad
        ]
        if not critic_params:
            raise RuntimeError("causal fake-score has no trainable parameters")
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=self.critic_lr,
            betas=(
                float(config.get("beta1", 0.9)),
                float(config.get("beta2", 0.999)),
            ),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )

        if self._deferred_resume_path:
            self._resume_fake_score(self._deferred_resume_path)

        self._init_dmd_local_visualization()

        if self.is_main_process:
            dmd_support = self.dmd.score_sigma_support()
            fake_support = self.dmd.score_sigma_support(fake_score=True)
            print(
                "[Stage2 causal TF-DMD] ready | "
                f"generator_every={self.dfake_gen_update_ratio} "
                "real_video_cfg_range="
                f"[{self.dmd.real_video_guidance_scale_min:g},"
                f"{self.dmd.real_video_guidance_scale_max:g}] "
                "real_audio_cfg_range="
                f"[{self.dmd.real_audio_guidance_scale_min:g},"
                f"{self.dmd.real_audio_guidance_scale_max:g}] "
                f"DMD_sigma={self.dmd.dmd_timestep_sampler_type} "
                f"shift={self.dmd.dmd_sigma_shift:g} support={dmd_support} "
                f"fake_sigma={self.dmd.fake_score_timestep_sampler_type} "
                f"shift={self.dmd.fake_score_sigma_shift:g} support={fake_support} "
                f"real={self.dmd.real_score_type}, "
                "fake=trainable causal TF score"
            )

    def _build_causal_fake_score(self, checkpoint_path: str) -> torch.nn.Module:
        from ltx_causal import CausalLTXModel

        if self.is_main_process:
            print(
                "[Stage2 causal TF-DMD] Building causal fake-score from "
                f"{checkpoint_path}"
            )
        model = CausalLTXModel(self.ar_teacher._make_model_config())
        fake_score = self.ar_teacher._make_wrapper(model).to(self.device, self.dtype)
        state_dict = LTX2ARTeacher._load_checkpoint_state_dict(checkpoint_path)
        missing, unexpected = fake_score.load_state_dict(state_dict, strict=False)
        real_missing = [key for key in missing if "mask_builder" not in key]
        if real_missing:
            raise RuntimeError(
                "Causal fake-score initialization is incomplete; missing keys include "
                f"{real_missing[:10]}"
            )
        if self.is_main_process and unexpected:
            print(
                f"[Stage2 causal TF-DMD] fake-score unexpected keys "
                f"({len(unexpected)}): {unexpected[:10]}"
            )
        fake_score.requires_grad_(True)
        if self.config.get("gradient_checkpointing", False):
            fake_score.enable_gradient_checkpointing()
        fake_score.train()
        return fake_score

    def _apply_lr(self) -> None:
        scale = self._lr_scale()
        for param_group in self.generator_optimizer.param_groups:
            param_group["lr"] = self.gen_lr * scale
        if hasattr(self, "critic_optimizer"):
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = self.critic_lr * scale

    def _sample_generator_tf_prediction(
        self,
        *,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        sigma_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_video = add_noise(
            clean_video,
            torch.randn_like(clean_video),
            sigma_batch[:, None],
        )
        noisy_audio = add_noise(
            clean_audio,
            torch.randn_like(clean_audio),
            sigma_batch[:, None],
        )
        return self._tf_predict_noisy(
            model=self.ar_teacher._generator,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=conditional_dict,
            sigma=sigma_batch[:, None],
        )

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)

    def train_one_step(self) -> None:
        self.ar_teacher._generator.train()
        self.causal_teacher.eval()
        self.dmd.fake_score.train()
        self._assert_teacher_frozen(check_grads=False)
        self._apply_lr()
        self.dmd.current_step = self.step

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        batch = next(self.dataloader)
        clean_video, clean_audio, text_prompts = self._encode_batch(batch)
        clean_audio = self._align_audio(clean_video, clean_audio)
        with torch.no_grad():
            conditional_dict = self.ar_teacher.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self._get_unconditional_dict(len(text_prompts))

        sigma_batch = self.dmd.sample_few_step_sigma(clean_video.shape[0])
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(sigma_batch, src=0)
        train_generator = self.step % self.dfake_gen_update_ratio == 0
        run_local_visualization = self._should_run_dmd_local_visualization(
            train_generator=train_generator
        )

        generator_loss_value = 0.0
        generator_grad_norm = 0.0
        generator_log: Dict[str, Any] = {}
        local_visualization_context: Optional[Dict[str, Any]] = None

        if train_generator:
            self.dmd.fake_score.requires_grad_(False)
            self.causal_teacher.requires_grad_(False)
            self.generator_optimizer.zero_grad(set_to_none=True)
            pred_video, pred_audio = self._sample_generator_tf_prediction(
                clean_video=clean_video,
                clean_audio=clean_audio,
                conditional_dict=conditional_dict,
                sigma_batch=sigma_batch,
            )
            generator_loss, generator_log = self.dmd.compute_distribution_matching_loss(
                pred_x0_video=pred_video,
                pred_x0_audio=pred_audio,
                clean_video_context=clean_video,
                clean_audio_context=clean_audio,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                return_video_predictions=(
                    run_local_visualization and self.is_main_process
                ),
            )
            if run_local_visualization and self.is_main_process:
                local_visualization_context = (
                    self._capture_dmd_local_visualization(
                        generator_video=pred_video,
                        generator_log=generator_log,
                        generator_sigma_mean=float(
                            sigma_batch.float().mean().item()
                        ),
                        prompt=text_prompts[0],
                        step=self.step,
                    )
                )
            generator_loss.backward()
            self._assert_teacher_frozen(check_grads=True)
            if hasattr(self.ar_teacher._generator, "clip_grad_norm_"):
                generator_grad_norm = self.ar_teacher._generator.clip_grad_norm_(
                    self.max_grad_norm
                )
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ar_teacher._generator.parameters(), self.max_grad_norm
                )
            self.generator_optimizer.step()
            generator_loss_value = float(generator_loss.item())
            del generator_loss, pred_video, pred_audio
            torch.cuda.empty_cache()

        self.dmd.fake_score.requires_grad_(True)
        self.causal_teacher.requires_grad_(False)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pred_video_det, pred_audio_det = self._sample_generator_tf_prediction(
                clean_video=clean_video,
                clean_audio=clean_audio,
                conditional_dict=conditional_dict,
                sigma_batch=sigma_batch,
            )
            pred_video_det = pred_video_det.detach()
            pred_audio_det = pred_audio_det.detach()

        critic_loss, critic_log = self.dmd.critic_loss(
            pred_x0_video_det=pred_video_det,
            pred_x0_audio_det=pred_audio_det,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=conditional_dict,
        )
        critic_loss.backward()
        self._assert_teacher_frozen(check_grads=True)
        if hasattr(self.dmd.fake_score, "clip_grad_norm_"):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(self.max_grad_norm)
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()

        if run_local_visualization:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self._run_dmd_local_visualization(local_visualization_context)
            del local_visualization_context
            torch.cuda.empty_cache()

        if self.is_main_process:
            log_data = {
                "train/critic_loss": float(critic_loss.item()),
                "train/critic_video_loss": critic_log["critic_video_loss"],
                "train/critic_audio_loss": critic_log["critic_audio_loss"],
                "train/critic_sigma_mean": critic_log["critic_sigma_mean"],
                "train/critic_grad_norm": self._scalar(critic_grad_norm),
                "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                "train/generator_sigma_mean": float(sigma_batch.float().mean().item()),
                "train/real_audio_cfg": float(self.dmd.real_audio_guidance_scale),
            }
            if train_generator:
                log_data.update(
                    {
                        "train/generator_loss": generator_loss_value,
                        "train/generator_video_dmd_loss": self._scalar(
                            generator_log["video_dmd_loss"]
                        ),
                        "train/generator_audio_dmd_loss": self._scalar(
                            generator_log["audio_dmd_loss"]
                        ),
                        "train/generator_grad_norm": self._scalar(generator_grad_norm),
                        "train/generator_lr": self.generator_optimizer.param_groups[0][
                            "lr"
                        ],
                        "train/dmd_gradient_norm_video": generator_log[
                            "dmdtrain_gradient_norm_video"
                        ],
                        "train/dmd_gradient_norm_audio": generator_log[
                            "dmdtrain_gradient_norm_audio"
                        ],
                        "train/dmd_sigma_mean": generator_log[
                            "alignment/video_sigma_mean"
                        ],
                        "train/real_video_cfg": generator_log[
                            "real_video_guidance_scale"
                        ],
                        "train/real_audio_cfg": generator_log[
                            "real_audio_guidance_scale"
                        ],
                    }
                )
            wandb.log(log_data, step=self.step)
            if self.step < 10 or self.step % 50 == 0:
                branch = "G+C" if train_generator else "C"
                print(
                    f"[Stage2 causal TF-DMD][{self.step}] {branch} | "
                    f"gen={generator_loss_value:.5f} "
                    f"critic={critic_loss.item():.5f} "
                    f"sigma={sigma_batch.float().mean().item():.4f} "
                    f"g_grad={self._scalar(generator_grad_norm):.4f} "
                    f"c_grad={self._scalar(critic_grad_norm):.4f}"
                )

        del (
            clean_video,
            clean_audio,
            conditional_dict,
            unconditional_dict,
            pred_video_det,
            pred_audio_det,
            critic_loss,
        )
        torch.cuda.empty_cache()

    def save(self) -> None:
        if self.is_main_process:
            print("[Stage2 causal TF-DMD] Gathering student/fake-score states...")
        generator_state = fsdp_state_dict(self.ar_teacher._generator)
        fake_score_state = fsdp_state_dict(self.dmd.fake_score)
        state_dict = {
            "generator": generator_state,
            "fake_score": fake_score_state,
            "step": self.step,
            "training_mode": self.training_mode,
        }
        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path, f"checkpoint_{self.step:06d}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"[Stage2 causal TF-DMD] Model saved to {checkpoint_path}")
        del generator_state, fake_score_state, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def _resume_from_checkpoint(self, resume_path: str) -> None:
        # super().__init__ invokes this before the causal fake-score is built.
        # Load the student immediately and defer the fake-score state.
        Stage2TFDCMTrainer._resume_from_checkpoint(self, resume_path)
        self._deferred_resume_path = resume_path

    def _resume_fake_score(self, resume_path: str) -> None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        if "fake_score" not in checkpoint:
            raise ValueError("causal TF-DMD resume checkpoint has no fake_score state")
        self.dmd.fake_score.load_state_dict(checkpoint["fake_score"], strict=False)
        del checkpoint
        if self.is_main_process:
            print("[Stage2 causal TF-DMD] fake-score resume state loaded")

    def train(self) -> None:
        max_steps = int(self.config.get("max_steps", 30_000))
        save_iters = int(self.config.get("log_iters", 800))
        while self.step < max_steps:
            self.train_one_step()
            if (
                not self.config.get("no_save", False)
                and self.step > 0
                and self.step % save_iters == 0
            ):
                self.save()
                self.ar_teacher._generator.train()
                self.dmd.fake_score.train()
            if (
                self.benchmark_enabled
                and self.benchmark_prompts
                and self.step > 0
                and self.step % self.benchmark_iters == 0
            ):
                self._run_benchmark_and_log()
                self.ar_teacher._generator.train()
                self.causal_teacher.eval()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: causal TF-DMD")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True
    trainer = Stage2CausalTFDMDTrainer(config)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
