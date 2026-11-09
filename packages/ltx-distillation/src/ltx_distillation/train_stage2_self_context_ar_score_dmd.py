"""Stage-2 self-context DMD with matched causal AR real/fake scores.

This experiment combines two isolated, previously validated paths:

* the student's two-pass self-context KV-cache rollout/replay; and
* causal packed teacher-forcing real/fake score forwards.

The detached pass-1 student rollout is the clean half of every score forward.
Consequently noisy block ``k`` in the student, real score, and fake score is
conditioned on the same generated latent prefix ``x_hat[:k]``.  The packed TF
mask still hides clean block ``k`` and all future blocks, so this alignment
does not leak the current target into either score prediction.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from ltx_distillation.train_stage2_causal_tf_dmd import (
    Stage2CausalTFDMDTrainer,
)
from ltx_distillation.train_stage2_self_context_dmd import (
    Stage2SelfContextDMDTrainer,
)


class Stage2SelfContextARScoreDMDTrainer(Stage2CausalTFDMDTrainer):
    """Self-context student replay plus causal AR-score DMD."""

    training_mode = "self_context_ar_score_dmd"

    # Reuse the exact student rollout/replay implementation from the isolated
    # bidirectional-score experiment. Only the score-gradient method below is
    # changed. Keeping one implementation prevents the KV-cache, padding, and
    # replay semantics from drifting between the two experiments.
    _causal_blocks = Stage2SelfContextDMDTrainer._causal_blocks
    _synchronized_rollout_blocks = (
        Stage2SelfContextDMDTrainer._synchronized_rollout_blocks
    )
    _pad_frames = staticmethod(Stage2SelfContextDMDTrainer._pad_frames)
    _generator_kv_block_forward = (
        Stage2SelfContextDMDTrainer._generator_kv_block_forward
    )
    _add_context_residual = staticmethod(
        Stage2SelfContextDMDTrainer._add_context_residual
    )
    _refresh_generated_context = (
        Stage2SelfContextDMDTrainer._refresh_generated_context
    )
    _build_noisy_gt_inputs = Stage2SelfContextDMDTrainer._build_noisy_gt_inputs
    _run_detached_self_context_rollout = (
        Stage2SelfContextDMDTrainer._run_detached_self_context_rollout
    )
    _replay_dmd_and_backward = (
        Stage2SelfContextDMDTrainer._replay_dmd_and_backward
    )
    _run_local_visualization = (
        Stage2SelfContextDMDTrainer._run_local_visualization
    )
    _scalar = staticmethod(Stage2SelfContextDMDTrainer._scalar)

    def __init__(self, config: DictConfig):
        if not bool(config.get("self_context_replay_enabled", True)):
            raise ValueError("self-context AR-score DMD requires replay")
        if not bool(config.get("same_step_across_blocks", True)):
            raise ValueError(
                "self-context AR-score DMD requires "
                "same_step_across_blocks=true"
            )
        if bool(config.get("svi_enabled", False)):
            raise ValueError(
                "the initial self-context AR-score DMD experiment keeps SVI off"
            )

        super().__init__(config)
        self.replay_check_tolerance = float(
            config.get("self_context_replay_check_tolerance", 5e-3)
        )
        self.strict_replay_check = bool(
            config.get("self_context_strict_replay_check", False)
        )
        self.local_visualization_iters = int(
            config.get("local_visualization_iters", 0)
        )
        self.local_video_fps = int(
            config.get(
                "local_video_fps",
                config.get("benchmark_video_fps", 24),
            )
        )
        if (
            self.local_visualization_iters > 0
            and self.is_main_process
            and not hasattr(self, "video_vae")
        ):
            self._build_vae_decoders()

        if self.is_main_process:
            print(
                "[Stage2 self-context AR-score DMD] ready | "
                "student=GT-noised chunk + generated-prefix KV | "
                "real/fake=causal packed TF with the same generated prefix | "
                "two_pass_replay=true | "
                "real_cfg_ranges="
                f"video[{self.dmd.real_video_guidance_scale_min:g},"
                f"{self.dmd.real_video_guidance_scale_max:g}] "
                f"audio[{self.dmd.real_audio_guidance_scale_min:g},"
                f"{self.dmd.real_audio_guidance_scale_max:g}] | "
                f"local_video_every={self.local_visualization_iters}",
                flush=True,
            )

    @torch.no_grad()
    def _compute_detached_causal_dmd_gradient(
        self,
        *,
        rollout_video: torch.Tensor,
        rollout_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        real_video_frames: int,
        real_audio_frames: int,
        capture_video_predictions: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
        Optional[Dict[str, Any]],
    ]:
        """Evaluate causal scores using the student rollout as clean prefix."""

        batch_size = int(rollout_video.shape[0])
        generated_video = rollout_video[:, :real_video_frames]
        generated_audio = rollout_audio[:, :real_audio_frames]

        score_sigma = self.dmd.sample_dmd_score_sigma(batch_size)
        video_sigma, audio_sigma = self.dmd.expand_sigma_to_frames(
            score_sigma,
            num_video_frames=real_video_frames,
            num_audio_frames=real_audio_frames,
        )
        noisy_video = self.dmd.add_noise(
            generated_video.flatten(0, 1),
            torch.randn_like(generated_video).flatten(0, 1),
            video_sigma.flatten(0, 1),
        ).unflatten(0, (batch_size, real_video_frames))
        noisy_audio = self.dmd.add_noise(
            generated_audio,
            torch.randn_like(generated_audio),
            audio_sigma,
        )

        grad_video, grad_audio, log_dict = (
            self.dmd._compute_causal_tf_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                generated_clean_video=generated_video,
                generated_clean_audio=generated_audio,
                # This is the defining change from the old AR-score TF-DMD:
                # all three models condition on the same detached student
                # prefix instead of GT. Packed masks hide the current block.
                clean_video_context=generated_video,
                clean_audio_context=generated_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=None,
                audio_loss_mask=None,
                return_video_predictions=capture_video_predictions,
            )
        )

        visualization_context = None
        if capture_video_predictions:
            pred_real_video = log_dict.pop("_pred_real_video_x0")
            pred_fake_video = log_dict.pop("_pred_fake_video_x0")
            visualization_context = {
                "generator_video": generated_video[:1].detach().cpu(),
                "real_score_video_x0": pred_real_video[:1].detach().cpu(),
                "fake_score_video_x0": pred_fake_video[:1].detach().cpu(),
                "video_sigma_mean": float(video_sigma.float().mean().item()),
            }

        log_dict.update(
            {
                "alignment/video_sigma_mean": video_sigma.float().mean().item(),
                "alignment/audio_sigma_mean": audio_sigma.float().mean().item(),
                "alignment/generated_prefix_context": 1.0,
            }
        )

        # Dummy suffix blocks synchronize student FSDP call counts only. Score
        # models never see them; replay receives zero gradient on that suffix.
        grad_video_full = torch.zeros_like(rollout_video)
        grad_audio_full = torch.zeros_like(rollout_audio)
        grad_video_full[:, :real_video_frames] = grad_video
        grad_audio_full[:, :real_audio_frames] = grad_audio
        return (
            grad_video_full,
            grad_audio_full,
            log_dict,
            visualization_context,
        )

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
            conditional_dict = self.ar_teacher.text_encoder(
                text_prompts=text_prompts
            )
        unconditional_dict = self._get_unconditional_dict(len(text_prompts))

        sigma_batch = self.dmd.sample_few_step_sigma(clean_video.shape[0])
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(sigma_batch, src=0)
        real_video_frames = int(clean_video.shape[1])
        real_audio_frames = int(clean_audio.shape[1])

        blocks, real_block_count = self._synchronized_rollout_blocks(
            real_video_frames
        )
        rollout_video_frames = int(blocks[-1].video_end)
        rollout_audio_frames = int(blocks[-1].audio_end)
        generator_video_padded = self._pad_frames(
            clean_video, rollout_video_frames
        )
        generator_audio_padded = self._pad_frames(
            clean_audio, rollout_audio_frames
        )
        video_sigma = sigma_batch[:, None].expand(
            clean_video.shape[0], rollout_video_frames
        ).contiguous()
        audio_sigma = sigma_batch[:, None].expand(
            clean_video.shape[0], rollout_audio_frames
        ).contiguous()

        train_generator = self.step % self.dfake_gen_update_ratio == 0
        run_local_visualization = bool(
            train_generator
            and self.local_visualization_iters > 0
            and self.step > 0
            and self.step % self.local_visualization_iters == 0
            and not bool(self.config.get("no_visualize", False))
        )
        generator_metrics: Dict[str, float] = {
            "loss": 0.0,
            "video_loss": 0.0,
            "audio_loss": 0.0,
            "replay_max_abs": 0.0,
            "replay_mean_abs": 0.0,
            "num_blocks": 0.0,
        }
        generator_log: Dict[str, Any] = {}
        generator_grad_norm: Any = 0.0
        local_visualization_context: Optional[Dict[str, Any]] = None

        if train_generator:
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.dmd.fake_score.requires_grad_(False)
            self.causal_teacher.requires_grad_(False)
            self.generator_optimizer.zero_grad(set_to_none=True)

            noisy_video, noisy_audio = self._build_noisy_gt_inputs(
                generator_video_source=generator_video_padded,
                generator_audio_source=generator_audio_padded,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )
            pass1_video, pass1_audio, blocks = (
                self._run_detached_self_context_rollout(
                    noisy_video=noisy_video,
                    noisy_audio=noisy_audio,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                    conditional_dict=conditional_dict,
                    video_context_residual=None,
                    audio_context_residual=None,
                    blocks=blocks,
                )
            )
            (
                dmd_grad_video,
                dmd_grad_audio,
                generator_log,
                local_visualization_context,
            ) = self._compute_detached_causal_dmd_gradient(
                rollout_video=pass1_video,
                rollout_audio=pass1_audio,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
                capture_video_predictions=(
                    run_local_visualization and self.is_main_process
                ),
            )
            if local_visualization_context is not None:
                local_visualization_context.update(
                    {
                        "prompt": text_prompts[0],
                        "step": int(self.step),
                        "real_video_frames": int(real_video_frames),
                    }
                )
            generator_metrics = self._replay_dmd_and_backward(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                pass1_video=pass1_video,
                pass1_audio=pass1_audio,
                dmd_grad_video=dmd_grad_video,
                dmd_grad_audio=dmd_grad_audio,
                blocks=blocks,
                video_context_residual=None,
                audio_context_residual=None,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )

            self._assert_teacher_frozen(check_grads=True)
            if hasattr(self.ar_teacher._generator, "clip_grad_norm_"):
                generator_grad_norm = (
                    self.ar_teacher._generator.clip_grad_norm_(
                        self.max_grad_norm
                    )
                )
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ar_teacher._generator.parameters(),
                    self.max_grad_norm,
                )
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            del (
                noisy_video,
                noisy_audio,
                pass1_video,
                pass1_audio,
                dmd_grad_video,
                dmd_grad_audio,
            )
            torch.cuda.empty_cache()

        # Train the causal fake score on a fresh student trajectory. Its clean
        # half is the very same detached trajectory it is asked to denoise;
        # the TF mask prevents each noisy block from seeing its own clean target.
        self.dmd.fake_score.requires_grad_(True)
        self.causal_teacher.requires_grad_(False)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            critic_noisy_video, critic_noisy_audio = self._build_noisy_gt_inputs(
                generator_video_source=generator_video_padded,
                generator_audio_source=generator_audio_padded,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )
            critic_video, critic_audio, _ = (
                self._run_detached_self_context_rollout(
                    noisy_video=critic_noisy_video,
                    noisy_audio=critic_noisy_audio,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                    conditional_dict=conditional_dict,
                    video_context_residual=None,
                    audio_context_residual=None,
                    blocks=blocks,
                )
            )
            critic_video_real = critic_video[:, :real_video_frames].detach()
            critic_audio_real = critic_audio[:, :real_audio_frames].detach()

        critic_loss, critic_log = self.dmd.critic_loss(
            pred_x0_video_det=critic_video_real,
            pred_x0_audio_det=critic_audio_real,
            clean_video_context=critic_video_real,
            clean_audio_context=critic_audio_real,
            conditional_dict=conditional_dict,
            video_loss_mask=None,
            audio_loss_mask=None,
        )
        critic_loss.backward()
        self._assert_teacher_frozen(check_grads=True)
        if hasattr(self.dmd.fake_score, "clip_grad_norm_"):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(
                self.max_grad_norm
            )
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        critic_loss_value = float(critic_loss.detach().item())

        if run_local_visualization:
            self.critic_optimizer.zero_grad(set_to_none=True)
            self._run_local_visualization(local_visualization_context)
            del local_visualization_context
            torch.cuda.empty_cache()

        if self.is_main_process:
            log_data = {
                "train/critic_loss": critic_loss_value,
                "train/critic_video_loss": critic_log["critic_video_loss"],
                "train/critic_audio_loss": critic_log["critic_audio_loss"],
                "train/critic_sigma_mean": critic_log["critic_sigma_mean"],
                "train/critic_grad_norm": self._scalar(critic_grad_norm),
                "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                "train/generator_sigma_mean": float(
                    sigma_batch.float().mean().item()
                ),
                "train/real_video_cfg": float(
                    self.dmd.real_video_guidance_scale
                ),
                "train/real_audio_cfg": float(
                    self.dmd.real_audio_guidance_scale
                ),
                "train/self_context_enabled": 1.0,
                "train/ar_score_generated_prefix_context": 1.0,
                "train/self_context_real_blocks": float(real_block_count),
                "train/self_context_synchronized_blocks": float(len(blocks)),
                "train/self_context_real_video_frames": float(
                    real_video_frames
                ),
                "train/self_context_rollout_video_frames": float(
                    rollout_video_frames
                ),
            }
            if train_generator:
                log_data.update(
                    {
                        "train/generator_loss": generator_metrics["loss"],
                        "train/generator_video_dmd_loss": generator_metrics[
                            "video_loss"
                        ],
                        "train/generator_audio_dmd_loss": generator_metrics[
                            "audio_loss"
                        ],
                        "train/generator_grad_norm": self._scalar(
                            generator_grad_norm
                        ),
                        "train/generator_lr": self.generator_optimizer.param_groups[
                            0
                        ]["lr"],
                        "train/self_context_num_blocks": generator_metrics[
                            "num_blocks"
                        ],
                        "train/self_context_replay_max_abs": generator_metrics[
                            "replay_max_abs"
                        ],
                        "train/self_context_replay_mean_abs": generator_metrics[
                            "replay_mean_abs"
                        ],
                        "train/dmd_gradient_norm_video": generator_log.get(
                            "dmdtrain_gradient_norm_video", 0.0
                        ),
                        "train/dmd_gradient_norm_audio": generator_log.get(
                            "dmdtrain_gradient_norm_audio", 0.0
                        ),
                        "train/dmd_sigma_mean": generator_log.get(
                            "alignment/video_sigma_mean", 0.0
                        ),
                        "train/real_video_cfg": generator_log.get(
                            "real_video_guidance_scale",
                            self.dmd.real_video_guidance_scale,
                        ),
                        "train/real_audio_cfg": generator_log.get(
                            "real_audio_guidance_scale",
                            self.dmd.real_audio_guidance_scale,
                        ),
                    }
                )
            wandb.log(log_data, step=self.step)
            if self.step < 10 or self.step % 50 == 0:
                branch = "G+C" if train_generator else "C"
                print(
                    f"[Stage2 self-context AR-score DMD][{self.step}] "
                    f"{branch} | gen={generator_metrics['loss']:.5f} "
                    f"critic={critic_loss_value:.5f} "
                    f"sigma={sigma_batch.float().mean().item():.4f} "
                    f"replay_max={generator_metrics['replay_max_abs']:.3e}",
                    flush=True,
                )

        del (
            clean_video,
            clean_audio,
            conditional_dict,
            unconditional_dict,
            generator_video_padded,
            generator_audio_padded,
            blocks,
            critic_noisy_video,
            critic_noisy_audio,
            critic_video,
            critic_audio,
            critic_video_real,
            critic_audio_real,
            critic_loss,
        )
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: self-context DMD with causal AR scores"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True
    trainer = Stage2SelfContextARScoreDMDTrainer(config)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
