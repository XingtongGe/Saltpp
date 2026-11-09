"""Causal teacher-forced DMD with a causal fake and configurable real score.

The fake score receives packed ``[clean GT | noisy candidate]`` inputs and
teacher-forcing AV masks. The real score can either use the identical causal
conditioning or evaluate the full noisy clip with a frozen bidirectional base.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ltx_distillation.dmd_stage2 import DMDStage2


class CausalTFDMD(DMDStage2):
    """DMDStage2 variant with causal fake and AR/bidirectional real score."""

    def __init__(self, args, device: torch.device):
        super().__init__(args, device)
        self.real_score_type = str(
            getattr(args, "real_score_type", "causal_ar")
        ).lower()
        if self.real_score_type not in {"causal_ar", "bidirectional"}:
            raise ValueError(
                "real_score_type must be 'causal_ar' or 'bidirectional', got "
                f"{self.real_score_type!r}"
            )

    def _causal_tf_score_forward(
        self,
        *,
        model: torch.nn.Module,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict x0 with a packed teacher-forcing causal score model."""
        batch_size = noisy_video.shape[0]
        clean_video_sigma = torch.zeros(
            batch_size,
            clean_video_context.shape[1],
            device=noisy_video.device,
            dtype=video_sigma.dtype,
        )
        clean_audio_sigma = torch.zeros(
            batch_size,
            clean_audio_context.shape[1],
            device=noisy_audio.device,
            dtype=audio_sigma.dtype,
        )
        return model(
            _tf_kwargs={
                "noisy_video": noisy_video,
                "clean_video": clean_video_context,
                "conditional_dict": conditional_dict,
                "noisy_timestep": video_sigma.to(self.dtype),
                "clean_timestep": clean_video_sigma.to(self.dtype),
                "noisy_audio": noisy_audio,
                "clean_audio": clean_audio_context,
                "noisy_audio_timestep": audio_sigma.to(self.dtype),
                "clean_audio_timestep": clean_audio_sigma.to(self.dtype),
                "use_causal_timestep": False,
            }
        )

    def _bidirectional_score_forward(
        self,
        *,
        model: torch.nn.Module,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict x0 with a full-clip bidirectional score model."""
        return model(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

    @torch.no_grad()
    def _compute_causal_tf_kl_grad(
        self,
        *,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        generated_clean_video: torch.Tensor,
        generated_clean_audio: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
        normalization: bool = True,
        return_video_predictions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.real_score is None or self.fake_score is None:
            raise RuntimeError("Causal TF-DMD score models have not been attached")

        # The fake score always tracks the generator under the identical clean
        # GT prefix and TF mask. The real branch is selected below.
        pred_fake_video, pred_fake_audio = self._causal_tf_score_forward(
            model=self.fake_score,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video_context=clean_video_context,
            clean_audio_context=clean_audio_context,
            conditional_dict=conditional_dict,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )
        if getattr(self, "real_score_type", "causal_ar") == "bidirectional":
            pred_real_cond_video, pred_real_cond_audio = (
                self._bidirectional_score_forward(
                    model=self.real_score,
                    noisy_video=noisy_video,
                    noisy_audio=noisy_audio,
                    conditional_dict=conditional_dict,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                )
            )
            pred_real_uncond_video, pred_real_uncond_audio = (
                self._bidirectional_score_forward(
                    model=self.real_score,
                    noisy_video=noisy_video,
                    noisy_audio=noisy_audio,
                    conditional_dict=unconditional_dict,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                )
            )
        else:
            pred_real_cond_video, pred_real_cond_audio = (
                self._causal_tf_score_forward(
                    model=self.real_score,
                    noisy_video=noisy_video,
                    noisy_audio=noisy_audio,
                    clean_video_context=clean_video_context,
                    clean_audio_context=clean_audio_context,
                    conditional_dict=conditional_dict,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                )
            )
            pred_real_uncond_video, pred_real_uncond_audio = (
                self._causal_tf_score_forward(
                    model=self.real_score,
                    noisy_video=noisy_video,
                    noisy_audio=noisy_audio,
                    clean_video_context=clean_video_context,
                    clean_audio_context=clean_audio_context,
                    conditional_dict=unconditional_dict,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                )
            )

        real_video_guidance_scale = self.sample_real_video_guidance_scale(
            pred_real_cond_video.device
        )
        real_audio_guidance_scale = self.sample_real_audio_guidance_scale(
            pred_real_cond_audio.device
        )
        pred_real_video = pred_real_uncond_video + real_video_guidance_scale * (
            pred_real_cond_video - pred_real_uncond_video
        )
        pred_real_audio = pred_real_uncond_audio + real_audio_guidance_scale * (
            pred_real_cond_audio - pred_real_uncond_audio
        )

        grad_video = pred_fake_video - pred_real_video
        grad_audio = pred_fake_audio - pred_real_audio

        if normalization:
            p_real_video = generated_clean_video - pred_real_video
            if video_loss_mask is not None:
                video_mask = video_loss_mask.to(p_real_video.dtype)[
                    ..., None, None, None
                ]
                active = (
                    video_mask.expand_as(p_real_video)
                    .sum(dim=[1, 2, 3, 4], keepdim=True)
                    .clamp_min(1.0)
                )
                normalizer_video = (torch.abs(p_real_video) * video_mask).sum(
                    dim=[1, 2, 3, 4], keepdim=True
                ) / active
            else:
                normalizer_video = torch.abs(p_real_video).mean(
                    dim=[1, 2, 3, 4], keepdim=True
                )
            grad_video = grad_video / (normalizer_video + 1e-8)

            p_real_audio = generated_clean_audio - pred_real_audio
            if audio_loss_mask is not None:
                audio_mask = audio_loss_mask.to(p_real_audio.dtype).unsqueeze(-1)
                active = (
                    audio_mask.expand_as(p_real_audio)
                    .sum(dim=[1, 2], keepdim=True)
                    .clamp_min(1.0)
                )
                normalizer_audio = (torch.abs(p_real_audio) * audio_mask).sum(
                    dim=[1, 2], keepdim=True
                ) / active
            else:
                normalizer_audio = torch.abs(p_real_audio).mean(
                    dim=[1, 2], keepdim=True
                )
            grad_audio = grad_audio / (normalizer_audio + 1e-8)

        grad_video = torch.nan_to_num(grad_video)
        grad_audio = torch.nan_to_num(grad_audio)
        log_dict = {
            "dmdtrain_gradient_norm_video": torch.mean(
                torch.abs(grad_video)
            ).item(),
            "dmdtrain_gradient_norm_audio": torch.mean(
                torch.abs(grad_audio)
            ).item(),
            "real_score_video": torch.mean(torch.abs(pred_real_video)).item(),
            "real_score_audio": torch.mean(torch.abs(pred_real_audio)).item(),
            "fake_score_video": torch.mean(torch.abs(pred_fake_video)).item(),
            "fake_score_audio": torch.mean(torch.abs(pred_fake_audio)).item(),
            "real_video_guidance_scale": real_video_guidance_scale,
            "real_audio_guidance_scale": real_audio_guidance_scale,
        }
        if return_video_predictions:
            # Visualization-only payload. Ordinary causal TF-DMD callers keep
            # this disabled and therefore retain no prediction tensors.
            log_dict["_pred_real_video_x0"] = pred_real_video.detach()
            log_dict["_pred_fake_video_x0"] = pred_fake_video.detach()
        return grad_video, grad_audio, log_dict

    def compute_distribution_matching_loss(
        self,
        *,
        pred_x0_video: torch.Tensor,
        pred_x0_audio: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
        return_video_predictions: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply causal TF real/fake score difference to the student x0."""
        batch_size, video_frames = pred_x0_video.shape[:2]
        audio_frames = pred_x0_audio.shape[1]

        with torch.no_grad():
            score_sigma = self.sample_dmd_score_sigma(batch_size)
            video_sigma, audio_sigma = self.expand_sigma_to_frames(
                score_sigma,
                num_video_frames=video_frames,
                num_audio_frames=audio_frames,
            )
            noisy_video = self.add_noise(
                pred_x0_video.flatten(0, 1),
                torch.randn_like(pred_x0_video).flatten(0, 1),
                video_sigma.flatten(0, 1),
            ).unflatten(0, (batch_size, video_frames))
            noisy_audio = self.add_noise(
                pred_x0_audio,
                torch.randn_like(pred_x0_audio),
                audio_sigma,
            )
            grad_video, grad_audio, log_dict = self._compute_causal_tf_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                generated_clean_video=pred_x0_video,
                generated_clean_audio=pred_x0_audio,
                clean_video_context=clean_video_context,
                clean_audio_context=clean_audio_context,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=video_loss_mask,
                audio_loss_mask=audio_loss_mask,
                return_video_predictions=return_video_predictions,
            )

        video_block_weights = self._compute_block_weights(video_frames)
        audio_block_weights = self._compute_block_weights(audio_frames, is_audio=True)
        video_diff = (
            pred_x0_video.double()
            - (pred_x0_video.double() - grad_video.double()).detach()
        )
        video_loss = 0.5 * self._masked_weighted_mean(
            (video_diff**2).mean(dim=[2, 3, 4]),
            video_block_weights.unsqueeze(0),
            video_loss_mask,
        )
        audio_diff = (
            pred_x0_audio.double()
            - (pred_x0_audio.double() - grad_audio.double()).detach()
        )
        audio_loss = 0.5 * self._masked_weighted_mean(
            (audio_diff**2).mean(dim=2),
            audio_block_weights.unsqueeze(0),
            audio_loss_mask,
        )

        video_weight, audio_weight = self.get_loss_weights()
        total_loss = video_weight * video_loss + audio_weight * audio_loss
        log_dict.update(
            {
                "video_dmd_loss": video_loss.detach(),
                "audio_dmd_loss": audio_loss.detach(),
                "video_loss_weight": video_weight,
                "audio_loss_weight": audio_weight,
                "alignment/video_sigma_mean": video_sigma.float().mean().item(),
                "alignment/audio_sigma_mean": audio_sigma.float().mean().item(),
            }
        )
        return total_loss, log_dict

    def critic_loss(
        self,
        *,
        pred_x0_video_det: torch.Tensor,
        pred_x0_audio_det: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train causal fake-score under the same GT TF context as real-score."""
        if self.fake_score is None:
            raise RuntimeError("Causal TF-DMD fake score has not been attached")
        batch_size, video_frames = pred_x0_video_det.shape[:2]
        audio_frames = pred_x0_audio_det.shape[1]
        score_sigma = self.sample_fake_score_sigma(batch_size)
        video_sigma, audio_sigma = self.expand_sigma_to_frames(
            score_sigma,
            num_video_frames=video_frames,
            num_audio_frames=audio_frames,
        )
        video_noise = torch.randn_like(pred_x0_video_det)
        audio_noise = torch.randn_like(pred_x0_audio_det)
        noisy_video = self.add_noise(
            pred_x0_video_det.flatten(0, 1),
            video_noise.flatten(0, 1),
            video_sigma.flatten(0, 1),
        ).unflatten(0, (batch_size, video_frames))
        noisy_audio = self.add_noise(
            pred_x0_audio_det,
            audio_noise,
            audio_sigma,
        )
        pred_video, pred_audio = self._causal_tf_score_forward(
            model=self.fake_score,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video_context=clean_video_context,
            clean_audio_context=clean_audio_context,
            conditional_dict=conditional_dict,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        video_sigma_bc = video_sigma[..., None, None, None].double().clamp_min(1e-8)
        video_flow = ((noisy_video.double() - pred_video.double()) / video_sigma_bc).to(
            self.dtype
        )
        audio_sigma_bc = audio_sigma.unsqueeze(-1).double().clamp_min(1e-8)
        audio_flow = ((noisy_audio.double() - pred_audio.double()) / audio_sigma_bc).to(
            self.dtype
        )
        video_loss = self._masked_denoising_loss(
            target=pred_x0_video_det,
            prediction=pred_video,
            noise=video_noise,
            flow_pred=video_flow,
            timestep=video_sigma,
            mask=video_loss_mask,
        )
        audio_loss = self._masked_denoising_loss(
            target=pred_x0_audio_det,
            prediction=pred_audio,
            noise=audio_noise,
            flow_pred=audio_flow,
            timestep=audio_sigma,
            mask=audio_loss_mask,
        )
        video_weight, audio_weight = self.get_loss_weights()
        return video_weight * video_loss + audio_weight * audio_loss, {
            "critic_video_loss": video_loss.item(),
            "critic_audio_loss": audio_loss.item(),
            "critic_sigma_mean": video_sigma.float().mean().item(),
        }
