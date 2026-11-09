"""Focused checks for generated-prefix causal AR-score DMD."""

from __future__ import annotations

import torch

from ltx_distillation.train_stage2_self_context_ar_score_dmd import (
    Stage2SelfContextARScoreDMDTrainer,
)


class _RecordingCausalDMD:
    min_step = 100
    max_step = 100

    def __init__(self) -> None:
        self.seen_video_frames = 0
        self.context_is_generated_video = False
        self.context_is_generated_audio = False

    @staticmethod
    def sample_dmd_score_sigma(batch_size: int) -> torch.Tensor:
        return torch.full((batch_size,), 0.1)

    @staticmethod
    def expand_sigma_to_frames(
        sigma_batch: torch.Tensor,
        num_video_frames: int,
        num_audio_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(sigma_batch.shape[0])
        return (
            sigma_batch.view(batch_size, 1).expand(batch_size, num_video_frames),
            sigma_batch.view(batch_size, 1).expand(batch_size, num_audio_frames),
        )

    @staticmethod
    def add_noise(
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        while sigma.ndim < original.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * original + sigma * noise

    def _compute_causal_tf_kl_grad(
        self,
        *,
        generated_clean_video: torch.Tensor,
        generated_clean_audio: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        return_video_predictions: bool,
        **kwargs,
    ):
        self.seen_video_frames = int(generated_clean_video.shape[1])
        self.context_is_generated_video = (
            clean_video_context.data_ptr() == generated_clean_video.data_ptr()
        )
        self.context_is_generated_audio = (
            clean_audio_context.data_ptr() == generated_clean_audio.data_ptr()
        )
        logs = {}
        if return_video_predictions:
            logs["_pred_real_video_x0"] = generated_clean_video + 1.0
            logs["_pred_fake_video_x0"] = generated_clean_video + 2.0
        return (
            torch.ones_like(generated_clean_video),
            torch.ones_like(generated_clean_audio),
            logs,
        )


def test_scores_share_generated_prefix_and_padding_gets_zero_gradient() -> None:
    trainer = Stage2SelfContextARScoreDMDTrainer.__new__(
        Stage2SelfContextARScoreDMDTrainer
    )
    trainer.device = torch.device("cpu")
    trainer.dmd = _RecordingCausalDMD()

    rollout_video = torch.randn(1, 7, 3, 2, 2)
    rollout_audio = torch.randn(1, 9, 4)
    grad_video, grad_audio, logs, visualization = (
        trainer._compute_detached_causal_dmd_gradient(
            rollout_video=rollout_video,
            rollout_audio=rollout_audio,
            conditional_dict={},
            unconditional_dict={},
            real_video_frames=4,
            real_audio_frames=5,
            capture_video_predictions=True,
        )
    )

    assert trainer.dmd.seen_video_frames == 4
    assert trainer.dmd.context_is_generated_video
    assert trainer.dmd.context_is_generated_audio
    assert torch.count_nonzero(grad_video[:, :4]).item() > 0
    assert torch.count_nonzero(grad_audio[:, :5]).item() > 0
    assert torch.count_nonzero(grad_video[:, 4:]).item() == 0
    assert torch.count_nonzero(grad_audio[:, 5:]).item() == 0
    assert logs["alignment/generated_prefix_context"] == 1.0
    assert visualization is not None
    assert visualization["generator_video"].shape[1] == 4
