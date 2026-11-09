"""Stage-3 self-forcing DMD with matched causal AR real/fake scores."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.sc_dmd import LTX2ShortcutConsistencyDMD


class SelfContextARScoreMixin:
    """Score every rollout chunk under the same detached generated prefix."""

    def _validate_self_context_ar_scores(self) -> None:
        if not self.real_score_use_causal_wrapper:
            raise ValueError(
                "self-context AR DMD requires real_score_use_causal_wrapper=true"
            )
        if not self.fake_score_use_causal_wrapper:
            raise ValueError(
                "self-context AR DMD requires fake_score_use_causal_wrapper=true"
            )
        if not self.enable_self_forcing:
            raise ValueError("self-context AR DMD requires self-forcing rollout")

    def _score_model_forward(
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
        """Parallel 2x TF score: noisy chunk k sees generated chunks < k."""
        if clean_audio_context is None or noisy_audio is None:
            raise ValueError("self-context AR score requires audio latents")
        batch_size = int(noisy_video.shape[0])
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
                "clean_video": clean_video_context.detach(),
                "conditional_dict": conditional_dict,
                "noisy_timestep": video_sigma.to(self.dtype),
                "clean_timestep": clean_video_sigma.to(self.dtype),
                "noisy_audio": noisy_audio,
                "clean_audio": clean_audio_context.detach(),
                "noisy_audio_timestep": audio_sigma.to(self.dtype),
                "clean_audio_timestep": clean_audio_sigma.to(self.dtype),
                "use_causal_timestep": False,
            }
        )


class LTX2SelfContextARDMD(SelfContextARScoreMixin, LTX2DMD):
    """Four-step Stage-3 DMD with matched AR real/fake score factorization."""

    def __init__(self, args, device: torch.device):
        super().__init__(args, device)
        self._validate_self_context_ar_scores()


class LTX2SelfContextARShortcutConsistencyDMD(
    SelfContextARScoreMixin,
    LTX2ShortcutConsistencyDMD,
):
    """Mixed-step SC-DMD variant using the identical AR score contract."""

    def __init__(self, args, device: torch.device):
        super().__init__(args, device)
        self._validate_self_context_ar_scores()
