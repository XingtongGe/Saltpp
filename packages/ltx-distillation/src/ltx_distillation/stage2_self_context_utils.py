"""Small, testable helpers for Stage-2 self-context DMD replay."""

from __future__ import annotations

import torch


def replay_dmd_surrogate_loss(
    prediction: torch.Tensor,
    dmd_gradient: torch.Tensor,
    frame_weights: torch.Tensor,
    denominator: int,
) -> torch.Tensor:
    """Attach a detached DMD gradient to a replay prediction.

    ``dmd_gradient`` is computed on the detached pass-1 rollout.  The replay
    forward evaluates the same point with autograd enabled.  This stop-gradient
    construction gives the replay prediction the same gradient direction while
    retaining the existing Stage-2 mean-over-latent-elements loss convention.
    """
    if prediction.shape != dmd_gradient.shape:
        raise ValueError(
            "prediction and dmd_gradient must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(dmd_gradient.shape)}"
        )
    if prediction.ndim < 3:
        raise ValueError("prediction must be [B, frames, ...]")
    if frame_weights.ndim != 1 or frame_weights.shape[0] != prediction.shape[1]:
        raise ValueError(
            "frame_weights must be one-dimensional and match the frame count"
        )
    if denominator <= 0:
        raise ValueError("denominator must be positive")

    prediction_f64 = prediction.double()
    target = (prediction_f64 - dmd_gradient.double()).detach()
    reduce_dims = tuple(range(2, prediction.ndim))
    per_frame = ((prediction_f64 - target) ** 2).mean(dim=reduce_dims)
    weights = frame_weights.to(device=prediction.device, dtype=per_frame.dtype)
    return 0.5 * (per_frame * weights.unsqueeze(0)).sum() / float(denominator)
