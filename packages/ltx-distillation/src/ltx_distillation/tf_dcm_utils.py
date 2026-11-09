"""Small, testable primitives for teacher-forcing discrete CM training."""

from __future__ import annotations

from itertools import pairwise
from typing import Sequence

import torch


def shift_rectified_flow_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the standard RF time shift ``s*t / (1 + (s-1)*t)``."""
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError(f"sigma shift must be positive, got {shift}")
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def dcm_sigma_path(
    uniform: torch.Tensor,
    *,
    total_steps: int,
    skipping_interval_steps: int,
    sigma_shift: float,
) -> torch.Tensor:
    """Construct shifted, descending dCM times for each batch item.

    ``uniform`` must be in ``[0, 1)`` and have shape ``[B, 1]``.  The
    returned tensor has shape ``[B, K+1]`` where ``K`` is the skipping
    interval.  This follows Causal rCM's TF-dCM time construction exactly:

    ``u ~ U(0, 1-K/N)``, ``s_k = 1-(u+k/N)``, ``t_k = shift(s_k)``.
    """
    if uniform.ndim != 2 or uniform.shape[1] != 1:
        raise ValueError(f"uniform must have shape [B, 1], got {tuple(uniform.shape)}")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if skipping_interval_steps <= 0 or skipping_interval_steps > total_steps:
        raise ValueError(
            "skipping_interval_steps must be in [1, total_steps], got "
            f"{skipping_interval_steps} for total_steps={total_steps}"
        )
    if bool(((uniform < 0.0) | (uniform >= 1.0)).any()):
        raise ValueError("uniform values must be in [0, 1)")

    du = 1.0 / float(total_steps)
    max_u = 1.0 - skipping_interval_steps * du
    u = uniform * max_u
    offsets = (
        torch.arange(
            skipping_interval_steps + 1,
            device=uniform.device,
            dtype=uniform.dtype,
        )
        * du
    )
    unshifted = 1.0 - (u + offsets.unsqueeze(0))
    return shift_rectified_flow_sigma(unshifted, sigma_shift)


def expand_sigma(sigma: torch.Tensor, frames: int) -> torch.Tensor:
    """Expand a scalar-per-sample sigma ``[B]`` or ``[B,1]`` to ``[B,F]``."""
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    if sigma.ndim != 2 or sigma.shape[1] != 1:
        raise ValueError(
            f"sigma must have shape [B] or [B, 1], got {tuple(sigma.shape)}"
        )
    return sigma.expand(-1, int(frames))


def _broadcast_sigma(
    sigma: torch.Tensor | float, reference: torch.Tensor
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(
        sigma, device=reference.device, dtype=reference.dtype
    )
    if sigma_tensor.ndim == 0:
        return sigma_tensor
    if sigma_tensor.ndim == 1:
        sigma_tensor = sigma_tensor[:, None]
    if sigma_tensor.shape[0] != reference.shape[0]:
        raise ValueError(
            f"sigma batch {sigma_tensor.shape[0]} does not match reference batch "
            f"{reference.shape[0]}"
        )
    if sigma_tensor.ndim != 2 or sigma_tensor.shape[1] not in (1, reference.shape[1]):
        raise ValueError(
            "sigma must have shape [B], [B, 1], or [B, F], got "
            f"{tuple(sigma_tensor.shape)} for reference {tuple(reference.shape)}"
        )
    return sigma_tensor.view(
        sigma_tensor.shape[0], sigma_tensor.shape[1], *([1] * (reference.ndim - 2))
    )


def add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Rectified-flow interpolation ``x_sigma=(1-sigma)*x0+sigma*eps``."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {clean.shape} vs {noise.shape}")
    sigma_bc = _broadcast_sigma(sigma, clean)
    return (1.0 - sigma_bc) * clean + sigma_bc * noise


def flow_velocity_from_x0(
    noisy: torch.Tensor,
    pred_x0: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Recover RF velocity from an x0-predicting model."""
    sigma_bc = _broadcast_sigma(sigma, noisy)
    if bool((sigma_bc <= 0).any()):
        raise ValueError("flow velocity is undefined at sigma <= 0")
    return (noisy.float() - pred_x0.float()) / sigma_bc.float()


def teacher_euler_step(
    noisy: torch.Tensor,
    pred_x0: torch.Tensor,
    sigma_from: torch.Tensor | float,
    sigma_to: torch.Tensor | float,
) -> torch.Tensor:
    """Advance an RF state from ``sigma_from`` to the lower ``sigma_to``."""
    velocity = flow_velocity_from_x0(noisy, pred_x0, sigma_from)
    sigma_from_bc = _broadcast_sigma(sigma_from, noisy).float()
    sigma_to_bc = _broadcast_sigma(sigma_to, noisy).float()
    if bool((sigma_to_bc > sigma_from_bc).any()):
        raise ValueError("teacher Euler step requires sigma_to <= sigma_from")
    return (noisy.float() + (sigma_to_bc - sigma_from_bc) * velocity).to(noisy.dtype)


def classifier_free_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine conditional and unconditional x0/velocity predictions."""
    if conditional.shape != unconditional.shape:
        raise ValueError(
            f"CFG shape mismatch: {conditional.shape} vs {unconditional.shape}"
        )
    return unconditional + float(guidance_scale) * (conditional - unconditional)


def validate_benchmark_schedules(
    schedules: Sequence[Sequence[int | float]],
    *,
    num_train_timestep: int,
) -> list[list[float]]:
    """Validate Stage3-style benchmark timestep schedules."""
    validated: list[list[float]] = []
    seen_counts: set[int] = set()
    for schedule in schedules:
        values = [float(value) for value in schedule]
        if len(values) < 2:
            raise ValueError(
                "each benchmark schedule needs at least one denoising step"
            )
        if values[-1] != 0.0:
            raise ValueError(f"benchmark schedule must end at 0: {values}")
        if any(value < 0 or value > num_train_timestep for value in values):
            raise ValueError(
                f"benchmark schedule is outside [0, {num_train_timestep}]: {values}"
            )
        if any(a <= b for a, b in pairwise(values)):
            raise ValueError(
                f"benchmark schedule must be strictly descending: {values}"
            )
        step_count = len(values) - 1
        if step_count in seen_counts:
            raise ValueError(f"duplicate benchmark denoising step count: {step_count}")
        seen_counts.add(step_count)
        validated.append(values)
    if not validated:
        raise ValueError("benchmark schedules must not be empty")
    return validated
