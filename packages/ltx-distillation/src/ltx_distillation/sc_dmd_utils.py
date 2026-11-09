"""Pure schedule and flow helpers for shortcut-consistency training."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def validate_mixed_step_config(
    step_lists: Sequence[Sequence[float]],
    probabilities: Sequence[float],
    num_train_timestep: int = 1000,
) -> Tuple[List[List[float]], List[float]]:
    """Validate and normalize mixed denoising schedules."""
    schedules = [[float(value) for value in schedule] for schedule in step_lists]
    probs = [float(value) for value in probabilities]
    if not schedules:
        raise ValueError("mixed_denoising_step_lists must not be empty")
    if len(schedules) != len(probs):
        raise ValueError(
            "mixed_denoising_step_lists and mixed_step_probs must have equal length"
        )
    if any(probability <= 0 for probability in probs):
        raise ValueError("mixed_step_probs must contain only positive values")

    seen_step_counts = set()
    for schedule in schedules:
        if len(schedule) < 2:
            raise ValueError(
                "each mixed denoising schedule needs at least one step plus terminal 0"
            )
        if abs(schedule[-1]) > 1e-8:
            raise ValueError(f"mixed denoising schedule must end at 0: {schedule}")
        if any(value < 0 or value > num_train_timestep for value in schedule):
            raise ValueError(
                f"mixed denoising schedule is outside [0, {num_train_timestep}]: "
                f"{schedule}"
            )
        if any(left <= right for left, right in zip(schedule, schedule[1:])):
            raise ValueError(
                f"mixed denoising schedule must be strictly descending: {schedule}"
            )
        step_count = len(schedule) - 1
        if step_count in seen_step_counts:
            raise ValueError(f"duplicate mixed denoising step count: {step_count}")
        seen_step_counts.add(step_count)

    probability_sum = sum(probs)
    return schedules, [probability / probability_sum for probability in probs]


def shortcut_pair_candidates(
    sigmas: Sequence[float],
    exit_index: int,
    num_segments: int,
) -> List[Tuple[int, int]]:
    """Return valid ``(mid_index, end_index)`` pairs on a nested sigma grid."""
    values = [float(value) for value in sigmas]
    num_steps = len(values) - 1
    if num_steps < 2 or exit_index < 0 or exit_index >= num_steps - 1:
        return []
    if num_segments <= 0:
        raise ValueError("shortcut_num_segments must be positive")

    boundary_indices = sorted(
        {segment * num_steps // num_segments for segment in range(num_segments)}
        | {num_steps}
    )
    next_sigma = values[exit_index + 1]
    valid_end_indices = [
        index for index in boundary_indices if values[index] < next_sigma
    ]
    valid_end_indices = sorted(
        valid_end_indices,
        key=lambda index: values[index],
        reverse=True,
    )[:2]

    pairs = []
    sigma_start = values[exit_index]
    for end_index in valid_end_indices:
        sigma_end = values[end_index]
        for mid_index in range(exit_index + 1, num_steps):
            sigma_mid = values[mid_index]
            if sigma_end < sigma_mid < sigma_start:
                pairs.append((mid_index, end_index))
    return pairs


def flow_velocity_from_x0(
    noisy: torch.Tensor,
    pred_x0: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Recover flow velocity from an x0-prediction model."""
    return (noisy.float() - pred_x0.float()) / max(float(sigma), 1e-8)


def euler_endpoint(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    sigma_start: float,
    sigma_end: float,
) -> torch.Tensor:
    """Apply one Euler transport step in decreasing-sigma direction."""
    return latent.float() - (float(sigma_start) - float(sigma_end)) * velocity.float()
