"""Unit tests for mixed-step shortcut-consistency math."""

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "ltx_distillation"
    / "sc_dmd_utils.py"
)
SPEC = importlib.util.spec_from_file_location("sc_dmd_utils", MODULE_PATH)
sc_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sc_utils)


def test_validate_mixed_step_config_normalizes_probabilities():
    schedules, probabilities = sc_utils.validate_mixed_step_config(
        [[1000, 750, 500, 250, 0], [1000, 500, 0]],
        [2, 1],
    )

    assert schedules == [
        [1000.0, 750.0, 500.0, 250.0, 0.0],
        [1000.0, 500.0, 0.0],
    ]
    assert probabilities == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.parametrize(
    "schedules, probabilities",
    [
        ([[1000, 500]], [1]),
        ([[1000, 500, 0], [1000, 400, 0]], [1, 1]),
        ([[1000, 500, 600, 0]], [1]),
        ([[1000, 500, 0]], [0]),
    ],
)
def test_validate_mixed_step_config_rejects_invalid_input(
    schedules, probabilities
):
    with pytest.raises(ValueError):
        sc_utils.validate_mixed_step_config(schedules, probabilities)


def test_shortcut_pairs_use_nested_four_step_boundaries():
    sigmas = [1.0, 0.982, 0.960, 0.930, 0.889, 0.828, 0.727, 0.533, 0.0]

    pairs = sc_utils.shortcut_pair_candidates(sigmas, exit_index=1, num_segments=4)

    assert pairs
    assert {end_index for _, end_index in pairs}.issubset({4, 6})
    assert all(1 < mid_index < end_index for mid_index, end_index in pairs)
    assert sc_utils.shortcut_pair_candidates(
        sigmas, exit_index=7, num_segments=4
    ) == []


def test_direct_and_composed_endpoints_match_for_constant_velocity():
    latent = torch.tensor([[[1.25, -0.75]]])
    velocity = torch.tensor([[[0.4, -0.2]]])
    sigma_start, sigma_mid, sigma_end = 0.9, 0.6, 0.2

    direct = sc_utils.euler_endpoint(
        latent, velocity, sigma_start, sigma_end
    )
    middle = sc_utils.euler_endpoint(
        latent, velocity, sigma_start, sigma_mid
    )
    composed = sc_utils.euler_endpoint(
        middle, velocity, sigma_mid, sigma_end
    )

    assert torch.allclose(direct, composed)


def test_flow_velocity_round_trip():
    x0 = torch.tensor([[[0.25, -0.5]]])
    noise = torch.tensor([[[1.0, 0.75]]])
    sigma = 0.6
    noisy = (1 - sigma) * x0 + sigma * noise

    velocity = sc_utils.flow_velocity_from_x0(noisy, x0, sigma)

    assert torch.allclose(velocity, noise - x0)
