"""Unit tests for teacher-forcing discrete consistency math."""

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "src" / "ltx_distillation" / "tf_dcm_utils.py"
SPEC = importlib.util.spec_from_file_location("tf_dcm_utils", MODULE_PATH)
tf_dcm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tf_dcm)


def test_dcm_sigma_path_is_shifted_and_descending() -> None:
    path = tf_dcm.dcm_sigma_path(
        torch.tensor([[0.0], [0.5]]),
        total_steps=48,
        skipping_interval_steps=2,
        sigma_shift=8.0,
    )

    assert path.shape == (2, 3)
    assert torch.all(path[:, :-1] > path[:, 1:])
    assert torch.all((path >= 0.0) & (path <= 1.0))
    assert path[0, 0].item() == pytest.approx(1.0)


def test_teacher_euler_step_is_exact_for_straight_flow() -> None:
    x0 = torch.tensor([[[0.25, -0.5]]])
    noise = torch.tensor([[[1.0, 0.75]]])
    sigma_hi = torch.tensor([[0.8]])
    sigma_lo = torch.tensor([[0.3]])
    x_hi = tf_dcm.add_noise(x0, noise, sigma_hi)

    x_lo = tf_dcm.teacher_euler_step(x_hi, x0, sigma_hi, sigma_lo)
    expected = tf_dcm.add_noise(x0, noise, sigma_lo)

    assert torch.allclose(x_lo, expected)


def test_add_noise_accepts_stage3_per_frame_sigmas() -> None:
    clean = torch.zeros(2, 3, 1)
    noise = torch.ones_like(clean)
    sigmas = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    noisy = tf_dcm.add_noise(clean, noise, sigmas)

    assert torch.allclose(noisy[..., 0], sigmas)


def test_cfg_is_applied_independently() -> None:
    cond = torch.tensor([1.0, 3.0])
    uncond = torch.tensor([0.5, -1.0])

    guided = tf_dcm.classifier_free_guidance(cond, uncond, 4.0)

    assert torch.allclose(guided, uncond + 4.0 * (cond - uncond))


def test_validate_stage3_benchmark_schedules() -> None:
    schedules = tf_dcm.validate_benchmark_schedules(
        [
            [1000, 960, 889, 727, 0],
            [1000, 982, 960, 930, 889, 828, 727, 533, 0],
        ],
        num_train_timestep=1000,
    )

    assert [len(schedule) - 1 for schedule in schedules] == [4, 8]


@pytest.mark.parametrize(
    "schedule",
    [
        [1000, 500],
        [1000, 500, 600, 0],
        [1001, 0],
    ],
)
def test_validate_benchmark_schedule_rejects_bad_values(
    schedule: list[int],
) -> None:
    with pytest.raises(ValueError):
        tf_dcm.validate_benchmark_schedules([schedule], num_train_timestep=1000)
