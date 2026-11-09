"""Regression tests for Stage-2 self-context replay gradient attachment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "ltx_distillation"
    / "stage2_self_context_utils.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage2_self_context_utils_for_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_replay_surrogate_attaches_expected_dmd_gradient() -> None:
    prediction = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    dmd_gradient = torch.randn_like(prediction)
    frame_weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    denominator = prediction.shape[0] * prediction.shape[1]

    loss = MODULE.replay_dmd_surrogate_loss(
        prediction,
        dmd_gradient,
        frame_weights,
        denominator,
    )
    loss.backward()

    feature_count = prediction.shape[2]
    expected = (
        dmd_gradient
        * frame_weights.view(1, -1, 1)
        / float(denominator * feature_count)
    )
    assert torch.allclose(prediction.grad, expected)


def test_replay_surrogate_rejects_mismatched_shapes() -> None:
    prediction = torch.zeros(1, 2, 3)
    gradient = torch.zeros(1, 3, 3)
    try:
        MODULE.replay_dmd_surrogate_loss(
            prediction,
            gradient,
            torch.ones(2),
            denominator=2,
        )
    except ValueError as error:
        assert "identical shapes" in str(error)
    else:
        raise AssertionError("shape mismatch must raise ValueError")
