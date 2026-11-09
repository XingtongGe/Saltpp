"""Focused tests for matched causal teacher-forced DMD score forwards."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch

PACKAGES_ROOT = Path(__file__).parents[2]
DISTILLATION_SRC = PACKAGES_ROOT / "ltx-distillation" / "src"
for package_src in (
    DISTILLATION_SRC,
    PACKAGES_ROOT / "ltx-core" / "src",
    PACKAGES_ROOT / "ltx-causal" / "src",
    PACKAGES_ROOT / "ltx-pipelines" / "src",
):
    sys.path.insert(0, str(package_src))

# Avoid executing ltx_distillation/__init__.py, whose unrelated ODE imports
# require optional LMDB dependencies not needed by these unit tests.
package = types.ModuleType("ltx_distillation")
package.__path__ = [str(DISTILLATION_SRC / "ltx_distillation")]
sys.modules["ltx_distillation"] = package
module_path = DISTILLATION_SRC / "ltx_distillation" / "causal_tf_dmd.py"
spec = importlib.util.spec_from_file_location(
    "ltx_distillation.causal_tf_dmd", module_path
)
causal_tf_dmd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = causal_tf_dmd
spec.loader.exec_module(causal_tf_dmd)


class RecordingScore(torch.nn.Module):
    def __init__(self, values: dict[str, float]) -> None:
        super().__init__()
        self.values = values
        self.calls: list[dict[str, Any]] = []

    def forward(self, *, _tf_kwargs: dict[str, Any]):
        self.calls.append(_tf_kwargs)
        condition_name = str(_tf_kwargs["conditional_dict"]["name"])
        value = self.values[condition_name]
        return (
            torch.full_like(_tf_kwargs["noisy_video"], value),
            torch.full_like(_tf_kwargs["noisy_audio"], value),
        )


class RecordingBidirectionalScore(torch.nn.Module):
    def __init__(self, values: dict[str, float]) -> None:
        super().__init__()
        self.values = values
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        condition_name = str(kwargs["conditional_dict"]["name"])
        value = self.values[condition_name]
        return (
            torch.full_like(kwargs["noisy_image_or_video"], value),
            torch.full_like(kwargs["noisy_audio"], value),
        )


def _make_dmd() -> Any:
    dmd = causal_tf_dmd.CausalTFDMD.__new__(causal_tf_dmd.CausalTFDMD)
    torch.nn.Module.__init__(dmd)
    dmd.dtype = torch.float32
    dmd.real_video_guidance_scale = 4.0
    dmd.real_audio_guidance_scale = 4.0
    dmd.fake_score = RecordingScore({"cond": 2.0})
    dmd.real_score = RecordingScore({"cond": 1.0, "uncond": 0.0})
    return dmd


def _configure_score_sigma_sampler(dmd: Any, sampler_type: str) -> None:
    dmd.device = torch.device("cpu")
    dmd.dmd_timestep_sampler_type = sampler_type
    dmd.dmd_sigma_shift = 8.0
    dmd.dmd_sigma_min = 0.003
    dmd.dmd_sigma_max = 1.0
    dmd.dmd_sigma_bounds_space = "raw"
    dmd.fake_score_timestep_sampler_type = sampler_type
    dmd.fake_score_sigma_shift = 8.0
    dmd.fake_score_sigma_min = 0.003
    dmd.fake_score_sigma_max = 1.0
    dmd.fake_score_sigma_bounds_space = "raw"
    dmd.num_train_timestep = 1000
    dmd.min_step = 20
    dmd.max_step = 980
    dmd.register_buffer("sigma_lookup", torch.linspace(0.0, 1.0, 1001))


def test_uniform_shifted_score_sigma_matches_stage0_formula() -> None:
    dmd = _make_dmd()
    _configure_score_sigma_sampler(dmd, "uniform_shifted")
    torch.manual_seed(123)
    raw = 0.003 + (1.0 - 0.003) * torch.rand(8)
    expected = 8.0 * raw / (1.0 + 7.0 * raw)

    torch.manual_seed(123)
    actual = dmd.sample_dmd_score_sigma(8)

    assert torch.equal(actual, expected)


def test_causal_fake_loss_accepts_continuous_score_sigma() -> None:
    dmd = _make_dmd()
    _configure_score_sigma_sampler(dmd, "uniform_shifted")
    dmd.video_loss_weight = 1.0
    dmd.audio_loss_weight = 0.16
    dmd.audio_start_step = 0
    dmd.current_step = 0
    dmd.denoising_loss_func = lambda **kwargs: torch.nn.functional.mse_loss(
        kwargs["flow_pred"], kwargs["noise"] - kwargs["x"]
    )

    video = torch.randn(2, 3, 2, 2, 2)
    audio = torch.randn(2, 4, 3)
    loss, logs = dmd.critic_loss(
        pred_x0_video_det=video,
        pred_x0_audio_det=audio,
        clean_video_context=video,
        clean_audio_context=audio,
        conditional_dict={"name": "cond"},
    )

    assert torch.isfinite(loss)
    assert 0.0 < logs["critic_sigma_mean"] <= 1.0


def test_real_and_fake_scores_receive_identical_tf_context_and_mask_route() -> None:
    dmd = _make_dmd()
    noisy_video = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    noisy_audio = torch.randn(1, 4, 3, requires_grad=True)
    clean_video = torch.randn_like(noisy_video)
    clean_audio = torch.randn_like(noisy_audio)
    video_sigma = torch.full((1, 3), 0.6)
    audio_sigma = torch.full((1, 4), 0.6)

    grad_video, grad_audio, _ = dmd._compute_causal_tf_kl_grad(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        generated_clean_video=torch.zeros_like(noisy_video),
        generated_clean_audio=torch.zeros_like(noisy_audio),
        clean_video_context=clean_video,
        clean_audio_context=clean_audio,
        video_sigma=video_sigma,
        audio_sigma=audio_sigma,
        conditional_dict={"name": "cond"},
        unconditional_dict={"name": "uncond"},
        normalization=False,
    )

    # real CFG = 0 + 4*(1-0) = 4; fake = 2; DMD direction = fake-real = -2.
    assert torch.allclose(grad_video, torch.full_like(grad_video, -2.0))
    assert torch.allclose(grad_audio, torch.full_like(grad_audio, -2.0))
    assert not grad_video.requires_grad
    assert not grad_audio.requires_grad

    all_calls = dmd.fake_score.calls + dmd.real_score.calls
    assert len(all_calls) == 3
    for call in all_calls:
        assert call["clean_video"].data_ptr() == clean_video.data_ptr()
        assert call["clean_audio"].data_ptr() == clean_audio.data_ptr()
        assert torch.equal(call["noisy_timestep"], video_sigma)
        assert torch.equal(call["noisy_audio_timestep"], audio_sigma)
        assert torch.count_nonzero(call["clean_timestep"]) == 0
        assert torch.count_nonzero(call["clean_audio_timestep"]) == 0
        assert call["use_causal_timestep"] is False


def test_causal_real_audio_cfg_uses_sampled_scale_and_logs_it() -> None:
    dmd = _make_dmd()
    dmd.real_audio_guidance_scale_min = 3.0
    dmd.real_audio_guidance_scale_max = 3.0
    noisy_video = torch.randn(1, 3, 2, 2, 2)
    noisy_audio = torch.randn(1, 4, 3)
    clean_video = torch.randn_like(noisy_video)
    clean_audio = torch.randn_like(noisy_audio)

    grad_video, grad_audio, logs = dmd._compute_causal_tf_kl_grad(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        generated_clean_video=torch.zeros_like(noisy_video),
        generated_clean_audio=torch.zeros_like(noisy_audio),
        clean_video_context=clean_video,
        clean_audio_context=clean_audio,
        video_sigma=torch.full((1, 3), 0.6),
        audio_sigma=torch.full((1, 4), 0.6),
        conditional_dict={"name": "cond"},
        unconditional_dict={"name": "uncond"},
        normalization=False,
    )

    # Video scale remains 4: fake 2 - real 4 = -2. Audio samples 3:
    # fake 2 - real 3 = -1.
    assert torch.allclose(grad_video, torch.full_like(grad_video, -2.0))
    assert torch.allclose(grad_audio, torch.full_like(grad_audio, -1.0))
    assert logs["real_audio_guidance_scale"] == 3.0


def test_hybrid_real_is_bidirectional_while_fake_keeps_causal_tf_context() -> None:
    dmd = _make_dmd()
    dmd.real_score_type = "bidirectional"
    dmd.real_score = RecordingBidirectionalScore({"cond": 1.0, "uncond": 0.0})
    noisy_video = torch.randn(1, 3, 2, 2, 2)
    noisy_audio = torch.randn(1, 4, 3)
    clean_video = torch.randn_like(noisy_video)
    clean_audio = torch.randn_like(noisy_audio)
    video_sigma = torch.full((1, 3), 0.6)
    audio_sigma = torch.full((1, 4), 0.6)

    grad_video, grad_audio, _ = dmd._compute_causal_tf_kl_grad(
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        generated_clean_video=torch.zeros_like(noisy_video),
        generated_clean_audio=torch.zeros_like(noisy_audio),
        clean_video_context=clean_video,
        clean_audio_context=clean_audio,
        video_sigma=video_sigma,
        audio_sigma=audio_sigma,
        conditional_dict={"name": "cond"},
        unconditional_dict={"name": "uncond"},
        normalization=False,
    )

    assert torch.allclose(grad_video, torch.full_like(grad_video, -2.0))
    assert torch.allclose(grad_audio, torch.full_like(grad_audio, -2.0))
    assert len(dmd.fake_score.calls) == 1
    assert dmd.fake_score.calls[0]["clean_video"].data_ptr() == clean_video.data_ptr()
    assert len(dmd.real_score.calls) == 2
    for call in dmd.real_score.calls:
        assert "_tf_kwargs" not in call
        assert call["noisy_image_or_video"].data_ptr() == noisy_video.data_ptr()
        assert torch.equal(call["timestep"], video_sigma)


def test_causal_score_predictions_are_only_retained_when_requested() -> None:
    dmd = _make_dmd()
    video = torch.zeros(1, 3, 2, 2, 2)
    audio = torch.zeros(1, 4, 3)
    kwargs = dict(
        noisy_video=video,
        noisy_audio=audio,
        generated_clean_video=video,
        generated_clean_audio=audio,
        clean_video_context=video,
        clean_audio_context=audio,
        video_sigma=torch.full((1, 3), 0.6),
        audio_sigma=torch.full((1, 4), 0.6),
        conditional_dict={"name": "cond"},
        unconditional_dict={"name": "uncond"},
        normalization=False,
    )

    _, _, ordinary_logs = dmd._compute_causal_tf_kl_grad(**kwargs)
    assert "_pred_real_video_x0" not in ordinary_logs
    assert "_pred_fake_video_x0" not in ordinary_logs

    _, _, capture_logs = dmd._compute_causal_tf_kl_grad(
        **kwargs,
        return_video_predictions=True,
    )
    assert torch.equal(
        capture_logs["_pred_real_video_x0"],
        torch.full_like(video, 4.0),
    )
    assert torch.equal(
        capture_logs["_pred_fake_video_x0"],
        torch.full_like(video, 2.0),
    )
