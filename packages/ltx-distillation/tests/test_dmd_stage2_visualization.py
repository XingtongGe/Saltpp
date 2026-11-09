"""Regression coverage for optional Stage-2 DMD visualization tensors."""

from __future__ import annotations

import torch

from ltx_distillation.dmd_stage2 import DMDStage2


class _ConstantScore(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, *, noisy_image_or_video, noisy_audio, **kwargs):
        return (
            torch.full_like(noisy_image_or_video, self.value),
            torch.full_like(noisy_audio, self.value),
        )


def _dmd_stage2_stub() -> DMDStage2:
    module = DMDStage2.__new__(DMDStage2)
    torch.nn.Module.__init__(module)
    module.fake_score = _ConstantScore(2.0)
    module.real_score = _ConstantScore(1.0)
    module.real_video_guidance_scale = 3.0
    module.real_audio_guidance_scale = 5.0
    return module


def _dmd_sampler_stub(sampler_type: str) -> DMDStage2:
    module = DMDStage2.__new__(DMDStage2)
    torch.nn.Module.__init__(module)
    module.device = torch.device("cpu")
    module.dmd_timestep_sampler_type = sampler_type
    module.dmd_sigma_shift = 8.0
    module.dmd_sigma_min = 0.003
    module.dmd_sigma_max = 1.0
    module.dmd_sigma_bounds_space = "raw"
    module.num_train_timestep = 1000
    module.min_step = 20
    module.max_step = 980
    module.register_buffer("sigma_lookup", torch.linspace(0.0, 1.0, 1001))
    return module


def test_uniform_shifted_score_sigma_matches_stage0_formula() -> None:
    module = _dmd_sampler_stub("uniform_shifted")
    torch.manual_seed(123)
    raw = 0.003 + (1.0 - 0.003) * torch.rand(8)
    expected = 8.0 * raw / (1.0 + 7.0 * raw)

    torch.manual_seed(123)
    actual = module.sample_dmd_score_sigma(8)

    assert torch.equal(actual, expected)
    assert actual.shape == (8,)


def test_fake_score_sampler_can_keep_shift8_when_dmd_uses_shift5() -> None:
    module = _dmd_sampler_stub("uniform_shifted")
    module.dmd_sigma_shift = 5.0
    module.fake_score_timestep_sampler_type = "uniform_shifted"
    module.fake_score_sigma_shift = 8.0
    module.fake_score_sigma_min = 0.003
    module.fake_score_sigma_max = 1.0
    module.fake_score_sigma_bounds_space = "raw"

    torch.manual_seed(456)
    raw = 0.003 + (1.0 - 0.003) * torch.rand(8)
    expected_dmd = 5.0 * raw / (1.0 + 4.0 * raw)
    expected_fake = 8.0 * raw / (1.0 + 7.0 * raw)

    torch.manual_seed(456)
    actual_dmd = module.sample_dmd_score_sigma(8)
    torch.manual_seed(456)
    actual_fake = module.sample_fake_score_sigma(8)

    assert torch.equal(actual_dmd, expected_dmd)
    assert torch.equal(actual_fake, expected_fake)
    assert not torch.equal(actual_dmd, actual_fake)


def test_shifted_bounds_constrain_actual_sigma_without_clamp_mass() -> None:
    module = _dmd_sampler_stub("uniform_shifted")
    module.dmd_sigma_shift = 8.0
    module.dmd_sigma_min = 0.02
    module.dmd_sigma_max = 0.98
    module.dmd_sigma_bounds_space = "shifted"

    torch.manual_seed(789)
    unit_uniform = torch.rand(32)
    raw_min = 0.02 / (8.0 - 7.0 * 0.02)
    raw_max = 0.98 / (8.0 - 7.0 * 0.98)
    raw = raw_min + (raw_max - raw_min) * unit_uniform
    expected = 8.0 * raw / (1.0 + 7.0 * raw)

    torch.manual_seed(789)
    actual = module.sample_dmd_score_sigma(32)

    assert torch.equal(actual, expected)
    assert float(actual.min()) >= 0.02
    assert float(actual.max()) <= 0.98
    assert module.score_sigma_support() == (0.02, 0.98)


def test_legacy_score_sigma_sampler_still_uses_integer_lookup() -> None:
    module = _dmd_sampler_stub("ltx_scheduler")
    module.min_step = 100
    module.max_step = 100

    actual = module.sample_dmd_score_sigma(4)

    assert torch.equal(actual, torch.full((4,), 0.1))


def test_video_cfg_scale_can_be_uniformly_sampled_from_a_range() -> None:
    module = _dmd_stage2_stub()
    module.real_video_guidance_scale_min = 1.5
    module.real_video_guidance_scale_max = 3.5

    torch.manual_seed(123)
    expected = float(torch.empty(()).uniform_(1.5, 3.5).item())
    torch.manual_seed(123)
    actual = module.sample_real_video_guidance_scale(torch.device("cpu"))

    assert actual == expected
    assert 1.5 <= actual <= 3.5


def test_stage2_kl_predictions_are_only_returned_when_requested() -> None:
    module = _dmd_stage2_stub()
    video = torch.zeros(1, 2, 3, 2, 2)
    audio = torch.zeros(1, 4, 3)
    kwargs = {
        "noisy_video": video,
        "noisy_audio": audio,
        "clean_video": video,
        "clean_audio": audio,
        "video_sigma": torch.ones(1, 2),
        "audio_sigma": torch.ones(1, 4),
        "conditional_dict": {},
        "unconditional_dict": {},
        "normalization": False,
    }

    _, _, ordinary_logs = module._compute_kl_grad(**kwargs)
    assert "_pred_real_video_x0" not in ordinary_logs
    assert "_pred_fake_video_x0" not in ordinary_logs

    grad_video, _, capture_logs = module._compute_kl_grad(
        **kwargs,
        return_video_predictions=True,
    )
    assert torch.equal(grad_video, torch.ones_like(video))
    assert torch.equal(
        capture_logs["_pred_real_video_x0"],
        torch.ones_like(video),
    )
    assert torch.equal(
        capture_logs["_pred_fake_video_x0"],
        torch.full_like(video, 2.0),
    )
    assert capture_logs["real_video_guidance_scale"] == 3.0
