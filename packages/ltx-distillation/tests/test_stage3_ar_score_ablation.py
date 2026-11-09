"""Focused contracts for Stage-3 matched AR-score ablations."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

PACKAGES_ROOT = Path(__file__).parents[2]
DISTILLATION_SRC = PACKAGES_ROOT / "ltx-distillation" / "src"
for package_src in (
    DISTILLATION_SRC,
    PACKAGES_ROOT / "ltx-core" / "src",
    PACKAGES_ROOT / "ltx-causal" / "src",
    PACKAGES_ROOT / "ltx-pipelines" / "src",
):
    sys.path.insert(0, str(package_src))

# Avoid unrelated optional ODE/LMDB imports from ltx_distillation/__init__.py.
if "ltx_distillation" not in sys.modules:
    package = types.ModuleType("ltx_distillation")
    package.__path__ = [str(DISTILLATION_SRC / "ltx_distillation")]
    sys.modules["ltx_distillation"] = package

from ltx_distillation.data import (
    causal_aligned_audio_frame_count,
    collate_metadata_prompts,
)
from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.self_context_ar_dmd import SelfContextARScoreMixin


CONFIG_DIR = Path(__file__).parents[3] / "configs" / "stage2_context_aligned_dmd"


def _load_config_tree(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_name = config.pop("base_config_path", None)
    if base_name is None:
        return config
    base = _load_config_tree(path.parent / base_name)
    base.update(config)
    return base


def _uniform_sampler_stub() -> LTX2DMD:
    module = LTX2DMD.__new__(LTX2DMD)
    torch.nn.Module.__init__(module)
    module.device = torch.device("cpu")
    module.dmd_timestep_sampler_type = "uniform_shifted"
    module.dmd_sigma_shift = 8.0
    module.dmd_sigma_min = 0.02
    module.dmd_sigma_max = 0.98
    module.dmd_sigma_bounds_space = "shifted"
    module.fake_score_timestep_sampler_type = "uniform_shifted"
    module.fake_score_sigma_shift = 8.0
    module.fake_score_sigma_min = 0.02
    module.fake_score_sigma_max = 0.98
    module.fake_score_sigma_bounds_space = "shifted"
    return module


def test_stage3_shifted_sampler_has_requested_support_and_shared_clip_sigma() -> None:
    module = _uniform_sampler_stub()
    torch.manual_seed(123)
    video_sigma, audio_sigma, _, _ = module._sample_score_noise_levels(
        batch_size=4,
        num_video_frames=7,
        num_audio_frames=13,
        task_type="causal_av",
        fake_score=False,
    )

    assert video_sigma.shape == (4, 7)
    assert audio_sigma.shape == (4, 13)
    assert torch.equal(video_sigma, video_sigma[:, :1].expand_as(video_sigma))
    assert torch.equal(audio_sigma, video_sigma[:, :1].expand_as(audio_sigma))
    assert float(video_sigma.min()) >= 0.02
    assert float(video_sigma.max()) <= 0.98
    assert module.score_sigma_support() == (0.02, 0.98)


def test_dmd_and_fake_score_draw_independently_from_same_support() -> None:
    module = _uniform_sampler_stub()
    torch.manual_seed(321)
    dmd_sigma = module._sample_score_noise_levels(
        batch_size=8,
        num_video_frames=4,
        num_audio_frames=6,
        task_type="bidirectional_av",
        fake_score=False,
    )[0]
    fake_sigma = module._sample_score_noise_levels(
        batch_size=8,
        num_video_frames=4,
        num_audio_frames=6,
        task_type="bidirectional_av",
        fake_score=True,
    )[0]

    assert not torch.equal(dmd_sigma, fake_sigma)
    assert module.score_sigma_support(fake_score=True) == (0.02, 0.98)


def test_stage3_audio_cfg_samples_inside_configured_range() -> None:
    module = _uniform_sampler_stub()
    module.real_audio_guidance_scale = 3.0
    module.real_audio_guidance_scale_min = 2.0
    module.real_audio_guidance_scale_max = 4.0
    torch.manual_seed(7182)
    samples = [
        module.sample_real_audio_guidance_scale(torch.device("cpu"))
        for _ in range(32)
    ]
    assert min(samples) >= 2.0
    assert max(samples) <= 4.0
    assert min(samples) < 3.0 < max(samples)


def test_kv_block_rollout_can_stop_at_random_exit_or_run_full_schedule() -> None:
    module = LTX2DMD.__new__(LTX2DMD)
    torch.nn.Module.__init__(module)
    module.device = torch.device("cpu")
    module.dtype = torch.float32
    module.denoising_sigmas = torch.tensor([1.0, 0.96, 0.889, 0.727, 0.0])

    forward_calls = []

    def fake_forward(**kwargs):
        forward_calls.append(float(kwargs["timestep_block"][0, 0]))
        value = float(len(forward_calls))
        return (
            torch.full_like(kwargs["noisy_video_block"], value),
            torch.full_like(kwargs["noisy_audio_block"], value),
        )

    module._kv_block_forward = fake_forward
    module._renoise_block = lambda prediction, _next_sigma: prediction
    block = SimpleNamespace(video_frames=1, audio_frames=1)
    video_ref = torch.zeros(1, 1, 1, 1, 1)
    audio_ref = torch.zeros(1, 1, 1)

    exit_video, exit_audio = module._kv_denoise_block(
        block=block,
        kv_caches=[],
        conditional_dict={},
        prev_video_ref=video_ref,
        prev_audio_ref=audio_ref,
        requires_grad=False,
        exit_index=1,
    )
    assert torch.allclose(torch.tensor(forward_calls), torch.tensor([1.0, 0.96]))
    assert torch.equal(exit_video, torch.full_like(exit_video, 2.0))
    assert torch.equal(exit_audio, torch.full_like(exit_audio, 2.0))

    forward_calls.clear()
    full_video, full_audio = module._kv_denoise_block(
        block=block,
        kv_caches=[],
        conditional_dict={},
        prev_video_ref=video_ref,
        prev_audio_ref=audio_ref,
        requires_grad=False,
    )
    assert torch.allclose(
        torch.tensor(forward_calls),
        torch.tensor([1.0, 0.96, 0.889, 0.727]),
    )
    assert torch.equal(full_video, torch.full_like(full_video, 4.0))
    assert torch.equal(full_audio, torch.full_like(full_audio, 4.0))


class _CaptureTFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def forward(self, *, _tf_kwargs):
        self.kwargs = _tf_kwargs
        return _tf_kwargs["noisy_video"], _tf_kwargs["noisy_audio"]


def test_self_context_ar_score_uses_detached_rollout_as_clean_tf_half() -> None:
    mixin = SelfContextARScoreMixin()
    mixin.dtype = torch.float32
    model = _CaptureTFModel()
    noisy_video = torch.randn(1, 7, 2, 2, 2)
    noisy_audio = torch.randn(1, 11, 3)
    clean_video = torch.randn_like(noisy_video, requires_grad=True)
    clean_audio = torch.randn_like(noisy_audio, requires_grad=True)
    video_sigma = torch.full((1, 7), 0.5)
    audio_sigma = torch.full((1, 11), 0.5)

    out_video, out_audio = mixin._score_model_forward(
        model=model,
        noisy_video=noisy_video,
        noisy_audio=noisy_audio,
        clean_video_context=clean_video,
        clean_audio_context=clean_audio,
        conditional_dict={"video_context": torch.zeros(1, 1, 1)},
        video_sigma=video_sigma,
        audio_sigma=audio_sigma,
    )

    assert torch.equal(out_video, noisy_video)
    assert torch.equal(out_audio, noisy_audio)
    assert model.kwargs["clean_video"].data_ptr() == clean_video.data_ptr()
    assert not model.kwargs["clean_video"].requires_grad
    assert not model.kwargs["clean_audio"].requires_grad
    assert torch.count_nonzero(model.kwargs["clean_timestep"]) == 0
    assert torch.count_nonzero(model.kwargs["clean_audio_timestep"]) == 0
    assert model.kwargs["use_causal_timestep"] is False


def test_metadata_only_collate_matches_prompt_dataset_contract() -> None:
    batch = [
        {"prompt": "first", "num_frames": 121},
        {"caption": "second", "sample_frames": 193},
    ]
    assert collate_metadata_prompts(batch) == {
        "prompts": ["first", "second"],
        "num_frames": [121, 193],
    }


def test_pad_max_audio_crop_uses_causal_block_boundary() -> None:
    # The generic duration formula yields 143 here, but the packed causal TF
    # mask for 18 video frames has exactly 142 audio positions.
    assert causal_aligned_audio_frame_count(
        18,
        num_frame_per_block=3,
        first_block_video_frames=4,
    ) == 142


def test_stage2_rollout_score_configs_only_change_score_factorization() -> None:
    bi = _load_config_tree(
        CONFIG_DIR / "step2_rollout_from_tfdcm_4step.yaml"
    )
    ar = _load_config_tree(
        CONFIG_DIR / "step2_rollout_ar_dmd_4step.yaml"
    )

    for key in (
        "denoising_step_list",
        "dmd_timestep_sampler_type",
        "dmd_sigma_shift",
        "dmd_sigma_min",
        "dmd_sigma_max",
        "dmd_sigma_bounds_space",
        "fake_score_timestep_sampler_type",
        "fake_score_sigma_shift",
        "fake_score_sigma_min",
        "fake_score_sigma_max",
        "fake_score_sigma_bounds_space",
        "data_config_path",
        "benchmark_denoising_step_lists",
        "generator_lr",
        "critic_lr",
        "max_grad_norm",
        "seed",
    ):
        assert bi[key] == ar[key], key

    assert bi["real_task_type"] == bi["fake_task_type"] == "bidirectional_av"
    assert ar["real_task_type"] == ar["fake_task_type"] == "causal_av"
    assert bi["distillation_loss"] == "dmd"
    assert ar["distillation_loss"] == "self_context_ar_dmd"
    assert ar["fake_score_init_checkpoint"] == ar["generator_ckpt"]
    assert ar["generator_checkpoint_state_key"] == "generator"
    assert ar["fake_score_init_state_key"] == "fake_score"


def test_stage2_rollout_salt_mixed_step_config() -> None:
    salt = _load_config_tree(
        CONFIG_DIR / "step2_rollout_salt_mixed_step.yaml"
    )

    assert "stage2_step1_clean_prefix_ar_dmd" in salt["generator_ckpt"]
    assert salt["generator_checkpoint_state_key"] == "generator"
    assert salt["real_task_type"] == "bidirectional_av"
    assert salt["fake_task_type"] == "bidirectional_av"
    assert salt["fake_score_rollout_mode"] == "random_exit"
    assert salt["same_step_across_blocks"] is True
    assert salt["data_config_path"].endswith("data_mixture_example.yaml")
    assert salt["distillation_loss"] == "sc_dmd"
    assert salt["mixed_denoising_step_lists"] == [
        [1000, 982, 960, 930, 889, 828, 727, 533, 0],
        [1000, 960, 889, 727, 0],
        [1000, 889, 0],
    ]
    assert salt["mixed_step_probs"] == [0.4, 0.4, 0.2]
    assert salt["shortcut_consistency_weight"] == 0.1
    assert salt["shortcut_consistency_apply_steps"] == [8]
