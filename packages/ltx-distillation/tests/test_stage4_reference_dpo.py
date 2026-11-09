from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from ltx_distillation.stage4_reference_dpo import (
    PreferencePair,
    Stage4ReferenceDPO,
    Stage4Rollout,
    dpo_loss_and_replay_coefficient,
)
from ltx_distillation.stage35_data import (
    build_stage35_local_prompt_dataset,
)
from ltx_distillation.train_stage4_reference_dpo import (
    Stage4ReferenceDPOTrainer,
    load_embedded_causal_upsampler,
    validate_stage4b_config,
)


class _ToyCausalPolicy(torch.nn.Module):
    def __init__(self, value: float, *, trainable: bool) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(value, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.forward_records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def init_av_kv_caches(self, **kwargs):
        return {"calls": 0, "kwargs": kwargs}

    def forward(self, *, _kv_cache_kwargs):
        _kv_cache_kwargs["kv_caches"]["calls"] += 1
        video = _kv_cache_kwargs["noisy_video_block"]
        audio = _kv_cache_kwargs["noisy_audio_block"]
        self.forward_records.append(
            (
                _kv_cache_kwargs["timestep_block"].detach().clone(),
                video.detach().clone(),
            )
        )
        return self.scale * video, self.scale * audio


def _objective(
    policy,
    reference,
    *,
    dpo_pair_source: str = "online_policy",
    negative_video_anchor_weight: float = 0.0,
    negative_audio_anchor_weight: float = 0.0,
    positive_video_mix: float = 1.0,
    positive_audio_source: str = "refined",
) -> Stage4ReferenceDPO:
    return Stage4ReferenceDPO(
        policy=policy,
        reference=reference,
        official_refiner=None,
        causal_upsampler=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_frame_per_block=2,
        first_block_video_frames=2,
        dpo_beta=1.0,
        dpo_loss_weight=1.0,
        dpo_video_weight=1.0,
        dpo_audio_weight=0.0,
        negative_video_anchor_weight=negative_video_anchor_weight,
        negative_audio_anchor_weight=negative_audio_anchor_weight,
        dpo_pair_source=dpo_pair_source,
        positive_video_mix=positive_video_mix,
        positive_audio_source=positive_audio_source,
    )


def test_dpo_replay_coefficient_matches_closed_form():
    positive = torch.tensor(0.7, requires_grad=True)
    negative = torch.tensor(1.1, requires_grad=True)
    ref_positive = torch.tensor(0.6)
    ref_negative = torch.tensor(0.9)
    beta = 1.3
    weight = 0.4

    loss, coefficient, advantage, logit = (
        dpo_loss_and_replay_coefficient(
            online_positive_energy=positive,
            online_negative_energy=negative,
            reference_positive_energy=ref_positive,
            reference_negative_energy=ref_negative,
            beta=beta,
            loss_weight=weight,
        )
    )
    loss.backward()

    assert torch.allclose(positive.grad, coefficient)
    assert torch.allclose(negative.grad, -coefficient)
    assert torch.allclose(logit, beta * advantage)


def test_identical_online_and_reference_start_at_log_two():
    energy_pos = torch.tensor(0.25)
    energy_neg = torch.tensor(0.75)
    loss, coefficient, advantage, _ = dpo_loss_and_replay_coefficient(
        online_positive_energy=energy_pos,
        online_negative_energy=energy_neg,
        reference_positive_energy=energy_pos,
        reference_negative_energy=energy_neg,
        beta=1.0,
        loss_weight=1.0,
    )
    assert advantage.item() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(math.log(2.0))
    assert coefficient.item() == pytest.approx(0.5)


def test_blockwise_replay_matches_monolithic_dpo_gradient():
    policy = _ToyCausalPolicy(0.7, trainable=True)
    reference = _ToyCausalPolicy(0.7, trainable=False)
    objective = _objective(policy, reference)
    block = SimpleNamespace(
        video_start=0,
        video_end=2,
        video_frames=2,
        audio_start=0,
        audio_end=2,
        audio_frames=2,
    )
    positive_video = torch.tensor(
        [[[[[0.2]]], [[[0.4]]]]], dtype=torch.float32
    )
    negative_video = torch.tensor(
        [[[[[0.8]]], [[[1.0]]]]], dtype=torch.float32
    )
    positive_audio = torch.zeros((1, 2, 1), dtype=torch.float32)
    negative_audio = torch.ones((1, 2, 1), dtype=torch.float32)
    pair = PreferencePair(
        positive_video=positive_video,
        positive_audio=positive_audio,
        negative_video=negative_video,
        negative_audio=negative_audio,
        lr_video=torch.empty(0),
        relative_video_change=torch.tensor(0.0),
        relative_audio_change=torch.tensor(0.0),
        blocks=[block],
    )
    conditional = {"video_context": torch.zeros((1, 1, 1))}

    torch.manual_seed(123)
    metrics = objective.dpo_backward(
        pair=pair,
        conditional_dict=conditional,
        timestep=889,
        real_video_frames=2,
        real_audio_frames=2,
    )
    replay_gradient = policy.scale.grad.detach().clone()

    direct_scale = torch.tensor(0.7, requires_grad=True)
    torch.manual_seed(123)
    video_noise = torch.randn_like(negative_video)
    _ = torch.randn_like(negative_audio)
    sigma = 0.889
    noisy_positive = (
        (1.0 - sigma) * positive_video + sigma * video_noise
    )
    noisy_negative = (
        (1.0 - sigma) * negative_video + sigma * video_noise
    )
    online_positive = (
        direct_scale * noisy_positive - positive_video
    ).square().mean()
    online_negative = (
        direct_scale * noisy_negative - negative_video
    ).square().mean()
    with torch.no_grad():
        reference_positive = (
            0.7 * noisy_positive - positive_video
        ).square().mean()
        reference_negative = (
            0.7 * noisy_negative - negative_video
        ).square().mean()
    direct_loss = torch.nn.functional.softplus(
        -(
            (online_negative - online_positive)
            - (reference_negative - reference_positive)
        )
    )
    direct_loss.backward()

    assert metrics["dpo/loss"].item() == pytest.approx(math.log(2.0))
    assert metrics["dpo/replay_positive_error"].item() < 1e-7
    assert metrics["dpo/replay_negative_error"].item() < 1e-7
    assert metrics["dpo/effective_coefficient_q"].item() == pytest.approx(
        metrics["dpo/coefficient_q"].item()
    )
    assert replay_gradient.item() == pytest.approx(
        direct_scale.grad.item(), abs=1e-6
    )

    policy.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    scaled_metrics = objective.dpo_backward(
        pair=pair,
        conditional_dict=conditional,
        timestep=889,
        real_video_frames=2,
        real_audio_frames=2,
        gradient_scale=0.5,
    )
    assert policy.scale.grad.item() == pytest.approx(
        0.5 * replay_gradient.item(), abs=1e-6
    )
    assert scaled_metrics["dpo/loss"].item() == pytest.approx(
        metrics["dpo/loss"].item()
    )
    assert scaled_metrics["dpo/coefficient_q"].item() == pytest.approx(
        metrics["dpo/coefficient_q"].item()
    )
    assert scaled_metrics[
        "dpo/effective_coefficient_q"
    ].item() == pytest.approx(
        0.5 * metrics["dpo/coefficient_q"].item()
    )

    full_gradients = []
    for seed in (234, 345):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        objective.dpo_backward(
            pair=pair,
            conditional_dict=conditional,
            timestep=889,
            real_video_frames=2,
            real_audio_frames=2,
        )
        full_gradients.append(policy.scale.grad.detach().clone())

    policy.zero_grad(set_to_none=True)
    for seed in (234, 345):
        torch.manual_seed(seed)
        objective.dpo_backward(
            pair=pair,
            conditional_dict=conditional,
            timestep=889,
            real_video_frames=2,
            real_audio_frames=2,
            gradient_scale=0.5,
        )
    assert policy.scale.grad.item() == pytest.approx(
        torch.stack(full_gradients).mean().item(), abs=1e-6
    )


def test_microbatch_metric_aggregation_averages_values_and_maxes_memory():
    accumulated: dict[str, float] = {}
    Stage4ReferenceDPOTrainer._accumulate_microbatch_metrics(
        accumulated,
        {
            "dpo/loss": torch.tensor(0.4),
            "dpo/timestep": 889.0,
            "memory/test/peak_allocated_gib": 100.0,
        },
        mean_weight=0.5,
    )
    Stage4ReferenceDPOTrainer._accumulate_microbatch_metrics(
        accumulated,
        {
            "dpo/loss": torch.tensor(0.8),
            "dpo/timestep": 889.0,
            "memory/test/peak_allocated_gib": 120.0,
        },
        mean_weight=0.5,
    )

    assert accumulated["dpo/loss"] == pytest.approx(0.6)
    assert accumulated["dpo/timestep"] == pytest.approx(889.0)
    assert accumulated["memory/test/peak_allocated_gib"] == pytest.approx(
        120.0
    )


def test_anchored_replay_matches_augmented_objective_gradient():
    policy = _ToyCausalPolicy(0.7, trainable=True)
    reference = _ToyCausalPolicy(0.7, trainable=False)
    objective = _objective(
        policy,
        reference,
        negative_video_anchor_weight=0.2,
        negative_audio_anchor_weight=0.05,
    )
    block = SimpleNamespace(
        video_start=0,
        video_end=2,
        video_frames=2,
        audio_start=0,
        audio_end=2,
        audio_frames=2,
    )
    positive_video = torch.tensor(
        [[[[[0.2]]], [[[0.4]]]]], dtype=torch.float32
    )
    negative_video = torch.tensor(
        [[[[[0.8]]], [[[1.0]]]]], dtype=torch.float32
    )
    positive_audio = torch.zeros((1, 2, 1), dtype=torch.float32)
    negative_audio = torch.ones((1, 2, 1), dtype=torch.float32)
    pair = PreferencePair(
        positive_video=positive_video,
        positive_audio=positive_audio,
        negative_video=negative_video,
        negative_audio=negative_audio,
        lr_video=torch.empty(0),
        relative_video_change=torch.tensor(0.0),
        relative_audio_change=torch.tensor(0.0),
        blocks=[block],
    )
    conditional = {"video_context": torch.zeros((1, 1, 1))}

    torch.manual_seed(456)
    metrics = objective.dpo_backward(
        pair=pair,
        conditional_dict=conditional,
        timestep=889,
        real_video_frames=2,
        real_audio_frames=2,
    )
    replay_gradient = policy.scale.grad.detach().clone()

    direct_scale = torch.tensor(0.7, requires_grad=True)
    torch.manual_seed(456)
    video_noise = torch.randn_like(negative_video)
    audio_noise = torch.randn_like(negative_audio)
    sigma = 0.889
    noisy_positive_video = (
        (1.0 - sigma) * positive_video + sigma * video_noise
    )
    noisy_negative_video = (
        (1.0 - sigma) * negative_video + sigma * video_noise
    )
    noisy_negative_audio = (
        (1.0 - sigma) * negative_audio + sigma * audio_noise
    )
    online_positive_video = (
        direct_scale * noisy_positive_video - positive_video
    ).square().mean()
    online_negative_video = (
        direct_scale * noisy_negative_video - negative_video
    ).square().mean()
    online_negative_audio = (
        direct_scale * noisy_negative_audio - negative_audio
    ).square().mean()
    with torch.no_grad():
        reference_positive_video = (
            0.7 * noisy_positive_video - positive_video
        ).square().mean()
        reference_negative_video = (
            0.7 * noisy_negative_video - negative_video
        ).square().mean()
    direct_dpo = torch.nn.functional.softplus(
        -(
            (online_negative_video - online_positive_video)
            - (
                reference_negative_video - reference_positive_video
            )
        )
    )
    direct_loss = (
        direct_dpo
        + 0.2 * online_negative_video
        + 0.05 * online_negative_audio
    )
    direct_loss.backward()

    assert replay_gradient.item() == pytest.approx(
        direct_scale.grad.item(), abs=1e-6
    )
    assert metrics["optimization/surrogate_loss"].item() == pytest.approx(
        direct_loss.detach().item(), abs=1e-6
    )
    assert metrics[
        "dpo/effective_negative_video_coefficient"
    ].item() == pytest.approx(0.2 - 0.5)
    assert metrics[
        "dpo/effective_negative_audio_coefficient"
    ].item() == pytest.approx(0.05)


def test_candidate_cache_refresh_uses_clean_candidate_not_prediction():
    policy = _ToyCausalPolicy(0.2, trainable=True)
    reference = _ToyCausalPolicy(0.2, trainable=False)
    objective = _objective(policy, reference)
    block = SimpleNamespace(
        video_start=0,
        video_end=2,
        video_frames=2,
        audio_start=0,
        audio_end=2,
        audio_frames=2,
    )
    candidate_video = torch.tensor(
        [[[[[0.25]]], [[[0.75]]]]], dtype=torch.float32
    )
    candidate_audio = torch.zeros((1, 2, 1), dtype=torch.float32)
    conditional = {"video_context": torch.zeros((1, 1, 1))}

    objective.score_candidate(
        model=policy,
        candidate_video=candidate_video,
        candidate_audio=candidate_audio,
        video_noise=torch.ones_like(candidate_video),
        audio_noise=torch.ones_like(candidate_audio),
        conditional_dict=conditional,
        timestep=889,
        real_video_frames=2,
        real_audio_frames=2,
        blocks=[block],
    )

    assert len(policy.forward_records) == 2
    noisy_sigma, noisy_input = policy.forward_records[0]
    clean_sigma, clean_input = policy.forward_records[1]
    assert torch.allclose(
        noisy_sigma, torch.full_like(noisy_sigma, 0.889)
    )
    assert not torch.allclose(noisy_input, candidate_video)
    assert torch.count_nonzero(clean_sigma).item() == 0
    assert torch.equal(clean_input, candidate_video)


def test_stage4b_config_is_dpo_only_and_freezes_upsampler():
    root = Path(__file__).resolve().parents[3]
    path = root / "configs" / "stage3_scale_wise_960p" / "scale_wise_reference_dpo_960p.yaml"
    config = OmegaConf.load(path)
    validate_stage4b_config(config, check_assets=False)

    assert config.training_mode == "stage4_reference_dpo"
    assert config.gradient_accumulation_steps >= 1
    assert config.train_causal_upsampler is False
    assert config.dpo_pair_source == "frozen_reference"
    assert config.causal_upsampler_checkpoint_path.endswith(
        "ltx-2-causal-upscaler-x2.pt"
    )
    assert "real_score" not in config
    assert "fake_score" not in config
    assert "critic_lr" not in config

    single_microbatch_config = OmegaConf.create(
        OmegaConf.to_container(config)
    )
    single_microbatch_config.gradient_accumulation_steps = 1
    validate_stage4b_config(single_microbatch_config, check_assets=False)

    invalid_config = OmegaConf.create(OmegaConf.to_container(config))
    invalid_config.gradient_accumulation_steps = 0
    with pytest.raises(ValueError, match="must be >= 1"):
        validate_stage4b_config(invalid_config, check_assets=False)

    invalid_pair_source = OmegaConf.create(OmegaConf.to_container(config))
    invalid_pair_source.dpo_pair_source = "moving_reference"
    with pytest.raises(ValueError, match="dpo_pair_source"):
        validate_stage4b_config(invalid_pair_source, check_assets=False)


def test_anchored_stage4b_pilot_config():
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "configs"
        / "stage3_scale_wise_960p"
        / "scale_wise_anchored_reference_dpo_960p.yaml"
    )
    config = OmegaConf.load(path)
    validate_stage4b_config(config, check_assets=False)

    assert config.dpo_pair_source == "frozen_reference"
    assert config.positive_video_mix == pytest.approx(0.5)
    assert config.positive_audio_source == "negative"
    assert config.negative_video_anchor_weight == pytest.approx(0.2)
    assert config.negative_audio_anchor_weight == pytest.approx(0.05)
    assert config.generator_lr == pytest.approx(2e-6)
    assert config.max_steps == 300
    assert config.checkpoint_iters == 25
    assert config.benchmark_at_start is True
    assert config.stability_guard_enabled is True
    assert config.stage35_prompt_source == "metadata"
    assert (
        config.causal_upsampler_state_checkpoint
        == config.policy_init_checkpoint
    )


def test_local_prompt_stream_requires_no_object_store(tmp_path):
    prompt_list = tmp_path / "prompts.txt"
    prompt_list.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    case_root = tmp_path / "cases"
    for index, prompt in enumerate(("third prompt", "fourth\nprompt")):
        case_dir = case_root / f"case_{index:04d}"
        case_dir.mkdir(parents=True)
        (case_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    config = OmegaConf.create(
        {
            "stage35_local_prompt_files": [str(prompt_list)],
            "stage35_local_prompt_globs": [
                str(case_root / "case_*" / "prompt.txt")
            ],
            "stage35_local_prompt_epoch_size": 24,
            "num_frames": 121,
            "video_height": 480,
            "video_width": 832,
        }
    )

    dataset = build_stage35_local_prompt_dataset(config)
    assert len(dataset) == 24
    assert dataset[0]["caption"] == "first prompt"
    assert dataset[1]["caption"] == "second prompt"
    assert dataset[2]["caption"] == "third prompt"
    assert dataset[3]["caption"] == "fourth\nprompt"
    assert dataset[4]["caption"] == "first prompt"
    assert dataset[0]["sample_frames"] == 121
    assert dataset[0]["height"] == 480
    assert dataset[0]["width"] == 832


def test_embedded_causal_upsampler_restore(tmp_path):
    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(1.25)
        source.bias.fill_(-0.5)
        target.weight.zero_()
        target.bias.zero_()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "causal_upsampler": source.state_dict(),
            "step": 3200,
        },
        checkpoint,
    )

    step = load_embedded_causal_upsampler(target, checkpoint)
    assert step == 3200
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_preference_pair_source_selects_frozen_reference(monkeypatch):
    policy = _ToyCausalPolicy(0.7, trainable=True)
    reference = _ToyCausalPolicy(0.7, trainable=False)
    objective = _objective(
        policy,
        reference,
        dpo_pair_source="frozen_reference",
    )
    selected_models = []

    def fake_sample_scale_wise(*, model, **_kwargs):
        selected_models.append(model)
        raise RuntimeError("selection observed")

    monkeypatch.setattr(objective, "sample_scale_wise", fake_sample_scale_wise)
    with pytest.raises(RuntimeError, match="selection observed"):
        objective.build_preference_pair(
            video_shape=(1, 2, 1, 1, 1),
            audio_shape=(1, 2, 1),
            conditional_dict={},
            real_video_frames=2,
            real_audio_frames=2,
        )
    assert selected_models == [reference]


def test_partial_video_positive_keeps_negative_audio(monkeypatch):
    policy = _ToyCausalPolicy(0.7, trainable=True)
    reference = _ToyCausalPolicy(0.7, trainable=False)
    objective = _objective(
        policy,
        reference,
        dpo_pair_source="frozen_reference",
        positive_video_mix=0.5,
        positive_audio_source="negative",
    )
    block = SimpleNamespace(video_start=0, video_end=2)
    rollout = Stage4Rollout(
        hr_video=torch.ones((1, 2, 1, 1, 1)),
        hr_audio=torch.full((1, 2, 1), 2.0),
        lr_video=torch.zeros((1, 2, 1, 1, 1)),
        lr_audio=torch.zeros((1, 2, 1)),
        upscaled_video=torch.zeros((1, 2, 1, 1, 1)),
        blocks=[block],
    )
    monkeypatch.setattr(
        objective,
        "sample_scale_wise",
        lambda **_kwargs: rollout,
    )
    monkeypatch.setattr(
        "ltx_distillation.stage4_reference_dpo.official_refine_full",
        lambda _model, **kwargs: (
            torch.full_like(kwargs["initial_video"], 3.0),
            torch.full_like(kwargs["initial_audio"], 7.0),
        ),
    )

    pair = objective.build_preference_pair(
        video_shape=rollout.hr_video.shape,
        audio_shape=rollout.hr_audio.shape,
        conditional_dict={},
        real_video_frames=2,
        real_audio_frames=2,
    )
    assert torch.equal(pair.positive_video, torch.full_like(pair.positive_video, 2.0))
    assert torch.equal(pair.positive_audio, pair.negative_audio)
    assert pair.relative_video_change.item() == pytest.approx(1.0)
    assert pair.relative_audio_change.item() == pytest.approx(0.0)
    assert pair.refiner_relative_video_change.item() == pytest.approx(4.0)
    assert pair.refiner_relative_audio_change.item() == pytest.approx(6.25)


def test_scale_wise_rollout_defaults_to_online_policy(monkeypatch):
    policy = _ToyCausalPolicy(0.7, trainable=True)
    reference = _ToyCausalPolicy(0.7, trainable=False)
    objective = _objective(
        policy,
        reference,
        dpo_pair_source="frozen_reference",
    )
    selected_models = []
    block = SimpleNamespace(video_start=0, video_end=2)

    def fake_rollout_scale(*, model, video_shape, audio_shape, **_kwargs):
        selected_models.append(model)
        return (
            torch.zeros(tuple(video_shape)),
            torch.zeros(tuple(audio_shape)),
            [block],
        )

    monkeypatch.setattr(objective, "_rollout_scale", fake_rollout_scale)
    monkeypatch.setattr(
        objective,
        "causal_upscale",
        lambda low_video, _blocks: low_video,
    )
    objective.sample_scale_wise(
        video_shape=(1, 2, 1, 1, 1),
        audio_shape=(1, 2, 1),
        conditional_dict={},
    )
    assert selected_models == [policy, policy]

    selected_models.clear()
    objective.sample_scale_wise(
        video_shape=(1, 2, 1, 1, 1),
        audio_shape=(1, 2, 1),
        conditional_dict={},
        model=reference,
    )
    assert selected_models == [reference, reference]


def test_trainer_source_does_not_construct_dmd_or_critic():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "src"
        / "ltx_distillation"
        / "train_stage4_reference_dpo.py"
    ).read_text(encoding="utf-8")
    assert "LTX2DMD" not in source
    assert "critic_optimizer" not in source
    assert "fake_score" not in source
    assert "real_score" not in source
