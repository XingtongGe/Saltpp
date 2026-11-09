from dataclasses import dataclass
from types import MethodType

import torch
from omegaconf import OmegaConf

from ltx_distillation.stage4_scale_wise_dmd import LTX2ScaleWiseDMD
from ltx_distillation.train_stage4_scale_wise import Stage4ScaleWiseTrainer


@dataclass
class _Block:
    video_start: int
    video_end: int
    audio_start: int
    audio_end: int

    @property
    def video_frames(self):
        return self.video_end - self.video_start

    @property
    def audio_frames(self):
        return self.audio_end - self.audio_start


def _bare_stage4():
    module = object.__new__(LTX2ScaleWiseDMD)
    torch.nn.Module.__init__(module)
    module.device = torch.device("cpu")
    module.dtype = torch.float32
    module.denoising_sigmas = torch.tensor([1.0, 0.96, 0.0])
    module.context_noise = 0
    module.num_train_timestep = 1000
    module.same_step_across_blocks = True
    module.stage4_random_exit_rollout = False
    module.stage4_train_causal_upsampler = False
    module._capture_hr_dmd_visualization = False
    module._hr_dmd_visualization_context = None
    module.real_video_guidance_scale = 1.0
    module.real_video_guidance_scale_min = 1.0
    module.real_video_guidance_scale_max = 1.0
    module.real_audio_guidance_scale = 1.0
    module.real_audio_guidance_scale_min = 1.0
    module.real_audio_guidance_scale_max = 1.0
    return module


def test_fixed_length_batch_shape_ignores_source_duration_metadata():
    trainer = object.__new__(Stage4ScaleWiseTrainer)
    trainer.config = OmegaConf.create(
        {
            "num_frames": 121,
            "num_frame_per_block": 3,
            "first_block_video_frames": 4,
        }
    )
    trainer.variable_length_rollout = False
    trainer.world_size = 1
    trainer.device = torch.device("cpu")

    shapes = trainer._batch_shapes(
        {"num_frames": 193, "height": 480, "width": 832}
    )

    video_shape, audio_shape = shapes[:2]
    real_num_frames, rollout_num_frames = shapes[2:4]
    real_video_frames, real_audio_frames = shapes[4:6]
    assert video_shape == [1, 16, 128, 15, 26]
    assert audio_shape == [1, 126, 128]
    assert real_num_frames == rollout_num_frames == 121
    assert real_video_frames == 16
    assert real_audio_frames == 126


def test_scale_rollout_runs_both_calls_even_when_replay_exit_is_early():
    module = _bare_stage4()
    blocks = [_Block(0, 1, 0, 2), _Block(1, 4, 2, 5)]
    module._get_causal_blocks = MethodType(
        lambda self, frames: blocks, module
    )
    module._generate_exit_indices = MethodType(
        lambda self, count, num_denoising_steps: [0, 1], module
    )
    module._init_self_forcing_kv_caches = MethodType(
        lambda self, **kwargs: [], module
    )

    calls = []

    def forward(self, **kwargs):
        assert not torch.is_grad_enabled()
        calls.append(
            (
                kwargs["block"].video_start,
                float(kwargs["timestep_block"][0, 0]),
            )
        )
        return (
            kwargs["noisy_video_block"] * 0 + len(calls),
            kwargs["noisy_audio_block"] * 0 + len(calls),
        )

    refreshes = []
    module._kv_block_forward = MethodType(forward, module)
    module._kv_refresh_cache_for_block = MethodType(
        lambda self, **kwargs: refreshes.append(
            kwargs["block"].video_start
        ),
        module,
    )

    rollout = module._rollout_scale_no_grad(
        video_shape=[1, 4, 128, 1, 1],
        audio_shape=[1, 5, 128],
        conditional_dict={"video_context": torch.zeros(1, 8, 4)},
        sigmas=torch.tensor([1.0, 0.96, 0.0]),
    )

    assert [item[0] for item in calls] == [0, 0, 1, 1]
    assert torch.allclose(
        torch.tensor([item[1] for item in calls]),
        torch.tensor([1.0, 0.96, 1.0, 0.96]),
    )
    assert refreshes == [0, 1]
    assert rollout.exit_indices == [0, 1]
    assert rollout.replay_video[0].shape[1] == 1
    assert rollout.replay_video[1].shape[1] == 3
    # Both blocks use the second call as their actual clean endpoint.
    assert torch.all(rollout.video[:, :1] == 2)
    assert torch.all(rollout.video[:, 1:] == 4)


def test_v2_random_exit_stops_at_selected_node_and_refreshes_its_x0():
    module = _bare_stage4()
    module.stage4_random_exit_rollout = True
    blocks = [_Block(0, 1, 0, 2), _Block(1, 4, 2, 5)]
    module._get_causal_blocks = MethodType(
        lambda self, frames: blocks, module
    )
    module._generate_exit_indices = MethodType(
        lambda self, count, num_denoising_steps: [0, 1], module
    )
    module._init_self_forcing_kv_caches = MethodType(
        lambda self, **kwargs: [], module
    )

    calls = []

    def forward(self, **kwargs):
        calls.append(
            (
                kwargs["block"].video_start,
                float(kwargs["timestep_block"][0, 0]),
            )
        )
        value = len(calls)
        return (
            kwargs["noisy_video_block"] * 0 + value,
            kwargs["noisy_audio_block"] * 0 + value,
        )

    refreshes = []
    module._kv_block_forward = MethodType(forward, module)
    module._kv_refresh_cache_for_block = MethodType(
        lambda self, **kwargs: refreshes.append(
            (
                kwargs["block"].video_start,
                float(kwargs["x0_video_block"].mean()),
            )
        ),
        module,
    )

    rollout = module._rollout_scale_no_grad(
        video_shape=[1, 4, 128, 1, 1],
        audio_shape=[1, 5, 128],
        conditional_dict={"video_context": torch.zeros(1, 8, 4)},
        sigmas=torch.tensor([1.0, 0.96, 0.0]),
    )

    assert [item[0] for item in calls] == [0, 1, 1]
    assert torch.allclose(
        torch.tensor([item[1] for item in calls]),
        torch.tensor([1.0, 1.0, 0.96]),
    )
    assert refreshes == [(0, 1.0), (1, 3.0)]
    assert rollout.exit_indices == [0, 1]
    assert torch.all(rollout.video[:, :1] == 1)
    assert torch.all(rollout.video[:, 1:] == 3)


def test_v2_force_full_steps_keeps_benchmark_at_two_calls():
    module = _bare_stage4()
    module.stage4_random_exit_rollout = True
    blocks = [_Block(0, 1, 0, 2), _Block(1, 4, 2, 5)]
    module._get_causal_blocks = MethodType(
        lambda self, frames: blocks, module
    )
    module._generate_exit_indices = MethodType(
        lambda self, count, num_denoising_steps: [0, 0], module
    )
    module._init_self_forcing_kv_caches = MethodType(
        lambda self, **kwargs: [], module
    )

    calls = []

    def forward(self, **kwargs):
        calls.append(kwargs["block"].video_start)
        value = len(calls)
        return (
            kwargs["noisy_video_block"] * 0 + value,
            kwargs["noisy_audio_block"] * 0 + value,
        )

    module._kv_block_forward = MethodType(forward, module)
    module._kv_refresh_cache_for_block = MethodType(
        lambda self, **kwargs: None, module
    )

    rollout = module._rollout_scale_no_grad(
        video_shape=[1, 4, 128, 1, 1],
        audio_shape=[1, 5, 128],
        conditional_dict={"video_context": torch.zeros(1, 8, 4)},
        sigmas=torch.tensor([1.0, 0.96, 0.0]),
        force_full_steps=True,
    )

    assert calls == [0, 0, 1, 1]
    assert torch.all(rollout.video[:, :1] == 2)
    assert torch.all(rollout.video[:, 1:] == 4)


def test_causal_upscale_is_blockwise_and_crops_each_current_block():
    module = _bare_stage4()
    blocks = [_Block(0, 1, 0, 2), _Block(1, 4, 2, 5)]
    calls = []

    class Upsampler(torch.nn.Module):
        def forward(self, video, block_start, block_end):
            calls.append((block_start, block_end))
            shape = (
                video.shape[0],
                block_end - block_start,
                video.shape[2],
                video.shape[3] * 2,
                video.shape[4] * 2,
            )
            return torch.full(shape, block_end, dtype=video.dtype)

    module.causal_upsampler = Upsampler()
    video = torch.zeros(1, 4, 128, 2, 3)
    output = module._causal_upscale(video, blocks)

    assert calls == [(0, 1), (1, 4)]
    assert output.shape == (1, 4, 128, 4, 6)
    assert torch.all(output[:, :1] == 1)
    assert torch.all(output[:, 1:] == 4)


def test_generator_replay_only_tracks_selected_nodes_and_backpropagates():
    module = _bare_stage4()
    blocks = [_Block(0, 1, 0, 2), _Block(1, 4, 2, 5)]
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    module.generator = torch.nn.Linear(1, 1, bias=False)
    module.generator.weight = parameter
    module.video_loss_weight = 1.0
    module.audio_loss_weight = 0.16
    module.audio_start_step = 0
    module.current_step = 0
    module.block_weight_mode = "uniform"
    module.block_weight_min = 0.5
    module.num_frame_per_block = 3
    module.first_block_video_frames = 1
    module._init_self_forcing_kv_caches = MethodType(
        lambda self, **kwargs: [], module
    )
    module._kv_refresh_cache_for_block = MethodType(
        lambda self, **kwargs: None, module
    )

    grad_enabled_calls = []

    def forward(self, **kwargs):
        grad_enabled_calls.append(torch.is_grad_enabled())
        return (
            kwargs["noisy_video_block"] * parameter,
            kwargs["noisy_audio_block"] * parameter,
        )

    module._kv_block_forward = MethodType(forward, module)
    replay_video = [
        torch.ones(1, 1, 128, 1, 1),
        torch.ones(1, 3, 128, 1, 1),
    ]
    replay_audio = [
        torch.ones(1, 2, 128),
        torch.ones(1, 3, 128),
    ]
    rollout_video = torch.zeros(1, 4, 128, 1, 1)
    rollout_audio = torch.zeros(1, 5, 128)
    from ltx_distillation.stage4_scale_wise_dmd import _ScaleRollout

    rollout = _ScaleRollout(
        video=rollout_video,
        audio=rollout_audio,
        replay_video=replay_video,
        replay_audio=replay_audio,
        exit_indices=[0, 1],
        blocks=blocks,
    )
    total, _, _ = module._backward_scale_generator(
        rollout=rollout,
        sigmas=torch.tensor([1.0, 0.96, 0.0]),
        grad_video=torch.ones_like(rollout_video),
        grad_audio=torch.ones_like(rollout_audio),
        conditional_dict={"video_context": torch.zeros(1, 8, 4)},
        real_video_frames=4,
        real_audio_frames=5,
        scale_weight=0.5,
    )

    assert grad_enabled_calls == [True, True]
    assert total > 0
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert parameter.grad.abs().item() > 0


def test_first_hr_exit_reconnects_dmd_gradient_to_causal_upsampler():
    module = _bare_stage4()
    module.stage4_random_exit_rollout = True
    module.stage4_train_causal_upsampler = True
    blocks = [_Block(0, 1, 0, 2), _Block(1, 4, 2, 5)]

    generator_parameter = torch.nn.Parameter(torch.tensor(1.0))
    upsampler_parameter = torch.nn.Parameter(torch.tensor(2.0))
    module.generator = torch.nn.Linear(1, 1, bias=False)
    module.generator.weight = generator_parameter

    class Upsampler(torch.nn.Module):
        def forward(self, video, block_start, block_end):
            return (
                video[:, block_start:block_end] * upsampler_parameter
            )

    module.causal_upsampler = Upsampler()
    module.video_loss_weight = 1.0
    module.audio_loss_weight = 0.16
    module.audio_start_step = 0
    module.current_step = 0
    module.block_weight_mode = "uniform"
    module.block_weight_min = 0.5
    module.num_frame_per_block = 3
    module.first_block_video_frames = 1
    module._init_self_forcing_kv_caches = MethodType(
        lambda self, **kwargs: [], module
    )
    module._kv_refresh_cache_for_block = MethodType(
        lambda self, **kwargs: None, module
    )

    replay_values = []

    def forward(self, **kwargs):
        replay_values.append(
            kwargs["noisy_video_block"].detach().clone()
        )
        return (
            kwargs["noisy_video_block"] * generator_parameter,
            kwargs["noisy_audio_block"] * generator_parameter,
        )

    module._kv_block_forward = MethodType(forward, module)
    replay_video = [
        torch.full((1, 1, 128, 1, 1), 3.0),
        torch.full((1, 3, 128, 1, 1), 4.0),
    ]
    replay_audio = [
        torch.ones(1, 2, 128),
        torch.ones(1, 3, 128),
    ]
    rollout_video = torch.zeros(1, 4, 128, 1, 1)
    rollout_audio = torch.zeros(1, 5, 128)
    from ltx_distillation.stage4_scale_wise_dmd import _ScaleRollout

    rollout = _ScaleRollout(
        video=rollout_video,
        audio=rollout_audio,
        replay_video=replay_video,
        replay_audio=replay_audio,
        exit_indices=[0, 0],
        blocks=blocks,
    )
    source_video = torch.ones(1, 4, 128, 1, 1)
    module._backward_scale_generator(
        rollout=rollout,
        sigmas=torch.tensor([0.889, 0.727, 0.0]),
        grad_video=torch.ones_like(rollout_video),
        grad_audio=torch.ones_like(rollout_audio),
        conditional_dict={"video_context": torch.zeros(1, 8, 4)},
        real_video_frames=4,
        real_audio_frames=5,
        scale_weight=0.5,
        upsampler_source_video=source_video,
    )

    # The zero-valued reconnect must not perturb the exact no-grad rollout
    # input used by DMD replay.
    assert torch.equal(replay_values[0], replay_video[0])
    assert torch.equal(replay_values[1], replay_video[1])
    assert upsampler_parameter.grad is not None
    assert torch.isfinite(upsampler_parameter.grad)
    assert upsampler_parameter.grad.abs().item() > 0


def test_second_hr_exit_keeps_causal_upsampler_out_of_dmd_graph():
    module = _bare_stage4()
    module.stage4_random_exit_rollout = True
    module.stage4_train_causal_upsampler = True
    blocks = [_Block(0, 1, 0, 2)]

    generator_parameter = torch.nn.Parameter(torch.tensor(1.0))
    upsampler_parameter = torch.nn.Parameter(torch.tensor(2.0))
    module.generator = torch.nn.Linear(1, 1, bias=False)
    module.generator.weight = generator_parameter

    class Upsampler(torch.nn.Module):
        def forward(self, video, block_start, block_end):
            return (
                video[:, block_start:block_end] * upsampler_parameter
            )

    module.causal_upsampler = Upsampler()
    module.video_loss_weight = 1.0
    module.audio_loss_weight = 0.16
    module.audio_start_step = 0
    module.current_step = 0
    module.block_weight_mode = "uniform"
    module.block_weight_min = 0.5
    module.num_frame_per_block = 1
    module.first_block_video_frames = 1
    module._init_self_forcing_kv_caches = MethodType(
        lambda self, **kwargs: [], module
    )
    module._kv_refresh_cache_for_block = MethodType(
        lambda self, **kwargs: None, module
    )
    module._kv_block_forward = MethodType(
        lambda self, **kwargs: (
            kwargs["noisy_video_block"] * generator_parameter,
            kwargs["noisy_audio_block"] * generator_parameter,
        ),
        module,
    )

    from ltx_distillation.stage4_scale_wise_dmd import _ScaleRollout

    rollout = _ScaleRollout(
        video=torch.zeros(1, 1, 128, 1, 1),
        audio=torch.zeros(1, 2, 128),
        replay_video=[torch.ones(1, 1, 128, 1, 1)],
        replay_audio=[torch.ones(1, 2, 128)],
        exit_indices=[1],
        blocks=blocks,
    )
    module._backward_scale_generator(
        rollout=rollout,
        sigmas=torch.tensor([0.889, 0.727, 0.0]),
        grad_video=torch.ones_like(rollout.video),
        grad_audio=torch.ones_like(rollout.audio),
        conditional_dict={"video_context": torch.zeros(1, 8, 4)},
        real_video_frames=1,
        real_audio_frames=2,
        scale_weight=0.5,
        upsampler_source_video=None,
    )

    assert upsampler_parameter.grad is None
    assert generator_parameter.grad is not None


def test_dmd_score_predictions_are_returned_only_when_requested():
    module = _bare_stage4()
    module.real_video_guidance_scale = 1.0
    module.real_audio_guidance_scale = 1.0

    class ConstantScore(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)

        def forward(
            self,
            *,
            noisy_image_or_video,
            noisy_audio,
            **kwargs,
        ):
            return (
                torch.full_like(noisy_image_or_video, self.value),
                torch.full_like(noisy_audio, self.value),
            )

    module.fake_score = ConstantScore(1.0)
    module.real_score = ConstantScore(2.0)
    video = torch.zeros(1, 2, 128, 1, 1)
    audio = torch.zeros(1, 3, 128)
    video_sigma = torch.full((1, 2), 0.5)
    audio_sigma = torch.full((1, 3), 0.5)
    kwargs = {
        "noisy_video": video,
        "noisy_audio": audio,
        "clean_video": video,
        "clean_audio": audio,
        "video_sigma": video_sigma,
        "audio_sigma": audio_sigma,
        "conditional_dict": {},
        "unconditional_dict": {},
        "normalization": False,
    }

    _, _, ordinary_logs = module._compute_kl_grad(**kwargs)
    assert "_pred_real_video_x0" not in ordinary_logs
    assert "_pred_fake_video_x0" not in ordinary_logs

    _, _, capture_logs = module._compute_kl_grad(
        **kwargs,
        return_video_predictions=True,
    )
    assert torch.all(capture_logs["_pred_real_video_x0"] == 2)
    assert torch.all(capture_logs["_pred_fake_video_x0"] == 1)


def test_stage4_hr_capture_keeps_detached_visualization_on_cpu():
    module = _bare_stage4()
    module.real_task_type = "bidirectional"
    module.min_step = 0
    module.max_step = 1000
    module._process_timestep = MethodType(
        lambda self, timestep, task_type: timestep,
        module,
    )
    module._compute_audio_timestep = MethodType(
        lambda self, timestep, frames, task_type: torch.zeros(
            timestep.shape[0],
            frames,
            device=timestep.device,
            dtype=timestep.dtype,
        ),
        module,
    )
    module.timestep_to_sigma = MethodType(
        lambda self, timestep: timestep.float() * 0 + 0.5,
        module,
    )
    module.add_noise = MethodType(
        lambda self, original, noise, sigma: original,
        module,
    )

    return_flags = []

    def compute(self, **kwargs):
        return_flags.append(kwargs["return_video_predictions"])
        logs = {}
        if kwargs["return_video_predictions"]:
            logs["_pred_real_video_x0"] = (
                kwargs["clean_video"] + 1
            )
            logs["_pred_fake_video_x0"] = (
                kwargs["clean_video"] + 2
            )
        return (
            torch.zeros_like(kwargs["clean_video"]),
            torch.zeros_like(kwargs["clean_audio"]),
            logs,
        )

    module._compute_kl_grad = MethodType(compute, module)
    from ltx_distillation.stage4_scale_wise_dmd import _ScaleRollout

    rollout = _ScaleRollout(
        video=torch.zeros(1, 2, 128, 1, 1),
        audio=torch.zeros(1, 3, 128),
        replay_video=[],
        replay_audio=[],
        exit_indices=[],
        blocks=[],
    )
    _, _, logs = module._compute_scale_dmd_gradient(
        rollout=rollout,
        conditional_dict={},
        unconditional_dict={},
        real_video_frames=2,
        real_audio_frames=3,
        capture_video_predictions=True,
    )

    assert return_flags == [True]
    assert "_pred_real_video_x0" not in logs
    assert "_pred_fake_video_x0" not in logs
    context = module.pop_hr_dmd_visualization_context()
    assert context is not None
    assert context["generator_video"].device.type == "cpu"
    assert context["real_score_video_x0"].device.type == "cpu"
    assert context["fake_score_video_x0"].device.type == "cpu"
    assert torch.all(context["generator_video"] == 0)
    assert torch.all(context["real_score_video_x0"] == 1)
    assert torch.all(context["fake_score_video_x0"] == 2)
    assert module.pop_hr_dmd_visualization_context() is None
