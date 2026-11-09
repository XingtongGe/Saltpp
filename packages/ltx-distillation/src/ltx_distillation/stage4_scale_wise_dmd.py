"""Stage-4A scale-wise Self-Forcing DMD for LTX-2.

The four student calls keep the Stage-3 schedule but split it across spatial
scales:

    LR: 1000 -> 960 -> clean
    HR:  889 -> 727 -> clean

LR and HR use the same causal generator weights and independent AV KV caches.
The only cross-scale edge is a causal latent upsampler followed by fresh noise
at sigma=0.889.  By default the upsampler stays frozen.  Stage-4 V2 can
optionally update it from HR DMD only when the randomly selected HR exit is the
first denoising call.  Generator and fake-score objectives are executed
scale-major so the LR KV cache is released before the HR cache is allocated.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from ltx_distillation.causal_window_upsampler import (
    build_causal_upsampler_from_checkpoint,
)
from ltx_distillation.dmd import LTX2DMD


@dataclass
class _ScaleRollout:
    video: torch.Tensor
    audio: torch.Tensor
    replay_video: List[torch.Tensor]
    replay_audio: List[torch.Tensor]
    exit_indices: List[int]
    blocks: list


class LTX2ScaleWiseDMD(LTX2DMD):
    """DMD-only 2-step LR + 2-step HR causal generator objective."""

    EXPECTED_SCHEDULE = (1000, 960, 889, 727, 0)

    def __init__(self, args, device: torch.device):
        super().__init__(args, device)

        schedule = tuple(int(value) for value in args.denoising_step_list)
        if schedule != self.EXPECTED_SCHEDULE:
            raise ValueError(
                "Stage4A requires the fixed schedule "
                f"{list(self.EXPECTED_SCHEDULE)}, got {list(schedule)}"
            )
        if not bool(getattr(args, "strict_denoising_sigmas", False)):
            raise ValueError("Stage4A requires strict_denoising_sigmas=true")
        if not self.enable_self_forcing:
            raise ValueError(
                "Stage4A requires training_mode=causal_self_forcing"
            )
        if not self.generator_use_causal_wrapper:
            raise ValueError(
                "Stage4A requires generator_use_causal_wrapper=true"
            )
        if self.fake_score_use_causal_wrapper:
            raise ValueError(
                "Stage4A fake_score must remain bidirectional"
            )

        # The two scales deliberately do not share a continuous trajectory.
        # The LR endpoint is clean before causal upsampling, and HR starts by
        # adding fresh noise at 0.889.
        self.register_buffer(
            "stage4_lr_sigmas",
            torch.tensor([1.0, 0.960, 0.0], device=device),
        )
        self.register_buffer(
            "stage4_hr_sigmas",
            torch.tensor([0.889, 0.727, 0.0], device=device),
        )
        self.stage4_lr_dmd_weight = float(
            getattr(args, "stage4_lr_dmd_weight", 0.5)
        )
        self.stage4_hr_dmd_weight = float(
            getattr(args, "stage4_hr_dmd_weight", 0.5)
        )
        if self.stage4_lr_dmd_weight < 0 or self.stage4_hr_dmd_weight < 0:
            raise ValueError("Stage4A DMD scale weights must be non-negative")
        if self.stage4_lr_dmd_weight + self.stage4_hr_dmd_weight <= 0:
            raise ValueError("At least one Stage4A DMD scale weight must be > 0")

        # V1 always completes both denoising calls before computing DMD on the
        # final endpoint.  V2 follows Stage-3 selected-exit semantics: each
        # scale stops at a uniformly sampled node, then uses that same x0 for
        # DMD, clean-cache refresh, and the LR->HR transition.
        self.stage4_random_exit_rollout = bool(
            getattr(args, "stage4_random_exit_rollout", False)
        )
        if self.stage4_random_exit_rollout and not self.same_step_across_blocks:
            raise ValueError(
                "Stage4A V2 random-exit training requires "
                "same_step_across_blocks=true"
            )
        self.stage4_train_causal_upsampler = bool(
            getattr(args, "stage4_train_causal_upsampler", False)
        )
        if (
            self.stage4_train_causal_upsampler
            and not self.stage4_random_exit_rollout
        ):
            raise ValueError(
                "Stage4A trainable causal upsampler is only supported by V2; "
                "set stage4_random_exit_rollout=true"
            )

        self.causal_upsampler = None
        self._memory_hook = None
        self._capture_hr_dmd_visualization = False
        self._hr_dmd_visualization_context = None

    def init_models(self):
        """Build Stage-3 models, restore its critic, then add the upsampler."""

        super().init_models()
        args = self.args

        critic_checkpoint = getattr(
            args, "stage4_fake_score_checkpoint", None
        )
        if critic_checkpoint:
            self._load_stage3_fake_score(
                str(critic_checkpoint),
                strict=bool(
                    getattr(args, "stage4_fake_score_checkpoint_strict", True)
                ),
            )

        upsampler_device = (
            torch.device(f"cuda:{self.device}")
            if isinstance(self.device, int)
            else torch.device(self.device)
        )
        self.causal_upsampler = build_causal_upsampler_from_checkpoint(
            checkpoint_path=str(args.checkpoint_path),
            spatial_upsampler_path=str(args.spatial_upsampler_path),
            causal_upsampler_checkpoint_path=str(
                args.causal_upsampler_checkpoint_path
            ),
            device=upsampler_device,
            dtype=self.dtype,
            history_frames=getattr(
                args, "causal_upsampler_history_frames", None
            ),
            gradient_checkpointing=(
                self.stage4_train_causal_upsampler
                and bool(getattr(args, "gradient_checkpointing", True))
            ),
        )
        self.causal_upsampler.eval().requires_grad_(
            self.stage4_train_causal_upsampler
        )

    def _load_stage3_fake_score(
        self,
        checkpoint_path: str,
        *,
        strict: bool,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Stage4A fake-score checkpoint does not exist: {path}"
            )
        try:
            loaded = torch.load(
                path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise TypeError(
                f"Expected mapping checkpoint at {path}, got "
                f"{type(loaded).__name__}"
            )
        state = loaded.get("critic", loaded.get("fake_score"))
        if not isinstance(state, dict):
            raise KeyError(
                f"{path} contains neither a 'critic' nor 'fake_score' state"
            )

        remapped = {}
        for key, value in state.items():
            normalized = key
            if normalized.startswith("_fsdp_wrapped_module."):
                normalized = normalized[len("_fsdp_wrapped_module.") :]
            remapped[normalized] = value

        missing, unexpected = self.fake_score.load_state_dict(
            remapped, strict=strict
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "[Stage4A] fake_score loaded from "
                f"{path}: missing={len(missing)}, "
                f"unexpected={len(unexpected)}",
                flush=True,
            )
        del loaded, state, remapped

    def set_memory_hook(self, hook) -> None:
        """Install an optional trainer callback for phase-level CUDA peaks."""

        self._memory_hook = hook

    def set_hr_dmd_visualization_capture(self, enabled: bool) -> None:
        """Request one detached HR DMD triplet from the next generator pass."""

        self._capture_hr_dmd_visualization = bool(enabled)
        self._hr_dmd_visualization_context = None

    def pop_hr_dmd_visualization_context(self) -> Optional[Dict[str, Any]]:
        context = self._hr_dmd_visualization_context
        self._hr_dmd_visualization_context = None
        self._capture_hr_dmd_visualization = False
        return context

    def _mark_memory(self, phase: str) -> None:
        if self._memory_hook is not None:
            self._memory_hook(phase)

    @staticmethod
    def _sigma_matrix(
        sigma: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return sigma.to(
            device=reference.device, dtype=torch.float32
        ).expand(reference.shape[0], reference.shape[1])

    def _rollout_scale_no_grad(
        self,
        *,
        video_shape: Sequence[int],
        audio_shape: Sequence[int],
        conditional_dict: Dict[str, Any],
        sigmas: torch.Tensor,
        source_video: Optional[torch.Tensor] = None,
        source_audio: Optional[torch.Tensor] = None,
        force_full_steps: bool = False,
    ) -> _ScaleRollout:
        """Run one scale and retain one replay input per block.

        V1 completes both calls regardless of the sampled replay node.  V2
        stops at the sampled node during training so the rollout endpoint,
        DMD sample, replay prediction, and clean cache context are aligned.
        ``force_full_steps`` keeps benchmark/inference fixed at two calls.
        """

        if len(sigmas) != 3:
            raise ValueError(
                f"Each Stage4 scale must have two calls, got {len(sigmas) - 1}"
            )
        B, F_v = int(video_shape[0]), int(video_shape[1])
        F_a = int(audio_shape[1])
        video_tail = tuple(int(value) for value in video_shape[2:])
        audio_channels = int(audio_shape[2])
        blocks = self._get_causal_blocks(F_v)
        if not blocks:
            raise ValueError("Stage4A scale rollout sampled zero causal blocks")
        if blocks[-1].audio_end != F_a:
            raise ValueError(
                "Stage4A audio shape must use causal AV alignment: "
                f"shape has {F_a}, blocks require {blocks[-1].audio_end}"
            )
        if (source_video is None) != (source_audio is None):
            raise ValueError(
                "source_video and source_audio must either both be set or both be None"
            )
        if source_video is not None:
            if tuple(source_video.shape) != tuple(video_shape):
                raise ValueError(
                    "HR source video shape mismatch: "
                    f"{tuple(source_video.shape)} vs {tuple(video_shape)}"
                )
            if tuple(source_audio.shape) != tuple(audio_shape):
                raise ValueError(
                    "HR source audio shape mismatch: "
                    f"{tuple(source_audio.shape)} vs {tuple(audio_shape)}"
                )

        exit_indices = self._generate_exit_indices(
            len(blocks), num_denoising_steps=2
        )
        output_video = torch.empty(
            tuple(video_shape), device=self.device, dtype=self.dtype
        )
        output_audio = torch.empty(
            tuple(audio_shape), device=self.device, dtype=self.dtype
        )
        replay_video: List[torch.Tensor] = []
        replay_audio: List[torch.Tensor] = []
        text_seq_len = int(conditional_dict["video_context"].shape[1])

        with torch.no_grad():
            caches = self._init_self_forcing_kv_caches(
                batch_size=B,
                num_video_frames=F_v,
                num_audio_frames=F_a,
                text_seq_len=text_seq_len,
                video_frame_seqlen=int(video_shape[-2])
                * int(video_shape[-1]),
            )
            for block_index, block in enumerate(blocks):
                v_slice = slice(block.video_start, block.video_end)
                a_slice = slice(block.audio_start, block.audio_end)
                if source_video is None:
                    current_video = torch.randn(
                        (B, block.video_frames, *video_tail),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    current_audio = torch.randn(
                        (B, block.audio_frames, audio_channels),
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    initial_video_sigma = self._sigma_matrix(
                        sigmas[0], source_video[:, v_slice]
                    )
                    initial_audio_sigma = self._sigma_matrix(
                        sigmas[0], source_audio[:, a_slice]
                    )
                    current_video = self.add_noise(
                        source_video[:, v_slice],
                        torch.randn_like(source_video[:, v_slice]),
                        initial_video_sigma,
                    )
                    current_audio = self.add_noise(
                        source_audio[:, a_slice],
                        torch.randn_like(source_audio[:, a_slice]),
                        initial_audio_sigma,
                    )

                selected_video = None
                selected_audio = None
                for step_index, sigma in enumerate(sigmas[:-1]):
                    if step_index == exit_indices[block_index]:
                        selected_video = current_video.detach().clone()
                        selected_audio = current_audio.detach().clone()
                    video_sigma = self._sigma_matrix(sigma, current_video)
                    audio_sigma = self._sigma_matrix(sigma, current_audio)
                    pred_video, pred_audio = self._kv_block_forward(
                        noisy_video_block=current_video,
                        noisy_audio_block=current_audio,
                        timestep_block=video_sigma,
                        audio_timestep_block=audio_sigma,
                        kv_caches=caches,
                        block=block,
                        conditional_dict=conditional_dict,
                    )
                    if (
                        self.stage4_random_exit_rollout
                        and not force_full_steps
                        and step_index == exit_indices[block_index]
                    ):
                        break
                    if step_index + 1 < len(sigmas) - 1:
                        next_sigma = sigmas[step_index + 1]
                        current_video = self.add_noise(
                            pred_video,
                            torch.randn_like(pred_video),
                            self._sigma_matrix(next_sigma, pred_video),
                        )
                        current_audio = self.add_noise(
                            pred_audio,
                            torch.randn_like(pred_audio),
                            self._sigma_matrix(next_sigma, pred_audio),
                        )

                if selected_video is None or selected_audio is None:
                    raise RuntimeError("Stage4A failed to capture replay input")
                replay_video.append(selected_video)
                replay_audio.append(selected_audio)
                output_video[:, v_slice] = pred_video
                output_audio[:, a_slice] = pred_audio
                self._kv_refresh_cache_for_block(
                    x0_video_block=pred_video,
                    x0_audio_block=pred_audio,
                    kv_caches=caches,
                    block=block,
                    conditional_dict=conditional_dict,
                )

            del caches

        return _ScaleRollout(
            video=output_video.detach(),
            audio=output_audio.detach(),
            replay_video=replay_video,
            replay_audio=replay_audio,
            exit_indices=exit_indices,
            blocks=blocks,
        )

    def _compute_scale_dmd_gradient(
        self,
        *,
        rollout: _ScaleRollout,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        real_video_frames: Optional[int],
        real_audio_frames: Optional[int],
        capture_video_predictions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Evaluate frozen real/fake scores on one detached scale endpoint."""

        full_video_frames = int(rollout.video.shape[1])
        full_audio_frames = int(rollout.audio.shape[1])
        rvf = (
            full_video_frames
            if real_video_frames is None
            else max(1, min(int(real_video_frames), full_video_frames))
        )
        raf = (
            full_audio_frames
            if real_audio_frames is None
            else max(1, min(int(real_audio_frames), full_audio_frames))
        )
        clean_video = rollout.video[:, :rvf]
        clean_audio = rollout.audio[:, :raf]
        B = int(clean_video.shape[0])

        with torch.no_grad():
            video_timestep = torch.randint(
                0,
                self.num_train_timestep,
                (B, rvf),
                device=self.device,
                dtype=torch.long,
            )
            video_timestep = self._process_timestep(
                video_timestep, self.real_task_type
            ).clamp(self.min_step, self.max_step)
            audio_timestep = self._compute_audio_timestep(
                video_timestep, raf, task_type=self.real_task_type
            )
            video_sigma = self.timestep_to_sigma(video_timestep)
            audio_sigma = self.timestep_to_sigma(audio_timestep)
            noise_video = torch.randn_like(clean_video)
            noise_audio = torch.randn_like(clean_audio)
            noisy_video = self.add_noise(
                clean_video.flatten(0, 1),
                noise_video.flatten(0, 1),
                video_sigma.flatten(0, 1),
            ).unflatten(0, (B, rvf))
            noisy_audio = self.add_noise(
                clean_audio, noise_audio, audio_sigma
            )
            grad_video, grad_audio, logs = self._compute_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                clean_video=clean_video,
                clean_audio=clean_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=None,
                audio_loss_mask=None,
                return_video_predictions=capture_video_predictions,
            )
            if capture_video_predictions:
                pred_real_video = logs.pop("_pred_real_video_x0")
                pred_fake_video = logs.pop("_pred_fake_video_x0")
                self._hr_dmd_visualization_context = {
                    # Keep the visualization payload away from the following
                    # replay backward and critic GPU-memory peaks.
                    "generator_video": clean_video.detach().cpu(),
                    "real_score_video_x0": pred_real_video.detach().cpu(),
                    "fake_score_video_x0": pred_fake_video.detach().cpu(),
                    "video_sigma_mean": float(
                        video_sigma.float().mean().item()
                    ),
                }

        if rvf < full_video_frames:
            padded_video = torch.zeros_like(rollout.video)
            padded_video[:, :rvf] = grad_video
            grad_video = padded_video
        if raf < full_audio_frames:
            padded_audio = torch.zeros_like(rollout.audio)
            padded_audio[:, :raf] = grad_audio
            grad_audio = padded_audio
        logs["alignment/video_sigma_mean"] = (
            video_sigma.float().mean().item()
        )
        logs["alignment/audio_sigma_mean"] = (
            audio_sigma.float().mean().item()
        )
        return grad_video, grad_audio, logs

    def _backward_scale_generator(
        self,
        *,
        rollout: _ScaleRollout,
        sigmas: torch.Tensor,
        grad_video: torch.Tensor,
        grad_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        real_video_frames: Optional[int],
        real_audio_frames: Optional[int],
        scale_weight: float,
        upsampler_source_video: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """Replay one selected node per block and immediately backpropagate.

        ``upsampler_source_video`` is only supplied for the HR scale when the
        V2 exit is the first HR denoising call.  The no-grad rollout replay
        already contains the exact noisy input sampled from the causal
        upsampler output.  Recomputing the current upsampler block and adding
        ``(1-sigma) * (upscaled - upscaled.detach())`` preserves that replay
        value exactly while restoring the flow-matching input derivative
        ``d x_t / d x_0 = 1-sigma``.  HR DMD can therefore update the
        upsampler without retaining the full LR-to-HR rollout graph.
        """

        B, F_v = rollout.video.shape[:2]
        F_a = int(rollout.audio.shape[1])
        rvf = (
            F_v
            if real_video_frames is None
            else max(1, min(int(real_video_frames), F_v))
        )
        raf = (
            F_a
            if real_audio_frames is None
            else max(1, min(int(real_audio_frames), F_a))
        )
        video_block_weights = self._compute_block_weights(F_v)
        audio_block_weights = self._compute_block_weights(
            F_a, is_audio=True
        )
        if rvf < F_v:
            video_block_weights = video_block_weights.clone()
            video_block_weights[rvf:] = 0
        if raf < F_a:
            audio_block_weights = audio_block_weights.clone()
            audio_block_weights[raf:] = 0

        video_loss_weight, audio_loss_weight = self.get_loss_weights()
        text_seq_len = int(conditional_dict["video_context"].shape[1])
        caches = self._init_self_forcing_kv_caches(
            batch_size=B,
            num_video_frames=F_v,
            num_audio_frames=F_a,
            text_seq_len=text_seq_len,
            video_frame_seqlen=int(rollout.video.shape[-2])
            * int(rollout.video.shape[-1]),
        )
        fsdp_no_sync = getattr(self.generator, "no_sync", None)
        upsampler_no_sync = (
            getattr(self.causal_upsampler, "no_sync", None)
            if upsampler_source_video is not None
            else None
        )
        total = 0.0
        total_video = 0.0
        total_audio = 0.0

        for block_index, block in enumerate(rollout.blocks):
            is_last = block_index == len(rollout.blocks) - 1
            with ExitStack() as stack:
                stack.enter_context(
                    nullcontext()
                    if is_last or not callable(fsdp_no_sync)
                    else fsdp_no_sync()
                )
                stack.enter_context(
                    nullcontext()
                    if is_last or not callable(upsampler_no_sync)
                    else upsampler_no_sync()
                )
                replay_video = rollout.replay_video[block_index]
                replay_audio = rollout.replay_audio[block_index]
                sigma = sigmas[rollout.exit_indices[block_index]]
                if upsampler_source_video is not None:
                    if rollout.exit_indices[block_index] != 0:
                        raise ValueError(
                            "Causal upsampler HR-DMD propagation requires "
                            "the first HR exit for every block"
                        )
                    upscaled_block = self.causal_upsampler(
                        upsampler_source_video,
                        block_start=block.video_start,
                        block_end=block.video_end,
                    )
                    if tuple(upscaled_block.shape) != tuple(
                        replay_video.shape
                    ):
                        raise RuntimeError(
                            "Causal upsampler replay shape mismatch: "
                            f"{tuple(upscaled_block.shape)} vs "
                            f"{tuple(replay_video.shape)}"
                        )
                    clean_coefficient = (
                        1.0 - sigma.to(device=replay_video.device)
                    ).to(dtype=replay_video.dtype)
                    replay_video = replay_video + clean_coefficient * (
                        upscaled_block - upscaled_block.detach()
                    )
                pred_video, pred_audio = self._kv_block_forward(
                    noisy_video_block=replay_video,
                    noisy_audio_block=replay_audio,
                    timestep_block=self._sigma_matrix(
                        sigma, replay_video
                    ),
                    audio_timestep_block=self._sigma_matrix(
                        sigma, replay_audio
                    ),
                    kv_caches=caches,
                    block=block,
                    conditional_dict=conditional_dict,
                )

                video_slice = slice(block.video_start, block.video_end)
                target_video = (
                    pred_video.detach() - grad_video[:, video_slice]
                ).detach()
                video_per_frame = (
                    (pred_video.double() - target_video.double()) ** 2
                ).mean(dim=(2, 3, 4))
                video_loss = (
                    0.5
                    * (
                        video_per_frame
                        * video_block_weights[video_slice].unsqueeze(0)
                    ).sum()
                    / float(B * rvf)
                )

                audio_slice = slice(block.audio_start, block.audio_end)
                target_audio = (
                    pred_audio.detach() - grad_audio[:, audio_slice]
                ).detach()
                audio_per_frame = (
                    (pred_audio.double() - target_audio.double()) ** 2
                ).mean(dim=2)
                audio_loss = (
                    0.5
                    * (
                        audio_per_frame
                        * audio_block_weights[audio_slice].unsqueeze(0)
                    ).sum()
                    / float(B * raf)
                )
                block_loss = scale_weight * (
                    video_loss_weight * video_loss
                    + audio_loss_weight * audio_loss
                )
                block_loss.backward()
                total += float(block_loss.detach().item())
                total_video += float(video_loss.detach().item())
                total_audio += float(audio_loss.detach().item())

                self._kv_refresh_cache_for_block(
                    x0_video_block=rollout.video[:, video_slice],
                    x0_audio_block=rollout.audio[:, audio_slice],
                    kv_caches=caches,
                    block=block,
                    conditional_dict=conditional_dict,
                )
                if upsampler_source_video is not None:
                    del upscaled_block

        del caches
        return total, total_video, total_audio

    @torch.no_grad()
    def _causal_upscale(
        self,
        video: torch.Tensor,
        blocks: list,
    ) -> torch.Tensor:
        if self.causal_upsampler is None:
            raise RuntimeError("Stage4A causal upsampler is not initialized")
        B, F_v, C, H, W = video.shape
        output = torch.empty(
            (B, F_v, C, H * 2, W * 2),
            device=video.device,
            dtype=video.dtype,
        )
        for block in blocks:
            output[:, block.video_start : block.video_end] = (
                self.causal_upsampler(
                    video,
                    block_start=block.video_start,
                    block_end=block.video_end,
                )
            )
        return output

    @staticmethod
    def _prefix_logs(
        prefix: str, logs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {f"{prefix}/{key}": value for key, value in logs.items()}

    def generator_loss(
        self,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
        real_video_frames: Optional[int] = None,
        real_audio_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Backpropagate LR DMD, release LR cache, then backpropagate HR DMD."""

        del clean_video, clean_audio
        logs: Dict[str, Any] = {}

        lr_rollout = self._rollout_scale_no_grad(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            sigmas=self.stage4_lr_sigmas,
        )
        lr_rollout_steps = int(lr_rollout.exit_indices[0]) + 1
        self._mark_memory("generator/lr_rollout")
        lr_grad_video, lr_grad_audio, lr_dmd_logs = (
            self._compute_scale_dmd_gradient(
                rollout=lr_rollout,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )
        )
        self._mark_memory("generator/lr_dmd_target")
        lr_total, lr_video_loss, lr_audio_loss = (
            self._backward_scale_generator(
                rollout=lr_rollout,
                sigmas=self.stage4_lr_sigmas,
                grad_video=lr_grad_video,
                grad_audio=lr_grad_audio,
                conditional_dict=conditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
                scale_weight=self.stage4_lr_dmd_weight,
            )
        )
        self._mark_memory("generator/lr_backward")

        hr_source_video = self._causal_upscale(
            lr_rollout.video, lr_rollout.blocks
        )
        hr_source_audio = lr_rollout.audio
        hr_video_shape = list(hr_source_video.shape)
        hr_audio_shape = list(hr_source_audio.shape)
        del lr_grad_video, lr_grad_audio
        del lr_rollout.replay_video, lr_rollout.replay_audio
        self._mark_memory("generator/scale_transition")

        hr_rollout = self._rollout_scale_no_grad(
            video_shape=hr_video_shape,
            audio_shape=hr_audio_shape,
            conditional_dict=conditional_dict,
            sigmas=self.stage4_hr_sigmas,
            source_video=hr_source_video,
            source_audio=hr_source_audio,
        )
        hr_rollout_steps = int(hr_rollout.exit_indices[0]) + 1
        train_upsampler_from_hr = bool(
            self.stage4_train_causal_upsampler
            and hr_rollout.exit_indices[0] == 0
        )
        upsampler_source_video = (
            lr_rollout.video
            if train_upsampler_from_hr
            else None
        )
        del hr_source_video, hr_source_audio, lr_rollout
        self._mark_memory("generator/hr_rollout")
        hr_grad_video, hr_grad_audio, hr_dmd_logs = (
            self._compute_scale_dmd_gradient(
                rollout=hr_rollout,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
                capture_video_predictions=(
                    self._capture_hr_dmd_visualization
                ),
            )
        )
        self._mark_memory("generator/hr_dmd_target")
        hr_total, hr_video_loss, hr_audio_loss = (
            self._backward_scale_generator(
                rollout=hr_rollout,
                sigmas=self.stage4_hr_sigmas,
                grad_video=hr_grad_video,
                grad_audio=hr_grad_audio,
                conditional_dict=conditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
                scale_weight=self.stage4_hr_dmd_weight,
                upsampler_source_video=upsampler_source_video,
            )
        )
        self._mark_memory("generator/hr_backward")

        logs.update(self._prefix_logs("lr", lr_dmd_logs))
        logs.update(self._prefix_logs("hr", hr_dmd_logs))
        logs.update(
            {
                "lr/video_dmd_loss": lr_video_loss,
                "lr/audio_dmd_loss": lr_audio_loss,
                "hr/video_dmd_loss": hr_video_loss,
                "hr/audio_dmd_loss": hr_audio_loss,
                "lr/dmd_weight": self.stage4_lr_dmd_weight,
                "hr/dmd_weight": self.stage4_hr_dmd_weight,
                "stage4/lr_steps": 2,
                "stage4/hr_steps": 2,
                "stage4/hr_height": int(hr_rollout.video.shape[-2]),
                "stage4/hr_width": int(hr_rollout.video.shape[-1]),
                "stage4/random_exit_rollout": float(
                    self.stage4_random_exit_rollout
                ),
                "stage4/train_causal_upsampler": float(
                    self.stage4_train_causal_upsampler
                ),
                "stage4/upsampler_hr_dmd_active": float(
                    train_upsampler_from_hr
                ),
                "stage4/generator_lr_rollout_steps": float(
                    lr_rollout_steps
                ),
                "stage4/generator_hr_rollout_steps": float(
                    hr_rollout_steps
                ),
            }
        )
        total = lr_total + hr_total
        del hr_grad_video, hr_grad_audio, hr_rollout
        if upsampler_source_video is not None:
            del upsampler_source_video
        return torch.tensor(total, device=self.device), logs

    def _critic_loss_on_sample(
        self,
        *,
        generated_video: torch.Tensor,
        generated_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        real_video_frames: Optional[int],
        real_audio_frames: Optional[int],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train the shared bidirectional fake score on one scale."""

        F_v_full = int(generated_video.shape[1])
        F_a_full = int(generated_audio.shape[1])
        rvf = (
            F_v_full
            if real_video_frames is None
            else max(1, min(int(real_video_frames), F_v_full))
        )
        raf = (
            F_a_full
            if real_audio_frames is None
            else max(1, min(int(real_audio_frames), F_a_full))
        )
        generated_video = generated_video[:, :rvf]
        generated_audio = generated_audio[:, :raf]
        B = int(generated_video.shape[0])

        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            (B, rvf),
            device=self.device,
            dtype=torch.long,
        )
        critic_timestep = self._process_timestep(
            critic_timestep, self.fake_task_type
        ).clamp(self.min_step, self.max_step)
        critic_sigma = self.timestep_to_sigma(critic_timestep)
        if self._is_bidirectional_task(self.fake_task_type):
            audio_timestep = critic_timestep[:, 0]
            audio_sigma = critic_sigma[:, :1].expand(B, raf)
        else:
            audio_timestep = self._compute_audio_timestep(
                critic_timestep, raf, task_type=self.fake_task_type
            ).clamp(self.min_step, self.max_step)
            audio_sigma = self.timestep_to_sigma(audio_timestep)

        noise_video = torch.randn_like(generated_video)
        noise_audio = torch.randn_like(generated_audio)
        noisy_video = self.add_noise(
            generated_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            critic_sigma.flatten(0, 1),
        ).unflatten(0, (B, rvf))
        noisy_audio = self.add_noise(
            generated_audio, noise_audio, audio_sigma
        )
        pred_video, pred_audio = self.fake_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=critic_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        video_sigma = (
            critic_sigma.flatten(0, 1)
            .double()
            .reshape(-1, 1, 1, 1)
            .clamp_min(1e-8)
        )
        audio_sigma_broadcast = (
            audio_sigma.double().unsqueeze(-1).clamp_min(1e-8)
        )
        flow_video = (
            (
                noisy_video.flatten(0, 1).double()
                - pred_video.flatten(0, 1).double()
            )
            / video_sigma
        ).to(self.dtype)
        flow_audio = (
            (noisy_audio.double() - pred_audio.double())
            / audio_sigma_broadcast
        ).to(self.dtype)
        video_loss = self._compute_masked_denoising_loss(
            target=generated_video,
            prediction=pred_video,
            noise=noise_video,
            flow_pred=flow_video.unflatten(0, (B, rvf)),
            timestep=critic_timestep,
            mask=None,
        )
        audio_loss = self._compute_masked_denoising_loss(
            target=generated_audio,
            prediction=pred_audio,
            noise=noise_audio,
            flow_pred=flow_audio,
            timestep=audio_timestep,
            mask=None,
        )
        video_weight, audio_weight = self.get_loss_weights()
        total = video_weight * video_loss + audio_weight * audio_loss
        return total, {
            "critic_video_loss": video_loss.detach(),
            "critic_audio_loss": audio_loss.detach(),
        }

    def critic_loss(
        self,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
        real_video_frames: Optional[int] = None,
        real_audio_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Backpropagate fake-score FM loss at LR and HR sequentially."""

        del unconditional_dict, clean_video, clean_audio
        lr_rollout = self._rollout_scale_no_grad(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            sigmas=self.stage4_lr_sigmas,
        )
        self._mark_memory("critic/lr_rollout")
        lr_loss, lr_logs = self._critic_loss_on_sample(
            generated_video=lr_rollout.video,
            generated_audio=lr_rollout.audio,
            conditional_dict=conditional_dict,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
        )
        weighted_lr_loss = self.stage4_lr_dmd_weight * lr_loss
        weighted_lr_loss.backward()
        lr_rollout_steps = int(lr_rollout.exit_indices[0]) + 1
        self._mark_memory("critic/lr_backward")

        hr_source_video = self._causal_upscale(
            lr_rollout.video, lr_rollout.blocks
        )
        hr_source_audio = lr_rollout.audio
        del lr_rollout
        hr_rollout = self._rollout_scale_no_grad(
            video_shape=list(hr_source_video.shape),
            audio_shape=list(hr_source_audio.shape),
            conditional_dict=conditional_dict,
            sigmas=self.stage4_hr_sigmas,
            source_video=hr_source_video,
            source_audio=hr_source_audio,
        )
        del hr_source_video, hr_source_audio
        self._mark_memory("critic/hr_rollout")
        hr_loss, hr_logs = self._critic_loss_on_sample(
            generated_video=hr_rollout.video,
            generated_audio=hr_rollout.audio,
            conditional_dict=conditional_dict,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
        )
        weighted_hr_loss = self.stage4_hr_dmd_weight * hr_loss
        weighted_hr_loss.backward()
        hr_rollout_steps = int(hr_rollout.exit_indices[0]) + 1
        self._mark_memory("critic/hr_backward")

        logs = {}
        logs.update(self._prefix_logs("lr", lr_logs))
        logs.update(self._prefix_logs("hr", hr_logs))
        logs.update(
            {
                "stage4/random_exit_rollout": float(
                    self.stage4_random_exit_rollout
                ),
                "stage4/critic_lr_rollout_steps": float(
                    lr_rollout_steps
                ),
                "stage4/critic_hr_rollout_steps": float(
                    hr_rollout_steps
                ),
            }
        )
        total = (
            float(weighted_lr_loss.detach().item())
            + float(weighted_hr_loss.detach().item())
        )
        del lr_loss, hr_loss, weighted_lr_loss, weighted_hr_loss, hr_rollout
        return torch.tensor(total, device=self.device), logs

    @torch.no_grad()
    def sample_scale_wise(
        self,
        *,
        video_shape: Sequence[int],
        audio_shape: Sequence[int],
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the exact four-call Stage4A inference graph for benchmark."""

        lr_rollout = self._rollout_scale_no_grad(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            sigmas=self.stage4_lr_sigmas,
            force_full_steps=True,
        )
        hr_source_video = self._causal_upscale(
            lr_rollout.video, lr_rollout.blocks
        )
        hr_rollout = self._rollout_scale_no_grad(
            video_shape=list(hr_source_video.shape),
            audio_shape=list(lr_rollout.audio.shape),
            conditional_dict=conditional_dict,
            sigmas=self.stage4_hr_sigmas,
            source_video=hr_source_video,
            source_audio=lr_rollout.audio,
            force_full_steps=True,
        )
        return hr_rollout.video, hr_rollout.audio, lr_rollout.video
