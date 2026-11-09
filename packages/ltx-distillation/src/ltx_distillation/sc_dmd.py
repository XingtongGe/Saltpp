"""Mixed-step Shortcut-Consistency DMD for Stage-3 self-forcing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.sc_dmd_utils import (
    euler_endpoint,
    flow_velocity_from_x0,
    shortcut_pair_candidates,
    validate_mixed_step_config,
)


class LTX2ShortcutConsistencyDMD(LTX2DMD):
    """Stage-3 DMD with mixed 2/4/8-step rollout and latent endpoint SC loss."""

    def __init__(self, args, device: torch.device):
        super().__init__(args, device)
        if not self.enable_self_forcing:
            raise ValueError("sc_dmd requires causal self-forcing training")
        if not bool(getattr(args, "strict_denoising_sigmas", False)):
            raise ValueError("sc_dmd currently requires strict_denoising_sigmas=true")

        raw_step_lists = getattr(args, "mixed_denoising_step_lists", None)
        raw_probabilities = getattr(args, "mixed_step_probs", None)
        if raw_step_lists is None or raw_probabilities is None:
            raise ValueError("sc_dmd requires mixed_denoising_step_lists and mixed_step_probs")

        step_lists, probabilities = validate_mixed_step_config(
            list(raw_step_lists),
            list(raw_probabilities),
            num_train_timestep=self.num_train_timestep,
        )
        self.mixed_denoising_step_lists = step_lists
        self.mixed_step_probabilities = probabilities
        self.mixed_denoising_sigmas = [
            torch.tensor(
                [value / float(self.num_train_timestep) for value in schedule],
                dtype=torch.float32,
                device=device,
            )
            for schedule in step_lists
        ]
        self.mixed_step_counts = [len(schedule) - 1 for schedule in step_lists]

        self.shortcut_consistency_weight = float(
            getattr(args, "shortcut_consistency_weight", 0.2)
        )
        self.shortcut_apply_steps = {
            int(value) for value in getattr(args, "shortcut_consistency_apply_steps", [8])
        }
        self.shortcut_num_segments = int(getattr(args, "shortcut_num_segments", 4))
        self.shortcut_time_weight = str(
            getattr(args, "shortcut_time_weight", "one_minus_sigma")
        ).lower()
        if self.shortcut_time_weight not in {"uniform", "one_minus_sigma"}:
            raise ValueError("shortcut_time_weight must be uniform or one_minus_sigma")
        unknown_steps = self.shortcut_apply_steps.difference(self.mixed_step_counts)
        if unknown_steps:
            raise ValueError(
                "shortcut_consistency_apply_steps contains unavailable schedules: "
                f"{sorted(unknown_steps)}"
            )

        self._active_mixed_step_count = len(self.denoising_sigmas) - 1
        self._active_mixed_schedule_index = -1

    def _sample_schedule_index(self) -> int:
        if not dist.is_initialized() or dist.get_rank() == 0:
            index = torch.multinomial(
                torch.tensor(
                    self.mixed_step_probabilities,
                    device=self.device,
                    dtype=torch.float32,
                ),
                num_samples=1,
            ).to(dtype=torch.long)
        else:
            index = torch.empty(1, device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(index, src=0)
        return int(index.item())

    @contextmanager
    def _sampled_mixed_schedule(self):
        schedule_index = self._sample_schedule_index()
        old_sigmas = self.denoising_sigmas
        old_step_count = self._active_mixed_step_count
        old_schedule_index = self._active_mixed_schedule_index
        self.denoising_sigmas = self.mixed_denoising_sigmas[schedule_index]
        self._active_mixed_step_count = self.mixed_step_counts[schedule_index]
        self._active_mixed_schedule_index = schedule_index
        try:
            yield self._active_mixed_step_count
        finally:
            self.denoising_sigmas = old_sigmas
            self._active_mixed_step_count = old_step_count
            self._active_mixed_schedule_index = old_schedule_index

    def generator_loss(self, *args, **kwargs):
        with self._sampled_mixed_schedule() as step_count:
            loss, log_dict = super().generator_loss(*args, **kwargs)
            log_dict["mixed_steps/generator"] = step_count
            return loss, log_dict

    def critic_loss(self, *args, **kwargs):
        with self._sampled_mixed_schedule() as step_count:
            loss, log_dict = super().critic_loss(*args, **kwargs)
            log_dict["mixed_steps/critic"] = step_count
            return loss, log_dict

    def _begin_shortcut_consistency(self, num_denoising_steps: int):
        if (
            self.shortcut_consistency_weight <= 0
            or self._active_mixed_step_count not in self.shortcut_apply_steps
        ):
            return None
        if num_denoising_steps != self._active_mixed_step_count:
            raise RuntimeError("active mixed schedule changed during the two-pass rollout")
        return {
            "pairs": 0,
            "shared_pair": None,
            "video_loss": 0.0,
            "audio_loss": 0.0,
            "weighted_loss": 0.0,
            "sigma_start": 0.0,
            "sigma_mid": 0.0,
            "sigma_end": 0.0,
        }

    def _sample_shortcut_pair(
        self,
        *,
        block_index: int,
        exit_index: int,
    ) -> Optional[Tuple[int, int]]:
        sigma_values = self.denoising_sigmas.detach().cpu().tolist()
        candidates = shortcut_pair_candidates(
            sigma_values,
            exit_index=exit_index,
            num_segments=self.shortcut_num_segments,
        )
        if not candidates:
            return None

        if not dist.is_initialized() or dist.get_rank() == 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(getattr(self.args, "seed", 0))
                + int(self.current_step) * 1_000_003
                + int(block_index) * 10_007
                + int(exit_index) * 101
                + int(self._active_mixed_step_count)
            )
            choice = torch.randint(
                0,
                len(candidates),
                (1,),
                generator=generator,
                dtype=torch.long,
            ).to(self.device)
        else:
            choice = torch.empty(1, device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(choice, src=0)
        return candidates[int(choice.item())]

    def _time_weight(self, sigma_start: float) -> float:
        if self.shortcut_time_weight == "one_minus_sigma":
            return max(0.0, 1.0 - float(sigma_start))
        return 1.0

    def _build_shortcut_consistency_target(
        self,
        *,
        context: Dict[str, Any],
        block_index: int,
        exit_index: int,
        block,
        noisy_video: torch.Tensor,
        noisy_audio: Optional[torch.Tensor],
        pred_video_x0: torch.Tensor,
        pred_audio_x0: Optional[torch.Tensor],
        sigma_start: torch.Tensor,
        kv_caches,
        conditional_dict: Dict[str, Any],
    ):
        pair = context["shared_pair"] if self.same_step_across_blocks else None
        if pair is None:
            pair = self._sample_shortcut_pair(
                block_index=block_index,
                exit_index=exit_index,
            )
            if self.same_step_across_blocks:
                context["shared_pair"] = pair
        if pair is None:
            return None
        mid_index, end_index = pair
        sigma_start_value = float(sigma_start.item())
        sigma_mid = float(self.denoising_sigmas[mid_index].item())
        sigma_end = float(self.denoising_sigmas[end_index].item())
        if self._time_weight(sigma_start_value) <= 0:
            return None

        video_v1 = flow_velocity_from_x0(noisy_video, pred_video_x0, sigma_start_value)
        video_mid = euler_endpoint(
            noisy_video, video_v1, sigma_start_value, sigma_mid
        ).to(dtype=noisy_video.dtype)

        audio_mid = None
        if noisy_audio is not None and pred_audio_x0 is not None:
            audio_v1 = flow_velocity_from_x0(noisy_audio, pred_audio_x0, sigma_start_value)
            audio_mid = euler_endpoint(
                noisy_audio, audio_v1, sigma_start_value, sigma_mid
            ).to(dtype=noisy_audio.dtype)

        video_mid_sigma = torch.full(
            (video_mid.shape[0], video_mid.shape[1]),
            sigma_mid,
            device=self.device,
            dtype=self.denoising_sigmas.dtype,
        )
        audio_mid_sigma = None
        if audio_mid is not None:
            audio_mid_sigma = torch.full(
                (audio_mid.shape[0], audio_mid.shape[1]),
                sigma_mid,
                device=self.device,
                dtype=self.denoising_sigmas.dtype,
            )

        with torch.no_grad():
            video_mid_x0, audio_mid_x0 = self._kv_block_forward(
                noisy_video_block=video_mid,
                noisy_audio_block=audio_mid,
                timestep_block=video_mid_sigma,
                audio_timestep_block=audio_mid_sigma,
                kv_caches=kv_caches,
                block=block,
                conditional_dict=conditional_dict,
            )
            video_v2 = flow_velocity_from_x0(video_mid, video_mid_x0, sigma_mid)
            video_target = euler_endpoint(
                video_mid, video_v2, sigma_mid, sigma_end
            ).to(dtype=noisy_video.dtype)

            audio_target = None
            if audio_mid is not None and audio_mid_x0 is not None:
                audio_v2 = flow_velocity_from_x0(audio_mid, audio_mid_x0, sigma_mid)
                audio_target = euler_endpoint(
                    audio_mid, audio_v2, sigma_mid, sigma_end
                ).to(dtype=noisy_audio.dtype)

        return {
            "video": video_target.detach(),
            "audio": audio_target.detach() if audio_target is not None else None,
            "sigma_start": sigma_start_value,
            "sigma_mid": sigma_mid,
            "sigma_end": sigma_end,
        }

    def _compute_shortcut_consistency_block_loss(
        self,
        *,
        context: Dict[str, Any],
        target,
        block,
        noisy_video: torch.Tensor,
        noisy_audio: Optional[torch.Tensor],
        pred_video_x0: torch.Tensor,
        pred_audio_x0: Optional[torch.Tensor],
        video_block_weights: torch.Tensor,
        audio_block_weights: Optional[torch.Tensor],
        video_denom: int,
        audio_denom: int,
        video_loss_weight: float,
        audio_loss_weight: float,
    ):
        if target is None:
            return None

        sigma_start = target["sigma_start"]
        sigma_end = target["sigma_end"]
        time_weight = self._time_weight(sigma_start)

        video_v1 = flow_velocity_from_x0(noisy_video, pred_video_x0, sigma_start)
        video_direct = euler_endpoint(
            noisy_video, video_v1, sigma_start, sigma_end
        )
        video_diff = video_direct - target["video"].float()
        video_per_frame = video_diff.square().mean(dim=[2, 3, 4])
        video_slice = slice(block.video_start, block.video_end)
        video_sc_loss = (
            video_per_frame * video_block_weights[video_slice].unsqueeze(0)
        ).sum() / float(video_denom)

        audio_sc_loss = None
        if (
            noisy_audio is not None
            and pred_audio_x0 is not None
            and target["audio"] is not None
            and audio_block_weights is not None
        ):
            audio_v1 = flow_velocity_from_x0(noisy_audio, pred_audio_x0, sigma_start)
            audio_direct = euler_endpoint(
                noisy_audio, audio_v1, sigma_start, sigma_end
            )
            audio_diff = audio_direct - target["audio"].float()
            audio_per_frame = audio_diff.square().mean(dim=2)
            audio_end = min(block.audio_end, audio_block_weights.shape[0])
            if audio_end > block.audio_start:
                audio_slice = slice(block.audio_start, audio_end)
                audio_sc_loss = (
                    audio_per_frame[:, : audio_end - block.audio_start]
                    * audio_block_weights[audio_slice].unsqueeze(0)
                ).sum() / float(audio_denom)

        combined = video_loss_weight * video_sc_loss
        if audio_sc_loss is not None:
            combined = combined + audio_loss_weight * audio_sc_loss
        weighted = self.shortcut_consistency_weight * time_weight * combined

        context["pairs"] += 1
        context["video_loss"] += float(video_sc_loss.detach().item())
        if audio_sc_loss is not None:
            context["audio_loss"] += float(audio_sc_loss.detach().item())
        context["weighted_loss"] += float(weighted.detach().item())
        context["sigma_start"] += sigma_start
        context["sigma_mid"] += target["sigma_mid"]
        context["sigma_end"] += sigma_end
        return weighted

    def _finalize_shortcut_consistency_logs(
        self,
        context: Optional[Dict[str, Any]],
        log_dict: Dict[str, Any],
    ) -> None:
        if context is None:
            log_dict["shortcut/active"] = 0
            return
        pairs = int(context["pairs"])
        log_dict["shortcut/active"] = 1
        log_dict["shortcut/pairs"] = pairs
        log_dict["shortcut/video_loss"] = context["video_loss"]
        log_dict["shortcut/audio_loss"] = context["audio_loss"]
        log_dict["shortcut/weighted_loss"] = context["weighted_loss"]
        log_dict["shortcut/weight"] = self.shortcut_consistency_weight
        if pairs > 0:
            log_dict["shortcut/sigma_start"] = context["sigma_start"] / pairs
            log_dict["shortcut/sigma_mid"] = context["sigma_mid"] / pairs
            log_dict["shortcut/sigma_end"] = context["sigma_end"] / pairs
