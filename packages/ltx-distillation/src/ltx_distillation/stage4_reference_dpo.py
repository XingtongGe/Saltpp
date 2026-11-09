"""Reference-based diffusion preference optimization for Stage 4B.

This module intentionally contains no DMD real-score/fake-score models.  It
implements the first Stage-4B validation setting:

* a frozen causal upsampler and a configurable pair generator (the frozen
  causal reference in the stable baseline) generate a detached 2-step-LR +
  2-step-HR negative sample;
* the released bidirectional LTX-2 refiner turns that negative into a detached
  positive sample;
* a trainable causal policy and a frozen causal reference policy score both
  candidates at one shared HR node (889 or 727);
* the exact DPO gradient is replayed block by block so only one causal block
  retains an autograd graph at a time.

The denoising energy is an x0-prediction MSE surrogate rather than an exact
diffusion log probability.  Candidate-clean cache refresh makes the score a
causal factorization of each fixed candidate instead of another rollout.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ltx_causal.attention.mask_builder import compute_av_blocks
from ltx_causal.wrapper import build_block_ref_from_prev
from ltx_distillation.stage35_refiner import (
    add_noise,
    full_sigma,
    init_kv_caches,
    kv_block_forward,
    official_refine_full,
    refresh_clean_cache,
)


@dataclass
class Stage4Rollout:
    """Detached result of the fixed deployment graph."""

    hr_video: torch.Tensor
    hr_audio: torch.Tensor
    lr_video: torch.Tensor
    lr_audio: torch.Tensor
    upscaled_video: torch.Tensor
    blocks: list


@dataclass
class CandidateEnergy:
    """Full-clip causal denoising energy."""

    total: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor


@dataclass
class PreferencePair:
    """Detached negative and official-refiner positive candidates."""

    positive_video: torch.Tensor
    positive_audio: torch.Tensor
    negative_video: torch.Tensor
    negative_audio: torch.Tensor
    lr_video: torch.Tensor
    relative_video_change: torch.Tensor
    relative_audio_change: torch.Tensor
    blocks: list
    refiner_relative_video_change: Optional[torch.Tensor] = None
    refiner_relative_audio_change: Optional[torch.Tensor] = None


def dpo_loss_and_replay_coefficient(
    *,
    online_positive_energy: torch.Tensor,
    online_negative_energy: torch.Tensor,
    reference_positive_energy: torch.Tensor,
    reference_negative_energy: torch.Tensor,
    beta: float,
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted DPO loss, replay coefficient, advantage and logit.

    With

    ``advantage = (E_online_neg - E_online_pos)
                 - (E_ref_neg - E_ref_pos)``

    the objective is ``loss_weight * softplus(-beta * advantage)``.  Its
    gradient can be reproduced exactly by replaying

    ``+q * E_online_pos`` and ``-q * E_online_neg``

    where ``q`` is detached from the energy graph.
    """

    beta_value = float(beta)
    weight_value = float(loss_weight)
    online_margin = online_negative_energy - online_positive_energy
    reference_margin = reference_negative_energy - reference_positive_energy
    advantage = online_margin - reference_margin
    logit = beta_value * advantage
    loss = weight_value * F.softplus(-logit)
    coefficient = (
        weight_value
        * beta_value
        * torch.sigmoid(-logit)
    ).detach()
    return loss, coefficient, advantage, logit


class Stage4ReferenceDPO:
    """Memory-bounded Stage-4B rollout, pair construction and DPO scoring."""

    EXPECTED_SCHEDULE = (1000, 960, 889, 727, 0)

    def __init__(
        self,
        *,
        policy,
        reference,
        official_refiner,
        causal_upsampler,
        device: torch.device,
        dtype: torch.dtype,
        num_frame_per_block: int = 3,
        first_block_video_frames: int = 4,
        enable_chunk_recycling: bool = False,
        dpo_beta: float = 1.0,
        dpo_loss_weight: float = 1.0,
        dpo_video_weight: float = 1.0,
        dpo_audio_weight: float = 0.0,
        negative_video_anchor_weight: float = 0.0,
        negative_audio_anchor_weight: float = 0.0,
        dpo_pair_source: str = "online_policy",
        positive_video_mix: float = 1.0,
        positive_audio_source: str = "refined",
        dpo_timestep_candidates: Sequence[int] = (889, 727),
        refiner_sigmas: Sequence[float] = (
            0.909375,
            0.725,
            0.421875,
            0.0,
        ),
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.official_refiner = official_refiner
        self.causal_upsampler = causal_upsampler
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_frame_per_block = int(num_frame_per_block)
        self.first_block_video_frames = int(first_block_video_frames)
        self.enable_chunk_recycling = bool(enable_chunk_recycling)
        self.dpo_beta = float(dpo_beta)
        self.dpo_loss_weight = float(dpo_loss_weight)
        self.dpo_video_weight = float(dpo_video_weight)
        self.dpo_audio_weight = float(dpo_audio_weight)
        self.negative_video_anchor_weight = float(
            negative_video_anchor_weight
        )
        self.negative_audio_anchor_weight = float(
            negative_audio_anchor_weight
        )
        self.dpo_pair_source = str(dpo_pair_source).strip().lower()
        self.positive_video_mix = float(positive_video_mix)
        self.positive_audio_source = str(positive_audio_source).strip().lower()

        if self.dpo_beta <= 0:
            raise ValueError("dpo_beta must be > 0")
        if self.dpo_loss_weight < 0:
            raise ValueError("dpo_loss_weight must be >= 0")
        if self.dpo_video_weight < 0 or self.dpo_audio_weight < 0:
            raise ValueError("DPO modality weights must be non-negative")
        if self.dpo_video_weight + self.dpo_audio_weight <= 0:
            raise ValueError("At least one DPO modality weight must be > 0")
        if (
            self.negative_video_anchor_weight < 0
            or self.negative_audio_anchor_weight < 0
        ):
            raise ValueError("Negative anchor weights must be non-negative")
        if self.dpo_pair_source not in {
            "online_policy",
            "frozen_reference",
        }:
            raise ValueError(
                "dpo_pair_source must be online_policy or frozen_reference"
            )
        if not 0.0 < self.positive_video_mix <= 1.0:
            raise ValueError("positive_video_mix must be in (0, 1]")
        if self.positive_audio_source not in {"negative", "refined"}:
            raise ValueError(
                "positive_audio_source must be negative or refined"
            )

        candidate_steps = tuple(int(value) for value in dpo_timestep_candidates)
        if not candidate_steps:
            raise ValueError("dpo_timestep_candidates must not be empty")
        if any(value not in (889, 727) for value in candidate_steps):
            raise ValueError(
                "Stage4B DPO candidates must be drawn from HR nodes 889/727"
            )
        self.dpo_timestep_candidates = candidate_steps

        self.lr_sigmas = torch.tensor(
            [1.0, 0.960, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        self.hr_sigmas = torch.tensor(
            [0.889, 0.727, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        self.refiner_sigmas = torch.tensor(
            list(refiner_sigmas),
            device=self.device,
            dtype=torch.float32,
        )
        if (
            self.refiner_sigmas.numel() != 4
            or float(self.refiner_sigmas[-1].item()) != 0.0
            or not bool(
                torch.all(
                    self.refiner_sigmas[:-1] > self.refiner_sigmas[1:]
                )
            )
        ):
            raise ValueError(
                "refiner_sigmas must contain three descending steps and terminal 0"
            )

    def _blocks(self, total_video_frames: int) -> list:
        blocks = compute_av_blocks(
            total_video_latent_frames=int(total_video_frames),
            num_frame_per_block=self.num_frame_per_block,
            first_block_video_frames=self.first_block_video_frames,
        )
        if not blocks:
            raise ValueError("Stage4B sampled zero causal blocks")
        return blocks

    @staticmethod
    def _validate_shapes(
        *,
        video_shape: Sequence[int],
        audio_shape: Sequence[int],
        blocks: list,
    ) -> None:
        if int(video_shape[0]) != 1 or int(audio_shape[0]) != 1:
            raise ValueError("Stage4B currently requires batch_size=1")
        if int(video_shape[1]) != int(blocks[-1].video_end):
            raise ValueError("video shape does not match causal block layout")
        if int(audio_shape[1]) != int(blocks[-1].audio_end):
            raise ValueError(
                "audio shape must use causal AV alignment: "
                f"shape has {audio_shape[1]}, blocks require "
                f"{blocks[-1].audio_end}"
            )

    @staticmethod
    def _valid_frames(start: int, end: int, real_frames: int) -> int:
        return max(0, min(int(end), int(real_frames)) - int(start))

    def _ref_video_block(
        self,
        *,
        previous_video: Optional[torch.Tensor],
        current_block_size: int,
        template: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.enable_chunk_recycling:
            return None

        return build_block_ref_from_prev(
            prev_chunk_latent=previous_video,
            current_block_size=int(current_block_size),
            template=template,
        )

    @torch.no_grad()
    def _rollout_scale(
        self,
        *,
        model,
        video_shape: Sequence[int],
        audio_shape: Sequence[int],
        conditional_dict: Dict[str, Any],
        sigmas: torch.Tensor,
        source_video: Optional[torch.Tensor] = None,
        source_audio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Run exactly two self-forcing calls per causal block."""

        if len(sigmas) != 3:
            raise ValueError("Each Stage4B scale must contain two calls")
        if (source_video is None) != (source_audio is None):
            raise ValueError(
                "source_video and source_audio must both be set or both be None"
            )

        shape_video = tuple(int(value) for value in video_shape)
        shape_audio = tuple(int(value) for value in audio_shape)
        blocks = self._blocks(shape_video[1])
        self._validate_shapes(
            video_shape=shape_video,
            audio_shape=shape_audio,
            blocks=blocks,
        )
        if source_video is not None:
            if tuple(source_video.shape) != shape_video:
                raise ValueError("source_video shape mismatch")
            if tuple(source_audio.shape) != shape_audio:
                raise ValueError("source_audio shape mismatch")

        batch_size = shape_video[0]
        output_video = torch.empty(
            shape_video, device=self.device, dtype=self.dtype
        )
        output_audio = torch.empty(
            shape_audio, device=self.device, dtype=self.dtype
        )
        caches = init_kv_caches(
            model,
            batch_size=batch_size,
            total_video_frames=shape_video[1],
            total_audio_frames=shape_audio[1],
            text_seq_len=int(conditional_dict["video_context"].shape[1]),
            video_frame_seqlen=shape_video[-2] * shape_video[-1],
            device=self.device,
            dtype=self.dtype,
        )

        previous_video = None
        for block in blocks:
            video_slice = slice(block.video_start, block.video_end)
            audio_slice = slice(block.audio_start, block.audio_end)
            if source_video is None:
                current_video = torch.randn(
                    (
                        batch_size,
                        block.video_frames,
                        *shape_video[2:],
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
                current_audio = torch.randn(
                    (
                        batch_size,
                        block.audio_frames,
                        shape_audio[2],
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                current_video = add_noise(
                    source_video[:, video_slice],
                    torch.randn_like(source_video[:, video_slice]),
                    sigmas[0],
                )
                current_audio = add_noise(
                    source_audio[:, audio_slice],
                    torch.randn_like(source_audio[:, audio_slice]),
                    sigmas[0],
                )

            ref_video = self._ref_video_block(
                previous_video=previous_video,
                current_block_size=block.video_frames,
                template=output_video,
            )
            for step_index, sigma in enumerate(sigmas[:-1]):
                video_sigma = full_sigma(
                    sigma,
                    batch_size=batch_size,
                    frames=current_video.shape[1],
                    device=self.device,
                    dtype=sigmas.dtype,
                )
                audio_sigma = full_sigma(
                    sigma,
                    batch_size=batch_size,
                    frames=current_audio.shape[1],
                    device=self.device,
                    dtype=sigmas.dtype,
                )
                prediction_video, prediction_audio = kv_block_forward(
                    model,
                    video=current_video,
                    audio=current_audio,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                    conditional_dict=conditional_dict,
                    kv_caches=caches,
                    block=block,
                    ref_video_block=ref_video,
                )
                if step_index + 1 < len(sigmas) - 1:
                    next_sigma = sigmas[step_index + 1]
                    current_video = add_noise(
                        prediction_video,
                        torch.randn_like(prediction_video),
                        next_sigma,
                    )
                    current_audio = add_noise(
                        prediction_audio,
                        torch.randn_like(prediction_audio),
                        next_sigma,
                    )

            output_video[:, video_slice] = prediction_video
            output_audio[:, audio_slice] = prediction_audio
            refresh_clean_cache(
                model,
                video_x0=prediction_video,
                audio_x0=prediction_audio,
                conditional_dict=conditional_dict,
                kv_caches=caches,
                block=block,
                sigma_dtype=sigmas.dtype,
                ref_video_block=ref_video,
            )
            previous_video = prediction_video.detach()

        del caches
        return output_video.detach(), output_audio.detach(), blocks

    @torch.no_grad()
    def causal_upscale(
        self,
        low_video: torch.Tensor,
        blocks: list,
    ) -> torch.Tensor:
        output = torch.empty(
            (
                low_video.shape[0],
                low_video.shape[1],
                low_video.shape[2],
                low_video.shape[3] * 2,
                low_video.shape[4] * 2,
            ),
            device=low_video.device,
            dtype=low_video.dtype,
        )
        for block in blocks:
            output[:, block.video_start : block.video_end] = (
                self.causal_upsampler(
                    low_video,
                    block_start=int(block.video_start),
                    block_end=int(block.video_end),
                )
            )
        return output.detach()

    @torch.no_grad()
    def sample_scale_wise(
        self,
        *,
        video_shape: Sequence[int],
        audio_shape: Sequence[int],
        conditional_dict: Dict[str, Any],
        model=None,
    ) -> Stage4Rollout:
        """Generate one detached full 2LR+2HR deployment sample.

        ``model=None`` intentionally means the online policy so inference and
        benchmark callers retain deployment semantics.  Preference-pair
        construction passes its generator explicitly.
        """

        rollout_model = self.policy if model is None else model

        lr_video, lr_audio, blocks = self._rollout_scale(
            model=rollout_model,
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            sigmas=self.lr_sigmas,
        )
        upscaled_video = self.causal_upscale(lr_video, blocks)
        hr_video, hr_audio, hr_blocks = self._rollout_scale(
            model=rollout_model,
            video_shape=upscaled_video.shape,
            audio_shape=lr_audio.shape,
            conditional_dict=conditional_dict,
            sigmas=self.hr_sigmas,
            source_video=upscaled_video,
            source_audio=lr_audio,
        )
        return Stage4Rollout(
            hr_video=hr_video,
            hr_audio=hr_audio,
            lr_video=lr_video,
            lr_audio=lr_audio,
            upscaled_video=upscaled_video,
            blocks=hr_blocks,
        )

    @torch.no_grad()
    def build_preference_pair(
        self,
        *,
        video_shape: Sequence[int],
        audio_shape: Sequence[int],
        conditional_dict: Dict[str, Any],
        real_video_frames: int,
        real_audio_frames: int,
    ) -> PreferencePair:
        """Create detached ``positive=refine(negative)`` HR candidates."""

        pair_generator = (
            self.reference
            if self.dpo_pair_source == "frozen_reference"
            else self.policy
        )
        rollout = self.sample_scale_wise(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            model=pair_generator,
        )
        rvf = max(
            1, min(int(real_video_frames), int(rollout.hr_video.shape[1]))
        )
        raf = max(
            1, min(int(real_audio_frames), int(rollout.hr_audio.shape[1]))
        )
        negative_video_real = rollout.hr_video[:, :rvf]
        negative_audio_real = rollout.hr_audio[:, :raf]

        refiner_initial_video = add_noise(
            negative_video_real,
            torch.randn_like(negative_video_real),
            self.refiner_sigmas[0],
        )
        refiner_initial_audio = add_noise(
            negative_audio_real,
            torch.randn_like(negative_audio_real),
            self.refiner_sigmas[0],
        )
        refined_video_real, refined_audio_real = official_refine_full(
            self.official_refiner,
            initial_video=refiner_initial_video,
            initial_audio=refiner_initial_audio,
            conditional_dict=conditional_dict,
            sigmas=self.refiner_sigmas,
        )

        positive_video_real = negative_video_real + self.positive_video_mix * (
            refined_video_real - negative_video_real
        )
        if self.positive_audio_source == "negative":
            positive_audio_real = negative_audio_real
        else:
            positive_audio_real = refined_audio_real

        # Preserve padded frames from the causal rollout.  They are excluded
        # from the energy denominator but may still be visited by a rank whose
        # local real sequence is shorter than the synchronized rollout shape.
        positive_video = rollout.hr_video.clone()
        positive_audio = rollout.hr_audio.clone()
        positive_video[:, :rvf] = positive_video_real
        positive_audio[:, :raf] = positive_audio_real
        relative_video_change = (
            (positive_video_real.float() - negative_video_real.float())
            .square()
            .mean()
            / negative_video_real.float().square().mean().clamp_min(1e-12)
        )
        relative_audio_change = (
            (positive_audio_real.float() - negative_audio_real.float())
            .square()
            .mean()
            / negative_audio_real.float().square().mean().clamp_min(1e-12)
        )
        refiner_relative_video_change = (
            (refined_video_real.float() - negative_video_real.float())
            .square()
            .mean()
            / negative_video_real.float().square().mean().clamp_min(1e-12)
        )
        refiner_relative_audio_change = (
            (refined_audio_real.float() - negative_audio_real.float())
            .square()
            .mean()
            / negative_audio_real.float().square().mean().clamp_min(1e-12)
        )
        return PreferencePair(
            positive_video=positive_video.detach(),
            positive_audio=positive_audio.detach(),
            negative_video=rollout.hr_video.detach(),
            negative_audio=rollout.hr_audio.detach(),
            lr_video=rollout.lr_video.detach(),
            relative_video_change=relative_video_change.detach(),
            relative_audio_change=relative_audio_change.detach(),
            blocks=rollout.blocks,
            refiner_relative_video_change=(
                refiner_relative_video_change.detach()
            ),
            refiner_relative_audio_change=(
                refiner_relative_audio_change.detach()
            ),
        )

    def sample_synced_dpo_timestep(self) -> int:
        """Sample one HR DPO node on rank 0 and broadcast it."""

        if not dist.is_initialized() or dist.get_rank() == 0:
            index = torch.randint(
                0,
                len(self.dpo_timestep_candidates),
                (1,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            index = torch.zeros(1, device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(index, src=0)
        return self.dpo_timestep_candidates[int(index.item())]

    @staticmethod
    def _block_energies(
        *,
        prediction_video: torch.Tensor,
        prediction_audio: torch.Tensor,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        valid_video_frames: int,
        valid_audio_frames: int,
        video_denominator: float,
        audio_denominator: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_video_frames > 0:
            video_per_frame = (
                prediction_video[:, :valid_video_frames].float()
                - clean_video[:, :valid_video_frames].float()
            ).square().mean(dim=(2, 3, 4))
            video_energy = video_per_frame.sum() / video_denominator
        else:
            video_energy = prediction_video.float().sum() * 0.0

        if valid_audio_frames > 0:
            audio_per_frame = (
                prediction_audio[:, :valid_audio_frames].float()
                - clean_audio[:, :valid_audio_frames].float()
            ).square().mean(dim=2)
            audio_energy = audio_per_frame.sum() / audio_denominator
        else:
            audio_energy = prediction_audio.float().sum() * 0.0
        return video_energy, audio_energy

    @torch.no_grad()
    def score_candidate(
        self,
        *,
        model,
        candidate_video: torch.Tensor,
        candidate_audio: torch.Tensor,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: int,
        real_video_frames: int,
        real_audio_frames: int,
        blocks: list,
    ) -> CandidateEnergy:
        """Score one fixed candidate with candidate-clean causal caches."""

        sigma = torch.tensor(
            float(timestep) / 1000.0,
            device=self.device,
            dtype=torch.float32,
        )
        batch_size = int(candidate_video.shape[0])
        rvf = max(
            1, min(int(real_video_frames), int(candidate_video.shape[1]))
        )
        raf = max(
            1, min(int(real_audio_frames), int(candidate_audio.shape[1]))
        )
        caches = init_kv_caches(
            model,
            batch_size=batch_size,
            total_video_frames=int(candidate_video.shape[1]),
            total_audio_frames=int(candidate_audio.shape[1]),
            text_seq_len=int(conditional_dict["video_context"].shape[1]),
            video_frame_seqlen=(
                int(candidate_video.shape[-2])
                * int(candidate_video.shape[-1])
            ),
            device=self.device,
            dtype=self.dtype,
        )
        video_total = torch.zeros((), device=self.device, dtype=torch.float32)
        audio_total = torch.zeros((), device=self.device, dtype=torch.float32)
        previous_video = None
        for block in blocks:
            video_slice = slice(block.video_start, block.video_end)
            audio_slice = slice(block.audio_start, block.audio_end)
            clean_video = candidate_video[:, video_slice]
            clean_audio = candidate_audio[:, audio_slice]
            noisy_video = add_noise(
                clean_video, video_noise[:, video_slice], sigma
            )
            noisy_audio = add_noise(
                clean_audio, audio_noise[:, audio_slice], sigma
            )
            ref_video = self._ref_video_block(
                previous_video=previous_video,
                current_block_size=block.video_frames,
                template=candidate_video,
            )
            prediction_video, prediction_audio = kv_block_forward(
                model,
                video=noisy_video,
                audio=noisy_audio,
                video_sigma=full_sigma(
                    sigma,
                    batch_size=batch_size,
                    frames=block.video_frames,
                    device=self.device,
                    dtype=torch.float32,
                ),
                audio_sigma=full_sigma(
                    sigma,
                    batch_size=batch_size,
                    frames=block.audio_frames,
                    device=self.device,
                    dtype=torch.float32,
                ),
                conditional_dict=conditional_dict,
                kv_caches=caches,
                block=block,
                ref_video_block=ref_video,
            )
            valid_video = self._valid_frames(
                block.video_start, block.video_end, rvf
            )
            valid_audio = self._valid_frames(
                block.audio_start, block.audio_end, raf
            )
            video_energy, audio_energy = self._block_energies(
                prediction_video=prediction_video,
                prediction_audio=prediction_audio,
                clean_video=clean_video,
                clean_audio=clean_audio,
                valid_video_frames=valid_video,
                valid_audio_frames=valid_audio,
                video_denominator=float(batch_size * rvf),
                audio_denominator=float(batch_size * raf),
            )
            video_total = video_total + video_energy
            audio_total = audio_total + audio_energy
            refresh_clean_cache(
                model,
                video_x0=clean_video,
                audio_x0=clean_audio,
                conditional_dict=conditional_dict,
                kv_caches=caches,
                block=block,
                sigma_dtype=torch.float32,
                ref_video_block=ref_video,
            )
            previous_video = clean_video.detach()

        del caches
        total = (
            self.dpo_video_weight * video_total
            + self.dpo_audio_weight * audio_total
        )
        return CandidateEnergy(
            total=total.detach(),
            video=video_total.detach(),
            audio=audio_total.detach(),
        )

    def backward_candidate(
        self,
        *,
        candidate_video: torch.Tensor,
        candidate_audio: torch.Tensor,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: int,
        real_video_frames: int,
        real_audio_frames: int,
        blocks: list,
        video_coefficient: torch.Tensor | float,
        audio_coefficient: torch.Tensor | float,
        synchronize_last_block: bool,
    ) -> CandidateEnergy:
        """Replay one online-policy energy and immediately backpropagate it."""

        sigma = torch.tensor(
            float(timestep) / 1000.0,
            device=self.device,
            dtype=torch.float32,
        )
        batch_size = int(candidate_video.shape[0])
        rvf = max(
            1, min(int(real_video_frames), int(candidate_video.shape[1]))
        )
        raf = max(
            1, min(int(real_audio_frames), int(candidate_audio.shape[1]))
        )
        caches = init_kv_caches(
            self.policy,
            batch_size=batch_size,
            total_video_frames=int(candidate_video.shape[1]),
            total_audio_frames=int(candidate_audio.shape[1]),
            text_seq_len=int(conditional_dict["video_context"].shape[1]),
            video_frame_seqlen=(
                int(candidate_video.shape[-2])
                * int(candidate_video.shape[-1])
            ),
            device=self.device,
            dtype=self.dtype,
        )
        no_sync = getattr(self.policy, "no_sync", None)
        video_scalar = 0.0
        audio_scalar = 0.0
        previous_video = None

        for block_index, block in enumerate(blocks):
            is_final_sync = bool(
                synchronize_last_block and block_index == len(blocks) - 1
            )
            sync_context = (
                nullcontext()
                if is_final_sync or not callable(no_sync)
                else no_sync()
            )
            with sync_context:
                video_slice = slice(block.video_start, block.video_end)
                audio_slice = slice(block.audio_start, block.audio_end)
                clean_video = candidate_video[:, video_slice]
                clean_audio = candidate_audio[:, audio_slice]
                noisy_video = add_noise(
                    clean_video, video_noise[:, video_slice], sigma
                )
                noisy_audio = add_noise(
                    clean_audio, audio_noise[:, audio_slice], sigma
                )
                ref_video = self._ref_video_block(
                    previous_video=previous_video,
                    current_block_size=block.video_frames,
                    template=candidate_video,
                )
                prediction_video, prediction_audio = kv_block_forward(
                    self.policy,
                    video=noisy_video,
                    audio=noisy_audio,
                    video_sigma=full_sigma(
                        sigma,
                        batch_size=batch_size,
                        frames=block.video_frames,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    audio_sigma=full_sigma(
                        sigma,
                        batch_size=batch_size,
                        frames=block.audio_frames,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    conditional_dict=conditional_dict,
                    kv_caches=caches,
                    block=block,
                    ref_video_block=ref_video,
                )
                valid_video = self._valid_frames(
                    block.video_start, block.video_end, rvf
                )
                valid_audio = self._valid_frames(
                    block.audio_start, block.audio_end, raf
                )
                video_energy, audio_energy = self._block_energies(
                    prediction_video=prediction_video,
                    prediction_audio=prediction_audio,
                    clean_video=clean_video,
                    clean_audio=clean_audio,
                    valid_video_frames=valid_video,
                    valid_audio_frames=valid_audio,
                    video_denominator=float(batch_size * rvf),
                    audio_denominator=float(batch_size * raf),
                )
                block_loss = (
                    video_coefficient * video_energy
                    + audio_coefficient * audio_energy
                )
                block_loss.backward()
                video_scalar += float(video_energy.detach().item())
                audio_scalar += float(audio_energy.detach().item())
                refresh_clean_cache(
                    self.policy,
                    video_x0=clean_video,
                    audio_x0=clean_audio,
                    conditional_dict=conditional_dict,
                    kv_caches=caches,
                    block=block,
                    sigma_dtype=torch.float32,
                    ref_video_block=ref_video,
                )
                previous_video = clean_video.detach()
                del prediction_video, prediction_audio, block_loss

        del caches
        video_total = torch.tensor(
            video_scalar, device=self.device, dtype=torch.float32
        )
        audio_total = torch.tensor(
            audio_scalar, device=self.device, dtype=torch.float32
        )
        return CandidateEnergy(
            total=(
                self.dpo_video_weight * video_total
                + self.dpo_audio_weight * audio_total
            ),
            video=video_total,
            audio=audio_total,
        )

    def dpo_backward(
        self,
        *,
        pair: PreferencePair,
        conditional_dict: Dict[str, Any],
        timestep: int,
        real_video_frames: int,
        real_audio_frames: int,
        gradient_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor | float]:
        """Compute coefficients no-grad, then replay the exact online gradient."""

        if int(timestep) not in self.dpo_timestep_candidates:
            raise ValueError(
                f"DPO timestep {timestep} is not in "
                f"{self.dpo_timestep_candidates}"
            )
        gradient_scale_value = float(gradient_scale)
        if not math.isfinite(gradient_scale_value) or gradient_scale_value <= 0:
            raise ValueError(
                "gradient_scale must be finite and > 0, got "
                f"{gradient_scale_value}"
            )
        video_noise = torch.randn_like(pair.negative_video)
        audio_noise = torch.randn_like(pair.negative_audio)

        # The pair shares both timestep and noise.  The reference and online
        # policy also receive the exact same noisy tensors and clean caches.
        reference_positive = self.score_candidate(
            model=self.reference,
            candidate_video=pair.positive_video,
            candidate_audio=pair.positive_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
            blocks=pair.blocks,
        )
        reference_negative = self.score_candidate(
            model=self.reference,
            candidate_video=pair.negative_video,
            candidate_audio=pair.negative_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
            blocks=pair.blocks,
        )
        online_positive = self.score_candidate(
            model=self.policy,
            candidate_video=pair.positive_video,
            candidate_audio=pair.positive_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
            blocks=pair.blocks,
        )
        online_negative = self.score_candidate(
            model=self.policy,
            candidate_video=pair.negative_video,
            candidate_audio=pair.negative_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
            blocks=pair.blocks,
        )
        loss, coefficient, advantage, logit = (
            dpo_loss_and_replay_coefficient(
                online_positive_energy=online_positive.total,
                online_negative_energy=online_negative.total,
                reference_positive_energy=reference_positive.total,
                reference_negative_energy=reference_negative.total,
                beta=self.dpo_beta,
                loss_weight=self.dpo_loss_weight,
            )
        )
        # Preserve the original replay path bit-for-bit when accumulation is
        # disabled.  For accumulation, each fully synchronized micro-batch
        # contributes 1 / accumulation_steps of its DPO gradient.
        replay_coefficient = (
            coefficient
            if gradient_scale_value == 1.0
            else coefficient * gradient_scale_value
        )

        # Positive never synchronizes.  The final negative block performs the
        # single FSDP synchronization for the complete DPO update.
        positive_video_coefficient = (
            replay_coefficient * self.dpo_video_weight
        )
        positive_audio_coefficient = (
            replay_coefficient * self.dpo_audio_weight
        )
        negative_video_coefficient = (
            -replay_coefficient * self.dpo_video_weight
            + gradient_scale_value * self.negative_video_anchor_weight
        )
        negative_audio_coefficient = (
            -replay_coefficient * self.dpo_audio_weight
            + gradient_scale_value * self.negative_audio_anchor_weight
        )

        replay_positive = self.backward_candidate(
            candidate_video=pair.positive_video,
            candidate_audio=pair.positive_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
            blocks=pair.blocks,
            video_coefficient=positive_video_coefficient,
            audio_coefficient=positive_audio_coefficient,
            synchronize_last_block=False,
        )
        replay_negative = self.backward_candidate(
            candidate_video=pair.negative_video,
            candidate_audio=pair.negative_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
            blocks=pair.blocks,
            video_coefficient=negative_video_coefficient,
            audio_coefficient=negative_audio_coefficient,
            synchronize_last_block=True,
        )

        negative_video_anchor_loss = (
            self.negative_video_anchor_weight * online_negative.video
        )
        negative_audio_anchor_loss = (
            self.negative_audio_anchor_weight * online_negative.audio
        )
        surrogate_loss = (
            loss + negative_video_anchor_loss + negative_audio_anchor_loss
        )

        return {
            "dpo/loss": loss.detach(),
            "dpo/timestep": float(timestep),
            "dpo/sigma": float(timestep) / 1000.0,
            "dpo/coefficient_q": coefficient.detach(),
            "dpo/effective_coefficient_q": replay_coefficient.detach(),
            "dpo/effective_positive_video_coefficient": (
                positive_video_coefficient.detach()
            ),
            "dpo/effective_positive_audio_coefficient": (
                positive_audio_coefficient.detach()
            ),
            "dpo/effective_negative_video_coefficient": (
                negative_video_coefficient.detach()
                if torch.is_tensor(negative_video_coefficient)
                else float(negative_video_coefficient)
            ),
            "dpo/effective_negative_audio_coefficient": (
                negative_audio_coefficient.detach()
                if torch.is_tensor(negative_audio_coefficient)
                else float(negative_audio_coefficient)
            ),
            "dpo/advantage": advantage.detach(),
            "dpo/logit": logit.detach(),
            "dpo/preference_probability": torch.sigmoid(logit).detach(),
            "dpo/preference_accuracy": (advantage > 0).float().detach(),
            "dpo/online_margin": (
                online_negative.total - online_positive.total
            ).detach(),
            "dpo/reference_margin": (
                reference_negative.total - reference_positive.total
            ).detach(),
            "dpo/online_positive": online_positive.total,
            "dpo/online_negative": online_negative.total,
            "dpo/reference_positive": reference_positive.total,
            "dpo/reference_negative": reference_negative.total,
            "dpo/online_positive_video": online_positive.video,
            "dpo/online_negative_video": online_negative.video,
            "dpo/reference_positive_video": reference_positive.video,
            "dpo/reference_negative_video": reference_negative.video,
            "dpo/online_positive_audio": online_positive.audio,
            "dpo/online_negative_audio": online_negative.audio,
            "dpo/reference_positive_audio": reference_positive.audio,
            "dpo/reference_negative_audio": reference_negative.audio,
            "anchor/negative_video_loss": negative_video_anchor_loss.detach(),
            "anchor/negative_audio_loss": negative_audio_anchor_loss.detach(),
            "optimization/surrogate_loss": surrogate_loss.detach(),
            "dpo/replay_positive_error": (
                replay_positive.total - online_positive.total
            ).abs(),
            "dpo/replay_negative_error": (
                replay_negative.total - online_negative.total
            ).abs(),
            "pair/relative_video_change": pair.relative_video_change,
            "pair/relative_audio_change": pair.relative_audio_change,
            "pair/refiner_relative_video_change": (
                pair.refiner_relative_video_change
                if pair.refiner_relative_video_change is not None
                else pair.relative_video_change
            ),
            "pair/refiner_relative_audio_change": (
                pair.refiner_relative_audio_change
                if pair.refiner_relative_audio_change is not None
                else pair.relative_audio_change
            ),
        }
