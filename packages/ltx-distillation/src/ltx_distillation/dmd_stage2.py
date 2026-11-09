"""
LTX-2 Stage 2 DMD Module: Teacher Forcing + Distribution Matching Distillation.

This module implements the DMD loss path for Stage 2 of our coarse-to-fine
streaming AR distillation pipeline. The key design difference from
``ltx_distillation.dmd.LTX2DMD`` is that this module **does NOT own the
generator**. The generator forward is performed externally by the trainer
(via the Stage 0 ``LTX2ARTeacher`` TF doubled-sequence path) and the
resulting ``pred_x0_video`` / ``pred_x0_audio`` are passed in.

Responsibilities of this module:
- Own ``real_score`` (frozen bidirectional teacher) and ``fake_score``
  (trainable bidirectional critic), both initialised from the LTX-2 base
  checkpoint (CausVid-style hybrid setup: causal generator + bidirectional
  teacher/critic for training stability).
- Compute the DMD distribution-matching loss given the generator's pred_x0.
- Compute the critic (fake_score) loss given a detached pred_x0.

What this module deliberately leaves out:
- Generator forward / FSDP wrapping / optimizer (lives in trainer)
- Self-forcing rollout / KV cache (only used in Stage 3)
- ODE-LMDB / backward simulation handling (only used in Stage 1 bidir DMD)
- SVI online error bank (planned for v2, not in this initial Stage 2)

All math (kl-grad normalisation, CFG on real_score, flow-pred critic loss)
follows ``LTX2DMD`` exactly so that Stage 2 → Stage 3 transitions are
loss-family consistent.
"""
from typing import Tuple, Dict, Any, Optional, List
import math

import torch
import torch.distributed as dist
import torch.nn as nn

from ltx_core.components.schedulers import LTX2Scheduler

from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper
from ltx_distillation.loss import get_denoising_loss

try:
    from ltx_causal.attention.mask_builder import compute_av_blocks
except ImportError:
    compute_av_blocks = None


class DMDStage2(nn.Module):
    """Stage 2 DMD module: real/fake score + DMD loss (no generator).

    Designed to be plugged into a trainer that owns the causal AR generator
    separately (Stage 0 ``LTX2ARTeacher``) and supplies pre-computed
    ``pred_x0_video`` / ``pred_x0_audio`` to the loss methods.

    Naming convention preserved from ``ltx_distillation.dmd`` for clarity:
    - ``real_score``: frozen LTX-2 base model used as the DMD teacher (CFG)
    - ``fake_score``: trainable critic that learns the generator's output
      distribution
    """

    def __init__(self, args, device: torch.device):
        super().__init__()

        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        # Task types. Stage 2 keeps generator causal (we don't reference it
        # here, but we record the type for block-aware utilities) and pins
        # real/fake score to bidirectional for training stability.
        self.generator_task_type = getattr(args, "generator_task_type", "causal_av")
        self.real_task_type = getattr(args, "real_task_type", "bidirectional_av")
        self.fake_task_type = getattr(args, "fake_task_type", "bidirectional_av")

        # DMD hyperparameters.
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_video_guidance_scale = getattr(args, "real_video_guidance_scale", 5.0)
        self.real_video_guidance_scale_min = float(
            getattr(
                args,
                "real_video_guidance_scale_min",
                self.real_video_guidance_scale,
            )
        )
        self.real_video_guidance_scale_max = float(
            getattr(
                args,
                "real_video_guidance_scale_max",
                self.real_video_guidance_scale,
            )
        )
        if (
            self.real_video_guidance_scale_min <= 0.0
            or self.real_video_guidance_scale_max
            < self.real_video_guidance_scale_min
        ):
            raise ValueError(
                "real video CFG range must satisfy "
                "0 < real_video_guidance_scale_min <= "
                "real_video_guidance_scale_max"
            )
        self.real_audio_guidance_scale = getattr(args, "real_audio_guidance_scale", 9.0)
        self.real_audio_guidance_scale_min = float(
            getattr(
                args,
                "real_audio_guidance_scale_min",
                self.real_audio_guidance_scale,
            )
        )
        self.real_audio_guidance_scale_max = float(
            getattr(
                args,
                "real_audio_guidance_scale_max",
                self.real_audio_guidance_scale,
            )
        )
        if (
            self.real_audio_guidance_scale_min <= 0.0
            or self.real_audio_guidance_scale_max
            < self.real_audio_guidance_scale_min
        ):
            raise ValueError(
                "real audio CFG range must satisfy "
                "0 < real_audio_guidance_scale_min <= "
                "real_audio_guidance_scale_max"
            )
        self.dmd_timestep_sampler_type = str(
            getattr(args, "dmd_timestep_sampler_type", "ltx_scheduler")
        )
        if self.dmd_timestep_sampler_type not in {
            "ltx_scheduler",
            "uniform_shifted",
        }:
            raise ValueError(
                "dmd_timestep_sampler_type must be 'ltx_scheduler' or "
                f"'uniform_shifted', got {self.dmd_timestep_sampler_type!r}"
            )
        self.dmd_sigma_shift = float(
            getattr(args, "dmd_sigma_shift", getattr(args, "sigma_shift", 1.0))
        )
        self.dmd_sigma_min = float(
            getattr(args, "dmd_sigma_min", getattr(args, "sigma_min", 0.0))
        )
        self.dmd_sigma_max = float(
            getattr(args, "dmd_sigma_max", getattr(args, "sigma_max", 1.0))
        )
        self.dmd_sigma_bounds_space = str(
            getattr(args, "dmd_sigma_bounds_space", "raw")
        ).lower()
        if self.dmd_sigma_bounds_space not in {"raw", "shifted"}:
            raise ValueError(
                "dmd_sigma_bounds_space must be 'raw' or 'shifted', got "
                f"{self.dmd_sigma_bounds_space!r}"
            )
        if self.dmd_sigma_shift <= 0.0:
            raise ValueError("dmd_sigma_shift must be positive")
        if not 0.0 <= self.dmd_sigma_min < self.dmd_sigma_max <= 1.0:
            raise ValueError(
                "dmd sigma bounds must satisfy "
                "0 <= dmd_sigma_min < dmd_sigma_max <= 1"
            )

        # Fake-score denoising may use a different noise marginal from the
        # generator-side DMD gradient.  Every field falls back to its DMD
        # counterpart so existing Stage-2 configs retain their exact behavior.
        self.fake_score_timestep_sampler_type = str(
            getattr(
                args,
                "fake_score_timestep_sampler_type",
                self.dmd_timestep_sampler_type,
            )
        )
        if self.fake_score_timestep_sampler_type not in {
            "ltx_scheduler",
            "uniform_shifted",
        }:
            raise ValueError(
                "fake_score_timestep_sampler_type must be 'ltx_scheduler' or "
                "'uniform_shifted', got "
                f"{self.fake_score_timestep_sampler_type!r}"
            )
        self.fake_score_sigma_shift = float(
            getattr(args, "fake_score_sigma_shift", self.dmd_sigma_shift)
        )
        self.fake_score_sigma_min = float(
            getattr(args, "fake_score_sigma_min", self.dmd_sigma_min)
        )
        self.fake_score_sigma_max = float(
            getattr(args, "fake_score_sigma_max", self.dmd_sigma_max)
        )
        self.fake_score_sigma_bounds_space = str(
            getattr(
                args,
                "fake_score_sigma_bounds_space",
                self.dmd_sigma_bounds_space,
            )
        ).lower()
        if self.fake_score_sigma_bounds_space not in {"raw", "shifted"}:
            raise ValueError(
                "fake_score_sigma_bounds_space must be 'raw' or 'shifted', got "
                f"{self.fake_score_sigma_bounds_space!r}"
            )
        if self.fake_score_sigma_shift <= 0.0:
            raise ValueError("fake_score_sigma_shift must be positive")
        if not (
            0.0
            <= self.fake_score_sigma_min
            < self.fake_score_sigma_max
            <= 1.0
        ):
            raise ValueError(
                "fake-score sigma bounds must satisfy "
                "0 <= fake_score_sigma_min < fake_score_sigma_max <= 1"
            )

        # Loss weighting.
        self.video_loss_weight = getattr(args, "video_loss_weight", 1.0)
        self.audio_loss_weight = getattr(args, "audio_loss_weight", 1.0)
        self.audio_start_step = getattr(args, "audio_start_step", 0)

        # Few-step generator schedule. We always use strict mode here:
        # ``denoising_step_list`` already encodes the shifted timesteps used
        # at inference time (e.g. shift=8). t/1000 is used directly as the
        # flow-matching sigma.
        if not getattr(args, "strict_denoising_sigmas", True):
            raise ValueError(
                "DMDStage2 requires strict_denoising_sigmas=true; the "
                "non-strict LMDB-aligned mode is intentionally not supported."
            )
        self.denoising_sigmas = torch.tensor(
            [t / 1000.0 for t in args.denoising_step_list],
            dtype=torch.float32,
            device=device,
        )

        # Sigma lookup table (integer timestep 0..N → sigma) for the DMD
        # noise level. Mirrors LTX2DMD.timestep_to_sigma.
        scheduler = LTX2Scheduler()
        full_sigmas = scheduler.execute(steps=self.num_train_timestep).to(device)
        self.register_buffer(
            "sigma_lookup",
            full_sigmas.flip(0),  # index 0 = sigma~0 (clean), index N = sigma~1 (noise)
        )

        # Denoising loss for the critic.
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

        # Block geometry (needed when block-aware weighting is requested).
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.first_block_video_frames = getattr(args, "first_block_video_frames", 4)
        self.block_weight_mode = getattr(args, "block_weight_mode", "uniform")
        self.block_weight_min = getattr(args, "block_weight_min", 0.5)

        # Placeholders for sub-modules. The trainer creates each one with
        # create_ltx2_wrapper(...), FSDP-wraps it, then assigns the wrapped
        # result back here. See the note above ``# Model initialisation``.
        self.real_score: Optional[LTX2DiffusionWrapper] = None
        self.fake_score: Optional[LTX2DiffusionWrapper] = None

        # Trainer pokes this in each step for two-phase audio weighting.
        self.current_step: int = 0

    # ------------------------------------------------------------------ #
    # Model initialisation                                                #
    # ------------------------------------------------------------------ #
    # NOTE: real_score and fake_score are NOT built here. The trainer
    # builds each one via create_ltx2_wrapper(...) and FSDP-wraps it
    # immediately, then assigns the wrapped module back to
    # self.real_score / self.fake_score. This interleaved build→wrap→build
    # pattern keeps GPU peak memory at ~1 full model copy instead of 2,
    # which matters on H200 140G when stacking generator + EMA + real +
    # fake all-in-one before any wrap.
    #
    # See `Stage2Trainer.__init__` for the exact order:
    #   1. build generator (via ar_teacher) → FSDP wrap → free full copy
    #   2. build text_encoder → FSDP wrap
    #   3. build real_score → FSDP wrap (share state_dict cache across both)
    #   4. build fake_score → FSDP wrap

    # ------------------------------------------------------------------ #
    # Schedule / noise helpers (mirror LTX2DMD)                           #
    # ------------------------------------------------------------------ #
    def get_loss_weights(self) -> Tuple[float, float]:
        video_w = self.video_loss_weight
        audio_w = self.audio_loss_weight
        if self.audio_start_step > 0 and self.current_step < self.audio_start_step:
            audio_w = 0.0
        return video_w, audio_w

    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Integer timestep ∈ [0, N] → sigma via pre-computed lookup."""
        t_clamped = timestep.long().clamp(0, self.num_train_timestep)
        return self.sigma_lookup[t_clamped]

    def _sample_score_sigma(
        self,
        batch_size: int,
        *,
        sampler_type: str,
        sigma_shift: float,
        sigma_min: float,
        sigma_max: float,
        bounds_space: str,
    ) -> torch.Tensor:
        """Sample one score-noise sigma per item from a configured marginal."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if sampler_type == "ltx_scheduler":
            timestep = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
            return self.timestep_to_sigma(timestep)

        raw_min, raw_max = sigma_min, sigma_max
        if bounds_space == "shifted":
            # Invert sigma = shift*raw / (1 + (shift-1)*raw). Sampling
            # uniformly between these inverse endpoints preserves the usual
            # uniform-then-shift distribution without creating clamp mass at
            # either requested post-shift boundary.
            raw_min = self._inverse_sigma_shift(sigma_min, sigma_shift)
            raw_max = self._inverse_sigma_shift(sigma_max, sigma_shift)

        raw = torch.rand(
            batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        raw = raw_min + (raw_max - raw_min) * raw
        if sigma_shift == 1.0:
            return raw
        return sigma_shift * raw / (1.0 + (sigma_shift - 1.0) * raw)

    @staticmethod
    def _inverse_sigma_shift(shifted_sigma: float, sigma_shift: float) -> float:
        """Map a post-shift sigma boundary back to raw-sigma space."""
        denominator = sigma_shift - (sigma_shift - 1.0) * shifted_sigma
        if denominator <= 0.0:
            raise ValueError(
                "invalid shifted sigma boundary: inverse denominator must be positive"
            )
        return shifted_sigma / denominator

    @staticmethod
    def _apply_sigma_shift(raw_sigma: float, sigma_shift: float) -> float:
        return sigma_shift * raw_sigma / (
            1.0 + (sigma_shift - 1.0) * raw_sigma
        )

    def score_sigma_support(self, *, fake_score: bool = False) -> Tuple[float, float]:
        """Return the actual post-shift sigma support for logging/tests."""
        if fake_score:
            sigma_min = self.fake_score_sigma_min
            sigma_max = self.fake_score_sigma_max
            sigma_shift = self.fake_score_sigma_shift
            bounds_space = self.fake_score_sigma_bounds_space
        else:
            sigma_min = self.dmd_sigma_min
            sigma_max = self.dmd_sigma_max
            sigma_shift = self.dmd_sigma_shift
            bounds_space = self.dmd_sigma_bounds_space
        if bounds_space == "shifted":
            return sigma_min, sigma_max
        return (
            self._apply_sigma_shift(sigma_min, sigma_shift),
            self._apply_sigma_shift(sigma_max, sigma_shift),
        )

    def sample_dmd_score_sigma(self, batch_size: int) -> torch.Tensor:
        """Sample one score-noise sigma per item for the generator DMD loss.

        ``ltx_scheduler`` preserves the legacy behavior: sample an integer
        timestep in ``[2%, 98%]`` and index the fixed LTX scheduler lookup.

        ``uniform_shifted`` matches Stage-0's uniform-shifted marginal:

        ``raw ~ U(raw_min, raw_max)``
        ``sigma = shift * raw / (1 + (shift - 1) * raw)``

        With ``dmd_sigma_bounds_space=raw``, the configured min/max are the
        raw bounds (legacy behavior). With ``shifted``, they define the actual
        output support and are inverse-mapped to ``raw_min/raw_max`` first.

        The caller expands this scalar to every video/audio frame, preserving
        Stage-2's same-step-across-blocks training contract.
        """
        return self._sample_score_sigma(
            batch_size,
            sampler_type=self.dmd_timestep_sampler_type,
            sigma_shift=self.dmd_sigma_shift,
            sigma_min=self.dmd_sigma_min,
            sigma_max=self.dmd_sigma_max,
            bounds_space=getattr(self, "dmd_sigma_bounds_space", "raw"),
        )

    def sample_fake_score_sigma(self, batch_size: int) -> torch.Tensor:
        """Sample the fake-score denoising sigma independently of DMD."""
        return self._sample_score_sigma(
            batch_size,
            sampler_type=self.fake_score_timestep_sampler_type,
            sigma_shift=self.fake_score_sigma_shift,
            sigma_min=self.fake_score_sigma_min,
            sigma_max=self.fake_score_sigma_max,
            bounds_space=getattr(self, "fake_score_sigma_bounds_space", "raw"),
        )

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching interpolation: ``x_t = (1 - sigma) * x_0 + sigma * eps``."""
        if sigma.dim() == 1:
            sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
        elif sigma.dim() == 2:
            sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
        sigma = sigma.to(dtype=original.dtype)
        return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)

    def sample_real_video_guidance_scale(self, device: torch.device) -> float:
        """Sample one video CFG scale for the current DMD score evaluation.

        When no range is configured, both endpoints fall back to the legacy
        fixed ``real_video_guidance_scale`` and this method is deterministic.
        A scalar is intentionally shared by the whole local batch so the CFG
        augmentation remains one coherent training condition per DMD update.
        """
        lower = float(
            getattr(
                self,
                "real_video_guidance_scale_min",
                self.real_video_guidance_scale,
            )
        )
        upper = float(
            getattr(
                self,
                "real_video_guidance_scale_max",
                self.real_video_guidance_scale,
            )
        )
        if lower == upper:
            return lower
        return float(
            torch.empty((), device=device, dtype=torch.float32)
            .uniform_(lower, upper)
            .item()
        )

    def sample_real_audio_guidance_scale(self, device: torch.device) -> float:
        """Sample one clip-level audio CFG scale for a real-score update."""
        lower = float(
            getattr(
                self,
                "real_audio_guidance_scale_min",
                self.real_audio_guidance_scale,
            )
        )
        upper = float(
            getattr(
                self,
                "real_audio_guidance_scale_max",
                self.real_audio_guidance_scale,
            )
        )
        if lower == upper:
            return lower
        return float(
            torch.empty((), device=device, dtype=torch.float32)
            .uniform_(lower, upper)
            .item()
        )

    def sample_few_step_sigma(self, batch_size: int) -> torch.Tensor:
        """Sample one sigma per batch item from ``denoising_sigmas[:-1]``.

        Stage 2 design choice: the entire 5s clip uses the SAME sigma per
        batch item (block-uniform AND across-blocks-uniform). This keeps the
        first version simple and matches the inference path's per-block
        timestep that is shared across all generated blocks.

        Returns:
            Tensor [B] of sigma values.
        """
        num_steps = len(self.denoising_sigmas)
        if num_steps <= 1:
            return torch.zeros(batch_size, device=self.device, dtype=self.denoising_sigmas.dtype)
        idx = torch.randint(
            0, num_steps - 1,  # exclude the final endpoint (sigma=0)
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        return self.denoising_sigmas[idx]

    def expand_sigma_to_frames(
        self,
        sigma_batch: torch.Tensor,
        num_video_frames: int,
        num_audio_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand a per-batch sigma to per-frame sigma tensors.

        Returns:
            (video_sigma [B, F_v], audio_sigma [B, F_a]) -- both filled with
            the same per-batch sigma value.
        """
        B = sigma_batch.shape[0]
        video_sigma = sigma_batch.view(B, 1).expand(B, num_video_frames).contiguous()
        audio_sigma = sigma_batch.view(B, 1).expand(B, num_audio_frames).contiguous()
        return video_sigma, audio_sigma

    # ------------------------------------------------------------------ #
    # Block-aware loss weighting (uniform default, ramp optional)         #
    # ------------------------------------------------------------------ #
    def _get_causal_blocks(self, num_video_frames: int):
        if compute_av_blocks is None:
            raise ImportError("Causal block utilities require the ltx-causal package")
        return compute_av_blocks(
            total_video_latent_frames=num_video_frames,
            num_frame_per_block=self.num_frame_per_block,
            first_block_video_frames=self.first_block_video_frames,
        )

    def _compute_block_weights(self, num_frames: int, *, is_audio: bool = False) -> torch.Tensor:
        """Per-frame weights for the block-aware overexposure suppression.

        Uniform by default (returns all-ones). When ``block_weight_mode ==
        'linear_ramp'``, early blocks get ``block_weight_min`` ramping
        linearly to 1.0 at the last block.
        """
        if self.block_weight_mode in {"uniform", "none"}:
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        # Causal block geometry.
        if is_audio:
            # Audio frames span the same number of blocks as video; use video
            # block count from the standard frame ratio.
            est_video_frames = max(
                self.first_block_video_frames,
                math.ceil((num_frames - 1) / 25) * self.num_frame_per_block + 1,
            )
            blocks = self._get_causal_blocks(est_video_frames)
            blocks = [block for block in blocks if block.audio_start < num_frames]
        else:
            blocks = self._get_causal_blocks(num_frames)
            blocks = [block for block in blocks if block.video_start < num_frames]
        n_blocks = len(blocks)

        if n_blocks <= 1:
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        weights = torch.ones(num_frames, device=self.device, dtype=torch.float64)
        for blk_idx, block in enumerate(blocks):
            w = self.block_weight_min + (1.0 - self.block_weight_min) * blk_idx / (n_blocks - 1)
            start = block.audio_start if is_audio else block.video_start
            end = min(block.audio_end if is_audio else block.video_end, num_frames)
            weights[start:end] = w
        return weights

    @staticmethod
    def _masked_weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return (values * weights).mean()
        mask_f = mask.to(values.dtype)
        weighted = values * weights * mask_f
        denom = (weights * mask_f).sum().clamp_min(1.0)
        return weighted.sum() / denom

    # ------------------------------------------------------------------ #
    # Core DMD: KL gradient (mirrors LTX2DMD._compute_kl_grad)            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_kl_grad(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
        normalization: bool = True,
        return_video_predictions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute DMD KL gradient (Eq. 7 in the DMD paper).

        Returns:
            grad_video, grad_audio, log_dict
        """
        # Fake score (single forward, no CFG).
        pred_fake_video, pred_fake_audio = self.fake_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        # Real score with CFG (two forwards: cond + uncond).
        pred_real_cond_video, pred_real_cond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )
        pred_real_uncond_video, pred_real_uncond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )
        real_video_guidance_scale = self.sample_real_video_guidance_scale(
            pred_real_cond_video.device
        )
        real_audio_guidance_scale = self.sample_real_audio_guidance_scale(
            pred_real_cond_audio.device
        )

        # CFG: out = cond + (scale - 1) * (cond - uncond).
        pred_real_video = pred_real_cond_video + (real_video_guidance_scale - 1) * (
            pred_real_cond_video - pred_real_uncond_video
        )
        pred_real_audio = pred_real_cond_audio + (real_audio_guidance_scale - 1) * (
            pred_real_cond_audio - pred_real_uncond_audio
        )

        # DMD gradient (Eq. 7).
        grad_video = pred_fake_video - pred_real_video
        grad_audio = pred_fake_audio - pred_real_audio

        # Per-sample normalisation (Eq. 8).
        if normalization:
            p_real_video = clean_video - pred_real_video
            if video_loss_mask is not None:
                vmask = video_loss_mask.to(p_real_video.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                v_active = vmask.expand_as(p_real_video).sum(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1.0)
                normalizer_video = (torch.abs(p_real_video) * vmask).sum(dim=[1, 2, 3, 4], keepdim=True)
                normalizer_video = normalizer_video / v_active
            else:
                normalizer_video = torch.abs(p_real_video).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad_video = grad_video / (normalizer_video + 1e-8)

            p_real_audio = clean_audio - pred_real_audio
            if audio_loss_mask is not None:
                amask = audio_loss_mask.to(p_real_audio.dtype).unsqueeze(-1)
                a_active = amask.expand_as(p_real_audio).sum(dim=[1, 2], keepdim=True).clamp_min(1.0)
                normalizer_audio = (torch.abs(p_real_audio) * amask).sum(dim=[1, 2], keepdim=True)
                normalizer_audio = normalizer_audio / a_active
            else:
                normalizer_audio = torch.abs(p_real_audio).mean(dim=[1, 2], keepdim=True)
            grad_audio = grad_audio / (normalizer_audio + 1e-8)

        grad_video = torch.nan_to_num(grad_video)
        grad_audio = torch.nan_to_num(grad_audio)

        log_dict = {
            "dmdtrain_gradient_norm_video": torch.mean(torch.abs(grad_video)).item(),
            "dmdtrain_gradient_norm_audio": torch.mean(torch.abs(grad_audio)).item(),
            "real_score_video": torch.mean(torch.abs(pred_real_video)).item(),
            "real_score_audio": torch.mean(torch.abs(pred_real_audio)).item(),
            "fake_score_video": torch.mean(torch.abs(pred_fake_video)).item(),
            "fake_score_audio": torch.mean(torch.abs(pred_fake_audio)).item(),
            "real_video_guidance_scale": real_video_guidance_scale,
            "real_audio_guidance_scale": real_audio_guidance_scale,
        }
        if return_video_predictions:
            # Visualization-only payload. Existing Stage-2 callers keep the
            # default disabled, so ordinary training retains no extra tensors.
            log_dict["_pred_real_video_x0"] = pred_real_video.detach()
            log_dict["_pred_fake_video_x0"] = pred_fake_video.detach()
        return grad_video, grad_audio, log_dict

    # ------------------------------------------------------------------ #
    # Public: generator-side DMD loss                                     #
    # ------------------------------------------------------------------ #
    def compute_distribution_matching_loss(
        self,
        pred_x0_video: torch.Tensor,
        pred_x0_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
        return_video_predictions: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the DMD loss given the generator's pred_x0.

        The generator forward (TF doubled-sequence path) is performed by the
        trainer; we receive the predicted x0 for video and audio and treat
        them as the ``video_latent`` / ``audio_latent`` arguments to the
        standard DMD loss formulation. The KL gradient is applied through
        a stop-gradient trick so backward flows into pred_x0.

        Args:
            pred_x0_video: [B, F_v, C, H, W] generator's x0 prediction (with grad)
            pred_x0_audio: [B, F_a, C_a] generator's x0 prediction (with grad)
            conditional_dict / unconditional_dict: text conditioning
            video_loss_mask / audio_loss_mask: optional per-frame mask; if
                None, the loss covers all frames.

        Returns:
            (total_loss, log_dict)
        """
        B, F_v = pred_x0_video.shape[:2]
        F_a = pred_x0_audio.shape[1]

        with torch.no_grad():
            # One independently sampled DMD-noise sigma per item, broadcast
            # across every chunk and both modalities.
            score_sigma = self.sample_dmd_score_sigma(B)
            video_sigma, audio_sigma = self.expand_sigma_to_frames(
                score_sigma,
                num_video_frames=F_v,
                num_audio_frames=F_a,
            )

            # Direct-noise variant (the only one we support in Stage 2).
            noise_video = torch.randn_like(pred_x0_video)
            noise_audio = torch.randn_like(pred_x0_audio)

            noisy_video = self.add_noise(
                pred_x0_video.flatten(0, 1),
                noise_video.flatten(0, 1),
                video_sigma.flatten(0, 1),
            ).unflatten(0, (B, F_v))
            noisy_audio = self.add_noise(
                pred_x0_audio, noise_audio, audio_sigma
            )

            grad_video, grad_audio, log_dict = self._compute_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                clean_video=pred_x0_video,
                clean_audio=pred_x0_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=video_loss_mask,
                audio_loss_mask=audio_loss_mask,
                return_video_predictions=return_video_predictions,
            )

        # Block-aware per-frame weights.
        video_block_w = self._compute_block_weights(F_v)
        audio_block_w = self._compute_block_weights(F_a, is_audio=True)

        # Stop-gradient trick: backward pushes (pred_x0 - target) into the
        # generator, where target = pred_x0 - grad (frozen).
        video_diff = pred_x0_video.double() - (pred_x0_video.double() - grad_video.double()).detach()
        video_per_frame = (video_diff ** 2).mean(dim=[2, 3, 4])
        video_loss = 0.5 * self._masked_weighted_mean(
            video_per_frame,
            video_block_w.unsqueeze(0),
            video_loss_mask,
        )

        audio_diff = pred_x0_audio.double() - (pred_x0_audio.double() - grad_audio.double()).detach()
        audio_per_frame = (audio_diff ** 2).mean(dim=2)
        audio_loss = 0.5 * self._masked_weighted_mean(
            audio_per_frame,
            audio_block_w.unsqueeze(0),
            audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict["video_dmd_loss"] = video_loss.detach()
        log_dict["audio_dmd_loss"] = audio_loss.detach()
        log_dict["video_loss_weight"] = video_w
        log_dict["audio_loss_weight"] = audio_w
        log_dict["alignment/video_sigma_mean"] = video_sigma.float().mean().item()
        log_dict["alignment/audio_sigma_mean"] = audio_sigma.float().mean().item()
        return total_loss, log_dict

    # ------------------------------------------------------------------ #
    # Public: critic-side loss (fake_score training)                      #
    # ------------------------------------------------------------------ #
    def critic_loss(
        self,
        pred_x0_video_det: torch.Tensor,
        pred_x0_audio_det: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train ``fake_score`` to denoise the generator's outputs.

        The trainer is responsible for providing freshly-detached pred_x0
        (one generator no_grad forward per critic step, matching Stage 3's
        ``dfake_gen_update_ratio`` discipline).

        Args:
            pred_x0_video_det: [B, F_v, C, H, W] detached generator output
            pred_x0_audio_det: [B, F_a, C_a] detached generator output
            conditional_dict: text conditioning (uncond not needed for critic)
            video_loss_mask / audio_loss_mask: optional per-frame mask

        Returns:
            (total_loss, log_dict)
        """
        B, F_v = pred_x0_video_det.shape[:2]
        F_a = pred_x0_audio_det.shape[1]

        # Fake-score denoising has its own configurable noise marginal. This
        # ablation keeps it at shift=8 while generator DMD uses shift=5.
        score_sigma = self.sample_fake_score_sigma(B)
        critic_sigma, audio_critic_sigma = self.expand_sigma_to_frames(
            score_sigma,
            num_video_frames=F_v,
            num_audio_frames=F_a,
        )

        noise_video = torch.randn_like(pred_x0_video_det)
        noise_audio = torch.randn_like(pred_x0_audio_det)

        noisy_generated_video = self.add_noise(
            pred_x0_video_det.flatten(0, 1),
            noise_video.flatten(0, 1),
            critic_sigma.flatten(0, 1),
        ).unflatten(0, (B, F_v))
        noisy_generated_audio = self.add_noise(
            pred_x0_audio_det, noise_audio, audio_critic_sigma
        )

        # Fake-score predicts x0 from noisy generator output.
        pred_video, pred_audio = self.fake_score(
            noisy_image_or_video=noisy_generated_video,
            conditional_dict=conditional_dict,
            timestep=critic_sigma,
            noisy_audio=noisy_generated_audio,
            audio_timestep=audio_critic_sigma,
        )

        # Convert x0 prediction to flow prediction for the velocity-style
        # denoising loss. The 1/sigma factor gives implicit 1/sigma^2
        # gradient weighting, making the critic accurate at low-noise
        # timesteps (critical for DMD).
        video_sigma_4d = critic_sigma.flatten(0, 1).double().reshape(-1, 1, 1, 1).clamp_min(1e-8)
        flow_pred_video = (
            (noisy_generated_video.flatten(0, 1).double() - pred_video.flatten(0, 1).double())
            / video_sigma_4d
        ).to(self.dtype)

        audio_sigma_2d = audio_critic_sigma.double().unsqueeze(-1).clamp_min(1e-8)
        flow_pred_audio = (
            (noisy_generated_audio.double() - pred_audio.double())
            / audio_sigma_2d
        ).to(self.dtype)

        video_loss = self._masked_denoising_loss(
            target=pred_x0_video_det,
            prediction=pred_video,
            noise=noise_video,
            flow_pred=flow_pred_video.unflatten(0, (B, F_v)),
            timestep=critic_sigma,
            mask=video_loss_mask,
        )
        audio_loss = self._masked_denoising_loss(
            target=pred_x0_audio_det,
            prediction=pred_audio,
            noise=noise_audio,
            flow_pred=flow_pred_audio,
            timestep=audio_critic_sigma,
            mask=audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict = {
            "critic_video_loss": video_loss.item(),
            "critic_audio_loss": audio_loss.item(),
            "critic_sigma_mean": critic_sigma.float().mean().item(),
        }
        return total_loss, log_dict

    def _masked_denoising_loss(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: Optional[torch.Tensor],
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Velocity / x0 denoising loss with optional per-frame mask."""
        if mask is None:
            return self.denoising_loss_func(
                x=target,
                x_pred=prediction,
                noise=noise,
                noise_pred=None,
                alphas_cumprod=None,
                timestep=timestep,
                flow_pred=flow_pred,
            )

        loss_type = str(getattr(self.args, "denoising_loss_type", "velocity")).lower()
        if loss_type == "x0":
            diff = (target.double() - prediction.double()) ** 2
        elif loss_type in {"velocity", "flow"}:
            pred = flow_pred.double() if flow_pred is not None else (noise.double() - prediction.double())
            diff = (pred - (noise.double() - target.double())) ** 2
        else:
            raise NotImplementedError(
                f"Stage 2 critic loss does not support denoising_loss_type={loss_type}"
            )

        reduce_dims = tuple(range(2, diff.dim()))
        per_frame = diff.mean(dim=reduce_dims)
        return self._masked_weighted_mean(
            per_frame,
            torch.ones_like(per_frame, dtype=per_frame.dtype),
            mask,
        ).to(target.dtype)
