"""
LTX-2 DMD (Distribution Matching Distillation) Module.

This module implements DMD for LTX-2 audio-video joint generation,
adapted from CausVid's DMD implementation.

Key differences from CausVid:
- Handles both video and audio modalities jointly
- Uses LTX-2's sigma-based timestep format
- Supports audio-video time alignment
"""

from contextlib import nullcontext
from typing import Tuple, Dict, Any, Optional, List
import gc
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ltx_core.loader.registry import StateDictRegistry

from ltx_core.components.schedulers import LTX2Scheduler

from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper, create_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper, create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper, create_vae_wrappers
from ltx_distillation.loss import get_denoising_loss
try:
    from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
    from ltx_causal.attention.mask_builder import compute_av_blocks
    from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
except ImportError:
    CausalLTX2DiffusionWrapper = None
    compute_av_blocks = None
    CausalLTXModel = None
    CausalLTXModelConfig = None


class LTX2DMD(nn.Module):
    """
    DMD (Distribution Matching Distillation) module for LTX-2.

    Implements the DMD algorithm for distilling a multi-step diffusion model
    to a few-step model, supporting audio-video joint generation.

    The module contains three diffusion models:
    - generator: Student model being trained
    - real_score: Teacher model (frozen)
    - fake_score: Critic model for discriminating real vs fake

    Training alternates between:
    1. Generator training: minimize DMD loss (KL divergence from teacher)
    2. Critic training: learn to distinguish generator outputs from teacher
    """

    # Audio-video time alignment constants
    VIDEO_LATENT_FPS = 3.0  # 24fps / 8
    AUDIO_LATENT_FPS = 25.0  # 16kHz / 160 / 4

    def __init__(self, args, device: torch.device):
        """
        Initialize the DMD module.

        Args:
            args: Configuration object with:
                - checkpoint_path: Path to LTX-2 checkpoint
                - gemma_path: Path to Gemma text encoder
                - denoising_step_list: List of denoising timesteps
                - num_train_timestep: Total training timesteps
                - real_video_guidance_scale: CFG scale for teacher (video)
                - real_audio_guidance_scale: CFG scale for teacher (audio)
                - gradient_checkpointing: Enable gradient checkpointing
                - mixed_precision: Use bfloat16
                - denoising_loss_type: Type of denoising loss
                - video_shape: [B, F, C, H, W]
                - audio_shape: [B, F_a, C]
            device: Target device
        """
        super().__init__()

        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        # Task types
        self.generator_task_type = getattr(args, "generator_task_type", args.generator_task)
        self.real_task_type = getattr(args, "real_task_type", args.generator_task)
        self.fake_task_type = getattr(args, "fake_task_type", args.generator_task)
        self.training_mode = getattr(args, "training_mode", "bidirectional")
        self.enable_self_forcing = "self_forcing" in str(self.training_mode).lower()
        inferred_causal = (
            "causal" in str(self.training_mode).lower()
            or "causal" in str(self.generator_task_type).lower()
            or "causal" in str(self.real_task_type).lower()
            or "causal" in str(self.fake_task_type).lower()
        )
        self.use_causal_wrapper = bool(getattr(args, "use_causal_wrapper", inferred_causal))
        # Per-model causal wrapper switches (CausVid-style hybrid default).
        # By default:
        # - generator follows global use_causal_wrapper
        # - real/fake follow their task types, enabling bidirectional teacher/critic
        self.generator_use_causal_wrapper = bool(
            getattr(args, "generator_use_causal_wrapper", self.use_causal_wrapper)
        )
        self.real_score_use_causal_wrapper = bool(
            getattr(args, "real_score_use_causal_wrapper", "causal" in str(self.real_task_type).lower())
        )
        self.fake_score_use_causal_wrapper = bool(
            getattr(args, "fake_score_use_causal_wrapper", "causal" in str(self.fake_task_type).lower())
        )
        self.alignment_rounding = str(getattr(args, "alignment_rounding", "round")).lower()
        if self.alignment_rounding not in {"round", "floor", "ceil"}:
            raise ValueError(
                f"Invalid alignment_rounding={self.alignment_rounding}, expected round|floor|ceil"
            )
        if (
            self.generator_use_causal_wrapper
            or self.real_score_use_causal_wrapper
            or self.fake_score_use_causal_wrapper
        ) and CausalLTX2DiffusionWrapper is None:
            raise ImportError(
                "Causal wrapper requires ltx-causal package. "
                "Install with: pip install -e packages/ltx-causal"
            )
        if self.enable_self_forcing and not self.generator_use_causal_wrapper:
            raise ValueError("Stage3 Self-Forcing requires generator_use_causal_wrapper=true")
        self.self_forcing_min_generated_blocks = getattr(
            args, "self_forcing_min_generated_blocks", None
        )
        self.self_forcing_max_generated_blocks = getattr(
            args, "self_forcing_max_generated_blocks", None
        )
        self.self_forcing_loss_scope = str(
            getattr(args, "self_forcing_loss_scope", "last_block")
        ).lower()
        if self.self_forcing_loss_scope != "last_block":
            raise ValueError(
                f"Invalid self_forcing_loss_scope={self.self_forcing_loss_scope}, "
                "only last_block is currently supported"
            )

        # === KV-cache Self-Forcing rollout configuration ===
        # When enabled, the rollout runs through a single shared KV-cache
        # (matching the inference path), giving train/inference alignment and
        # O(N) compute per rollout.
        self.enable_kv_cache = bool(getattr(args, "enable_kv_cache", True))
        if not self.enable_kv_cache:
            raise ValueError(
                "enable_kv_cache=false is no longer supported: legacy prefix_rerun "
                "has been retired. Please set enable_kv_cache=true."
            )
        # Integer in [0, num_train_timestep]; encodes the sigma at which the
        # block-end cache REFRESH forward is run. 0 = pure normalisation
        # (refresh with the clean x0 + sigma=0). >0 = light noise injection
        # for exposure-bias mitigation (try 100/250/500 in v2).
        self.context_noise = int(getattr(args, "context_noise", 0))
        if not (0 <= self.context_noise <= getattr(args, "num_train_timestep", 1000)):
            raise ValueError(
                f"Invalid context_noise={self.context_noise}, expected 0..num_train_timestep"
            )
        self.self_forcing_variant = str(
            getattr(args, "self_forcing_variant", "two_pass")
        ).lower()
        if self.self_forcing_variant not in {"two_pass"}:
            raise ValueError(
                f"Invalid self_forcing_variant={self.self_forcing_variant}, "
                "expected two_pass (blockwise path retired)"
            )
        self.same_step_across_blocks = bool(getattr(args, "same_step_across_blocks", True))
        # Distribution used to train the fake score model. ``full`` preserves
        # the historical behavior: every generated chunk completes the entire
        # student denoising schedule. ``random_exit`` matches the generator DMD
        # path by stopping every chunk at a sampled exit step and writing that
        # x0 prediction into the KV cache for subsequent chunks.
        self.fake_score_rollout_mode = str(
            getattr(args, "fake_score_rollout_mode", "full")
        ).lower()
        if self.fake_score_rollout_mode not in {"full", "random_exit"}:
            raise ValueError(
                "fake_score_rollout_mode must be full or random_exit, got "
                f"{self.fake_score_rollout_mode!r}"
            )
        if self.fake_score_rollout_mode == "random_exit" and not self.enable_self_forcing:
            raise ValueError(
                "fake_score_rollout_mode=random_exit requires self-forcing training"
            )

        # Initialize models (will be populated by _init_models or external loading)
        self.generator: LTX2DiffusionWrapper = None
        self.real_score: LTX2DiffusionWrapper = None
        self.fake_score: LTX2DiffusionWrapper = None
        self.text_encoder: GemmaTextEncoderWrapper = None
        self.video_vae: VideoVAEWrapper = None
        self.audio_vae: AudioVAEWrapper = None

        # DMD hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_video_guidance_scale = getattr(args, "real_video_guidance_scale", 3.0)
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
                "real video CFG range must satisfy 0 < min <= max"
            )
        self.real_audio_guidance_scale = getattr(args, "real_audio_guidance_scale", 7.0)
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
            raise ValueError("real audio CFG range must satisfy 0 < min <= max")

        # Score-noise marginals. The default preserves the existing Stage-3
        # integer-timestep + native LTX scheduler behavior. New controlled
        # ablations can independently select Stage-0/2-style uniform-shifted
        # sigmas with bounds expressed in either raw or shifted space.
        self.dmd_timestep_sampler_type = str(
            getattr(args, "dmd_timestep_sampler_type", "ltx_scheduler")
        )
        self.dmd_sigma_shift = float(getattr(args, "dmd_sigma_shift", 1.0))
        self.dmd_sigma_min = float(getattr(args, "dmd_sigma_min", 0.0))
        self.dmd_sigma_max = float(getattr(args, "dmd_sigma_max", 1.0))
        self.dmd_sigma_bounds_space = str(
            getattr(args, "dmd_sigma_bounds_space", "raw")
        ).lower()
        self.fake_score_timestep_sampler_type = str(
            getattr(
                args,
                "fake_score_timestep_sampler_type",
                self.dmd_timestep_sampler_type,
            )
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
        for prefix in ("dmd", "fake_score"):
            sampler_type = getattr(self, f"{prefix}_timestep_sampler_type")
            sigma_shift = getattr(self, f"{prefix}_sigma_shift")
            sigma_min = getattr(self, f"{prefix}_sigma_min")
            sigma_max = getattr(self, f"{prefix}_sigma_max")
            bounds_space = getattr(self, f"{prefix}_sigma_bounds_space")
            if sampler_type not in {"ltx_scheduler", "uniform_shifted"}:
                raise ValueError(
                    f"{prefix}_timestep_sampler_type must be ltx_scheduler or "
                    f"uniform_shifted, got {sampler_type!r}"
                )
            if sigma_shift <= 0.0:
                raise ValueError(f"{prefix}_sigma_shift must be positive")
            if not 0.0 <= sigma_min < sigma_max <= 1.0:
                raise ValueError(
                    f"{prefix} sigma bounds must satisfy 0 <= min < max <= 1"
                )
            if bounds_space not in {"raw", "shifted"}:
                raise ValueError(
                    f"{prefix}_sigma_bounds_space must be raw or shifted"
                )

        # DMD latent noise mode for KL gradient computation.
        # "direct_noise": add Gaussian noise at target sigma (standard DMD)
        # "teacher_denoise": teacher denoises from high noise to target sigma
        self.dmd_latent_mode = getattr(args, "dmd_latent_mode", "direct_noise")

        # Video/Audio loss weighting for ablation experiments.
        # video_loss_weight + audio_loss_weight need not sum to 1.
        # Supports two-phase training: video-only phase then joint phase.
        self.video_loss_weight = getattr(args, "video_loss_weight", 1.0)
        self.audio_loss_weight = getattr(args, "audio_loss_weight", 1.0)
        # Two-phase: if audio_start_step > 0, audio_loss_weight=0 until that step
        self.audio_start_step = getattr(args, "audio_start_step", 0)

        # Denoising sigmas for the few-step generator schedule. Two modes:
        #
        # - Legacy (strict_denoising_sigmas=False): subsample denoising_step_list
        #   to the closest sigma on the LTX2Scheduler(40) grid, so the schedule
        #   matches the ODE trajectories stored in LMDB. Required when
        #   backward_simulation=False (the values are aligned to the LMDB pairs).
        #
        # - Strict (strict_denoising_sigmas=True): use t/1000 directly as the
        #   flow-matching sigma (add_noise: x_t=(1-σ)x_0+σε). denoising_step_list
        #   already encodes the shifted timesteps (t = 1000 * shifted_sigma), so
        #   this reproduces the exact schedule the list was computed with
        #   (e.g. Causal-Forcing shift=8). Appropriate for data-free
        #   (backward_simulation=True) runs where no LMDB alignment is needed.
        if getattr(args, "strict_denoising_sigmas", False):
            self.denoising_sigmas = torch.tensor(
                [t / 1000.0 for t in args.denoising_step_list],
                dtype=torch.float32,
                device=device,
            )
        else:
            _ode_num_steps = getattr(args, "num_inference_steps", 40)
            _full_sigmas = LTX2Scheduler().execute(steps=_ode_num_steps)
            _denoising_sigmas = []
            for t in args.denoising_step_list:
                target_sigma = t / 1000.0
                idx = (_full_sigmas - target_sigma).abs().argmin().item()
                _denoising_sigmas.append(_full_sigmas[idx])
            self.denoising_sigmas = torch.stack(_denoising_sigmas).to(device)

        # Pre-compute sigma lookup table for random timestep → sigma conversion.
        # This matches CausVid's approach where scheduler.add_noise() internally
        # does argmin lookup against the scheduler's sigma schedule.
        # We compute a 1001-entry table (timestep 0..1000) using the native
        # LTX2Scheduler's shifted+stretched sigmoid formula.
        # sigma_lookup[t] gives the actual sigma for integer timestep t.
        scheduler = LTX2Scheduler()
        full_sigmas = scheduler.execute(steps=self.num_train_timestep).to(device)  # [1001] values
        # full_sigmas goes from ~1.0 (noise) to 0.0 (clean), same order as timesteps 1000→0
        # We need sigma_lookup[t] where t=0 → sigma=0 (clean) and t=1000 → sigma≈1 (noise)
        # full_sigmas is ordered: sigma[0]=high (noise), sigma[-1]=0 (clean)
        # So sigma_lookup[t] = full_sigmas[1000 - t] maps t=1000→full_sigmas[0], t=0→full_sigmas[1000]
        self.register_buffer(
            'sigma_lookup',
            full_sigmas.flip(0),  # Reverse so index 0=clean(σ≈0), index 1000=noise(σ≈1)
        )

        # Teacher denoise config (only used when dmd_latent_mode == "teacher_denoise")
        if self.dmd_latent_mode == "teacher_denoise":
            self.teacher_num_steps = getattr(args, "teacher_num_steps", 40)
            # How many teacher schedule steps above target_sigma to start from.
            # E.g. offset=5 means: start 5 steps before target in the sigma schedule,
            # so the teacher only runs ~5 Euler steps regardless of target sigma.
            # Smaller = faster + more student structure preserved.
            # Larger = more "on teacher trajectory" but slower.
            self.teacher_start_offset = getattr(args, "teacher_start_offset", 5)
            # Pre-compute fine-grained teacher sigma schedule.
            # teacher_sigmas[0] ≈ 1.0 (noise), teacher_sigmas[-1] = 0.0 (clean)
            teacher_sigmas = LTX2Scheduler().execute(steps=self.teacher_num_steps).to(device)
            self.register_buffer('teacher_sigmas', teacher_sigmas)

        # Loss function
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

        # Block-aware loss weighting for over-exposure suppression.
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.first_block_video_frames = getattr(args, "first_block_video_frames", 1)
        self.block_weight_mode = getattr(args, "block_weight_mode", "uniform")
        self.block_weight_min = getattr(args, "block_weight_min", 0.5)

        # Inference pipeline (lazy init)
        self.inference_pipeline = None

        # Current training step (updated by trainer)
        self.current_step = 0

    def get_loss_weights(self) -> Tuple[float, float]:
        """Get current video/audio loss weights based on training step."""
        video_w = self.video_loss_weight
        audio_w = self.audio_loss_weight
        if self.audio_start_step > 0 and self.current_step < self.audio_start_step:
            audio_w = 0.0
        return video_w, audio_w

    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert integer timestep (0-1000) to sigma using LTX2Scheduler's lookup table.

        Uses a pre-computed lookup table from the native LTX2Scheduler (shifted+stretched
        sigmoid schedule) instead of a linear t/1000 mapping. This matches CausVid's
        approach where scheduler.add_noise() does internal argmin lookup.

        Args:
            timestep: Integer timestep tensor [B, F] in range [0, num_train_timestep]

        Returns:
            Sigma tensor with same shape, values in [0, 1]
        """
        # Clamp to valid range and index into pre-computed lookup table
        t_clamped = timestep.long().clamp(0, self.num_train_timestep)
        return self.sigma_lookup[t_clamped]

    @staticmethod
    def _apply_sigma_shift(raw_sigma: float, sigma_shift: float) -> float:
        return sigma_shift * raw_sigma / (
            1.0 + (sigma_shift - 1.0) * raw_sigma
        )

    @staticmethod
    def _inverse_sigma_shift(
        shifted_sigma: float,
        sigma_shift: float,
    ) -> float:
        denominator = sigma_shift - (sigma_shift - 1.0) * shifted_sigma
        if denominator <= 0.0:
            raise ValueError("invalid shifted sigma boundary")
        return shifted_sigma / denominator

    def _sample_uniform_shifted_sigma(
        self,
        batch_size: int,
        *,
        sigma_shift: float,
        sigma_min: float,
        sigma_max: float,
        bounds_space: str,
    ) -> torch.Tensor:
        raw_min, raw_max = sigma_min, sigma_max
        if bounds_space == "shifted":
            raw_min = self._inverse_sigma_shift(sigma_min, sigma_shift)
            raw_max = self._inverse_sigma_shift(sigma_max, sigma_shift)
        unit = torch.rand(
            batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        raw = raw_min + (raw_max - raw_min) * unit
        if sigma_shift == 1.0:
            return raw
        return sigma_shift * raw / (
            1.0 + (sigma_shift - 1.0) * raw
        )

    def score_sigma_support(
        self,
        *,
        fake_score: bool = False,
    ) -> Tuple[float, float]:
        prefix = "fake_score" if fake_score else "dmd"
        sigma_min = float(getattr(self, f"{prefix}_sigma_min"))
        sigma_max = float(getattr(self, f"{prefix}_sigma_max"))
        sigma_shift = float(getattr(self, f"{prefix}_sigma_shift"))
        bounds_space = str(getattr(self, f"{prefix}_sigma_bounds_space"))
        if bounds_space == "shifted":
            return sigma_min, sigma_max
        return (
            self._apply_sigma_shift(sigma_min, sigma_shift),
            self._apply_sigma_shift(sigma_max, sigma_shift),
        )

    def _sample_score_noise_levels(
        self,
        *,
        batch_size: int,
        num_video_frames: int,
        num_audio_frames: int,
        task_type: str,
        fake_score: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return video/audio sigma plus legacy loss-timestep tensors.

        ``uniform_shifted`` samples one scalar per item and expands it over all
        chunks and both modalities. DMD and fake-score call this independently.
        ``ltx_scheduler`` exactly preserves the previous Stage-3 behavior.
        """
        prefix = "fake_score" if fake_score else "dmd"
        sampler_type = str(getattr(self, f"{prefix}_timestep_sampler_type"))
        if sampler_type == "uniform_shifted":
            scalar_sigma = self._sample_uniform_shifted_sigma(
                batch_size,
                sigma_shift=float(getattr(self, f"{prefix}_sigma_shift")),
                sigma_min=float(getattr(self, f"{prefix}_sigma_min")),
                sigma_max=float(getattr(self, f"{prefix}_sigma_max")),
                bounds_space=str(getattr(self, f"{prefix}_sigma_bounds_space")),
            )
            video_sigma = scalar_sigma[:, None].expand(
                batch_size,
                num_video_frames,
            ).contiguous()
            audio_sigma = scalar_sigma[:, None].expand(
                batch_size,
                num_audio_frames,
            ).contiguous()
            # Velocity/x0 losses do not consume timestep numerically. Passing
            # sigma keeps the returned contract meaningful for new samplers.
            return video_sigma, audio_sigma, video_sigma, audio_sigma

        video_timestep = torch.randint(
            0,
            self.num_train_timestep,
            (batch_size, num_video_frames),
            device=self.device,
            dtype=torch.long,
        )
        video_timestep = self._process_timestep(video_timestep, task_type)
        video_timestep = video_timestep.clamp(self.min_step, self.max_step)
        audio_timestep = self._compute_audio_timestep(
            video_timestep,
            num_audio_frames,
            task_type=task_type,
        )
        video_sigma = self.timestep_to_sigma(video_timestep)
        audio_sigma = self.timestep_to_sigma(audio_timestep)
        return video_sigma, audio_sigma, video_timestep, audio_timestep

    def sample_real_video_guidance_scale(self, device: torch.device) -> float:
        lower = float(self.real_video_guidance_scale_min)
        upper = float(self.real_video_guidance_scale_max)
        if lower == upper:
            return lower
        return float(
            torch.empty((), device=device, dtype=torch.float32)
            .uniform_(lower, upper)
            .item()
        )

    def sample_real_audio_guidance_scale(self, device: torch.device) -> float:
        lower = float(self.real_audio_guidance_scale_min)
        upper = float(self.real_audio_guidance_scale_max)
        if lower == upper:
            return lower
        return float(
            torch.empty((), device=device, dtype=torch.float32)
            .uniform_(lower, upper)
            .item()
        )

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using flow matching interpolation.

        Flow matching formula: x_t = (1 - sigma) * x_0 + sigma * epsilon

        Args:
            original: Clean samples x_0, shape [B, ...]
            noise: Gaussian noise epsilon, shape [B, ...]
            sigma: Noise level, shape [B] or [B, T] or scalar

        Returns:
            Noisy samples x_t
        """
        # Reshape sigma for broadcasting
        if sigma.dim() == 1:
            # [B] -> [B, 1, 1, 1, ...] for proper broadcasting
            sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
        elif sigma.dim() == 2:
            # [B, T] -> [B, T, 1, 1, ...] for video/audio
            sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
        sigma = sigma.to(dtype=original.dtype)
        return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)

    def init_models(self):
        """
        Initialize all models from checkpoints.

        This method should be called BEFORE FSDP wrapping in distributed training.
        Models must exist before they can be wrapped with FSDP.
        """
        args = self.args

        def _init_log(message: str) -> None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[DMDInit] {message}", flush=True)

        # Get video dimensions from config
        video_height = getattr(args, "video_height", 512)
        video_width = getattr(args, "video_width", 768)

        # Create diffusion wrappers per model (CausVid-style hybrid setup):
        # generator can be causal while real/fake remain bidirectional.
        if isinstance(self.device, int):
            target_device = f"cuda:{self.device}"
        else:
            target_device = str(self.device)

        def _load_checkpoint_state_dict(
            checkpoint_path: str,
            state_key: Optional[str] = None,
        ) -> dict:
            # The Stage-0/2 training checkpoints are ~71 GiB because they hold
            # two 19B state dicts. Never retain their top-level dictionaries:
            # four local ranks doing so can exceed host RAM before FSDP starts.
            # mmap keeps tensor storage file-backed and selecting one state key
            # lets the unselected generator/EMA/fake mapping be released as
            # soon as this helper returns.
            cache_key = (checkpoint_path, state_key)
            if checkpoint_path.endswith(".safetensors"):
                if cache_key in checkpoint_state_cache:
                    return checkpoint_state_cache[cache_key]
                from safetensors.torch import load_file
                loaded = load_file(checkpoint_path)
                if state_key is not None:
                    raise ValueError(
                        f"state_key={state_key!r} is invalid for safetensors "
                        f"checkpoint {checkpoint_path}"
                    )
                checkpoint_state_cache[cache_key] = loaded
                return loaded

            try:
                loaded = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
            except TypeError:
                # Compatibility fallback for older PyTorch versions. Current
                # causal-self-flow supports mmap + weights_only.
                loaded = torch.load(checkpoint_path, map_location="cpu")
            if state_key is not None:
                if not isinstance(loaded, dict) or state_key not in loaded:
                    available = list(loaded) if isinstance(loaded, dict) else []
                    raise KeyError(
                        f"checkpoint {checkpoint_path} has no state key "
                        f"{state_key!r}; available={available[:20]}"
                    )
                loaded = loaded[state_key]
            elif isinstance(loaded, dict) and "generator" in loaded:
                loaded = loaded["generator"]
            elif isinstance(loaded, dict) and "model" in loaded:
                loaded = loaded["model"]
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                loaded = loaded["state_dict"]
            return loaded

        def _remap_state_dict_keys(state_dict: dict) -> dict:
            if not state_dict:
                return state_dict

            non_transformer_prefixes = (
                "vae.", "audio_vae.", "vocoder.",
                "model.vae.", "model.audio_vae.", "model.vocoder.",
            )
            remapped_non_transformer_prefixes = (
                "model.audio_embeddings_connector.",
                "model.video_embeddings_connector.",
            )

            sample_keys = list(state_dict.keys())[:20]
            has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in sample_keys)
            if not has_diffusion_model:
                has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in state_dict)

            if has_diffusion_model:
                remapped = {}
                for k, v in state_dict.items():
                    if not k.startswith("model.diffusion_model."):
                        continue
                    new_key = "model." + k[len("model.diffusion_model."):]
                    if any(new_key.startswith(p) for p in remapped_non_transformer_prefixes):
                        continue
                    remapped[new_key] = v
                return remapped

            first_key = next(iter(state_dict))
            if first_key.startswith("model.velocity_model."):
                return {
                    "model." + k[len("model.velocity_model."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("model.velocity_model.")
                }
            if first_key.startswith("model."):
                return {
                    k: v for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in non_transformer_prefixes)
                }
            return {
                "model." + k: v
                for k, v in state_dict.items()
                if not any(k.startswith(p) for p in non_transformer_prefixes)
            }

        def _build_wrapper(use_causal: bool):
            if use_causal:
                _init_log("build causal wrapper start")
                causal_config = CausalLTXModelConfig(
                    num_frame_per_block=self.num_frame_per_block,
                    first_block_video_frames=self.first_block_video_frames,
                    enable_causal_log_rescale=getattr(args, "enable_causal_log_rescale", False),
                    num_audio_sink_tokens=getattr(args, "num_audio_sink_tokens", 0),
                )
                model = CausalLTXModel(causal_config).to(device=target_device, dtype=self.dtype)
                wrapper = CausalLTX2DiffusionWrapper(
                    model=model,
                    video_height=video_height,
                    video_width=video_width,
                    num_frame_per_block=self.num_frame_per_block,
                    disable_causal_mask=getattr(args, "disable_causal_mask", False),
                    num_audio_sink_tokens=getattr(args, "num_audio_sink_tokens", 0),
                    first_block_video_frames=self.first_block_video_frames,
                )
                state_dict = _remap_state_dict_keys(
                    _load_checkpoint_state_dict(args.checkpoint_path)
                )
                _init_log("load causal wrapper base state done")
                missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
                real_missing = [
                    k for k in missing
                    if "mask_builder" not in k and "audio_sink_tokens" not in k and "causal_gate" not in k
                ]
                if real_missing:
                    print(
                        f"[Stage3] Causal init from {args.checkpoint_path}: "
                        f"missing={len(real_missing)} unexpected={len(unexpected)}"
                    )
                _init_log("build causal wrapper done")
                return wrapper
            _init_log("build bidirectional wrapper start")
            return create_ltx2_wrapper(
                checkpoint_path=args.checkpoint_path,
                gemma_path=args.gemma_path,
                device=self.device,
                dtype=self.dtype,
                video_height=video_height,
                video_width=video_width,
                registry=shared_registry,
            )

        checkpoint_state_cache: Dict[tuple, dict] = {}
        shared_registry = StateDictRegistry()
        _init_log("generator wrapper init start")
        self.generator = _build_wrapper(self.generator_use_causal_wrapper)
        _init_log("generator wrapper init done")
        _init_log("real_score wrapper init start")
        self.real_score = _build_wrapper(self.real_score_use_causal_wrapper)
        _init_log("real_score wrapper init done")
        _init_log("fake_score wrapper init start")
        self.fake_score = _build_wrapper(self.fake_score_use_causal_wrapper)
        _init_log("fake_score wrapper init done")

        # The causal wrappers have copied the official base weights already.
        # Drop the direct safetensors mapping before loading text/VAE and the
        # much larger Stage-0/2 handoff checkpoints.
        checkpoint_state_cache.clear()
        gc.collect()

        _init_log("text encoder init start")
        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=args.checkpoint_path,
            gemma_path=args.gemma_path,
            device=self.device,
            dtype=self.dtype,
            registry=shared_registry,
        )
        _init_log("text encoder init done")

        _init_log("vae init start")
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=args.checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            registry=shared_registry,
        )
        _init_log("vae init done")

        # Set gradients
        self.generator.set_module_grad(args.generator_grad)
        self.real_score.set_module_grad(args.real_score_grad)
        self.fake_score.set_module_grad(args.fake_score_grad)
        self.text_encoder.requires_grad_(False)
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # Checkpoint loading with priority:
        #   resume_checkpoint > generator_ckpt > stage1_ckpt_path
        stage1_ckpt = getattr(args, "stage1_ckpt_path", None)
        stage1_strict = getattr(args, "stage1_ckpt_strict", False)
        generator_ckpt = getattr(args, "generator_ckpt", None)
        generator_ckpt_strict = getattr(args, "generator_ckpt_strict", False)

        if generator_ckpt:
            print(f"Loading pretrained generator from {generator_ckpt}")
            generator_state_key = getattr(
                args,
                "generator_checkpoint_state_key",
                None,
            )
            gen_sd = _load_checkpoint_state_dict(
                generator_ckpt,
                state_key=generator_state_key,
            )
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys(gen_sd)
            missing_g, unexpected_g = self.generator.load_state_dict(gen_sd, strict=generator_ckpt_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            sink_key = None
            for k in gen_sd:
                if "audio_sink_tokens" in k:
                    sink_key = k
                    break
            if sink_key is not None:
                for pname, param in self.generator.named_parameters():
                    if "audio_sink_tokens" in pname:
                        assert param.shape == gen_sd[sink_key].shape, (
                            f"[Stage3] Sink token shape mismatch in generator: "
                            f"model={param.shape} vs ckpt={gen_sd[sink_key].shape}"
                        )
                        break
            print("[Stage3] Generator checkpoint load complete")
            del gen_sd
            gc.collect()

        elif stage1_ckpt:
            print(f"[Stage2] Loading Stage 1 checkpoint from {stage1_ckpt}")
            gen_sd = _load_checkpoint_state_dict(
                stage1_ckpt,
                state_key=getattr(args, "stage1_checkpoint_state_key", None),
            )
            # Stage 3 configs may point stage1_ckpt_path at either:
            # 1. a causal/ODE checkpoint already keyed as model.*
            # 2. a bidirectional DMD checkpoint keyed as model.velocity_model.*
            # The causal generator expects model.* keys, so remap before load.
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys(gen_sd)
            missing_g, unexpected_g = self.generator.load_state_dict(gen_sd, strict=stage1_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            # CausVid-style hybrid setup: only load Stage1 ckpt into fake_score
            # when fake_score itself is causal.
            if self.fake_score_use_causal_wrapper:
                missing_f, unexpected_f = self.fake_score.load_state_dict(gen_sd, strict=stage1_strict)
                real_missing_f = [k for k in missing_f if "mask_builder" not in k]
                if real_missing_f:
                    print(f"  [fake_score] missing keys ({len(real_missing_f)}): {real_missing_f[:10]}...")
                if unexpected_f:
                    print(f"  [fake_score] unexpected keys ({len(unexpected_f)}): {unexpected_f[:10]}...")
            else:
                print("[Stage2] fake_score is bidirectional, skip Stage1 causal ckpt load for fake_score")

            # Validate sink token shape consistency
            sink_key = None
            for k in gen_sd:
                if "audio_sink_tokens" in k:
                    sink_key = k
                    break
            if sink_key is not None:
                models_to_check = [("generator", self.generator)]
                if self.fake_score_use_causal_wrapper:
                    models_to_check.append(("fake_score", self.fake_score))
                for name, model in models_to_check:
                    for pname, param in model.named_parameters():
                        if "audio_sink_tokens" in pname:
                            assert param.shape == gen_sd[sink_key].shape, (
                                f"[Stage2] Sink token shape mismatch in {name}: "
                                f"model={param.shape} vs ckpt={gen_sd[sink_key].shape}"
                            )
                            break
            print("[Stage2] Stage1 checkpoint load complete")
            del gen_sd
            gc.collect()

        # ------------------------------------------------------------------ #
        # Optional pluggable bidirectional teacher for real/fake score.       #
        # By default real_score/fake_score are initialized from the base LTX  #
        # checkpoint (args.checkpoint_path) inside _build_wrapper. If a        #
        # bidirectional teacher checkpoint is provided (e.g. an SFT'd LTX-2    #
        # bidirectional model), load it into real_score and fake_score here.  #
        # Per-model overrides take precedence over the shared key.            #
        # ------------------------------------------------------------------ #
        shared_teacher_ckpt = getattr(args, "bidirectional_teacher_checkpoint", None)
        real_init_ckpt = getattr(args, "real_score_init_checkpoint", None) or shared_teacher_ckpt
        fake_init_ckpt = getattr(args, "fake_score_init_checkpoint", None) or shared_teacher_ckpt
        teacher_strict = bool(getattr(args, "bidirectional_teacher_strict", False))

        real_state_key = getattr(args, "real_score_init_state_key", None)
        fake_state_key = getattr(args, "fake_score_init_state_key", None)

        def _load_teacher_into(
            module,
            ckpt_path,
            name,
            *,
            state_key,
            use_causal_wrapper,
        ):
            print(
                f"[Teacher] Loading {name} from {ckpt_path} "
                f"state_key={state_key!r}"
            )
            sd = _load_checkpoint_state_dict(ckpt_path, state_key=state_key)
            if use_causal_wrapper:
                sd = _remap_state_dict_keys(sd)
            missing, unexpected = module.load_state_dict(sd, strict=teacher_strict)
            real_missing = [k for k in missing if "mask_builder" not in k]
            if real_missing:
                print(f"  [{name}] missing keys ({len(real_missing)}): {real_missing[:10]}...")
            if unexpected:
                print(f"  [{name}] unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
            print(f"[Teacher] {name} teacher load complete")
            del sd
            gc.collect()

        if real_init_ckpt:
            _load_teacher_into(
                self.real_score,
                real_init_ckpt,
                "real_score",
                state_key=real_state_key,
                use_causal_wrapper=self.real_score_use_causal_wrapper,
            )
        if fake_init_ckpt:
            _load_teacher_into(
                self.fake_score,
                fake_init_ckpt,
                "fake_score",
                state_key=fake_state_key,
                use_causal_wrapper=self.fake_score_use_causal_wrapper,
            )

        checkpoint_state_cache.clear()
        gc.collect()

    def _round_align(self, value: float) -> int:
        if self.alignment_rounding == "floor":
            return int(torch.floor(torch.tensor(value)).item())
        if self.alignment_rounding == "ceil":
            return int(torch.ceil(torch.tensor(value)).item())
        return int(round(value))

    @staticmethod
    def _is_bidirectional_task(task_type: Optional[str]) -> bool:
        return "bidirectional" in str(task_type).lower()

    @staticmethod
    def _is_causal_task(task_type: Optional[str]) -> bool:
        return "causal" in str(task_type).lower()

    def _get_causal_blocks(self, num_video_frames: int):
        if compute_av_blocks is None:
            raise ImportError("Causal block utilities require the ltx-causal package")
        return compute_av_blocks(
            total_video_latent_frames=num_video_frames,
            num_frame_per_block=self.num_frame_per_block,
            first_block_video_frames=getattr(self, 'first_block_video_frames', 1),
        )

    def _build_current_block_masks(
        self,
        num_video_frames: int,
        num_audio_frames: int,
        block_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        blocks = self._get_causal_blocks(num_video_frames)
        batch_size = block_indices.shape[0]

        video_mask = torch.zeros(
            batch_size, num_video_frames, device=block_indices.device, dtype=torch.bool
        )
        audio_mask = torch.zeros(
            batch_size, num_audio_frames, device=block_indices.device, dtype=torch.bool
        )

        for batch_idx, block_idx in enumerate(block_indices.tolist()):
            block = blocks[block_idx]
            video_mask[batch_idx, block.video_start:block.video_end] = True
            audio_end = min(block.audio_end, num_audio_frames)
            if audio_end > block.audio_start:
                audio_mask[batch_idx, block.audio_start:audio_end] = True

        return video_mask, audio_mask

    def _sample_causal_training_blocks(
        self,
        batch_size: int,
        num_video_frames: int,
    ) -> torch.Tensor:
        blocks = self._get_causal_blocks(num_video_frames)
        if len(blocks) <= 1:
            raise ValueError(
                f"Causal training requires at least one standard block, got {num_video_frames} video frames"
            )
        return torch.randint(
            1,
            len(blocks),
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def _sample_causal_supervision_timesteps(
        self,
        batch_size: int,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_timestep = torch.zeros(
            video_mask.shape,
            device=self.device,
            dtype=torch.long,
        )
        audio_timestep = torch.zeros(
            audio_mask.shape,
            device=self.device,
            dtype=torch.long,
        )

        sampled = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        for batch_idx in range(batch_size):
            video_timestep[batch_idx, video_mask[batch_idx]] = sampled[batch_idx]
            audio_timestep[batch_idx, audio_mask[batch_idx]] = sampled[batch_idx]

        return video_timestep, audio_timestep

    def _prepare_causal_generator_inputs(
        self,
        clean_video: torch.Tensor,
        clean_audio: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_video_frames = clean_video.shape[:2]
        num_audio_frames = clean_audio.shape[1] if clean_audio is not None else 0

        block_indices = self._sample_causal_training_blocks(batch_size, num_video_frames)
        video_mask, audio_mask = self._build_current_block_masks(
            num_video_frames=num_video_frames,
            num_audio_frames=num_audio_frames,
            block_indices=block_indices,
        )

        num_steps = len(self.denoising_sigmas)
        if num_steps > 1:
            sampled_indices = torch.randint(
                0,
                num_steps - 1,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            sampled_indices = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        sampled_sigmas = self.denoising_sigmas[sampled_indices]

        video_sigma = torch.zeros(
            batch_size, num_video_frames, device=self.device, dtype=self.denoising_sigmas.dtype
        )
        for batch_idx in range(batch_size):
            video_sigma[batch_idx, video_mask[batch_idx]] = sampled_sigmas[batch_idx]

        noise_video = torch.randn_like(clean_video)
        noisy_video = self.add_noise(
            clean_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            video_sigma.flatten(0, 1),
        ).unflatten(0, (batch_size, num_video_frames))

        if clean_audio is None:
            return noisy_video, None, video_sigma, None, video_mask, None

        audio_sigma = torch.zeros(
            batch_size, num_audio_frames, device=self.device, dtype=self.denoising_sigmas.dtype
        )
        for batch_idx in range(batch_size):
            audio_sigma[batch_idx, audio_mask[batch_idx]] = sampled_sigmas[batch_idx]

        noise_audio = torch.randn_like(clean_audio)
        noisy_audio = self.add_noise(clean_audio, noise_audio, audio_sigma)

        return noisy_video, noisy_audio, video_sigma, audio_sigma, video_mask, audio_mask

    def _sample_synced_int(self, min_value: int, max_value: int) -> int:
        if min_value > max_value:
            raise ValueError(f"Invalid synced sampling range [{min_value}, {max_value}]")
        if dist.is_initialized():
            if dist.get_rank() == 0:
                sampled = torch.randint(
                    min_value,
                    max_value + 1,
                    (1,),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                sampled = torch.empty((1,), device=self.device, dtype=torch.long)
            dist.broadcast(sampled, src=0)
            return int(sampled.item())
        return int(
            torch.randint(
                min_value,
                max_value + 1,
                (1,),
                device=self.device,
                dtype=torch.long,
            ).item()
        )

    def _get_self_forcing_rollout_blocks(self, num_video_frames: int):
        blocks = self._get_causal_blocks(num_video_frames)
        if not blocks:
            raise ValueError(
                f"Self-forcing requires at least one causal block, got {num_video_frames} video frames"
            )

        total_blocks = len(blocks)
        max_cfg = self.self_forcing_max_generated_blocks
        if max_cfg is None:
            max_blocks = total_blocks
        else:
            max_blocks = min(total_blocks, max(1, int(max_cfg)))

        min_cfg = self.self_forcing_min_generated_blocks
        if min_cfg is None:
            min_blocks = max_blocks
        else:
            min_blocks = min(max_blocks, max(1, int(min_cfg)))

        rollout_blocks = self._sample_synced_int(min_blocks, max_blocks)
        # Pure autoregressive rollout: block 0 is generated as well.
        return blocks[:rollout_blocks]

    def _build_masks_for_blocks(
        self,
        batch_size: int,
        num_video_frames: int,
        num_audio_frames: int,
        blocks,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        video_mask = torch.zeros(
            batch_size, num_video_frames, device=self.device, dtype=torch.bool
        )
        audio_mask = None
        if num_audio_frames > 0:
            audio_mask = torch.zeros(
                batch_size, num_audio_frames, device=self.device, dtype=torch.bool
            )

        for block in blocks:
            video_mask[:, block.video_start:block.video_end] = True
            if audio_mask is not None:
                audio_end = min(block.audio_end, num_audio_frames)
                if audio_end > block.audio_start:
                    audio_mask[:, block.audio_start:audio_end] = True

        return video_mask, audio_mask

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    def _renoise_block(self, clean_block: Optional[torch.Tensor], next_sigma: torch.Tensor) -> Optional[torch.Tensor]:
        if clean_block is None:
            return None
        sigma = next_sigma.to(device=clean_block.device, dtype=clean_block.dtype).expand(
            clean_block.shape[0], clean_block.shape[1]
        )
        return self.add_noise(clean_block, torch.randn_like(clean_block), sigma)

    # ================================================================
    # KV-cache Self-Forcing Rollout (Phase 5, default path)
    # ================================================================

    def _init_self_forcing_kv_caches(
        self,
        batch_size: int,
        num_video_frames: int,
        num_audio_frames: int,
        text_seq_len: int,
        video_frame_seqlen: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Allocate per-layer KV caches sized for the entire rollout.

        We bypass FSDP / DDP wrappers and call ``init_av_kv_caches`` directly
        on the unwrapped causal wrapper because it only allocates buffers
        (no model forward, no parameter all-gather).

        ``video_frame_seqlen`` is the runtime tokens-per-frame (= latent H * W).
        It MUST be passed so the cache buffer matches the actual resolution;
        otherwise it falls back to ``model.config.video_frame_seqlen`` (the
        default resolution) and the per-block K/V write will size-mismatch.
        """
        wrapper = self._unwrap_module(self.generator)
        init_fn = getattr(wrapper, "init_av_kv_caches", None)
        if init_fn is None:
            raise RuntimeError(
                "Generator wrapper does not expose init_av_kv_caches; "
                "ensure the causal wrapper is built from CausalLTX2DiffusionWrapper "
                "with KV-cache support (Phase 4)."
            )
        return init_fn(
            batch_size=batch_size,
            num_video_frames=num_video_frames,
            num_audio_frames=num_audio_frames,
            text_seq_len=text_seq_len,
            device=self.device,
            dtype=self.dtype,
            video_frame_seqlen=video_frame_seqlen,
        )

    def _generate_exit_indices(self, num_blocks: int, num_denoising_steps: int) -> List[int]:
        """Sample one exit denoising step per block, synchronized across ranks."""
        if self.same_step_across_blocks:
            if not dist.is_initialized() or dist.get_rank() == 0:
                idx = torch.randint(
                    low=0,
                    high=num_denoising_steps,
                    size=(1,),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                idx = torch.empty((1,), device=self.device, dtype=torch.long)
            if dist.is_initialized():
                dist.broadcast(idx, src=0)
            return [int(idx.item())] * num_blocks

        if not dist.is_initialized() or dist.get_rank() == 0:
            idx = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            idx = torch.empty((num_blocks,), device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return idx.tolist()

    def _kv_block_forward(
        self,
        *,
        noisy_video_block: torch.Tensor,
        noisy_audio_block: Optional[torch.Tensor],
        timestep_block: torch.Tensor,
        audio_timestep_block: Optional[torch.Tensor],
        kv_caches: List[Dict[str, Any]],
        block,
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One block forward through the wrapper's KV-cache codepath.

        Returns x0 predictions for the block. The caller controls the grad
        context (no_grad for prefix-fill / non-last blocks / refresh, grad-
        enabled for the last block's denoising steps).
        """
        return self.generator(
            _kv_cache_kwargs=dict(
                noisy_video_block=noisy_video_block,
                noisy_audio_block=noisy_audio_block,
                conditional_dict=conditional_dict,
                timestep_block=timestep_block,
                audio_timestep_block=audio_timestep_block,
                kv_caches=kv_caches,
                current_video_start_frame=block.video_start,
                current_audio_start_frame=block.audio_start,
            )
        )

    def _kv_prefill_clean_block(
        self,
        *,
        clean_video_block: torch.Tensor,
        clean_audio_block: Optional[torch.Tensor],
        kv_caches: List[Dict[str, Any]],
        block,
        conditional_dict: Dict[str, Any],
    ) -> None:
        """Populate cache slots for a clean prefix block (sigma=0, no_grad)."""
        batch_size = clean_video_block.shape[0]
        sigma_dtype = self.denoising_sigmas.dtype
        video_sigma = torch.zeros(
            (batch_size, clean_video_block.shape[1]),
            device=self.device, dtype=sigma_dtype,
        )
        audio_sigma = None
        if clean_audio_block is not None:
            audio_sigma = torch.zeros(
                (batch_size, clean_audio_block.shape[1]),
                device=self.device, dtype=sigma_dtype,
            )
        with torch.no_grad():
            self._kv_block_forward(
                noisy_video_block=clean_video_block,
                noisy_audio_block=clean_audio_block,
                timestep_block=video_sigma,
                audio_timestep_block=audio_sigma,
                kv_caches=kv_caches,
                block=block,
                conditional_dict=conditional_dict,
            )

    def _kv_refresh_cache_for_block(
        self,
        *,
        x0_video_block: torch.Tensor,
        x0_audio_block: Optional[torch.Tensor],
        kv_caches: List[Dict[str, Any]],
        block,
        conditional_dict: Dict[str, Any],
    ) -> None:
        """Run an extra no_grad forward to overwrite the block's cache slot
        with K/V from a normalised input ("clean x0 + sigma=ctx_t").

        This forward is ALWAYS run (even when context_noise == 0), to make
        the cache state deterministic and exactly match what inference will
        feed into downstream blocks. See KV_CACHE_MIGRATION_PLAN.md sec. 7.
        """
        batch_size = x0_video_block.shape[0]
        sigma_dtype = self.denoising_sigmas.dtype
        ctx_t = float(self.context_noise) / float(self.num_train_timestep)
        video_ctx_sigma = torch.full(
            (batch_size, x0_video_block.shape[1]),
            ctx_t, device=self.device, dtype=sigma_dtype,
        )
        audio_ctx_sigma = None
        if x0_audio_block is not None:
            audio_ctx_sigma = torch.full(
                (batch_size, x0_audio_block.shape[1]),
                ctx_t, device=self.device, dtype=sigma_dtype,
            )

        if ctx_t > 0.0:
            ctx_video = self.add_noise(
                x0_video_block.detach(),
                torch.randn_like(x0_video_block),
                video_ctx_sigma,
            )
            ctx_audio = None
            if x0_audio_block is not None:
                ctx_audio = self.add_noise(
                    x0_audio_block.detach(),
                    torch.randn_like(x0_audio_block),
                    audio_ctx_sigma,
                )
        else:
            ctx_video = x0_video_block.detach()
            ctx_audio = x0_audio_block.detach() if x0_audio_block is not None else None

        with torch.no_grad():
            self._kv_block_forward(
                noisy_video_block=ctx_video,
                noisy_audio_block=ctx_audio,
                timestep_block=video_ctx_sigma,
                audio_timestep_block=audio_ctx_sigma,
                kv_caches=kv_caches,
                block=block,
                conditional_dict=conditional_dict,
            )

    def _kv_denoise_block(
        self,
        *,
        block,
        kv_caches: List[Dict[str, Any]],
        conditional_dict: Dict[str, Any],
        prev_video_ref: torch.Tensor,
        prev_audio_ref: Optional[torch.Tensor],
        requires_grad: bool,
        exit_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run multi-step DMD denoising for a single block under the KV cache.

        Each denoising step calls the wrapper at the same (current_video_start_frame,
        current_audio_start_frame) offset; cache writes overwrite the same slot
        so the final K/V correspond to the last denoising step's input.

        For the LAST rollout block (requires_grad=True) the entire denoising
        chain is autograd-tracked so loss can flow back through every step.

        When ``exit_index`` is provided, stop immediately after that model
        forward and return its x0 prediction. Otherwise return the full-
        schedule clean x0 prediction for the block.
        """
        batch_size = prev_video_ref.shape[0]
        sigma_dtype = self.denoising_sigmas.dtype
        video_tail_shape = prev_video_ref.shape[2:]
        num_denoising_steps = len(self.denoising_sigmas) - 1
        if exit_index is not None and not 0 <= int(exit_index) < num_denoising_steps:
            raise ValueError(
                f"exit_index must be in [0, {num_denoising_steps}), got "
                f"{exit_index}"
            )

        current_video = torch.randn(
            (batch_size, block.video_frames, *video_tail_shape),
            device=self.device, dtype=self.dtype,
        )
        current_audio = None
        if prev_audio_ref is not None:
            current_audio = torch.randn(
                (batch_size, block.audio_frames, prev_audio_ref.shape[2]),
                device=self.device, dtype=self.dtype,
            )

        grad_ctx = nullcontext() if requires_grad else torch.no_grad()
        with grad_ctx:
            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                video_sigma = sigma.to(device=self.device, dtype=sigma_dtype).expand(
                    batch_size, current_video.shape[1]
                )
                audio_sigma = None
                if current_audio is not None:
                    audio_sigma = sigma.to(device=self.device, dtype=sigma_dtype).expand(
                        batch_size, current_audio.shape[1]
                    )

                pred_video, pred_audio = self._kv_block_forward(
                    noisy_video_block=current_video,
                    noisy_audio_block=current_audio,
                    timestep_block=video_sigma,
                    audio_timestep_block=audio_sigma,
                    kv_caches=kv_caches,
                    block=block,
                    conditional_dict=conditional_dict,
                )

                if exit_index is not None and sigma_idx == int(exit_index):
                    return pred_video, pred_audio

                next_sigma = self.denoising_sigmas[sigma_idx + 1]
                if float(next_sigma.item()) > 0.0:
                    current_video = self._renoise_block(pred_video, next_sigma)
                    if current_audio is not None:
                        current_audio = self._renoise_block(pred_audio, next_sigma)
                else:
                    current_video = pred_video
                    current_audio = pred_audio if current_audio is not None else None

        return current_video, current_audio

    def _run_self_forcing_rollout_kv_cache(
        self,
        *,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        rollout_mode: str = "full",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Detached KV-cache Self-Forcing rollout used by fake-score training.

        Layout:
          1. Allocate ONE shared KV cache for the entire rollout
             for all generated blocks.
          2. For each rollout block b:
             - ``full`` runs the complete denoising schedule.
             - ``random_exit`` stops at the sampled exit forward.
             - Run a no_grad cache REFRESH forward to normalise the cache
               slot to the chosen x0 + sigma=ctx_t K/V, so the next block sees
               the same exit distribution.
          3. Concatenate the detached video/audio rollout tensors.

        Causality is enforced by the cache offsets: each block forward only
        sees [0, end) of the cache, where end = the previous block's write
        boundary.

        Returns:
            (rollout_video, rollout_audio, video_loss_mask, audio_loss_mask, log)
        """
        rollout_mode = str(rollout_mode).lower()
        if rollout_mode not in {"full", "random_exit"}:
            raise ValueError(
                f"rollout_mode must be full or random_exit, got {rollout_mode!r}"
            )

        total_video_frames = int(video_shape[1])
        rollout_blocks = self._get_self_forcing_rollout_blocks(total_video_frames)
        if not rollout_blocks:
            raise ValueError("Self-forcing rollout sampled zero blocks")
        num_denoising_steps = len(self.denoising_sigmas) - 1
        if num_denoising_steps <= 0:
            raise ValueError("denoising_sigmas must have at least 2 entries")
        if rollout_mode == "random_exit":
            exit_indices = self._generate_exit_indices(
                len(rollout_blocks), num_denoising_steps
            )
        else:
            exit_indices = [num_denoising_steps - 1] * len(rollout_blocks)

        # Total cache capacity covers all sampled blocks.
        f_v_total = rollout_blocks[-1].video_end
        f_a_total = rollout_blocks[-1].audio_end if audio_shape is not None else 0
        text_seq_len = conditional_dict["video_context"].shape[1]
        batch_size = int(video_shape[0])

        kv_caches = self._init_self_forcing_kv_caches(
            batch_size=batch_size,
            num_video_frames=f_v_total,
            num_audio_frames=f_a_total,
            text_seq_len=text_seq_len,
            video_frame_seqlen=int(video_shape[-2]) * int(video_shape[-1]),
        )

        # === Roll out all sampled blocks from noise ===
        rollout_video = torch.zeros(
            (batch_size, f_v_total, *video_shape[2:]),
            device=self.device, dtype=self.dtype,
        )
        rollout_audio = None
        if audio_shape is not None:
            rollout_audio = torch.zeros(
                (batch_size, f_a_total, audio_shape[2]),
                device=self.device, dtype=self.dtype,
            )

        for idx, block in enumerate(rollout_blocks):
            x0_video, x0_audio = self._kv_denoise_block(
                block=block,
                kv_caches=kv_caches,
                conditional_dict=conditional_dict,
                prev_video_ref=rollout_video,
                prev_audio_ref=rollout_audio,
                requires_grad=False,
                exit_index=(exit_indices[idx] if rollout_mode == "random_exit" else None),
            )

            self._kv_refresh_cache_for_block(
                x0_video_block=x0_video.detach(),
                x0_audio_block=x0_audio.detach() if x0_audio is not None else None,
                kv_caches=kv_caches,
                block=block,
                conditional_dict=conditional_dict,
            )

            rollout_video[:, block.video_start:block.video_end] = x0_video.detach()
            if rollout_audio is not None and x0_audio is not None:
                rollout_audio[:, block.audio_start:block.audio_end] = x0_audio.detach()

        # === 3. Build loss masks for the LAST block ===
        num_audio_frames = rollout_audio.shape[1] if rollout_audio is not None else 0
        video_loss_mask, audio_loss_mask = self._build_masks_for_blocks(
            batch_size=rollout_video.shape[0],
            num_video_frames=rollout_video.shape[1],
            num_audio_frames=num_audio_frames,
            blocks=[rollout_blocks[-1]],
        )

        rollout_log = {
            "self_forcing_rollout_blocks": len(rollout_blocks),
            "self_forcing_rollout_video_frames": f_v_total,
            "self_forcing_rollout_audio_frames": num_audio_frames,
            "self_forcing_runtime": 1,  # 1=kv_cache (prefix_rerun retired)
            "self_forcing_context_noise": self.context_noise,
            "self_forcing_rollout_random_exit": int(rollout_mode == "random_exit"),
            "self_forcing_rollout_exit_index_mean": float(
                sum(exit_indices) / len(exit_indices)
            ),
            "self_forcing_rollout_steps_mean": float(
                sum(index + 1 for index in exit_indices) / len(exit_indices)
            ),
        }
        return rollout_video, rollout_audio, video_loss_mask, audio_loss_mask, rollout_log

    def _process_timestep(self, timestep: torch.Tensor, task_type: str) -> torch.Tensor:
        """
        Process timestep based on task type.

        For causal tasks, each block of num_frame_per_block frames shares the
        same timestep (noise level), matching CausVid semantics.

        Args:
            timestep: [B, F] tensor of timesteps
            task_type: "bidirectional_av", "bidirectional_video", "causal_av", etc.

        Returns:
            Processed timestep tensor
        """
        if self._is_bidirectional_task(task_type):
            for i in range(timestep.shape[0]):
                timestep[i] = timestep[i, 0]
            return timestep
        elif "causal" in task_type:
            result = timestep.clone()
            if result.shape[1] <= 1:
                return result
            fb = min(self.first_block_video_frames, result.shape[1])
            result[:, :fb] = result[:, 0:1].expand(-1, fb)
            idx = fb
            while idx < result.shape[1]:
                end = min(idx + self.num_frame_per_block, result.shape[1])
                result[:, idx:end] = result[:, idx:idx + 1].expand(-1, end - idx)
                idx = end
            return result
        else:
            return timestep

    def _compute_audio_timestep(
        self,
        video_timestep: torch.Tensor,
        num_audio_frames: int,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute audio timestep from video timestep.

        In bidirectional mode, all frames use the same timestep.
        In causal mode, audio frames inherit the timestep from their
        corresponding video block via the AV alignment ratio.
        """
        B = video_timestep.shape[0]
        num_video_frames = video_timestep.shape[1]
        mode = task_type or self.real_task_type

        if self._is_bidirectional_task(mode):
            return video_timestep[:, 0:1].expand(B, num_audio_frames)

        # Causal/non-bidirectional: map audio blocks to the video block sigma
        # defined by the causal wrapper's Global Prefix schedule.
        audio_timestep = torch.zeros(
            B, num_audio_frames, device=video_timestep.device, dtype=video_timestep.dtype
        )
        for block in self._get_causal_blocks(num_video_frames):
            if block.audio_start >= num_audio_frames:
                break
            audio_end = min(block.audio_end, num_audio_frames)
            if audio_end <= block.audio_start:
                continue
            audio_timestep[:, block.audio_start:audio_end] = video_timestep[
                :, block.video_start:block.video_start + 1
            ].expand(B, audio_end - block.audio_start)
        return audio_timestep

    @torch.no_grad()
    def _teacher_denoise_cfg_step(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One teacher denoising step with classifier-free guidance.

        Calls real_score twice (conditional + unconditional) and applies
        LTX-2's CFG formula with separate video/audio guidance scales.

        Returns:
            Tuple of (pred_video_x0, pred_audio_x0)
        """
        pred_cond_video, pred_cond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        pred_uncond_video, pred_uncond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        # CFG: output = cond + (scale - 1) * (cond - uncond)
        pred_video_x0 = pred_cond_video + (self.real_video_guidance_scale - 1) * (
            pred_cond_video - pred_uncond_video
        )
        pred_audio_x0 = pred_cond_audio + (self.real_audio_guidance_scale - 1) * (
            pred_cond_audio - pred_uncond_audio
        )

        return pred_video_x0, pred_audio_x0

    @torch.no_grad()
    def _get_noisy_latent_via_teacher_denoise(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        target_video_sigma: torch.Tensor,
        target_audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher denoises from high noise to target sigma, producing latents
        on the teacher's ODE trajectory instead of the Gaussian interpolation line.

        The number of Euler steps is determined solely by where target_sigma
        falls in the pre-computed teacher_sigmas schedule — no randomization.

        Args:
            clean_video: Generator-predicted clean video [B, F_v, C, H, W]
            clean_audio: Generator-predicted clean audio [B, F_a, C]
            target_video_sigma: Target sigma [B, F_v]
            target_audio_sigma: Target sigma [B, F_a]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings

        Returns:
            (noisy_video, noisy_audio, actual_video_sigma, actual_audio_sigma)
            where actual sigmas are the exact teacher schedule values at the
            stop point, guaranteed to match the returned latents' noise level.
        """
        B = clean_video.shape[0]
        F_v = clean_video.shape[1]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        teacher_sigmas = self.teacher_sigmas  # [N+1], descending: [0]≈1.0, [-1]=0.0
        offset = self.teacher_start_offset

        # In bidirectional mode all frames share the same sigma
        target_scalar = target_video_sigma[:, 0]  # [B]

        # Find the closest index in teacher_sigmas for each batch element
        target_idx = torch.argmin(
            (teacher_sigmas.unsqueeze(0) - target_scalar.unsqueeze(1)).abs(),
            dim=1,
        )  # [B]
        target_idx = target_idx.clamp(min=1)  # at least 1 denoising step

        # Start index: `offset` steps before target in the schedule.
        # start_idx < target_idx, so teacher_sigmas[start_idx] > teacher_sigmas[target_idx].
        # Clamped to 0 so we never go before the schedule start.
        start_idx = (target_idx - offset).clamp(min=0)  # [B]
        num_steps = target_idx - start_idx  # [B], exactly how many Euler steps each element needs

        # NCCL safety: all ranks must call real_score() the same number of times.
        max_steps = num_steps.max().item()
        if dist.is_initialized():
            max_steps_tensor = torch.tensor(max_steps, device=device, dtype=torch.long)
            dist.all_reduce(max_steps_tensor, op=dist.ReduceOp.MAX)
            max_steps = max_steps_tensor.item()

        # Add noise at each element's start sigma (not necessarily pure noise)
        noise_video = torch.randn_like(clean_video)
        noise_audio = torch.randn_like(clean_audio)

        start_sigma_per_elem = teacher_sigmas[start_idx]  # [B]
        s_v = start_sigma_per_elem.unsqueeze(1).expand(B, F_v)
        s_a = start_sigma_per_elem.unsqueeze(1).expand(B, F_a)

        current_video = self.add_noise(
            clean_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            s_v.flatten(0, 1),
        ).unflatten(0, (B, F_v))
        current_audio = self.add_noise(clean_audio, noise_audio, s_a)

        # Teacher Euler denoising: each element runs from its own start_idx to target_idx.
        # We iterate max_steps times; each element's absolute schedule index is start_idx + step_i.
        for step_i in range(max_steps):
            active = (step_i < num_steps)  # [B]

            # Each element may be at a different position in the schedule
            abs_idx = start_idx + step_i  # [B]
            cur_sigma = teacher_sigmas[abs_idx]    # [B]
            nxt_sigma = teacher_sigmas[(abs_idx + 1).clamp(max=len(teacher_sigmas) - 1)]  # [B]

            v_sigma = cur_sigma.unsqueeze(1).expand(B, F_v)
            a_sigma = cur_sigma.unsqueeze(1).expand(B, F_a)

            pred_v_x0, pred_a_x0 = self._teacher_denoise_cfg_step(
                noisy_video=current_video,
                noisy_audio=current_audio,
                video_sigma=v_sigma,
                audio_sigma=a_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

            # Euler step: v = (x_t - x_0) / sigma, x_{t'} = x_t + v * (sigma' - sigma)
            cur_sigma_bcast = cur_sigma.view(B, 1, 1, 1, 1)
            dsigma = (nxt_sigma - cur_sigma).view(B, 1, 1, 1, 1)
            vel_v = (current_video - pred_v_x0) / cur_sigma_bcast
            nxt_v = current_video + vel_v * dsigma

            cur_sigma_audio = cur_sigma.view(B, 1, 1)
            dsigma_audio = (nxt_sigma - cur_sigma).view(B, 1, 1)
            vel_a = (current_audio - pred_a_x0) / cur_sigma_audio
            nxt_a = current_audio + vel_a * dsigma_audio

            m_v = active.view(B, 1, 1, 1, 1).expand_as(current_video)
            m_a = active.view(B, 1, 1).expand_as(current_audio)
            current_video = torch.where(m_v, nxt_v, current_video)
            current_audio = torch.where(m_a, nxt_a, current_audio)

        # Actual sigma at the exact stopping point (from the teacher schedule)
        actual_sigma = teacher_sigmas[target_idx]  # [B]
        actual_video_sigma = actual_sigma.unsqueeze(1).expand(B, F_v)
        actual_audio_sigma = actual_sigma.unsqueeze(1).expand(B, F_a)

        return current_video, current_audio, actual_video_sigma, actual_audio_sigma

    def _score_model_forward(
        self,
        *,
        model: torch.nn.Module,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_video_context: torch.Tensor,
        clean_audio_context: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default full-clip score forward; AR subclasses override this hook."""
        del clean_video_context, clean_audio_context
        return model(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

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
        """
        Compute KL gradient for both video and audio.

        This implements Equation 7 from the DMD paper.

        Args:
            video_sigma: Noise level sigma [B, F_v], passed directly to score networks.
            audio_sigma: Noise level sigma [B, F_a], passed directly to score networks.
        """
        # Step 1: Fake score prediction
        pred_fake_video, pred_fake_audio = self._score_model_forward(
            model=self.fake_score,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=conditional_dict,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        # Step 2: Real score prediction with CFG
        pred_real_cond_video, pred_real_cond_audio = self._score_model_forward(
            model=self.real_score,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=conditional_dict,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        pred_real_uncond_video, pred_real_uncond_audio = self._score_model_forward(
            model=self.real_score,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video_context=clean_video,
            clean_audio_context=clean_audio,
            conditional_dict=unconditional_dict,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        # Apply CFG: output = cond + (scale - 1) * (cond - uncond)
        # This matches LTX-2's native CFGGuider.delta = (scale - 1) * (cond - uncond)
        # With video_scale=3.0: effective = 3.0*cond - 2.0*uncond
        # With audio_scale=7.0: effective = 7.0*cond - 6.0*uncond
        real_video_guidance_scale = self.sample_real_video_guidance_scale(
            pred_real_cond_video.device
        )
        real_audio_guidance_scale = self.sample_real_audio_guidance_scale(
            pred_real_cond_audio.device
        )
        pred_real_video = pred_real_cond_video + (real_video_guidance_scale - 1) * (
            pred_real_cond_video - pred_real_uncond_video
        )
        pred_real_audio = pred_real_cond_audio + (real_audio_guidance_scale - 1) * (
            pred_real_cond_audio - pred_real_uncond_audio
        )

        # Step 3: Compute DMD gradient
        grad_video = pred_fake_video - pred_real_video
        grad_audio = pred_fake_audio - pred_real_audio

        # Step 4: Gradient normalization (Eq. 8)
        if normalization:
            # Video normalization
            p_real_video = clean_video - pred_real_video
            if video_loss_mask is not None:
                video_mask = video_loss_mask.to(p_real_video.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Count all active latent elements, not just active frames.
                video_active = video_mask.expand_as(p_real_video).sum(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1.0)
                normalizer_video = (torch.abs(p_real_video) * video_mask).sum(dim=[1, 2, 3, 4], keepdim=True)
                normalizer_video = normalizer_video / video_active
            else:
                normalizer_video = torch.abs(p_real_video).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad_video = grad_video / (normalizer_video + 1e-8)

            # Audio normalization
            p_real_audio = clean_audio - pred_real_audio
            if audio_loss_mask is not None:
                audio_mask = audio_loss_mask.to(p_real_audio.dtype).unsqueeze(-1)
                audio_active = audio_mask.expand_as(p_real_audio).sum(dim=[1, 2], keepdim=True).clamp_min(1.0)
                normalizer_audio = (torch.abs(p_real_audio) * audio_mask).sum(dim=[1, 2], keepdim=True)
                normalizer_audio = normalizer_audio / audio_active
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
            # Optional visualization-only payload.  Normal training callers
            # leave this disabled, so no prediction tensor is retained.
            log_dict["_pred_real_video_x0"] = pred_real_video.detach()
            log_dict["_pred_fake_video_x0"] = pred_fake_video.detach()

        return grad_video, grad_audio, log_dict

    def _compute_block_weights(self, num_frames: int, *, is_audio: bool = False) -> torch.Tensor:
        """
        Compute per-frame loss weights based on block position.

        For "linear_ramp", early blocks get lower weight (block_weight_min)
        ramping linearly to 1.0 at the last block.
        For "uniform" or "none", returns all-ones.

        Returns:
            Tensor [num_frames] of per-frame weights on self.device.
        """
        if self.block_weight_mode == "uniform" or self.block_weight_mode == "none":
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        if self._is_causal_task(self.generator_task_type):
            blocks = self._get_causal_blocks(
                math.ceil((num_frames - 1) / 25) * self.num_frame_per_block + 1
            ) if is_audio else self._get_causal_blocks(num_frames)
            if is_audio:
                blocks = [block for block in blocks if block.audio_start < num_frames]
            else:
                blocks = [block for block in blocks if block.video_start < num_frames]
            n_blocks = len(blocks)
        else:
            nfpb = self.num_frame_per_block
            n_blocks = math.ceil(num_frames / nfpb)
            blocks = None

        if n_blocks <= 1:
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        weights = torch.ones(num_frames, device=self.device, dtype=torch.float64)
        if blocks is not None:
            for blk_idx, block in enumerate(blocks):
                w = self.block_weight_min + (1.0 - self.block_weight_min) * blk_idx / (n_blocks - 1)
                start = block.audio_start if is_audio else block.video_start
                end = min(block.audio_end if is_audio else block.video_end, num_frames)
                weights[start:end] = w
        else:
            for blk in range(n_blocks):
                start = blk * nfpb
                end = min(start + nfpb, num_frames)
                w = self.block_weight_min + (1.0 - self.block_weight_min) * blk / (n_blocks - 1)
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

    def _compute_masked_denoising_loss(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: Optional[torch.Tensor],
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
                f"Masked causal critic loss does not support denoising_loss_type={loss_type}"
            )

        reduce_dims = tuple(range(2, diff.dim()))
        per_frame = diff.mean(dim=reduce_dims)
        return self._masked_weighted_mean(
            per_frame,
            torch.ones_like(per_frame, dtype=per_frame.dtype),
            mask,
        ).to(target.dtype)

    def compute_distribution_matching_loss(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the DMD loss for video and audio jointly.

        Supports block-aware per-frame weighting for causal over-exposure suppression
        and causal block-wise timestep unification.

        Args:
            video_latent: Clean video latent [B, F, C, H, W]
            audio_latent: Clean audio latent [B, F_a, C]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings

        Returns:
            Tuple of (total_loss, log_dict)
        """
        B, F_v = video_latent.shape[:2]
        F_a = audio_latent.shape[1]

        with torch.no_grad():
            if (
                self.dmd_timestep_sampler_type == "ltx_scheduler"
                and video_loss_mask is not None
                and audio_loss_mask is not None
            ):
                video_timestep, audio_timestep = self._sample_causal_supervision_timesteps(
                    B,
                    video_loss_mask,
                    audio_loss_mask,
                )
                video_sigma = self.timestep_to_sigma(video_timestep)
                audio_sigma = self.timestep_to_sigma(audio_timestep)
            else:
                video_sigma, audio_sigma, _, _ = self._sample_score_noise_levels(
                    batch_size=B,
                    num_video_frames=F_v,
                    num_audio_frames=F_a,
                    task_type=self.real_task_type,
                    fake_score=False,
                )

            if self.dmd_latent_mode == "teacher_denoise":
                noisy_video, noisy_audio, video_sigma, audio_sigma = \
                    self._get_noisy_latent_via_teacher_denoise(
                        clean_video=video_latent,
                        clean_audio=audio_latent,
                        target_video_sigma=video_sigma,
                        target_audio_sigma=audio_sigma,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                    )
            else:
                noise_video = torch.randn_like(video_latent)
                noise_audio = torch.randn_like(audio_latent)

                noisy_video = self.add_noise(
                    video_latent.flatten(0, 1),
                    noise_video.flatten(0, 1),
                    video_sigma.flatten(0, 1),
                ).unflatten(0, (B, F_v))

                noisy_audio = self.add_noise(
                    audio_latent,
                    noise_audio,
                    audio_sigma,
                )

            grad_video, grad_audio, log_dict = self._compute_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                clean_video=video_latent,
                clean_audio=audio_latent,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=video_loss_mask,
                audio_loss_mask=audio_loss_mask,
            )

        # Block-aware per-frame loss weighting (over-exposure suppression)
        video_block_w = self._compute_block_weights(F_v)  # [F_v]
        audio_block_w = self._compute_block_weights(F_a, is_audio=True)  # [F_a]

        # Per-frame MSE then weight
        video_diff = video_latent.double() - (video_latent.double() - grad_video.double()).detach()
        video_per_frame = (video_diff ** 2).mean(dim=[2, 3, 4])  # [B, F_v]
        video_loss = 0.5 * self._masked_weighted_mean(
            video_per_frame,
            video_block_w.unsqueeze(0),
            video_loss_mask,
        )

        audio_diff = audio_latent.double() - (audio_latent.double() - grad_audio.double()).detach()
        audio_per_frame = (audio_diff ** 2).mean(dim=2)  # [B, F_a]
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

    def _initialize_inference_pipeline(self):
        """Initialize the inference pipeline for backward simulation."""
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVTrajectoryPipeline

        self.inference_pipeline = BidirectionalAVTrajectoryPipeline(
            generator=self.generator,
            add_noise_fn=self.add_noise,
            denoising_sigmas=self.denoising_sigmas,
        )

    @torch.no_grad()
    def _consistency_backward_simulation(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Simulate generator input using backward simulation.

        Returns trajectory of noisy inputs at each denoising step.

        Note: The generator is temporarily switched to eval() mode during
        backward simulation. This disables gradient checkpointing, which
        would otherwise conflict with FSDP under torch.no_grad() (checkpoint
        requires grad-enabled tensors). After simulation, the generator is
        restored to train() mode so that gradient checkpointing remains
        active for the subsequent gradient-enabled forward pass — essential
        for the 19B model's memory footprint.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # Temporarily disable gradient checkpointing by switching to eval().
        # Under @torch.no_grad(), FSDP + gradient checkpointing conflicts
        # because checkpoint requires grad-enabled tensors.
        self.generator.eval()
        try:
            result = self.inference_pipeline.inference_with_trajectory(
                video_noise=video_noise,
                audio_noise=audio_noise,
                conditional_dict=conditional_dict,
            )
        finally:
            # Restore train() so gradient checkpointing is active for
            # the gradient-enabled generator forward pass that follows.
            self.generator.train()

        return result

    def _run_generator(
        self,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
        self_forcing_rollout_mode: str = "full",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Run generator with backward simulation.

        Returns predicted clean video and audio.
        """
        B = video_shape[0]
        F_v = video_shape[1]
        F_a = audio_shape[1]

        video_loss_mask = None
        audio_loss_mask = None
        rollout_log: Dict[str, Any] = {}

        if self.enable_self_forcing:
            pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log = (
                self._run_self_forcing_rollout_kv_cache(
                    video_shape=video_shape,
                    audio_shape=audio_shape,
                    conditional_dict=conditional_dict,
                    rollout_mode=self_forcing_rollout_mode,
                )
            )
            return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

        # Step 1: Backward simulation or ODE data
        if getattr(self.args, "backward_simulation", True):
            if self._is_causal_task(self.generator_task_type):
                raise NotImplementedError(
                    "Causal Stage-3 DMD currently requires backward_simulation=false so training "
                    "matches the clean-prefix/current-block-noisy inference distribution."
                )
            video_noise = torch.randn(video_shape, device=self.device, dtype=self.dtype)
            audio_noise = torch.randn(audio_shape, device=self.device, dtype=self.dtype)

            simulated_video, simulated_audio = self._consistency_backward_simulation(
                video_noise=video_noise,
                audio_noise=audio_noise,
                conditional_dict=conditional_dict,
            )
        else:
            if self._is_causal_task(self.generator_task_type):
                noisy_video, noisy_audio, video_sigma, audio_sigma, video_loss_mask, audio_loss_mask = (
                    self._prepare_causal_generator_inputs(
                        clean_video=clean_video,
                        clean_audio=clean_audio,
                    )
                )
                pred_video, pred_audio = self.generator(
                    noisy_image_or_video=noisy_video,
                    conditional_dict=conditional_dict,
                    timestep=video_sigma,
                    noisy_audio=noisy_audio,
                    audio_timestep=audio_sigma,
                    use_causal_timestep=False,
                )
                return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

            # Use provided clean latents
            simulated_video = []
            simulated_audio = []

            for sigma in self.denoising_sigmas:
                noise_v = torch.randn(video_shape, device=self.device, dtype=self.dtype)
                noise_a = torch.randn(audio_shape, device=self.device, dtype=self.dtype)

                sigma_tensor = sigma * torch.ones([B, F_v], device=self.device)
                sigma_tensor_a = sigma * torch.ones([B, F_a], device=self.device)

                if sigma > 0:
                    noisy_video = self.add_noise(
                        clean_video.flatten(0, 1),
                        noise_v.flatten(0, 1),
                        sigma_tensor.flatten(0, 1),
                    ).unflatten(0, (B, F_v))
                    noisy_audio = self.add_noise(clean_audio, noise_a, sigma_tensor_a)
                else:
                    noisy_video = clean_video
                    noisy_audio = clean_audio

                simulated_video.append(noisy_video)
                simulated_audio.append(noisy_audio)

            simulated_video = torch.stack(simulated_video, dim=1)
            simulated_audio = torch.stack(simulated_audio, dim=1)

        # Step 2: Random timestep selection
        num_steps = len(self.denoising_sigmas)
        index = torch.randint(0, num_steps, [B, F_v], device=self.device, dtype=torch.long)
        index = self._process_timestep(index, self.generator_task_type)
        if self._is_bidirectional_task(self.generator_task_type):
            # Keep the Stage-1 bidirectional path byte-for-byte aligned with
            # the 88fb145 DMD fix semantics: one shared step per sample across
            # all video and audio frames.
            noisy_video = torch.gather(
                simulated_video,
                dim=1,
                index=index[:, :1, None, None, None, None].expand(-1, -1, F_v, *video_shape[2:]),
            ).squeeze(1)
            noisy_audio = torch.gather(
                simulated_audio,
                dim=1,
                index=index[:, :1, None, None].expand(-1, -1, F_a, audio_shape[2]),
            ).squeeze(1)

            sigma = self.denoising_sigmas[index[:, 0]]
            video_sigma = sigma.unsqueeze(1).expand(B, F_v)
            audio_sigma = sigma.unsqueeze(1).expand(B, F_a)
        else:
            audio_index = self._compute_audio_timestep(
                index, F_a, task_type=self.generator_task_type
            ).clamp(0, num_steps - 1)

            noisy_video = torch.gather(
                simulated_video, dim=1,
                index=index.reshape(B, 1, F_v, 1, 1, 1).expand(-1, -1, -1, *video_shape[2:])
            ).squeeze(1)

            noisy_audio = torch.gather(
                simulated_audio, dim=1,
                index=audio_index.reshape(B, 1, F_a, 1).expand(-1, -1, -1, audio_shape[2])
            ).squeeze(1)

            # Step 3: Generator prediction (per-frame/per-audio-frame sigma)
            video_sigma = self.denoising_sigmas[index]
            audio_sigma = self.denoising_sigmas[audio_index]

        pred_video, pred_audio = self.generator(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

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
        """
        Compute generator loss using DMD.

        Args:
            video_shape: [B, F, C, H, W]
            audio_shape: [B, F_a, C]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings
            clean_video: Clean video latent (optional, for non-backward-simulation)
            clean_audio: Clean audio latent (optional)
            real_video_frames: pad_max variable-length rollout only. When the
                rollout is padded to a cross-rank global-max length, this is this
                rank's REAL latent video frame count; the DMD loss is masked to
                the first ``real_video_frames`` frames so padded frames/blocks
                contribute zero gradient. ``None`` => use the full rollout length.
            real_audio_frames: analogous real latent audio frame count.

        Returns:
            Tuple of (loss, log_dict)
        """
        if self.enable_self_forcing:
            return self._generator_loss_self_forcing_two_pass(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )

        # Run generator
        pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log = self._run_generator(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            clean_video=clean_video,
            clean_audio=clean_audio,
        )

        # Compute DMD loss
        dmd_loss, log_dict = self.compute_distribution_matching_loss(
            video_latent=pred_video,
            audio_latent=pred_audio,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )
        log_dict.update(rollout_log)

        return dmd_loss, log_dict

    # Optional extension points for objectives that reuse the memory-optimized
    # two-pass rollout. The base DMD path intentionally keeps these disabled.
    def _begin_shortcut_consistency(self, num_denoising_steps: int):
        return None

    def _build_shortcut_consistency_target(self, **kwargs):
        return None

    def _compute_shortcut_consistency_block_loss(self, **kwargs):
        return None

    def _finalize_shortcut_consistency_logs(
        self,
        context: Optional[Dict[str, Any]],
        log_dict: Dict[str, Any],
    ) -> None:
        return

    def _prepare_generator_auxiliary_gradients(self, **kwargs):
        """Optional detached-rollout objective (for example LADD)."""
        return None

    def _compute_generator_auxiliary_block_loss(self, **kwargs):
        return None

    def _finalize_generator_auxiliary_logs(
        self,
        context: Optional[Dict[str, Any]],
        log_dict: Dict[str, Any],
    ) -> None:
        return

    def _compute_auxiliary_critic_loss(self, **kwargs):
        return None, {}

    def _generator_loss_self_forcing_two_pass(
        self,
        *,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor],
        clean_audio: Optional[torch.Tensor],
        real_video_frames: Optional[int] = None,
        real_audio_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Two-pass Self-Forcing generator loss (main path, memory-optimized).

        Pass 1 (no_grad):
          - Full KV-cache rollout to get detached block predictions for all blocks.
          - Compute full-video KL gradient on detached outputs.
        Pass 2 (grad, per block):
          - Replay identical rollout randomness (RNG restored).
          - Keep grad only on each block's sampled exit step.
          - Compute paired block loss and call backward immediately.
          - Refresh KV cache with detached Pass-1 outputs for bit-identical cache state.

        This keeps peak activation memory close to one block's exit-step graph.
        """
        B = int(video_shape[0])
        rollout_blocks = self._get_self_forcing_rollout_blocks(int(video_shape[1]))
        num_blocks = len(rollout_blocks)
        if num_blocks == 0:
            raise ValueError("Self-forcing two-pass sampled zero blocks")
        total_video_frames = rollout_blocks[-1].video_end

        total_audio_frames = rollout_blocks[-1].audio_end
        audio_channels = int(audio_shape[2]) if audio_shape is not None else None

        num_denoising_steps = len(self.denoising_sigmas) - 1
        if num_denoising_steps <= 0:
            raise ValueError("denoising_sigmas must have at least 2 entries.")
        shortcut_context = self._begin_shortcut_consistency(num_denoising_steps)
        shortcut_targets = {} if shortcut_context is not None else None

        text_seq_len = conditional_dict["video_context"].shape[1]
        video_tail_shape = tuple(video_shape[2:])

        # Save RNG so Pass 2 replays Pass 1 exactly.
        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = None
        if torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state(self.device)

        rollout_video_detached = torch.zeros(
            (B, total_video_frames, *video_tail_shape),
            device=self.device, dtype=self.dtype,
        )
        rollout_audio_detached = None
        if audio_channels is not None:
            rollout_audio_detached = torch.zeros(
                (B, total_audio_frames, audio_channels),
                device=self.device, dtype=self.dtype,
            )

        # ---- Pass 1: no_grad full rollout ----
        with torch.no_grad():
            kv_caches_p1 = self._init_self_forcing_kv_caches(
                batch_size=B,
                num_video_frames=total_video_frames,
                num_audio_frames=total_audio_frames,
                text_seq_len=text_seq_len,
                video_frame_seqlen=int(video_shape[-2]) * int(video_shape[-1]),
            )
            exit_indices_p1 = self._generate_exit_indices(num_blocks, num_denoising_steps)

            for b_idx, block in enumerate(rollout_blocks):
                v_len = block.video_end - block.video_start
                a_len = block.audio_end - block.audio_start

                noisy_v = torch.randn(
                    (B, v_len, *video_tail_shape),
                    device=self.device, dtype=self.dtype,
                )
                noisy_a = None
                if rollout_audio_detached is not None:
                    noisy_a = torch.randn(
                        (B, a_len, audio_channels),
                        device=self.device, dtype=self.dtype,
                    )

                this_exit = exit_indices_p1[b_idx]
                denoised_v_p1 = None
                denoised_a_p1 = None
                v_sigma = None
                a_sigma = None

                for step_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                    v_sigma = sigma.to(
                        device=self.device, dtype=self.denoising_sigmas.dtype
                    ).expand(B, v_len)
                    if noisy_a is not None:
                        a_sigma = sigma.to(
                            device=self.device, dtype=self.denoising_sigmas.dtype
                        ).expand(B, a_len)
                    else:
                        a_sigma = None

                    pred_v, pred_a = self._kv_block_forward(
                        noisy_video_block=noisy_v,
                        noisy_audio_block=noisy_a,
                        timestep_block=v_sigma,
                        audio_timestep_block=a_sigma,
                        kv_caches=kv_caches_p1,
                        block=block,
                        conditional_dict=conditional_dict,
                    )

                    if step_idx == this_exit:
                        denoised_v_p1 = pred_v
                        denoised_a_p1 = pred_a
                        if shortcut_context is not None:
                            shortcut_target = self._build_shortcut_consistency_target(
                                context=shortcut_context,
                                block_index=b_idx,
                                exit_index=this_exit,
                                block=block,
                                noisy_video=noisy_v,
                                noisy_audio=noisy_a,
                                pred_video_x0=pred_v,
                                pred_audio_x0=pred_a,
                                sigma_start=sigma,
                                kv_caches=kv_caches_p1,
                                conditional_dict=conditional_dict,
                            )
                            if shortcut_target is not None:
                                shortcut_targets[b_idx] = shortcut_target
                        break

                    next_sigma = self.denoising_sigmas[step_idx + 1]
                    if float(next_sigma.item()) > 0.0:
                        noisy_v = self._renoise_block(pred_v, next_sigma)
                        if pred_a is not None:
                            noisy_a = self._renoise_block(pred_a, next_sigma)
                    else:
                        noisy_v = pred_v
                        if pred_a is not None:
                            noisy_a = pred_a
                else:
                    denoised_v_p1 = pred_v
                    denoised_a_p1 = pred_a

                rollout_video_detached[:, block.video_start:block.video_end] = denoised_v_p1
                if rollout_audio_detached is not None and denoised_a_p1 is not None:
                    rollout_audio_detached[:, block.audio_start:block.audio_end] = denoised_a_p1

                # Context-noise refresh for next block.
                self._kv_refresh_cache_for_block(
                    x0_video_block=denoised_v_p1,
                    x0_audio_block=denoised_a_p1,
                    kv_caches=kv_caches_p1,
                    block=block,
                    conditional_dict=conditional_dict,
                )

            del kv_caches_p1
            torch.cuda.empty_cache()

            F_v = rollout_video_detached.shape[1]
            F_a = rollout_audio_detached.shape[1] if rollout_audio_detached is not None else 0

            # pad_max: real (unpadded) frame counts for this rank. Frames beyond
            # these are padding rolled out only to keep cross-rank collective
            # counts aligned; they must not contribute to the generator update.
            # 下界 video至少1帧，audio至少0帧，不过其实应该区别不大，不会真的这么短到触发边界条件
            rvf = F_v if real_video_frames is None else max(1, min(int(real_video_frames), F_v))
            raf = F_a if real_audio_frames is None else max(0, min(int(real_audio_frames), F_a))

            # Crop the padded suffix BEFORE running the (bidirectional) real/fake
            # score networks. Their full self-attention would otherwise let padded
            # frames leak into the KL gradient on the REAL frames. The KL-grad
            # forwards are single full-sequence passes, so cropping only changes
            # tensor sizes, not the number of FSDP collectives — cross-rank sync
            # is preserved.
            clean_video_kl = rollout_video_detached[:, :rvf]
            clean_audio_kl = (
                rollout_audio_detached[:, :raf]
                if (rollout_audio_detached is not None and raf > 0)
                else None
            )

            # KL gradient over detached REAL (unpadded) rollout.
            video_sigma_kl, audio_sigma_kl, _, _ = (
                self._sample_score_noise_levels(
                    batch_size=B,
                    num_video_frames=rvf,
                    num_audio_frames=raf,
                    task_type=self.real_task_type,
                    fake_score=False,
                )
            )

            if self.dmd_latent_mode == "teacher_denoise":
                noisy_video, noisy_audio, video_sigma_kl, audio_sigma_kl = (
                    self._get_noisy_latent_via_teacher_denoise(
                        clean_video=clean_video_kl,
                        clean_audio=clean_audio_kl,
                        target_video_sigma=video_sigma_kl,
                        target_audio_sigma=audio_sigma_kl,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                    )
                )
            else:
                noise_video = torch.randn_like(clean_video_kl)
                noisy_video = self.add_noise(
                    clean_video_kl.flatten(0, 1),
                    noise_video.flatten(0, 1),
                    video_sigma_kl.flatten(0, 1),
                ).unflatten(0, (B, rvf))
                if clean_audio_kl is not None:
                    noise_audio = torch.randn_like(clean_audio_kl)
                    noisy_audio = self.add_noise(
                        clean_audio_kl, noise_audio, audio_sigma_kl,
                    )
                else:
                    noisy_audio = clean_video_kl

            grad_video, grad_audio, log_dict = self._compute_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                clean_video=clean_video_kl,
                clean_audio=(
                    clean_audio_kl if clean_audio_kl is not None else clean_video_kl
                ),
                video_sigma=video_sigma_kl,
                audio_sigma=(
                    audio_sigma_kl if clean_audio_kl is not None else video_sigma_kl
                ),
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=None,
                audio_loss_mask=None,
            )
            # Scatter the real-length KL gradient back into a full-length buffer
            # (padded frames = 0) so Pass-2 per-block indexing grad_video[:, blk]
            # yields exactly zero DMD loss on padded blocks.
            if rvf < F_v:
                grad_video_full = torch.zeros_like(rollout_video_detached)
                grad_video_full[:, :rvf] = grad_video
                grad_video = grad_video_full
            if (
                grad_audio is not None
                and rollout_audio_detached is not None
                and raf < F_a
            ):
                grad_audio_full = torch.zeros_like(rollout_audio_detached)
                if raf > 0:
                    grad_audio_full[:, :raf] = grad_audio
                grad_audio = grad_audio_full

        auxiliary_context = self._prepare_generator_auxiliary_gradients(
            rollout_video=rollout_video_detached,
            rollout_audio=rollout_audio_detached,
            clean_video=clean_video,
            clean_audio=clean_audio,
            conditional_dict=conditional_dict,
            real_video_frames=rvf,
            real_audio_frames=raf,
        )

        # Restore RNG for Pass 2 deterministic replay.
        torch.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None:
            torch.cuda.set_rng_state(rng_state_cuda, self.device)

        # ---- Pass 2: per-block forward + backward ----
        # Causal block tiling is prefix-stable, so weights over F_v match those
        # over rvf on the real frames; zero the padded tail so padded blocks
        # contribute exactly zero loss while still running forward/backward
        # (keeping FSDP collective counts aligned across ranks). Denominators use
        # the REAL frame counts so real-frame loss scale is identical to an
        # unpadded rollout.
        video_block_w = self._compute_block_weights(F_v)
        if rvf < F_v:
            video_block_w = video_block_w.clone()
            video_block_w[rvf:] = 0
        audio_block_w = (
            self._compute_block_weights(F_a, is_audio=True) if F_a > 0 else None
        )
        if audio_block_w is not None and raf < F_a:
            audio_block_w = audio_block_w.clone()
            audio_block_w[raf:] = 0
        video_w, audio_w = self.get_loss_weights()
        video_denom = B * rvf
        audio_denom = B * raf if raf > 0 else 1

        total_loss_scalar = 0.0
        total_video_loss_scalar = 0.0
        total_audio_loss_scalar = 0.0

        kv_caches_p2 = self._init_self_forcing_kv_caches(
            batch_size=B,
            num_video_frames=total_video_frames,
            num_audio_frames=total_audio_frames,
            text_seq_len=text_seq_len,
            video_frame_seqlen=int(video_shape[-2]) * int(video_shape[-1]),
        )
        exit_indices_p2 = self._generate_exit_indices(num_blocks, num_denoising_steps)
        fsdp_no_sync = getattr(self.generator, "no_sync", None)

        for b_idx, block in enumerate(rollout_blocks):
            is_last_block = (b_idx == num_blocks - 1)
            if (not is_last_block) and callable(fsdp_no_sync):
                block_ctx = fsdp_no_sync()
            else:
                block_ctx = nullcontext()

            with block_ctx:
                v_len = block.video_end - block.video_start
                a_len = block.audio_end - block.audio_start

                noisy_v = torch.randn(
                    (B, v_len, *video_tail_shape),
                    device=self.device, dtype=self.dtype,
                )
                noisy_a = None
                if rollout_audio_detached is not None:
                    noisy_a = torch.randn(
                        (B, a_len, audio_channels),
                        device=self.device, dtype=self.dtype,
                    )

                this_exit = exit_indices_p2[b_idx]
                denoised_v = None
                denoised_a = None
                v_sigma = None
                a_sigma = None

                for step_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                    need_grad = (step_idx == this_exit)
                    grad_ctx = nullcontext() if need_grad else torch.no_grad()
                    v_sigma = sigma.to(
                        device=self.device, dtype=self.denoising_sigmas.dtype
                    ).expand(B, v_len)
                    if noisy_a is not None:
                        a_sigma = sigma.to(
                            device=self.device, dtype=self.denoising_sigmas.dtype
                        ).expand(B, a_len)
                    else:
                        a_sigma = None

                    with grad_ctx:
                        pred_v, pred_a = self._kv_block_forward(
                            noisy_video_block=noisy_v,
                            noisy_audio_block=noisy_a,
                            timestep_block=v_sigma,
                            audio_timestep_block=a_sigma,
                            kv_caches=kv_caches_p2,
                            block=block,
                            conditional_dict=conditional_dict,
                        )

                    if need_grad:
                        denoised_v = pred_v
                        denoised_a = pred_a
                        break

                    next_sigma = self.denoising_sigmas[step_idx + 1]
                    if float(next_sigma.item()) > 0.0:
                        noisy_v = self._renoise_block(pred_v.detach(), next_sigma)
                        if pred_a is not None:
                            noisy_a = self._renoise_block(pred_a.detach(), next_sigma)
                    else:
                        noisy_v = pred_v.detach()
                        if pred_a is not None:
                            noisy_a = pred_a.detach()
                else:
                    denoised_v = pred_v
                    denoised_a = pred_a

                slc_v = slice(block.video_start, block.video_end)
                target_v = (denoised_v.detach() - grad_video[:, slc_v]).detach()
                v_diff = denoised_v.double() - target_v.double()
                v_per_frame = (v_diff ** 2).mean(dim=[2, 3, 4])
                block_w_v = video_block_w[slc_v].unsqueeze(0)
                video_loss_block = 0.5 * (v_per_frame * block_w_v).sum() / float(video_denom)

                audio_loss_block = None
                if (
                    denoised_a is not None
                    and audio_block_w is not None
                    and grad_audio is not None
                    and F_a > 0
                ):
                    audio_end_clamped = min(block.audio_end, F_a)
                    if audio_end_clamped > block.audio_start:
                        slc_a = slice(block.audio_start, audio_end_clamped)
                        target_a = (denoised_a.detach() - grad_audio[:, slc_a]).detach()
                        a_diff = denoised_a.double() - target_a.double()
                        a_per_frame = (a_diff ** 2).mean(dim=2)
                        block_w_a = audio_block_w[slc_a].unsqueeze(0)
                        audio_loss_block = 0.5 * (a_per_frame * block_w_a).sum() / float(audio_denom)

                if audio_loss_block is not None:
                    loss_block = video_w * video_loss_block + audio_w * audio_loss_block
                else:
                    loss_block = video_w * video_loss_block

                if shortcut_context is not None:
                    shortcut_loss_block = self._compute_shortcut_consistency_block_loss(
                        context=shortcut_context,
                        target=shortcut_targets.get(b_idx),
                        block=block,
                        noisy_video=noisy_v,
                        noisy_audio=noisy_a,
                        pred_video_x0=denoised_v,
                        pred_audio_x0=denoised_a,
                        video_block_weights=video_block_w,
                        audio_block_weights=audio_block_w,
                        video_denom=video_denom,
                        audio_denom=audio_denom,
                        video_loss_weight=video_w,
                        audio_loss_weight=audio_w,
                    )
                    if shortcut_loss_block is not None:
                        loss_block = loss_block + shortcut_loss_block

                auxiliary_loss_block = self._compute_generator_auxiliary_block_loss(
                    context=auxiliary_context,
                    block=block,
                    pred_video_x0=denoised_v,
                    pred_audio_x0=denoised_a,
                    video_block_weights=video_block_w,
                    audio_block_weights=audio_block_w,
                    video_denom=video_denom,
                    audio_denom=audio_denom,
                    video_loss_weight=video_w,
                    audio_loss_weight=audio_w,
                )
                if auxiliary_loss_block is not None:
                    loss_block = loss_block + auxiliary_loss_block

                loss_block.backward()

                total_loss_scalar += float(loss_block.detach().item())
                total_video_loss_scalar += float(video_loss_block.detach().item())
                if audio_loss_block is not None:
                    total_audio_loss_scalar += float(audio_loss_block.detach().item())

                # Refresh with Pass-1 detached output to keep cache state bit-identical.
                detached_v_slice = rollout_video_detached[:, block.video_start:block.video_end]
                detached_a_slice = None
                if rollout_audio_detached is not None and F_a > 0:
                    audio_end_clamped = min(block.audio_end, F_a)
                    if audio_end_clamped > block.audio_start:
                        detached_a_slice = rollout_audio_detached[:, block.audio_start:audio_end_clamped]
                self._kv_refresh_cache_for_block(
                    x0_video_block=detached_v_slice,
                    x0_audio_block=detached_a_slice,
                    kv_caches=kv_caches_p2,
                    block=block,
                    conditional_dict=conditional_dict,
                )

                del denoised_v, target_v, v_diff, v_per_frame, video_loss_block, loss_block
                if audio_loss_block is not None:
                    del denoised_a, target_a, a_diff, a_per_frame, audio_loss_block

        log_dict["video_dmd_loss"] = total_video_loss_scalar
        log_dict["audio_dmd_loss"] = total_audio_loss_scalar
        log_dict["video_loss_weight"] = video_w
        log_dict["audio_loss_weight"] = audio_w
        log_dict["alignment/video_sigma_mean"] = video_sigma_kl.float().mean().item()
        log_dict["alignment/audio_sigma_mean"] = audio_sigma_kl.float().mean().item()
        log_dict["self_forcing_num_blocks"] = num_blocks
        log_dict["self_forcing_grad_blocks"] = num_blocks
        log_dict["self_forcing_video_frames"] = F_v
        log_dict["self_forcing_audio_frames"] = F_a
        log_dict["self_forcing_context_noise"] = self.context_noise
        self._finalize_shortcut_consistency_logs(shortcut_context, log_dict)
        self._finalize_generator_auxiliary_logs(auxiliary_context, log_dict)

        # Backward is already executed per block; return detached scalar.
        return torch.tensor(total_loss_scalar, device=self.device), log_dict

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
        """
        Compute critic (fake_score) loss.

        The critic learns to denoise generated samples.
        """
        # Step 1: Generate samples (no gradient)
        with torch.no_grad():
            (
                generated_video,
                generated_audio,
                video_loss_mask,
                audio_loss_mask,
                fake_rollout_log,
            ) = self._run_generator(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
                self_forcing_rollout_mode=self.fake_score_rollout_mode,
            )
            # 训fake model的时候不加mask
            video_loss_mask = None
            audio_loss_mask = None

            # pad_max: crop the padded tail BEFORE training the critic on it.
            # Frames beyond real_video_frames/real_audio_frames are rolled out
            # only to keep cross-rank collective counts aligned (see
            # train_one_step's rollout_num_frames all_reduce MAX); they are
            # pure self-forcing "hallucinated" continuation with no grounding
            # in the sample's real prompt/duration and must not leak into the
            # critic's training distribution. Cropping here only changes
            # tensor length, not the number of fake_score forward/backward
            # calls (still exactly one), so it is safe w.r.t. FSDP collectives.
            _F_v_full = generated_video.shape[1]
            _F_a_full = generated_audio.shape[1] if generated_audio is not None else 0
            _rvf = (
                _F_v_full if real_video_frames is None
                else max(1, min(int(real_video_frames), _F_v_full))
            )
            # Lower-bounded at 1 (not 0 like the generator side): critic_loss
            # always feeds a real tensor through fake_score/denoising loss
            # (no masked-skip path here), so keep at least 1 audio frame to
            # avoid a zero-length tensor producing NaN in .mean() reductions.
            _raf = (
                _F_a_full if real_audio_frames is None
                else max(1, min(int(real_audio_frames), _F_a_full)) if _F_a_full > 0 else 0
            )
            if _rvf < _F_v_full:
                generated_video = generated_video[:, :_rvf]
            if generated_audio is not None and _raf < _F_a_full:
                generated_audio = generated_audio[:, :_raf]

        B = generated_video.shape[0]
        F_v = generated_video.shape[1]
        F_a = generated_audio.shape[1]

        # Step 2: Sample critic timestep
        if (
            self.fake_score_timestep_sampler_type == "ltx_scheduler"
            and video_loss_mask is not None
            and audio_loss_mask is not None
        ):
            critic_timestep, audio_critic_timestep = self._sample_causal_supervision_timesteps(
                B,
                video_loss_mask,
                audio_loss_mask,
            )
            critic_sigma = self.timestep_to_sigma(critic_timestep)
            audio_critic_sigma = self.timestep_to_sigma(audio_critic_timestep)
        else:
            (
                critic_sigma,
                audio_critic_sigma,
                critic_timestep,
                audio_critic_timestep,
            ) = self._sample_score_noise_levels(
                batch_size=B,
                num_video_frames=F_v,
                num_audio_frames=F_a,
                task_type=self.fake_task_type,
                fake_score=True,
            )

        # Step 3: Add noise to generated samples
        noise_video = torch.randn_like(generated_video)
        noise_audio = torch.randn_like(generated_audio)

        noisy_generated_video = self.add_noise(
            generated_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            critic_sigma.flatten(0, 1),
        ).unflatten(0, (B, F_v))

        noisy_generated_audio = self.add_noise(
            generated_audio, noise_audio, audio_critic_sigma
        )

        # Step 4: Critic prediction
        pred_video, pred_audio = self._score_model_forward(
            model=self.fake_score,
            noisy_video=noisy_generated_video,
            noisy_audio=noisy_generated_audio,
            clean_video_context=generated_video,
            clean_audio_context=generated_audio,
            conditional_dict=conditional_dict,
            video_sigma=critic_sigma,
            audio_sigma=audio_critic_sigma,
        )

        # Step 5: Compute flow matching loss for critic
        # CausVid uses flow_pred = (xt - x0_pred) / sigma, NOT simple x0 MSE.
        # The 1/sigma factor gives implicit 1/sigma^2 gradient weighting,
        # making the critic accurate at low-noise timesteps (critical for DMD).
        # Float64 for numerical stability, then cast back (matches CausVid).
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

        # flow_true = noise - x0 (target flow)
        video_loss = self._compute_masked_denoising_loss(
            target=generated_video,
            prediction=pred_video,
            noise=noise_video,
            flow_pred=flow_pred_video.unflatten(0, (B, F_v)),
            timestep=critic_timestep,
            mask=video_loss_mask,
        )

        audio_loss = self._compute_masked_denoising_loss(
            target=generated_audio,
            prediction=pred_audio,
            noise=noise_audio,
            flow_pred=flow_pred_audio,
            timestep=audio_critic_timestep,
            mask=audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict = {
            "critic_video_loss": video_loss.item(),
            "critic_audio_loss": audio_loss.item(),
            "critic_sigma_mean": critic_sigma.float().mean().item(),
        }
        log_dict.update(
            {f"fake_{key}": value for key, value in fake_rollout_log.items()}
        )

        auxiliary_loss, auxiliary_log = self._compute_auxiliary_critic_loss(
            generated_video=generated_video,
            generated_audio=generated_audio,
            clean_video=clean_video,
            clean_audio=clean_audio,
            conditional_dict=conditional_dict,
        )
        if auxiliary_loss is not None:
            total_loss = total_loss + auxiliary_loss
        log_dict.update(auxiliary_log)

        return total_loss, log_dict
