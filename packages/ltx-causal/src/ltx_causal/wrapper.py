"""
CausalLTX2DiffusionWrapper: Diffusion wrapper for causal LTX-2.

This module provides:
- Interface compatible with ltx-distillation's LTX2DiffusionWrapper
- Support for ODE causal training via masking
- Returns x0 predictions (matching original X0Model behavior)

The wrapper bridges CausalLTXModel with the DMD training framework.
"""

import dataclasses
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist

from ltx_causal.config import (
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
    CausalMaskConfig,
)
from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
from ltx_causal.attention.mask_builder import (
    build_all_causal_masks,
    build_all_teacher_forcing_masks,
    compute_aligned_audio_frames,
)
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model


# ============================================================================
# AAPT-style chunk recycling helpers
# ============================================================================

def build_chunk_ref(
    clean_video: torch.Tensor,
    first_block_video_frames: int,
    num_frame_per_block: int,
) -> torch.Tensor:
    """Construct the per-frame ref tensor for channel-cat recycling (training).

    Convention (matches the AAPT recipe adapted to chunk-based AR):
      - chunk 0 positions [0, fbvf): ref = zeros (no past chunk).
      - chunk 1 positions [fbvf, fbvf+fpb):
          * if fbvf >= fpb: ref = clean_video[fbvf-fpb : fbvf] (last fpb of chunk 0).
          * else:           ref = broadcast(clean_video[fbvf-1]) repeated fpb times.
      - chunk N (N >= 2) positions: ref = clean_video shifted back by fpb frames
        (each chunk N looks at chunk N-1's full content).

    Args:
        clean_video: [B, F, C, H, W] ground-truth (clean) video latents
        first_block_video_frames: size of chunk 0 (e.g. 4)
        num_frame_per_block: size of all subsequent chunks (e.g. 3)

    Returns:
        ref: [B, F, C, H, W] same shape/device/dtype as ``clean_video``.
    """
    B, F, C, H, W = clean_video.shape
    fbvf = int(first_block_video_frames)
    fpb = int(num_frame_per_block)
    ref = torch.zeros_like(clean_video)

    if F <= fbvf:
        return ref

    chunk1_end = min(fbvf + fpb, F)
    chunk1_size = chunk1_end - fbvf

    if fbvf >= fpb:
        ref[:, fbvf:chunk1_end] = clean_video[:, fbvf - chunk1_size : fbvf]
    else:
        last_frame = clean_video[:, fbvf - 1 : fbvf]  # [B, 1, C, H, W]
        ref[:, fbvf:chunk1_end] = last_frame.expand(B, chunk1_size, C, H, W)

    if chunk1_end < F:
        ref[:, chunk1_end:F] = clean_video[:, chunk1_end - fpb : F - fpb]

    return ref


def build_block_ref_from_prev(
    prev_chunk_latent: Optional[torch.Tensor],
    current_block_size: int,
    template: torch.Tensor,
) -> torch.Tensor:
    """Construct the per-frame ref tensor for ONE block during AR inference.

    Args:
        prev_chunk_latent: [B, F_prev, C, H, W] previously generated chunk
            (model x0 output, detached). None for the first block.
        current_block_size: F_curr (= block.video_frames for current block)
        template: any [B, F_any, C, H, W] tensor used to derive
            (B, C, H, W, device, dtype) when prev_chunk_latent is None.

    Returns:
        ref: [B, F_curr, C, H, W]
    """
    B = template.shape[0]
    C = template.shape[2]
    H, W = template.shape[-2], template.shape[-1]
    device = template.device
    dtype = template.dtype

    if prev_chunk_latent is None:
        return torch.zeros(
            (B, current_block_size, C, H, W), device=device, dtype=dtype,
        )

    F_prev = prev_chunk_latent.shape[1]
    prev_cast = prev_chunk_latent.to(device=device, dtype=dtype)
    if F_prev >= current_block_size:
        return prev_cast[:, -current_block_size:].contiguous()

    last = prev_cast[:, -1:]  # [B, 1, C, H, W]
    return last.expand(B, current_block_size, C, H, W).contiguous()


class CausalLTX2DiffusionWrapper(nn.Module):
    """
    Wrapper for CausalLTXModel providing DMD-compatible interface.

    Returns x0 predictions (denoised latents), matching the behavior of
    the original LTX2DiffusionWrapper which uses X0Model internally.

    The conversion is: x0 = sample - velocity * sigma

    Example:
        wrapper = CausalLTX2DiffusionWrapper(model)
        video_x0, audio_x0 = wrapper(
            video_latent, audio_latent, timestep, conditional_dict
        )
    """

    # Time alignment constants (same as ltx-distillation)
    VIDEO_LATENT_FPS = VIDEO_LATENT_FPS
    AUDIO_LATENT_FPS = AUDIO_LATENT_FPS
    ALIGNMENT_RATIO = AUDIO_LATENT_FPS / VIDEO_LATENT_FPS

    # Latent dimensions
    VIDEO_FRAME_SEQLEN = 384  # For 512x768 with patch_size=1: (512/32)*(768/32)
    AUDIO_FRAME_SEQLEN = 1

    # Upper bound on how many (F_v, F_a, frame_seqlen, ...) combinations to
    # keep BlockMasks for. Dynamic resolution can otherwise grow unbounded.
    _MASK_CACHE_MAX = 128

    def __init__(
        self,
        model: CausalLTXModel,
        video_height: int = 512,
        video_width: int = 768,
        vae_spatial_compression: int = 32,
        num_frame_per_block: int = 3,
        disable_causal_mask: bool = False,
        num_audio_sink_tokens: int = 0,
        first_block_video_frames: int = 1,
    ):
        """
        Args:
            model: CausalLTXModel instance
            video_height: Video height in pixels
            video_width: Video width in pixels
            vae_spatial_compression: VAE spatial compression factor
            num_frame_per_block: Frames per causal generation block
            disable_causal_mask: If True, disable all causal masks (fully bidirectional,
                equivalent to original LTX-2). Used for ablation studies.
            num_audio_sink_tokens: Number of learnable sink tokens prepended to audio
            first_block_video_frames: Video frames in Block 0 (1 = single-frame
                Global Prefix; 4 = merge causal-fix with first standard block)
        """
        super().__init__()
        self.model = model
        self.video_height = video_height
        self.video_width = video_width
        self.vae_spatial_compression = vae_spatial_compression
        self.num_frame_per_block = num_frame_per_block
        self.first_block_video_frames = first_block_video_frames
        self.disable_causal_mask = disable_causal_mask
        self.num_audio_sink_tokens = num_audio_sink_tokens

        # Validate patch_size compatibility
        if hasattr(model, 'config') and hasattr(model.config, 'patch_size'):
            ps = model.config.patch_size
            if ps != (1, 1, 1):
                raise ValueError(
                    f"CausalLTX2DiffusionWrapper requires patch_size=(1,1,1), "
                    f"but model has patch_size={ps}. The causal masks assume "
                    f"video_frame_seqlen = (H/32)*(W/32) tokens per frame."
                )

        # Compute latent dimensions
        self.latent_height = video_height // vae_spatial_compression
        self.latent_width = video_width // vae_spatial_compression

        # Compute frame sequence length
        # LTX-2 uses patch_size=1 (no spatial grouping), so each spatial
        # position in the latent grid is one token.
        # For 512x768: (512/32) * (768/32) = 16 * 24 = 384
        self.video_frame_seqlen = self.latent_height * self.latent_width

        # Mask config with actual resolution parameters (not defaults)
        self.mask_config = CausalMaskConfig(
            video_frame_seqlen=self.video_frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            first_block_video_frames=first_block_video_frames,
            num_audio_sink_tokens=num_audio_sink_tokens,
        )

        # Cache masks per (shape, device, sink-token config) so training and
        # benchmark block shapes can reuse prebuilt masks instead of rebuilding.
        # OrderedDict + move_to_end for cheap LRU semantics.
        self._mask_cache: "OrderedDict[Tuple, Dict[str, Any]]" = OrderedDict()

        # Cache the recycling flag for cheap access from forward paths.
        self.enable_chunk_recycling = bool(
            getattr(getattr(model, "config", None), "enable_chunk_recycling", False)
        )
        # Keep the eval-only delegate out of nn.Module registration so FSDP/state_dict
        # do not treat it as part of the trainable causal model.
        self.__dict__["_bidirectional_delegate"] = None

    def set_bidirectional_delegate(self, delegate: nn.Module) -> None:
        """Install an explicit bidirectional delegate for eval-time fallback."""
        self.__dict__["_bidirectional_delegate"] = delegate

    def has_bidirectional_delegate(self) -> bool:
        """Return whether a delegate was explicitly installed."""
        return self._bidirectional_delegate is not None

    def set_module_grad(self, module_grad: Dict[str, bool]) -> None:
        """Set gradient requirements for model components."""
        if module_grad.get("model", True):
            self.model.requires_grad_(True)
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.model.enable_gradient_checkpointing()

    def _build_bidirectional_delegate(self) -> nn.Module:
        """Build an eval-only wrapper that uses the original bidirectional path."""
        from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper

        model_config = self.model.config
        velocity_model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=model_config.video_heads,
            attention_head_dim=model_config.video_d_head,
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            num_layers=model_config.num_layers,
            cross_attention_dim=model_config.cross_attention_dim,
            norm_eps=model_config.norm_eps,
            caption_channels=model_config.caption_channels,
            positional_embedding_theta=model_config.pe_theta,
            positional_embedding_max_pos=list(model_config.pe_max_pos),
            timestep_scale_multiplier=model_config.timestep_scale_multiplier,
            audio_num_attention_heads=model_config.audio_heads,
            audio_attention_head_dim=model_config.audio_d_head,
            audio_in_channels=model_config.in_channels,
            audio_out_channels=model_config.out_channels,
            audio_cross_attention_dim=model_config.audio_cross_attention_dim,
            audio_positional_embedding_max_pos=list(model_config.audio_pe_max_pos),
            av_ca_timestep_scale_multiplier=model_config.av_ca_timestep_scale_multiplier,
        ).to(device=next(self.model.parameters()).device, dtype=next(self.model.parameters()).dtype)

        state_dict = {
            key: value
            for key, value in self.model.state_dict().items()
            if "mask_builder" not in key and "causal_gate" not in key and "audio_sink_tokens" not in key
        }
        missing, unexpected = velocity_model.load_state_dict(state_dict, strict=False)
        real_missing = [key for key in missing if "mask_builder" not in key]
        if real_missing or unexpected:
            raise RuntimeError(
                "Failed to build bidirectional delegate from causal model: "
                f"missing={real_missing[:10]}, unexpected={unexpected[:10]}"
            )

        delegate = LTX2DiffusionWrapper(
            model=X0Model(velocity_model),
            video_height=self.video_height,
            video_width=self.video_width,
            vae_spatial_compression=self.vae_spatial_compression,
        )
        delegate.eval()
        return delegate

    def _get_bidirectional_delegate(self) -> nn.Module:
        if self._bidirectional_delegate is None:
            if dist.is_initialized():
                raise RuntimeError(
                    "Bidirectional delegate was not pre-installed before distributed/FSDP wrapping. "
                    "Lazy delegate construction calls self.model.state_dict(), which is unsafe under "
                    "FSDP and can leave unmanaged all-gathered parameter storage alive. Install the "
                    "delegate via set_bidirectional_delegate() before wrapping the causal generator."
                )
            self._bidirectional_delegate = self._build_bidirectional_delegate()
        return self._bidirectional_delegate

    def get_scheduler(self):
        """Get the scheduler for this model."""
        try:
            from ltx_distillation.schedulers import LTX2FlowScheduler
            return LTX2FlowScheduler()
        except ImportError:
            raise ImportError(
                "LTX2FlowScheduler not found. "
                "Please install ltx-distillation package."
            )

    def _process_timestep_causal(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process timestep for causal generation.

        Makes timesteps uniform within each generation block.
        Block 0: first_block_video_frames frames (all share one timestep).
        Blocks 1..N: groups of num_frame_per_block starting after Block 0.

        Args:
            timestep: [B, F] per-frame timesteps

        Returns:
            [B, F] timesteps with uniform values within blocks

        Example (first_block_video_frames=4, num_frame_per_block=3, 16 frames):
            Block 0: frames 0-3
            Block 1: frames 4-6
            Block 2: frames 7-9
            ...
            Input:  [t0, t1, t2, t3, t4, t5, t6, ...]
            Output: [t0, t0, t0, t0, t4, t4, t4, ...]
        """
        if timestep.ndim == 1:
            return timestep

        B, F = timestep.shape

        if F <= 1:
            return timestep

        result = timestep.new_zeros(B, F)

        # Block 0: first_block_video_frames frames share one timestep
        fb = min(self.first_block_video_frames, F)
        result[:, :fb] = timestep[:, 0:1].expand(B, fb)

        # Standard blocks: groups of num_frame_per_block
        block_size = self.num_frame_per_block
        idx = fb
        while idx < F:
            end = min(idx + block_size, F)
            result[:, idx:end] = timestep[:, idx:idx + 1].expand(B, end - idx)
            idx = end

        return result

    def _get_or_build_masks(
        self,
        num_video_frames: int,
        num_audio_frames: int,
        device: torch.device,
        frame_seqlen: int,
    ) -> Dict:
        """Get cached masks or build new ones.

        ``frame_seqlen`` is the runtime tokens-per-frame (= latent H * W) and
        is part of the cache key so dynamic-resolution training/inference reuses
        a separate BlockMask per spatial shape.

        When disable_causal_mask=True, returns all-None masks so attention
        falls through to standard bidirectional path (no mask), matching
        original LTX-2 behavior exactly.
        """
        if self.disable_causal_mask:
            return {'video_self': None, 'audio_self': None, 'a2v': None, 'v2a': None}

        cache_key = (
            num_video_frames,
            num_audio_frames,
            str(device),
            self.mask_config.num_frame_per_block,
            self.mask_config.num_audio_sink_tokens,
            int(frame_seqlen),
        )

        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            self._mask_cache.move_to_end(cache_key)
            return cached

        runtime_cfg = dataclasses.replace(
            self.mask_config, video_frame_seqlen=int(frame_seqlen),
        )
        built = build_all_causal_masks(
            num_video_frames, num_audio_frames,
            config=runtime_cfg, device=device,
        )
        self._mask_cache[cache_key] = built
        if len(self._mask_cache) > self._MASK_CACHE_MAX:
            self._mask_cache.popitem(last=False)
        return built

    @staticmethod
    def _reshape_sigma_for_broadcast(
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Reshape sigma tensor for broadcasting with target.

        Args:
            sigma: [B] or [B, F] sigma values
            target: Target tensor to broadcast against
                    Video: [B, F, C, H, W] or Audio: [B, F, C]

        Returns:
            Sigma reshaped for broadcasting
        """
        if sigma.ndim == 1:
            # [B] → add trailing dims to match target
            return sigma.view(-1, *([1] * (target.ndim - 1)))
        elif sigma.ndim == 2:
            # [B, F] → add trailing dims
            return sigma.view(sigma.shape[0], sigma.shape[1], *([1] * (target.ndim - 2)))
        else:
            return sigma

    def forward(
        self,
        *args,
        _tf_kwargs: Optional[Dict] = None,
        _kv_cache_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Dispatch to standard / teacher-forcing / KV-cache forward.

        Routing rules (checked in order):
          1. ``_tf_kwargs`` -> ``_forward_teacher_forcing`` (Method A 2x sequence).
          2. ``_kv_cache_kwargs`` -> ``_forward_with_cache`` (per-block KV cache
             for Self-Forcing rollout / KV-cache causal inference).
          3. Otherwise -> ``_forward_standard`` (full-sequence training).

        Routing through ``forward()`` ensures FSDP hooks fire correctly.
        """
        if _tf_kwargs is not None:
            return self._forward_teacher_forcing(**_tf_kwargs)
        if _kv_cache_kwargs is not None:
            return self._forward_with_cache(**_kv_cache_kwargs)
        return self._forward_standard(*args, **kwargs)

    def _forward_standard(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = True,
        force_bidirectional: bool = False,
        clean_video_for_ref: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass for training.

        Returns x0 predictions (denoised latents) matching the original
        X0Model wrapper behavior: x0 = sample - velocity * sigma.

        Args:
            noisy_image_or_video: Noisy video latent [B, F, C, H, W]
            conditional_dict: Text conditioning dictionary with:
                - video_context: [B, L, D] video text embeddings
                - audio_context: [B, L, D] audio text embeddings
                - video_context_mask: Optional attention mask
                - audio_context_mask: Optional attention mask
            timestep: Diffusion timestep (sigma) [B] or [B, F] (video timestep)
            noisy_audio: Noisy audio latent [B, F_a, C] (optional)
            audio_timestep: Audio diffusion timestep (sigma) [B] or [B, F_a] (optional)
                If not provided, uses video timestep expanded to audio frames
            use_causal_timestep: Whether to apply causal timestep processing

        Returns:
            (video_x0, audio_x0): Denoised x0 predictions.
        """
        B, F_v, C, H, W = noisy_image_or_video.shape
        device = noisy_image_or_video.device

        if (
            not self.training
            and (self.disable_causal_mask or force_bidirectional)
            and not dist.is_initialized()  # unsafe under FSDP: see _get_bootstrap_generator note
        ):
            delegate = self._get_bidirectional_delegate()
            delegate = delegate.to(device=device, dtype=noisy_image_or_video.dtype)
            return delegate(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                noisy_audio=noisy_audio,
                audio_timestep=audio_timestep,
            )

        aligned_audio = compute_aligned_audio_frames(
            F_v, self.num_frame_per_block, self.first_block_video_frames,
        )

        # Compute audio frames if not provided
        if noisy_audio is None:
            # Create audio with aligned frame count
            F_a = aligned_audio
            noisy_audio = torch.randn(B, F_a, C, device=device, dtype=noisy_image_or_video.dtype)
        else:
            F_a = noisy_audio.shape[1]

        # Validate audio frame count matches aligned count
        if F_a != aligned_audio:
            raise ValueError(
                f"Audio frames ({F_a}) does not match aligned count ({aligned_audio}) "
                f"for {F_v} video frames. With V_0 + A_0 in Block 0, the aligned "
                f"count should match ltx-core's raw audio count exactly."
            )

        # Process timestep
        if use_causal_timestep and timestep.ndim > 1:
            timestep = self._process_timestep_causal(timestep)

        # Extract context from conditional_dict
        video_context = conditional_dict.get('video_context')
        audio_context = conditional_dict.get('audio_context')
        video_context_mask = conditional_dict.get('video_context_mask')
        audio_context_mask = conditional_dict.get('audio_context_mask')

        # Handle shared context
        if video_context is None and 'context' in conditional_dict:
            video_context = conditional_dict['context']
        if audio_context is None:
            audio_context = video_context

        # Handle shared attention mask fallback
        shared_mask = conditional_dict.get('attention_mask')
        if video_context_mask is None and shared_mask is not None:
            video_context_mask = shared_mask
        if audio_context_mask is None and shared_mask is not None:
            audio_context_mask = shared_mask

        # Build or get cached masks. frame_seqlen is the runtime tokens-per-frame
        # so dynamic-resolution inputs hit a separate cache entry / BlockMask.
        H_lat, W_lat = noisy_image_or_video.shape[-2:]
        frame_seqlen = int(H_lat) * int(W_lat)
        masks = self._get_or_build_masks(F_v, F_a, device, frame_seqlen)

        # Build AAPT-style channel-cat ref when chunk recycling is enabled and
        # a clean reference is available. The standard (non-TF) path typically
        # does not have access to clean_video, so we fall back to a zero ref
        # tensor — equivalent to disabling recycling for this call.
        ref_video = None
        if self.enable_chunk_recycling:
            if clean_video_for_ref is not None:
                ref_video = build_chunk_ref(
                    clean_video_for_ref,
                    self.first_block_video_frames,
                    self.num_frame_per_block,
                ).to(dtype=noisy_image_or_video.dtype)
            else:
                ref_video = torch.zeros_like(noisy_image_or_video)

        # Call model (returns velocity predictions)
        video_velocity, audio_velocity = self.model(
            video_latent=noisy_image_or_video,
            audio_latent=noisy_audio,
            timesteps=timestep,
            video_context=video_context,
            audio_context=audio_context,
            video_context_mask=video_context_mask,
            audio_context_mask=audio_context_mask,
            audio_timesteps=audio_timestep,
            masks=masks,
            ref_video_latent=ref_video,
        )

        # === Convert velocity to x0 predictions ===
        # Matches X0Model / to_denoised: x0 = sample - velocity * sigma
        # Use float32 for numerical stability (matching to_denoised in ltx-core)
        compute_dtype = torch.float32

        # Video: x0 = noisy_video - velocity * sigma
        video_sigma = self._reshape_sigma_for_broadcast(timestep, noisy_image_or_video)
        video_x0 = (
            noisy_image_or_video.to(compute_dtype)
            - video_velocity.to(compute_dtype) * video_sigma.to(compute_dtype)
        ).to(noisy_image_or_video.dtype)

        # Audio: x0 = noisy_audio - velocity * sigma
        audio_sigma_raw = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = self._reshape_sigma_for_broadcast(audio_sigma_raw, noisy_audio)
        audio_x0 = (
            noisy_audio.to(compute_dtype)
            - audio_velocity.to(compute_dtype) * audio_sigma.to(compute_dtype)
        ).to(noisy_audio.dtype)

        return video_x0, audio_x0

    # ================================================================
    # Teacher Forcing Forward (Method A: 2x sequence)
    # ================================================================

    def _get_or_build_tf_masks(
        self,
        num_video_frames: int,
        num_audio_frames: int,
        device: torch.device,
        frame_seqlen: int,
    ) -> Dict:
        """Get cached teacher-forcing masks or build new ones.

        ``frame_seqlen`` is the runtime tokens-per-frame (= latent H * W).
        Same LRU semantics as ``_get_or_build_masks``.
        """
        if not hasattr(self, '_tf_mask_cache'):
            self._tf_mask_cache: "OrderedDict[Tuple, Dict]" = OrderedDict()

        cache_key = (
            num_video_frames,
            num_audio_frames,
            str(device),
            self.mask_config.num_frame_per_block,
            self.mask_config.num_audio_sink_tokens,
            int(frame_seqlen),
        )

        cached = self._tf_mask_cache.get(cache_key)
        if cached is not None:
            self._tf_mask_cache.move_to_end(cache_key)
            return cached

        runtime_cfg = dataclasses.replace(
            self.mask_config, video_frame_seqlen=int(frame_seqlen),
        )
        built = build_all_teacher_forcing_masks(
            num_video_frames, num_audio_frames,
            config=runtime_cfg, device=device,
        )
        self._tf_mask_cache[cache_key] = built
        if len(self._tf_mask_cache) > self._MASK_CACHE_MAX:
            self._tf_mask_cache.popitem(last=False)
        return built

    def _forward_teacher_forcing(
        self,
        noisy_video: torch.Tensor,
        clean_video: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        noisy_timestep: torch.Tensor,
        clean_timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
        noisy_audio_timestep: Optional[torch.Tensor] = None,
        clean_audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = True,
        return_features: bool = False,
        feature_layer_shallow: Optional[int] = None,
        feature_layer_deep: Optional[int] = None,
        return_velocity: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher Forcing forward pass returning x0 predictions.

        When ``return_features`` is True the call additionally returns a dict of
        intermediate hidden states (shallow/deep, video/audio) for Causal
        Self-Flow alignment, i.e. ``(video_x0, audio_x0, feats)``.
        When ``return_velocity`` is True, the raw velocity outputs are appended
        after the normal return values. This keeps the default x0-facing API
        unchanged while allowing Stage-0 flow-matching supervision to consume
        the model's native outputs without dividing by sigma.

        Passes [clean | noisy] 2x sequences through the model with
        teacher forcing masks and returns x0 for the noisy half only.

        Args:
            noisy_video: [B, F, C, H, W] noisy video latent
            clean_video: [B, F, C, H, W] clean video latent
            conditional_dict: Text conditioning
            noisy_timestep: [B] or [B, F] noisy sigma
            clean_timestep: [B] or [B, F] clean sigma (typically 0)
            noisy_audio: [B, F_a, C] noisy audio (optional)
            clean_audio: [B, F_a, C] clean audio (optional)
            noisy_audio_timestep: [B] or [B, F_a] noisy audio sigma
            clean_audio_timestep: [B] or [B, F_a] clean audio sigma

        Returns:
            (video_x0, audio_x0) for the noisy half only
        """
        B, F_v, C, H, W = noisy_video.shape
        device = noisy_video.device

        aligned_audio = compute_aligned_audio_frames(
            F_v, self.num_frame_per_block, self.first_block_video_frames,
        )

        if noisy_audio is None:
            F_a = aligned_audio
            noisy_audio = torch.randn(B, F_a, C, device=device, dtype=noisy_video.dtype)
            clean_audio = torch.zeros(B, F_a, C, device=device, dtype=clean_video.dtype)
        else:
            F_a = noisy_audio.shape[1]

        if F_a != aligned_audio:
            raise ValueError(
                f"Audio frames ({F_a}) does not match aligned count ({aligned_audio}) "
                f"for {F_v} video frames."
            )

        if clean_audio is None:
            clean_audio = torch.zeros_like(noisy_audio)

        # Process causal timesteps
        if use_causal_timestep and noisy_timestep.ndim > 1:
            noisy_timestep = self._process_timestep_causal(noisy_timestep)
        if use_causal_timestep and clean_timestep.ndim > 1:
            clean_timestep = self._process_timestep_causal(clean_timestep)

        # Extract context
        video_context = conditional_dict.get('video_context')
        audio_context = conditional_dict.get('audio_context')
        video_context_mask = conditional_dict.get('video_context_mask')
        audio_context_mask = conditional_dict.get('audio_context_mask')

        if video_context is None and 'context' in conditional_dict:
            video_context = conditional_dict['context']
        if audio_context is None:
            audio_context = video_context

        shared_mask = conditional_dict.get('attention_mask')
        if video_context_mask is None and shared_mask is not None:
            video_context_mask = shared_mask
        if audio_context_mask is None and shared_mask is not None:
            audio_context_mask = shared_mask

        # Build TF masks (runtime frame_seqlen for dynamic resolution).
        H_lat, W_lat = noisy_video.shape[-2:]
        frame_seqlen = int(H_lat) * int(W_lat)
        tf_masks = self._get_or_build_tf_masks(F_v, F_a, device, frame_seqlen)

        # AAPT-style ref for the noisy half. Built from the GT clean_video so
        # the ref content for chunk N is chunk N-1's clean frames. Inference
        # uses the previously generated x0 in the same position; here at TF
        # training we use GT (teacher-forcing for the ref signal too).
        # Match the noisy-half dtype so the add inside ref_patchify_proj fuses
        # without an upcast.
        ref_video = None
        if self.enable_chunk_recycling:
            ref_video = build_chunk_ref(
                clean_video,
                self.first_block_video_frames,
                self.num_frame_per_block,
            ).to(dtype=noisy_video.dtype)

        # Route through self.model(...) so FSDP hooks fire correctly
        model_out = self.model(
            _tf_kwargs=dict(
                noisy_video_latent=noisy_video,
                noisy_audio_latent=noisy_audio,
                clean_video_latent=clean_video,
                clean_audio_latent=clean_audio,
                noisy_timesteps=noisy_timestep,
                clean_timesteps=clean_timestep,
                video_context=video_context,
                audio_context=audio_context,
                video_context_mask=video_context_mask,
                audio_context_mask=audio_context_mask,
                noisy_audio_timesteps=noisy_audio_timestep,
                clean_audio_timesteps=clean_audio_timestep,
                tf_masks=tf_masks,
                ref_video_latent=ref_video,
                return_features=return_features,
                feature_layer_shallow=feature_layer_shallow,
                feature_layer_deep=feature_layer_deep,
            )
        )
        if return_features:
            video_velocity, audio_velocity, feats = model_out
        else:
            video_velocity, audio_velocity = model_out

        # Convert velocity to x0 (noisy half only)
        compute_dtype = torch.float32
        video_sigma = self._reshape_sigma_for_broadcast(noisy_timestep, noisy_video)
        video_x0 = (
            noisy_video.to(compute_dtype)
            - video_velocity.to(compute_dtype) * video_sigma.to(compute_dtype)
        ).to(noisy_video.dtype)

        audio_sigma_raw = noisy_audio_timestep if noisy_audio_timestep is not None else noisy_timestep
        audio_sigma = self._reshape_sigma_for_broadcast(audio_sigma_raw, noisy_audio)
        audio_x0 = (
            noisy_audio.to(compute_dtype)
            - audio_velocity.to(compute_dtype) * audio_sigma.to(compute_dtype)
        ).to(noisy_audio.dtype)

        if return_features:
            if return_velocity:
                return (
                    video_x0,
                    audio_x0,
                    feats,
                    video_velocity,
                    audio_velocity,
                )
            return video_x0, audio_x0, feats

        if return_velocity:
            return video_x0, audio_x0, video_velocity, audio_velocity

        return video_x0, audio_x0

    # ================================================================
    # KV-cache Causal Inference / Self-Forcing Training
    # ================================================================

    def init_av_kv_caches(
        self,
        batch_size: int,
        num_video_frames: int,
        num_audio_frames: Optional[int] = None,
        text_seq_len: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        video_frame_seqlen: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Allocate per-layer KV caches sized for a full causal rollout.

        Args:
            batch_size: B
            num_video_frames: total video frames covered by the rollout
                              (sum across all blocks)
            num_audio_frames: total audio frames covered by the rollout.
                              If None, derived from num_video_frames via the
                              standard block alignment.
            text_seq_len: length of text context (matched against
                          conditional_dict['video_context'].shape[1] in callers)
            device: cache buffer device (defaults to model device)
            dtype: cache buffer dtype (defaults to model parameter dtype, must
                   match the model's compute dtype for SDPA correctness)
            video_frame_seqlen: runtime tokens-per-frame (= latent H * W).
                If None, falls back to ``model.config.video_frame_seqlen``;
                pass the actual value when running dynamic-resolution
                inference so the cache buffer is sized correctly.

        Returns:
            List of per-layer cache dicts. See CausalLTXModel.init_av_kv_caches.
        """
        if num_audio_frames is None:
            num_audio_frames = compute_aligned_audio_frames(
                num_video_frames,
                self.num_frame_per_block,
                self.first_block_video_frames,
            )

        if device is None or dtype is None:
            ref_param = next(self.model.parameters())
            if device is None:
                device = ref_param.device
            if dtype is None:
                dtype = ref_param.dtype

        return self.model.init_av_kv_caches(
            batch_size=batch_size,
            max_video_frames=num_video_frames,
            max_audio_frames=num_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
            video_frame_seqlen=video_frame_seqlen,
        )

    def _forward_with_cache(
        self,
        noisy_video_block: torch.Tensor,
        noisy_audio_block: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep_block: torch.Tensor,
        kv_caches: List[Dict[str, Any]],
        current_video_start_frame: int,
        current_audio_start_frame: int,
        audio_timestep_block: Optional[torch.Tensor] = None,
        ref_video_block: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-block KV-cache forward, returning x0 predictions for the block.

        Inputs are CURRENT-BLOCK-shaped tensors:
          - noisy_video_block: [B, F_v_block, C, H, W]
          - noisy_audio_block: [B, F_a_block, C]
          - timestep_block: [B] or [B, F_v_block]    (uniform-per-block sigma)
          - audio_timestep_block: [B] or [B, F_a_block] (defaults to timestep_block)

        The caller is responsible for slicing inputs into block-shaped pieces
        and feeding consistent (current_video_start_frame, current_audio_start_frame)
        offsets so that cache writes line up with the standard block layout.

        No mask is built here: causality is enforced by the cache mechanism
        (each call only writes the block's K/V at the given offset and reads
        the prefix [0, end) populated by previous blocks).

        Returns:
            (video_x0_block, audio_x0_block) for the current block.
        """
        if audio_timestep_block is None:
            audio_timestep_block = timestep_block

        # Extract context (matches _forward_standard).
        video_context = conditional_dict.get('video_context')
        audio_context = conditional_dict.get('audio_context')
        video_context_mask = conditional_dict.get('video_context_mask')
        audio_context_mask = conditional_dict.get('audio_context_mask')

        if video_context is None and 'context' in conditional_dict:
            video_context = conditional_dict['context']
        if audio_context is None:
            audio_context = video_context

        shared_mask = conditional_dict.get('attention_mask')
        if video_context_mask is None and shared_mask is not None:
            video_context_mask = shared_mask
        if audio_context_mask is None and shared_mask is not None:
            audio_context_mask = shared_mask

        # Match dtype for the ref add inside ref_patchify_proj.
        ref_for_call = None
        if self.enable_chunk_recycling:
            if ref_video_block is None:
                # Block 0 or caller forgot to pass: degrade to zero ref.
                ref_for_call = torch.zeros_like(noisy_video_block)
            else:
                ref_for_call = ref_video_block.to(
                    device=noisy_video_block.device,
                    dtype=noisy_video_block.dtype,
                )

        # Route through self.model(...) so FSDP hooks fire correctly. The
        # model dispatcher detects kv_caches in kwargs and routes to its own
        # _forward_with_cache.
        video_velocity, audio_velocity = self.model(
            video_latent=noisy_video_block,
            audio_latent=noisy_audio_block,
            timesteps=timestep_block,
            audio_timesteps=audio_timestep_block,
            video_context=video_context,
            audio_context=audio_context,
            video_context_mask=video_context_mask,
            audio_context_mask=audio_context_mask,
            kv_caches=kv_caches,
            current_video_start_frame=current_video_start_frame,
            current_audio_start_frame=current_audio_start_frame,
            ref_video_latent=ref_for_call,
        )

        # Convert velocity -> x0 in float32 for numerical stability.
        compute_dtype = torch.float32

        video_sigma = self._reshape_sigma_for_broadcast(timestep_block, noisy_video_block)
        video_x0 = (
            noisy_video_block.to(compute_dtype)
            - video_velocity.to(compute_dtype) * video_sigma.to(compute_dtype)
        ).to(noisy_video_block.dtype)

        audio_sigma = self._reshape_sigma_for_broadcast(audio_timestep_block, noisy_audio_block)
        audio_x0 = (
            noisy_audio_block.to(compute_dtype)
            - audio_velocity.to(compute_dtype) * audio_sigma.to(compute_dtype)
        ).to(noisy_audio_block.dtype)

        return video_x0, audio_x0

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        video_height: int = 512,
        video_width: int = 768,
        num_frame_per_block: int = 3,
        disable_causal_mask: bool = False,
        enable_causal_log_rescale: bool = False,
        num_audio_sink_tokens: int = 0,
        first_block_video_frames: int = 1,
        enable_chunk_recycling: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CausalLTX2DiffusionWrapper":
        """
        Load wrapper from pretrained checkpoint.

        Args:
            checkpoint_path: Path to LTX-2 checkpoint
            video_height: Video height in pixels
            video_width: Video width in pixels
            num_frame_per_block: Frames per generation block
            disable_causal_mask: If True, disable all causal masks
            enable_causal_log_rescale: If True, apply log-ratio entropy rescaling
            num_audio_sink_tokens: Number of learnable sink tokens prepended to audio
            first_block_video_frames: Video frames in Block 0
            device: Target device
            dtype: Model dtype

        Returns:
            Initialized CausalLTX2DiffusionWrapper
        """
        config = CausalLTXModelConfig(
            num_frame_per_block=num_frame_per_block,
            first_block_video_frames=first_block_video_frames,
            enable_causal_log_rescale=enable_causal_log_rescale,
            num_audio_sink_tokens=num_audio_sink_tokens,
            enable_chunk_recycling=enable_chunk_recycling,
        )

        model = CausalLTXModel.from_pretrained(
            checkpoint_path, config, device, dtype
        )

        return cls(
            model=model,
            video_height=video_height,
            video_width=video_width,
            num_frame_per_block=num_frame_per_block,
            disable_causal_mask=disable_causal_mask,
            num_audio_sink_tokens=num_audio_sink_tokens,
            first_block_video_frames=first_block_video_frames,
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_causal_diffusion_wrapper(
    checkpoint_path: Optional[str] = None,
    model: Optional[CausalLTXModel] = None,
    **kwargs,
) -> CausalLTX2DiffusionWrapper:
    """
    Factory function to create CausalLTX2DiffusionWrapper.

    Args:
        checkpoint_path: Path to checkpoint (if model not provided)
        model: Pre-loaded CausalLTXModel (if checkpoint not provided)
        **kwargs: Additional arguments for wrapper

    Returns:
        Configured wrapper instance
    """
    if model is not None:
        return CausalLTX2DiffusionWrapper(model=model, **kwargs)
    elif checkpoint_path is not None:
        return CausalLTX2DiffusionWrapper.from_pretrained(checkpoint_path, **kwargs)
    else:
        raise ValueError("Either checkpoint_path or model must be provided")
