"""
CausalLTXModel: Full causal LTX-2 model for ODE causal training.

This module implements:
- Complete causal transformer with 48 layers
- Training mode: Flexattention with BlockMask for causal masking
- Weight loading from original LTX-2 checkpoints

Weight compatibility:
    All module names and state_dict keys match the original LTXModel exactly.
    The causal model adds block masks at runtime, which do not
    affect the checkpoint structure.

Based on LTX-2's LTXModel architecture (ltx_core/model/transformer/model.py).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from ltx_causal.config import (
    CausalMaskConfig,
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
)
from ltx_causal.attention.mask_builder import (
    AVCausalMaskBuilder,
    build_all_causal_masks,
    build_all_teacher_forcing_masks,
    compute_aligned_audio_frames,
    compute_av_blocks,
    compute_causal_log_scales,
)
from ltx_causal.transformer.causal_block import (
    CausalAVTransformerBlock,
    CausalTransformerArgs,
    TransformerConfig,
    rms_norm,
)
from ltx_causal.transformer.compat import (
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    causal_precompute_freqs_cis,
)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class CausalLTXModelConfig:
    """Configuration for CausalLTXModel."""

    # Model dimensions
    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_heads: int = 32
    audio_heads: int = 32
    video_d_head: int = 128
    audio_d_head: int = 64

    # Cross-attention context dimension
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048  # Also used as inner_dim for cross-modal RoPE

    # Patch embedding (LTX-2 uses patch_size=1 with nn.Linear)
    in_channels: int = 128
    out_channels: int = 128
    patch_size: Tuple[int, int, int] = (1, 1, 1)

    # Caption (text) projection
    caption_channels: int = 3840  # Gemma text encoder output dim

    # Position embedding
    pe_theta: float = 10000.0
    pe_max_pos: Tuple[int, int, int] = (20, 2048, 2048)
    audio_pe_max_pos: Tuple[int] = (20,)

    # Timestep embedding
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1

    # Normalization
    norm_eps: float = 1e-6

    # Causal generation
    num_frame_per_block: int = 3
    first_block_video_frames: int = 1

    # Audio sink tokens
    num_audio_sink_tokens: int = 0

    # RoPE
    rope_type: CausalRopeType = CausalRopeType.INTERLEAVED

    # Token sizes
    video_frame_seqlen: int = 384  # For 512x768: (512/32)*(768/32)
    audio_frame_seqlen: int = 1

    # Log-ratio entropy-aligned rescaling for causal attention outputs.
    # When True, each token's causal attention output is scaled by
    # log(1 + visible_tokens) / log(1 + total_tokens), compensating for
    # the information deficit caused by causal masking vs bidirectional.
    # No learnable parameters — purely structural rescaling.
    enable_causal_log_rescale: bool = False

    # AAPT-style channel-wise concat of the previous chunk's frames as
    # an extra condition for the current chunk. Implemented as a separate
    # zero-init Linear (`ref_patchify_proj`) whose output is added to the
    # patchified noisy video tokens. Mathematically equivalent to
    # concat([noisy, ref], dim=channel) -> single Linear, with the new
    # channel slice zero-initialized so the model loads cleanly from
    # checkpoints trained without this feature.
    # When False (default), the module is not created and there is no
    # runtime overhead.
    enable_chunk_recycling: bool = False


# ============================================================================
# CausalLTXModel
# ============================================================================

class CausalLTXModel(nn.Module):
    """
    Causal LTX-2 Model for ODE causal training via masking.

    This model is weight-compatible with the original LTX-2 checkpoint.
    All module names match the original LTXModel to enable direct checkpoint loading.

    Module name mapping to original LTXModel:
        patchify_proj           → nn.Linear(128, 4096)
        audio_patchify_proj     → nn.Linear(128, 2048)
        adaln_single            → AdaLayerNormSingle(4096, coefficient=6)
        audio_adaln_single      → AdaLayerNormSingle(2048, coefficient=6)
        caption_projection      → PixArtAlphaTextProjection(3840, 4096)
        audio_caption_projection → PixArtAlphaTextProjection(3840, 2048)
        av_ca_video_scale_shift_adaln_single → AdaLayerNormSingle(4096, coefficient=4)
        av_ca_audio_scale_shift_adaln_single → AdaLayerNormSingle(2048, coefficient=4)
        av_ca_a2v_gate_adaln_single          → AdaLayerNormSingle(4096, coefficient=1)
        av_ca_v2a_gate_adaln_single          → AdaLayerNormSingle(2048, coefficient=1)
        transformer_blocks      → ModuleList of CausalAVTransformerBlock
        scale_shift_table       → Parameter([2, 4096])
        norm_out                → LayerNorm(4096, elementwise_affine=False)
        proj_out                → nn.Linear(4096, 128)
        audio_scale_shift_table → Parameter([2, 2048])
        audio_norm_out          → LayerNorm(2048, elementwise_affine=False)
        audio_proj_out          → nn.Linear(2048, 128)
    """

    def __init__(self, config: CausalLTXModelConfig):
        super().__init__()
        self.config = config

        # Store key parameters for easy access
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = config.av_ca_timestep_scale_multiplier

        # === Patch Embedding (Linear, matching original) ===
        self.patchify_proj = nn.Linear(config.in_channels, config.video_dim, bias=True)
        self.audio_patchify_proj = nn.Linear(config.in_channels, config.audio_dim, bias=True)

        # === Optional: AAPT-style channel-concat ref projection ===
        # Separate Linear (no bias, zero-init) added to patchify_proj output.
        # Mathematically equivalent to channel-concat([noisy, ref], dim=C)
        # then a single Linear(2C, D) with the second half of weights zero-init.
        # When disabled, the module is absent and forward has zero overhead.
        if getattr(config, "enable_chunk_recycling", False):
            self.ref_patchify_proj = nn.Linear(
                config.in_channels, config.video_dim, bias=False,
            )
            nn.init.zeros_(self.ref_patchify_proj.weight)
        else:
            self.ref_patchify_proj = None

        # === AdaLN Timestep Embedding ===
        # Main AdaLN: outputs [N, 6*D] for per-block shift/scale/gate + [N, D] embedded_timestep
        self.adaln_single = AdaLayerNormSingle(config.video_dim, embedding_coefficient=6)
        self.audio_adaln_single = AdaLayerNormSingle(config.audio_dim, embedding_coefficient=6)

        # === Caption (Text) Projection ===
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=config.caption_channels,
            hidden_size=config.video_dim,
        )
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=config.caption_channels,
            hidden_size=config.audio_dim,
        )

        # === Cross-Attention AdaLN (for A2V / V2A) ===
        # 4 additional AdaLN modules for cross-modal timestep conditioning
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            config.video_dim, embedding_coefficient=4,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            config.audio_dim, embedding_coefficient=4,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            config.video_dim, embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            config.audio_dim, embedding_coefficient=1,
        )

        # === Transformer Blocks ===
        video_config = TransformerConfig(
            dim=config.video_dim,
            heads=config.video_heads,
            d_head=config.video_d_head,
            context_dim=config.cross_attention_dim,
        )

        audio_config = TransformerConfig(
            dim=config.audio_dim,
            heads=config.audio_heads,
            d_head=config.audio_d_head,
            context_dim=config.audio_cross_attention_dim,
        )

        # Name must be `transformer_blocks` to match original state_dict
        self.transformer_blocks = nn.ModuleList([
            CausalAVTransformerBlock(
                idx=i,
                video=video_config,
                audio=audio_config,
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
            )
            for i in range(config.num_layers)
        ])

        # === Output Layers (matching original names exactly) ===
        # Video output
        self.scale_shift_table = nn.Parameter(torch.empty(2, config.video_dim))
        self.norm_out = nn.LayerNorm(
            config.video_dim, elementwise_affine=False, eps=config.norm_eps
        )
        self.proj_out = nn.Linear(config.video_dim, config.out_channels)

        # Audio output
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, config.audio_dim))
        self.audio_norm_out = nn.LayerNorm(
            config.audio_dim, elementwise_affine=False, eps=config.norm_eps
        )
        self.audio_proj_out = nn.Linear(config.audio_dim, config.out_channels)

        # === Audio Sink Tokens ===
        if config.num_audio_sink_tokens > 0:
            self.audio_sink_tokens = nn.Parameter(
                torch.zeros(1, config.num_audio_sink_tokens, config.audio_dim)
            )
            nn.init.normal_(self.audio_sink_tokens, std=0.02)

        # === Mask Builder ===
        self.mask_builder = AVCausalMaskBuilder(
            video_frame_seqlen=config.video_frame_seqlen,
            audio_frame_seqlen=config.audio_frame_seqlen,
            num_frame_per_block=config.num_frame_per_block,
            num_audio_sink_tokens=config.num_audio_sink_tokens,
        )

        # Gradient checkpointing
        self.gradient_checkpointing = False

    # ================================================================
    # Timestep Processing
    # ================================================================

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings via AdaLayerNormSingle.

        Matches TransformerArgsPreprocessor._prepare_timestep exactly.

        Args:
            timestep: [B] or [B, F] raw sigma values
            adaln: AdaLayerNormSingle module
            batch_size: Batch size
            hidden_dtype: Target dtype

        Returns:
            (timestep_6d, embedded_timestep):
                timestep_6d: [B, L, 6*D] for block AdaLN
                embedded_timestep: [B, L, D] for output layer
        """
        timestep = timestep * self.timestep_scale_multiplier
        timestep_6d, embedded_timestep = adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        timestep_6d = timestep_6d.view(batch_size, -1, timestep_6d.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep_6d, embedded_timestep

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross-attention timestep embeddings.

        Matches MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep.

        Args:
            timestep: [B] or [B, F] raw sigma values
            cross_scale_shift_adaln: AdaLN for scale/shift (coefficient=4)
            cross_gate_adaln: AdaLN for gate (coefficient=1)
            batch_size: Batch size
            hidden_dtype: Target dtype

        Returns:
            (scale_shift_timestep, gate_timestep):
                scale_shift_timestep: [B, L, 4*D]
                gate_timestep: [B, L, D]
        """
        timestep_scaled = timestep * self.timestep_scale_multiplier
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

        scale_shift_timestep, _ = cross_scale_shift_adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(
            batch_size, -1, scale_shift_timestep.shape[-1]
        )

        gate_timestep, _ = cross_gate_adaln(
            timestep_scaled.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_timestep = gate_timestep.view(
            batch_size, -1, gate_timestep.shape[-1]
        )

        return scale_shift_timestep, gate_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        projection: PixArtAlphaTextProjection,
        target_dim: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Project and reshape text context.

        Args:
            context: [B, L_ctx, caption_channels] raw text embeddings
            projection: PixArtAlphaTextProjection module
            target_dim: Target hidden dimension
            batch_size: Batch size

        Returns:
            [B, L_ctx, target_dim] projected context
        """
        context = projection(context)
        context = context.view(batch_size, -1, target_dim)
        return context

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Prepare attention mask (convert bool/int to float mask)."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    # ================================================================
    # Output Processing
    # ================================================================

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output with timestep-conditioned modulation.

        Matches LTXModel._process_output exactly.

        Args:
            scale_shift_table: [2, D] learnable parameters
            norm_out: LayerNorm (elementwise_affine=False)
            proj_out: Linear projection to output channels
            x: [B, T, D] transformer output
            embedded_timestep: [B, L, D] from AdaLN

        Returns:
            [B, T, out_channels] projected output
        """
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    # ================================================================
    # Patchify helpers
    # ================================================================

    def _patchify_video(
        self,
        video_latent: torch.Tensor,
        ref_video_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patchify a video latent and optionally add the AAPT-style ref signal.

        Args:
            video_latent: [B, F, C, H, W]
            ref_video_latent: Optional [B, F, C, H, W] same shape; only consumed
                when ``self.ref_patchify_proj`` is not None. Out-of-distribution
                positions (e.g. chunk 0 with no past) should be zeros.

        Returns:
            video_tokens: [B, F*H*W, D]
        """
        B_v, F_v, C_v, H_v, W_v = video_latent.shape
        v_flat = video_latent.permute(0, 2, 1, 3, 4)            # [B, C, F, H, W]
        v_flat = v_flat.reshape(B_v, C_v, -1).permute(0, 2, 1)  # [B, F*H*W, C]
        out = self.patchify_proj(v_flat)                        # [B, T, D]

        if self.ref_patchify_proj is not None and ref_video_latent is not None:
            assert ref_video_latent.shape == video_latent.shape, (
                f"ref_video_latent shape {tuple(ref_video_latent.shape)} must "
                f"match video_latent shape {tuple(video_latent.shape)}"
            )
            r_flat = ref_video_latent.permute(0, 2, 1, 3, 4)
            r_flat = r_flat.reshape(B_v, C_v, -1).permute(0, 2, 1)
            ref_out = self.ref_patchify_proj(r_flat)

            # Sanity stats for chunk recycling (cached as GPU scalars; no sync
            # in the forward path). Captured INSIDE forward so under FSDP the
            # ref_patchify_proj weight is in its all-gathered state, giving
            # a meaningful global norm rather than a local shard norm.
            with torch.no_grad():
                self._last_ref_patch_norm = ref_out.detach().norm()
                self._last_noisy_patch_norm = out.detach().norm()
                self._last_ref_weight_norm = (
                    self.ref_patchify_proj.weight.detach().norm()
                )

            out = out + ref_out

        return out

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, *args, _tf_kwargs: Optional[Dict] = None, **kwargs):
        """Dispatch to standard / teacher-forcing / KV-cache forward.

        Routing rules (checked in order):
          1. ``_tf_kwargs`` provided -> ``_forward_teacher_forcing``.
          2. ``kv_caches`` in kwargs and not None -> ``_forward_with_cache``
             (per-block KV-cache codepath used by Self-Forcing rollout and
             KV-cache causal inference).
          3. Otherwise -> ``_forward_standard`` (full-sequence training).

        Routing through ``forward()`` is required so that FSDP hooks
        (parameter all-gather / release) fire correctly.
        """
        if _tf_kwargs is not None:
            return self._forward_teacher_forcing(**_tf_kwargs)

        kv_caches = kwargs.pop("kv_caches", None)
        if kv_caches is not None:
            return self._forward_with_cache(*args, kv_caches=kv_caches, **kwargs)

        # Strip cache-only kwargs in case caller forgot to remove them.
        kwargs.pop("current_video_start_frame", None)
        kwargs.pop("current_audio_start_frame", None)
        return self._forward_standard(*args, **kwargs)

    def _forward_standard(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timesteps: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        audio_timesteps: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, Any]] = None,
        ref_video_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass (training only).

        Args:
            video_latent: [B, F_v, C, H, W] video latent
            audio_latent: [B, F_a, C] audio latent
            timesteps: [B] or [B, F_v] diffusion timesteps (sigma values)
            video_context: [B, L_ctx, caption_channels] video text context
            audio_context: [B, L_ctx, caption_channels] audio text context
            video_context_mask: Optional attention mask for video context
            audio_context_mask: Optional attention mask for audio context
            audio_timesteps: [B] or [B, F_a] audio timesteps (optional, defaults to video)
            masks: Pre-computed causal masks (from build_all_causal_masks)

        Returns:
            (video_velocity, audio_velocity): Velocity predictions
        """
        B = video_latent.shape[0]
        device = video_latent.device
        hidden_dtype = video_latent.dtype

        # === Patch Embedding ===
        # Video: [B, F, C, H, W] → patchify → [B, T, C] → project → [B, T, D]
        # When ref_video_latent is provided and ref_patchify_proj is enabled,
        # the ref signal is added on top of the noisy patchify output.
        B_v, F_v, C_v, H_v, W_v = video_latent.shape
        video_x = self._patchify_video(video_latent, ref_video_latent)  # [B, T, D]
        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

        # Audio: [B, F_a, C] -> [B, F_a, audio_dim]
        audio_x = self.audio_patchify_proj(audio_latent)
        F_a_original = audio_x.shape[1]
        audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

        # === Prepend Audio Sink Tokens ===
        num_sink = self.config.num_audio_sink_tokens
        if num_sink > 0:
            sink_expanded = self.audio_sink_tokens.expand(B, -1, -1).to(audio_x.dtype)
            audio_x = torch.cat([sink_expanded, audio_x], dim=1)

        # === Context Projection ===
        video_ctx = self._prepare_context(
            video_context, self.caption_projection, self.config.video_dim, B
        )
        audio_ctx = self._prepare_context(
            audio_context, self.audio_caption_projection, self.config.audio_dim, B
        )

        # Prepare attention masks
        video_context_mask = self._prepare_attention_mask(video_context_mask, hidden_dtype)
        audio_context_mask = self._prepare_attention_mask(audio_context_mask, hidden_dtype)

        # === Timestep Embedding via AdaLayerNormSingle ===
        # Video timestep
        video_ts = timesteps  # [B] or [B, F_v]
        video_timestep_6d, video_embedded_ts = self._prepare_timestep(
            video_ts, self.adaln_single, B, hidden_dtype
        )

        # Audio timestep (defaults to video timestep if not provided)
        audio_ts = audio_timesteps if audio_timesteps is not None else timesteps

        # Expand audio timestep with sink entries (same as first frame's timestep)
        if num_sink > 0 and audio_ts.ndim == 2:
            # audio_ts is [B, F_a], prepend num_sink copies of first frame's value
            sink_ts = audio_ts[:, :1].expand(-1, num_sink)  # [B, num_sink]
            audio_ts_expanded = torch.cat([sink_ts, audio_ts], dim=1)  # [B, num_sink + F_a]
        else:
            audio_ts_expanded = audio_ts

        # Save original audio embedded timestep (without sinks) for output
        audio_timestep_6d, audio_embedded_ts_full = self._prepare_timestep(
            audio_ts_expanded, self.audio_adaln_single, B, hidden_dtype
        )
        # Strip sink entries from embedded timestep for output processing
        if num_sink > 0 and audio_embedded_ts_full.shape[1] > 1:
            audio_embedded_ts = audio_embedded_ts_full[:, num_sink:]
        else:
            audio_embedded_ts = audio_embedded_ts_full

        # === Cross-Attention Timesteps ===
        video_cross_ss, video_cross_gate = self._prepare_cross_attention_timestep(
            video_ts,
            self.av_ca_video_scale_shift_adaln_single,
            self.av_ca_a2v_gate_adaln_single,
            B, hidden_dtype,
        )
        audio_cross_ss, audio_cross_gate = self._prepare_cross_attention_timestep(
            audio_ts_expanded,
            self.av_ca_audio_scale_shift_adaln_single,
            self.av_ca_v2a_gate_adaln_single,
            B, hidden_dtype,
        )

        # === Expand per-frame timestep embeddings to per-token ===
        # When timesteps is [B, F_v] (per-frame), AdaLN output is [B, F_v, *]
        # but transformer blocks need [B, F_v*H*W, *] (per-token).
        # When timesteps is [B] (scalar), output is [B, 1, *] which broadcasts.
        # Audio has 1 token/frame so no expansion needed.
        frame_seqlen = H_v * W_v
        assert video_x.shape[1] == F_v * frame_seqlen, (
            f"video tokens mismatch: got {video_x.shape[1]}, expected {F_v * frame_seqlen}"
        )
        video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)

        # Reuse wrapper-provided masks whenever available. The wrapper already
        # builds them with num_audio_sink_tokens=num_sink, so sink tokens do not
        # require an unconditional rebuild here.
        if masks is None:
            num_video_frames = video_grid_sizes[0, 0].item()
            num_audio_frames = audio_grid_sizes[0, 0].item()  # Original count (without sinks)
            mask_config = CausalMaskConfig(
                video_frame_seqlen=frame_seqlen,
                num_frame_per_block=self.config.num_frame_per_block,
                first_block_video_frames=self.config.first_block_video_frames,
                num_audio_sink_tokens=num_sink,
            )
            masks = build_all_causal_masks(
                num_video_frames, num_audio_frames,
                config=mask_config,
                device=device,
            )

        # === Compute log-ratio scales for causal attention rescaling ===
        log_scales = None
        if self.config.enable_causal_log_rescale:
            blocks = compute_av_blocks(
                F_v, self.config.num_frame_per_block,
                self.config.first_block_video_frames,
            )
            log_scales = compute_causal_log_scales(
                blocks,
                video_frame_seqlen=frame_seqlen,
                audio_frame_seqlen=self.config.audio_frame_seqlen,
                device=device,
                num_audio_sink_tokens=num_sink,
            )

        # === Precompute RoPE ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
        )

        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )

        # Prepend identity RoPE for sink tokens (cos=1, sin=0 → no rotation)
        if num_sink > 0:
            audio_rope_dim = audio_pe[0].shape[-1]
            sink_cos = torch.ones(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[0].dtype)
            sink_sin = torch.zeros(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[1].dtype)
            audio_pe = (
                torch.cat([sink_cos, audio_pe[0]], dim=1),
                torch.cat([sink_sin, audio_pe[1]], dim=1),
            )

        # === Cross-attention RoPE ===
        # Original uses 1D temporal-only positions at audio_cross_attention_dim (2048).
        # cross_pe_max_pos = max(pe_max_pos[0], audio_pe_max_pos[0]) = max(20, 20) = 20
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0],
            self.config.audio_pe_max_pos[0],
        )
        # Video cross-PE: temporal positions from video grid (1D, video temporal)
        video_temporal_grid = torch.tensor(
            [[F_v]], device=device, dtype=torch.long
        )  # [1, 1]
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
            is_audio=False,  # Video temporal conversion
        )
        # Expand temporal PE to full video sequence: each frame's tokens share same temporal PE
        # video_cross_pe: [B, F_v, D] → need [B, F_v*H*W, D]
        video_cross_pe = (
            video_cross_pe[0].unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1].unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )

        # Audio cross-PE: temporal positions from audio grid (1D, audio temporal)
        # Use original audio frame count (without sinks) for cross-PE computation
        audio_temporal_grid = torch.tensor(
            [[F_a_original]], device=device, dtype=torch.long
        )  # [1, 1]
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,  # Audio temporal conversion
        )

        # Prepend identity RoPE for sink tokens in cross-PE
        if num_sink > 0:
            cross_rope_dim = audio_cross_pe[0].shape[-1]
            sink_cross_cos = torch.ones(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[0].dtype)
            sink_cross_sin = torch.zeros(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[1].dtype)
            audio_cross_pe = (
                torch.cat([sink_cross_cos, audio_cross_pe[0]], dim=1),
                torch.cat([sink_cross_sin, audio_cross_pe[1]], dim=1),
            )

        # === Prepare Transformer Args ===
        video_args = CausalTransformerArgs(
            x=video_x,
            timesteps=video_timestep_6d,
            positional_embeddings=video_pe,
            context=video_ctx,
            context_mask=video_context_mask,
            block_mask=masks.get('video_self'),
            cross_causal_mask=masks.get('a2v'),
            cross_positional_embeddings=video_cross_pe,
            cross_scale_shift_timestep=video_cross_ss,
            cross_gate_timestep=video_cross_gate,
            self_attn_log_scale=log_scales['video_self_scale'].to(hidden_dtype) if log_scales else None,
            cross_attn_log_scale=log_scales['a2v_scale'].to(hidden_dtype) if log_scales else None,
        )

        audio_args = CausalTransformerArgs(
            x=audio_x,
            timesteps=audio_timestep_6d,
            positional_embeddings=audio_pe,
            context=audio_ctx,
            context_mask=audio_context_mask,
            block_mask=masks.get('audio_self'),
            cross_causal_mask=masks.get('v2a'),
            cross_positional_embeddings=audio_cross_pe,
            cross_scale_shift_timestep=audio_cross_ss,
            cross_gate_timestep=audio_cross_gate,
            self_attn_log_scale=log_scales['audio_self_scale'].to(hidden_dtype) if log_scales else None,
            cross_attn_log_scale=log_scales['v2a_scale'].to(hidden_dtype) if log_scales else None,
        )

        # === Transformer Blocks ===
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block, video_args, audio_args,
                    use_reentrant=False,
                    determinism_check="none",
                )
            else:
                video_args, audio_args = block(video_args, audio_args)

        # === Strip sink tokens before output processing ===
        audio_out_x = audio_args.x
        if num_sink > 0:
            audio_out_x = audio_out_x[:, num_sink:]

        # === Output Layer (with timestep conditioning) ===
        video_out = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out,
            video_args.x, video_embedded_ts,
        )
        audio_out = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            audio_out_x, audio_embedded_ts,
        )

        # === Unpatchify Video ===
        # [B, T, C] → [B, F, H, W, C] → [B, F, C, H, W]
        F, H, W = F_v, H_v, W_v
        C_out = self.config.out_channels
        video_out = video_out.reshape(B, F, H, W, C_out)
        video_out = video_out.permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]

        return video_out, audio_out

    # ================================================================
    # Teacher Forcing Forward Pass (Method A: 2x sequence)
    # ================================================================

    def _forward_teacher_forcing(
        self,
        noisy_video_latent: torch.Tensor,
        noisy_audio_latent: torch.Tensor,
        clean_video_latent: torch.Tensor,
        clean_audio_latent: torch.Tensor,
        noisy_timesteps: torch.Tensor,
        clean_timesteps: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        noisy_audio_timesteps: Optional[torch.Tensor] = None,
        clean_audio_timesteps: Optional[torch.Tensor] = None,
        tf_masks: Optional[Dict[str, Any]] = None,
        ref_video_latent: Optional[torch.Tensor] = None,
        return_features: bool = False,
        feature_layer_shallow: Optional[int] = None,
        feature_layer_deep: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher Forcing forward pass with 2x sequence concatenation.

        When ``return_features`` is True, additionally returns intermediate
        hidden states (noisy half only) sampled at a shallow and a deep block,
        for Causal Self-Flow feature alignment. The shallow features keep grad
        (consumed by the student branch through a projector); the deep features
        are detached (used as the teacher target). Audio sink tokens are
        stripped from the audio features so they line up with the audio frames.

        ``ref_video_latent`` (when provided and ``ref_patchify_proj`` is enabled)
        is added on top of the patchified NOISY half only — the clean half does
        not receive ref injection because clean tokens themselves already encode
        the past frames via the [clean | noisy] attention path. This mirrors the
        inference behavior where ref is fed alongside the noisy current chunk.

        Concatenates [clean | noisy] along the sequence dimension, applies
        teacher forcing masks so noisy block k sees clean blocks 0..k-1
        but not clean block k, then returns only the noisy half output.

        Args:
            noisy_video_latent: [B, F_v, C, H, W] noisy video
            noisy_audio_latent: [B, F_a, C] noisy audio
            clean_video_latent: [B, F_v, C, H, W] clean video
            clean_audio_latent: [B, F_a, C] clean audio
            noisy_timesteps: [B] or [B, F_v] noisy diffusion sigma
            clean_timesteps: [B] or [B, F_v] clean sigma (typically 0)
            video_context: [B, L, D] video text embeddings
            audio_context: [B, L, D] audio text embeddings
            noisy_audio_timesteps: [B] or [B, F_a] noisy audio sigma
            clean_audio_timesteps: [B] or [B, F_a] clean audio sigma
            tf_masks: Pre-built teacher forcing masks

        Returns:
            (video_velocity, audio_velocity) for the noisy half only
        """
        B = noisy_video_latent.shape[0]
        device = noisy_video_latent.device
        hidden_dtype = noisy_video_latent.dtype

        # === Patchify both clean and noisy ===
        # Ref injection (if enabled and ref provided) only applies to the noisy
        # half. The clean half is patchified plainly so its role as ground-truth
        # past context (visible via attention) is unaffected.
        B_v, F_v, C_v, H_v, W_v = noisy_video_latent.shape
        frame_seqlen = H_v * W_v

        noisy_video_x = self._patchify_video(noisy_video_latent, ref_video_latent)
        clean_video_x = self._patchify_video(clean_video_latent, None)
        assert noisy_video_x.shape[1] == F_v * frame_seqlen, (
            f"noisy video tokens mismatch: got {noisy_video_x.shape[1]}, expected {F_v * frame_seqlen}"
        )

        noisy_audio_x = self.audio_patchify_proj(noisy_audio_latent)  # [B, F_a, D_a]
        clean_audio_x = self.audio_patchify_proj(clean_audio_latent)  # [B, F_a, D_a]
        F_a_original = noisy_audio_x.shape[1]

        # === Prepend Audio Sink Tokens to both halves ===
        num_sink = self.config.num_audio_sink_tokens
        if num_sink > 0:
            sink_expanded = self.audio_sink_tokens.expand(B, -1, -1).to(noisy_audio_x.dtype)
            noisy_audio_x = torch.cat([sink_expanded, noisy_audio_x], dim=1)
            clean_audio_x = torch.cat([sink_expanded.clone(), clean_audio_x], dim=1)

        # === Concatenate [clean | noisy] along sequence dim ===
        video_x = torch.cat([clean_video_x, noisy_video_x], dim=1)  # [B, 2*T_v, D]
        audio_x = torch.cat([clean_audio_x, noisy_audio_x], dim=1)  # [B, 2*T_a, D_a]

        T_v_tokens = noisy_video_x.shape[1]
        T_a_tokens = noisy_audio_x.shape[1]

        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)
        audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

        # === Context Projection ===
        video_ctx = self._prepare_context(
            video_context, self.caption_projection, self.config.video_dim, B
        )
        audio_ctx = self._prepare_context(
            audio_context, self.audio_caption_projection, self.config.audio_dim, B
        )
        video_context_mask = self._prepare_attention_mask(video_context_mask, hidden_dtype)
        audio_context_mask = self._prepare_attention_mask(audio_context_mask, hidden_dtype)

        # === Timestep Embedding for both halves ===
        # Concat [clean_ts | noisy_ts] along frame dim, then embed together
        def _concat_timesteps(clean_ts, noisy_ts, num_frames):
            if clean_ts.ndim == 1:
                clean_ts = clean_ts.unsqueeze(1).expand(-1, num_frames)
            if noisy_ts.ndim == 1:
                noisy_ts = noisy_ts.unsqueeze(1).expand(-1, num_frames)
            return torch.cat([clean_ts, noisy_ts], dim=1)  # [B, 2*F]

        video_ts_cat = _concat_timesteps(clean_timesteps, noisy_timesteps, F_v)
        video_timestep_6d, video_embedded_ts = self._prepare_timestep(
            video_ts_cat, self.adaln_single, B, hidden_dtype
        )

        noisy_ats = noisy_audio_timesteps if noisy_audio_timesteps is not None else noisy_timesteps
        clean_ats = clean_audio_timesteps if clean_audio_timesteps is not None else clean_timesteps
        audio_ts_cat = _concat_timesteps(clean_ats, noisy_ats, F_a_original + num_sink)
        audio_timestep_6d, audio_embedded_ts_full = self._prepare_timestep(
            audio_ts_cat, self.audio_adaln_single, B, hidden_dtype
        )

        # === Cross-Attention Timesteps ===
        video_cross_ss, video_cross_gate = self._prepare_cross_attention_timestep(
            video_ts_cat, self.av_ca_video_scale_shift_adaln_single,
            self.av_ca_a2v_gate_adaln_single, B, hidden_dtype,
        )
        audio_cross_ss, audio_cross_gate = self._prepare_cross_attention_timestep(
            audio_ts_cat, self.av_ca_audio_scale_shift_adaln_single,
            self.av_ca_v2a_gate_adaln_single, B, hidden_dtype,
        )

        # === Expand per-frame to per-token for video (2x frames) ===
        video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)

        # === Build or reuse TF masks ===
        if tf_masks is None:
            mask_config = CausalMaskConfig(
                video_frame_seqlen=frame_seqlen,
                num_frame_per_block=self.config.num_frame_per_block,
                first_block_video_frames=self.config.first_block_video_frames,
                num_audio_sink_tokens=num_sink,
            )
            tf_masks = build_all_teacher_forcing_masks(
                F_v, F_a_original, config=mask_config, device=device,
            )

        # === RoPE: same positions for both halves ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
        )
        # Duplicate for 2x sequence: same positional encoding for clean and noisy halves
        video_pe = (
            video_pe[0].repeat(1, 2, 1),
            video_pe[1].repeat(1, 2, 1),
        )

        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )
        if num_sink > 0:
            audio_rope_dim = audio_pe[0].shape[-1]
            sink_cos = torch.ones(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[0].dtype)
            sink_sin = torch.zeros(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[1].dtype)
            audio_pe = (
                torch.cat([sink_cos, audio_pe[0]], dim=1),
                torch.cat([sink_sin, audio_pe[1]], dim=1),
            )
        audio_pe = (
            audio_pe[0].repeat(1, 2, 1),
            audio_pe[1].repeat(1, 2, 1),
        )

        # === Cross-attention RoPE (also 2x) ===
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0], self.config.audio_pe_max_pos[0],
        )
        video_temporal_grid = torch.tensor([[F_v]], device=device, dtype=torch.long)
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid, self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta, max_pos=[cross_pe_max_pos],
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype, is_audio=False,
        )
        video_cross_pe = (
            video_cross_pe[0].unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1].unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )
        video_cross_pe = (
            video_cross_pe[0].repeat(1, 2, 1),
            video_cross_pe[1].repeat(1, 2, 1),
        )

        audio_temporal_grid = torch.tensor([[F_a_original]], device=device, dtype=torch.long)
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid, self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta, max_pos=[cross_pe_max_pos],
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype, is_audio=True,
        )
        if num_sink > 0:
            cross_rope_dim = audio_cross_pe[0].shape[-1]
            sink_cross_cos = torch.ones(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[0].dtype)
            sink_cross_sin = torch.zeros(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[1].dtype)
            audio_cross_pe = (
                torch.cat([sink_cross_cos, audio_cross_pe[0]], dim=1),
                torch.cat([sink_cross_sin, audio_cross_pe[1]], dim=1),
            )
        audio_cross_pe = (
            audio_cross_pe[0].repeat(1, 2, 1),
            audio_cross_pe[1].repeat(1, 2, 1),
        )

        # === Prepare Transformer Args (2x sequence) ===
        video_args = CausalTransformerArgs(
            x=video_x,
            timesteps=video_timestep_6d,
            positional_embeddings=video_pe,
            context=video_ctx,
            context_mask=video_context_mask,
            block_mask=tf_masks.get('video_self'),
            cross_causal_mask=tf_masks.get('a2v'),
            cross_positional_embeddings=video_cross_pe,
            cross_scale_shift_timestep=video_cross_ss,
            cross_gate_timestep=video_cross_gate,
            self_attn_log_scale=None,
            cross_attn_log_scale=None,
        )

        audio_args = CausalTransformerArgs(
            x=audio_x,
            timesteps=audio_timestep_6d,
            positional_embeddings=audio_pe,
            context=audio_ctx,
            context_mask=audio_context_mask,
            block_mask=tf_masks.get('audio_self'),
            cross_causal_mask=tf_masks.get('v2a'),
            cross_positional_embeddings=audio_cross_pe,
            cross_scale_shift_timestep=audio_cross_ss,
            cross_gate_timestep=audio_cross_gate,
            self_attn_log_scale=None,
            cross_attn_log_scale=None,
        )

        # === Transformer Blocks ===
        # Resolve Self-Flow feature-extraction layer indices (0-based block idx).
        num_layers = len(self.transformer_blocks)
        if return_features:
            shallow_idx = (
                feature_layer_shallow if feature_layer_shallow is not None
                else round(0.3 * num_layers)
            )
            deep_idx = (
                feature_layer_deep if feature_layer_deep is not None
                else round(0.7 * num_layers)
            )
            shallow_idx = max(0, min(shallow_idx, num_layers - 1))
            deep_idx = max(0, min(deep_idx, num_layers - 1))
        feat_video_shallow = feat_video_deep = None
        feat_audio_shallow = feat_audio_deep = None

        can_checkpoint = self.gradient_checkpointing and self.training
        for block_idx, block in enumerate(self.transformer_blocks):
            if can_checkpoint:
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block,
                    video_args,
                    audio_args,
                    use_reentrant=False,
                    determinism_check="none",
                )
            else:
                video_args, audio_args = block(video_args, audio_args)

            if return_features and (block_idx == shallow_idx or block_idx == deep_idx):
                # Noisy-half hidden states; strip audio sink tokens.
                v_feat = video_args.x[:, T_v_tokens:]
                a_feat = audio_args.x[:, T_a_tokens:]
                if num_sink > 0:
                    a_feat = a_feat[:, num_sink:]
                if block_idx == shallow_idx:
                    # Student branch consumes these -> keep grad.
                    feat_video_shallow = v_feat.clone()
                    feat_audio_shallow = a_feat.clone()
                if block_idx == deep_idx:
                    # Teacher target -> detach.
                    feat_video_deep = v_feat.detach().clone()
                    feat_audio_deep = a_feat.detach().clone()

        # === Take only noisy half ===
        video_noisy_x = video_args.x[:, T_v_tokens:]
        audio_noisy_x = audio_args.x[:, T_a_tokens:]

        # Strip sink tokens from noisy audio output
        if num_sink > 0:
            audio_noisy_x = audio_noisy_x[:, num_sink:]

        # Use only noisy-half embedded timesteps for output processing
        # embedded_ts is [B, 2*F, D], take the noisy half
        noisy_video_embedded_ts = video_embedded_ts[:, T_v_tokens:]
        noisy_audio_embedded_ts = audio_embedded_ts_full[:, T_a_tokens:]
        if num_sink > 0:
            noisy_audio_embedded_ts = noisy_audio_embedded_ts[:, num_sink:]

        # Expand noisy embedded_ts to per-token for output processing
        # noisy_video_embedded_ts is [B, F_v * H*W, D] already expanded
        # noisy_audio_embedded_ts is [B, F_a, D_a]

        # === Output Layer ===
        video_out = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out,
            video_noisy_x, noisy_video_embedded_ts,
        )
        audio_out = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            audio_noisy_x, noisy_audio_embedded_ts,
        )

        # === Unpatchify Video ===
        F, H, W = F_v, H_v, W_v
        C_out = self.config.out_channels
        video_out = video_out.reshape(B, F, H, W, C_out)
        video_out = video_out.permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]

        if return_features:
            feats = {
                "video_shallow": feat_video_shallow,
                "video_deep": feat_video_deep,
                "audio_shallow": feat_audio_shallow,
                "audio_deep": feat_audio_deep,
            }
            return video_out, audio_out, feats

        return video_out, audio_out

    # ================================================================
    # KV-cache Causal Inference / Self-Forcing Training
    # ================================================================
    #
    # Local design notes (differ from upstream OmniForcing a81e632 here):
    #
    # Audio sink tokens (num_audio_sink_tokens, default 16 in stage 2/3) are
    # prepended to the audio sequence in _forward_standard. To keep the
    # KV-cache codepath bit-equivalent to that path under block partition,
    # we apply the SAME sink prepend / identity-RoPE / output-strip logic,
    # but only on block 0 (current_video_start_frame == 0):
    #
    #   - On block 0:
    #       audio_x_block <- [sink_tokens, audio_latent_block_0]
    #       audio_pe_block <- [identity_pe_for_sink, rope(start_frame=0)]
    #       audio_cross_pe_block <- [identity_pe_for_sink, rope(start_frame=0)]
    #       audio_ts_block <- [num_sink * first_frame_ts, audio_ts_block]
    #       a_token_start = 0
    #       cache writes to [0, num_sink + L_a_block_0)
    #       audio_args.x[:, num_sink:] is fed to _process_output (strip sink)
    #
    #   - On block >= 1:
    #       no sink prepend / strip
    #       audio_pe / audio_cross_pe use start_frame=current_audio_start_frame
    #       a_token_start = num_sink + current_audio_start_frame * audio_frame_seqlen
    #       cache writes to [a_token_start, a_token_start + L_a_block)
    #
    # This ensures that across an autoregressive rollout, the audio_self /
    # a2v caches contain exactly the same K/V content (in the same order)
    # as a single full-sequence _forward_standard would produce, so the
    # train / inference paths are numerically aligned.
    # ================================================================

    def init_av_kv_caches(
        self,
        batch_size: int,
        max_video_frames: int,
        max_audio_frames: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        video_frame_seqlen: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Allocate per-layer KV caches for causal AV generation.

        Each layer's cache contains 6 buffers:
          - video_self: video self-attention K/V
          - audio_self: audio self-attention K/V (capacity includes sink slots)
          - a2v: cross-modal A2V K/V (audio K/V projected by audio_to_video_attn,
                 audio heads/d_head; capacity includes sink slots)
          - v2a: cross-modal V2A K/V (video K/V projected by video_to_audio_attn,
                 audio heads/d_head; no sink offset)
          - video_text: video-text crossattn cache (one-shot init)
          - audio_text: audio-text crossattn cache (one-shot init)

        Args:
            batch_size: B
            max_video_frames: total video frames covered by the rollout
                              (including all blocks)
            max_audio_frames: total audio frames covered by the rollout
                              (excluding sink tokens; sink slots are added
                              automatically via num_audio_sink_tokens config)
            text_seq_len: length of text context
            device: cache buffer device
            dtype: cache buffer dtype (must match the model's compute dtype
                   for the SDPA path)
        """
        num_sink = self.config.num_audio_sink_tokens
        runtime_video_frame_seqlen = (
            int(video_frame_seqlen)
            if video_frame_seqlen is not None
            else self.config.video_frame_seqlen
        )
        v_cache_tokens = max_video_frames * runtime_video_frame_seqlen
        a_cache_tokens = num_sink + max_audio_frames * self.config.audio_frame_seqlen

        v_h, v_d = self.config.video_heads, self.config.video_d_head
        a_h, a_d = self.config.audio_heads, self.config.audio_d_head

        def _alloc_self(B: int, T: int, H: int, D: int) -> Dict[str, Any]:
            return {
                "k": torch.zeros(B, T, H, D, device=device, dtype=dtype),
                "v": torch.zeros(B, T, H, D, device=device, dtype=dtype),
                "end": torch.tensor([0], dtype=torch.long, device=device),
            }

        def _alloc_text(B: int, T: int, H: int, D: int) -> Dict[str, Any]:
            return {
                "k": torch.zeros(B, T, H * D, device=device, dtype=dtype),
                "v": torch.zeros(B, T, H * D, device=device, dtype=dtype),
                "len": 0,
                "is_init": False,
            }

        caches: List[Dict[str, Any]] = []
        for _ in range(self.config.num_layers):
            caches.append({
                # video self-attention
                "video_self": _alloc_self(batch_size, v_cache_tokens, v_h, v_d),
                # audio self-attention (with sink offset)
                "audio_self": _alloc_self(batch_size, a_cache_tokens, a_h, a_d),
                # A2V: video Q reads audio K/V (audio heads/d_head, with sink offset)
                "a2v":        _alloc_self(batch_size, a_cache_tokens, a_h, a_d),
                # V2A: audio Q reads video K/V (audio heads/d_head, no sink)
                "v2a":        _alloc_self(batch_size, v_cache_tokens, a_h, a_d),
                # text caches
                "video_text": _alloc_text(batch_size, text_seq_len, v_h, v_d),
                "audio_text": _alloc_text(batch_size, text_seq_len, a_h, a_d),
            })
        return caches

    def _forward_with_cache(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timesteps: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        kv_caches: List[Dict[str, Any]],
        current_video_start_frame: int,
        current_audio_start_frame: int,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        audio_timesteps: Optional[torch.Tensor] = None,
        ref_video_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal autoregressive forward for ONE block, with KV caches.

        Inputs cover the CURRENT block only (block-shaped tensors):
          - video_latent: [B, F_v_block, C, H, W]
          - audio_latent: [B, F_a_block, C]
          - timesteps:    [B] or [B, F_v_block] sigma values
          - audio_timesteps: [B] or [B, F_a_block] (defaults to timesteps)

        Causality is enforced by the cache mechanism: each call writes the
        block's K/V at offsets derived from current_*_start_frame and reads
        cache up to the new end. Future tokens are not yet written.

        Returns:
            (video_velocity_block, audio_velocity_block) for the current block.
        """
        B = video_latent.shape[0]
        device = video_latent.device
        hidden_dtype = video_latent.dtype

        num_sink = self.config.num_audio_sink_tokens
        is_first_block = (current_video_start_frame == 0)

        # === Patch Embedding (current block only) ===
        # ref_video_latent (when provided) injects past-chunk info via
        # ref_patchify_proj added to the patchify output. See _patchify_video.
        B_v, F_v, C_v, H_v, W_v = video_latent.shape
        video_x = self._patchify_video(video_latent, ref_video_latent)
        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

        audio_x = self.audio_patchify_proj(audio_latent)
        F_a_original = audio_x.shape[1]
        audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

        # === Prepend Audio Sink Tokens (block 0 only) ===
        sink_active = num_sink > 0 and is_first_block
        if sink_active:
            sink_expanded = self.audio_sink_tokens.expand(B, -1, -1).to(audio_x.dtype)
            audio_x = torch.cat([sink_expanded, audio_x], dim=1)

        # === Context Projection ===
        video_ctx = self._prepare_context(
            video_context, self.caption_projection, self.config.video_dim, B
        )
        audio_ctx = self._prepare_context(
            audio_context, self.audio_caption_projection, self.config.audio_dim, B
        )
        video_context_mask = self._prepare_attention_mask(video_context_mask, hidden_dtype)
        audio_context_mask = self._prepare_attention_mask(audio_context_mask, hidden_dtype)

        # === Timestep Embedding ===
        video_ts = timesteps
        video_timestep_6d, video_embedded_ts = self._prepare_timestep(
            video_ts, self.adaln_single, B, hidden_dtype
        )

        audio_ts = audio_timesteps if audio_timesteps is not None else timesteps

        # On block 0, prepend num_sink copies of first frame's ts to audio_ts
        # (matches _forward_standard's sink-ts handling).
        if sink_active and audio_ts.ndim == 2:
            sink_ts = audio_ts[:, :1].expand(-1, num_sink)
            audio_ts_expanded = torch.cat([sink_ts, audio_ts], dim=1)
        else:
            audio_ts_expanded = audio_ts

        audio_timestep_6d, audio_embedded_ts_full = self._prepare_timestep(
            audio_ts_expanded, self.audio_adaln_single, B, hidden_dtype
        )
        if sink_active and audio_embedded_ts_full.shape[1] > 1:
            audio_embedded_ts = audio_embedded_ts_full[:, num_sink:]
        else:
            audio_embedded_ts = audio_embedded_ts_full

        # === Cross-Attention Timesteps ===
        video_cross_ss, video_cross_gate = self._prepare_cross_attention_timestep(
            video_ts,
            self.av_ca_video_scale_shift_adaln_single,
            self.av_ca_a2v_gate_adaln_single,
            B, hidden_dtype,
        )
        audio_cross_ss, audio_cross_gate = self._prepare_cross_attention_timestep(
            audio_ts_expanded,
            self.av_ca_audio_scale_shift_adaln_single,
            self.av_ca_v2a_gate_adaln_single,
            B, hidden_dtype,
        )

        # === Expand per-frame to per-token ===
        frame_seqlen = H_v * W_v
        assert video_x.shape[1] == F_v * frame_seqlen, (
            f"cache video tokens mismatch: got {video_x.shape[1]}, expected {F_v * frame_seqlen}"
        )
        video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)

        # === RoPE for the CURRENT BLOCK at the given start_frame ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=current_video_start_frame,
            rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
        )

        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=current_audio_start_frame,
            rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )

        # On block 0, prepend identity RoPE (cos=1, sin=0) for sink tokens.
        if sink_active:
            audio_rope_dim = audio_pe[0].shape[-1]
            sink_cos = torch.ones(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[0].dtype)
            sink_sin = torch.zeros(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[1].dtype)
            audio_pe = (
                torch.cat([sink_cos, audio_pe[0]], dim=1),
                torch.cat([sink_sin, audio_pe[1]], dim=1),
            )

        # === Cross-attention RoPE (current block only) ===
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0],
            self.config.audio_pe_max_pos[0],
        )
        # Video temporal cross-PE for current block
        video_temporal_grid = torch.tensor([[F_v]], device=device, dtype=torch.long)
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=current_video_start_frame,
            rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
            is_audio=False,
        )
        # Expand temporal -> per-token along spatial dim
        video_cross_pe = (
            video_cross_pe[0].unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1].unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )

        # Audio temporal cross-PE for current block
        audio_temporal_grid = torch.tensor([[F_a_original]], device=device, dtype=torch.long)
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=current_audio_start_frame,
            rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )
        if sink_active:
            cross_rope_dim = audio_cross_pe[0].shape[-1]
            sink_cross_cos = torch.ones(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[0].dtype)
            sink_cross_sin = torch.zeros(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[1].dtype)
            audio_cross_pe = (
                torch.cat([sink_cross_cos, audio_cross_pe[0]], dim=1),
                torch.cat([sink_cross_sin, audio_cross_pe[1]], dim=1),
            )

        # === Token offsets into cache ring buffers ===
        v_token_start = current_video_start_frame * frame_seqlen
        if sink_active:
            # Block 0: write sinks + frames starting at offset 0.
            a_token_start = 0
        else:
            # Block >= 1: skip sink slots, write at the post-sink audio offset.
            a_token_start = num_sink + current_audio_start_frame * self.config.audio_frame_seqlen

        # === Run transformer blocks with per-layer caches ===
        for layer_idx, block in enumerate(self.transformer_blocks):
            layer_caches = kv_caches[layer_idx]
            video_args = CausalTransformerArgs(
                x=video_x,
                timesteps=video_timestep_6d,
                positional_embeddings=video_pe,
                context=video_ctx,
                context_mask=video_context_mask,
                cross_positional_embeddings=video_cross_pe,
                cross_scale_shift_timestep=video_cross_ss,
                cross_gate_timestep=video_cross_gate,
                kv_cache_self=layer_caches["video_self"],
                kv_cache_cross=layer_caches["a2v"],          # video Q reads audio K/V
                crossattn_cache_text=layer_caches["video_text"],
                current_token_start=v_token_start,
                cross_token_start=a_token_start,
            )
            audio_args = CausalTransformerArgs(
                x=audio_x,
                timesteps=audio_timestep_6d,
                positional_embeddings=audio_pe,
                context=audio_ctx,
                context_mask=audio_context_mask,
                cross_positional_embeddings=audio_cross_pe,
                cross_scale_shift_timestep=audio_cross_ss,
                cross_gate_timestep=audio_cross_gate,
                kv_cache_self=layer_caches["audio_self"],
                kv_cache_cross=layer_caches["v2a"],          # audio Q reads video K/V
                crossattn_cache_text=layer_caches["audio_text"],
                current_token_start=a_token_start,
                cross_token_start=v_token_start,
            )
            video_args, audio_args = block(video_args, audio_args)
            video_x = video_args.x
            audio_x = audio_args.x

        # === Strip sink tokens before output processing (block 0 only) ===
        if sink_active:
            audio_x = audio_x[:, num_sink:]

        # === Output Layer ===
        video_out = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out,
            video_x, video_embedded_ts,
        )
        audio_out = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            audio_x, audio_embedded_ts,
        )

        # === Unpatchify Video (current block) ===
        F, H, W = F_v, H_v, W_v
        C_out = self.config.out_channels
        video_out = video_out.reshape(B, F, H, W, C_out)
        video_out = video_out.permute(0, 1, 4, 2, 3)

        return video_out, audio_out

    # ================================================================
    # Model Loading
    # ================================================================

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[CausalLTXModelConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CausalLTXModel":
        """
        Load model from pretrained LTX-2 checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration (uses defaults if None)
            device: Target device
            dtype: Model dtype

        Returns:
            Initialized CausalLTXModel with loaded weights
        """
        if config is None:
            config = CausalLTXModelConfig()

        model = cls(config)
        model = model.to(device=device, dtype=dtype)

        # Load checkpoint
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Load with strict=False to ignore mask_builder and optional sink-token keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Verify only expected keys are missing
        expected_missing = ['mask_builder']
        if config.num_audio_sink_tokens > 0:
            expected_missing.append('audio_sink_tokens')
        real_missing = [
            key for key in missing
            if not any(pat in key for pat in expected_missing)
        ]
        if real_missing:
            raise RuntimeError(
                f"Missing keys in checkpoint: {real_missing[:10]}. "
                f"Total missing: {len(real_missing)}. "
                f"This likely means the checkpoint is incompatible with CausalLTXModel."
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {unexpected[:10]}. "
                f"Total unexpected: {len(unexpected)}. "
                f"This likely means the checkpoint is incompatible with CausalLTXModel."
            )

        return model

    @staticmethod
    def _expand_per_frame_to_per_token(
        per_frame: torch.Tensor,
        frame_seqlen: int,
    ) -> torch.Tensor:
        """Expand per-frame tensor to per-token by repeating each frame's value.

        When timesteps is [B, F] (per-frame), AdaLN outputs [B, F, D].
        But transformer blocks expect [B, F*H*W, D] (per-token).
        Each frame's H*W tokens share the same timestep embedding.

        When timesteps is [B] (scalar), AdaLN outputs [B, 1, D] which
        broadcasts naturally, so no expansion is needed.

        Args:
            per_frame: [B, F, D] per-frame values
            frame_seqlen: Number of tokens per frame (H*W for video, 1 for audio)

        Returns:
            [B, F*frame_seqlen, D] per-token values
        """
        if per_frame.shape[1] <= 1 or frame_seqlen <= 1:
            return per_frame
        B, F, D = per_frame.shape
        return (
            per_frame.unsqueeze(2)
            .expand(-1, -1, frame_seqlen, -1)
            .reshape(B, F * frame_seqlen, D)
        )

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
