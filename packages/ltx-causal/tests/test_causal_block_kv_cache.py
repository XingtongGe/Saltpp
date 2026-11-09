"""Block-level Gate for the KV-cache migration (Phase 2).

Verifies that ``CausalAVTransformerBlock.forward()`` correctly routes the
new ``kv_cache_self / kv_cache_cross / crossattn_cache_text`` fields
through the four attention call sites (video self, audio self, A2V, V2A,
plus video-text and audio-text cross-attention).

Equivalence claim:
- Path A: block.forward with no caches and no masks -> full attention
  inside the block (within-sequence bidirectional).
- Path B: same inputs, kv_cache_self/kv_cache_cross/crossattn_cache_text
  set, empty caches, kv_start=0. The cache codepath writes K/V at
  [0, L) and reads back the same K/V it just wrote (no history) -> also
  full attention inside the block.

Both paths must produce identical output (up to fp32 roundoff).
"""

from __future__ import annotations

import pytest
import torch

from ltx_causal.transformer.causal_block import (
    CausalAVTransformerBlock,
    CausalTransformerArgs,
    TransformerConfig,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_block(
    video_dim: int,
    audio_dim: int,
    video_heads: int,
    audio_heads: int,
    video_d_head: int,
    audio_d_head: int,
    context_dim: int,
) -> CausalAVTransformerBlock:
    torch.manual_seed(0)
    block = CausalAVTransformerBlock(
        idx=0,
        video=TransformerConfig(
            dim=video_dim, heads=video_heads,
            d_head=video_d_head, context_dim=context_dim,
        ),
        audio=TransformerConfig(
            dim=audio_dim, heads=audio_heads,
            d_head=audio_d_head, context_dim=context_dim,
        ),
    )
    # CausalAVTransformerBlock leaves its scale_shift_table parameters
    # uninitialized (torch.empty). Fill them with small random values
    # to make the forward deterministic and finite.
    with torch.no_grad():
        for p in block.parameters():
            if p.requires_grad:
                p.data.normal_(0.0, 0.02)
    block.eval()
    # Disable diagnostic gate stats: register_hook on no-grad tensors raises.
    block._store_gate_stats = False
    return block


def _alloc_kv_self(B: int, T_max: int, H: int, D: int):
    return {
        "k": torch.zeros(B, T_max, H, D),
        "v": torch.zeros(B, T_max, H, D),
        "end": torch.tensor([0], dtype=torch.long),
    }


def _alloc_text_cache(B: int, T_text: int, H: int, D: int):
    return {
        "k": torch.zeros(B, T_text, H * D),
        "v": torch.zeros(B, T_text, H * D),
        "len": 0,
        "is_init": False,
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestBlockKVCacheRouting:
    """Sanity: kv_cache_* routing through the four attention sites."""

    def test_block_forward_with_caches_matches_no_cache(self):
        B = 1
        Vd, Vh, Vdh = 64, 4, 16
        Ad, Ah, Adh = 32, 4, 8
        Cd = 24
        Lv, La = 6, 5            # video / audio token counts
        Tctx = 8                 # text context length

        block = _make_block(Vd, Ad, Vh, Ah, Vdh, Adh, Cd)

        torch.manual_seed(42)
        vx = torch.randn(B, Lv, Vd)
        ax = torch.randn(B, La, Ad)
        v_ctx = torch.randn(B, Tctx, Cd)
        a_ctx = torch.randn(B, Tctx, Cd)

        # AdaLN timestep tensors have shape [B, L, 6 * dim] (modulator.outdim)
        # in the LTX-2 layout. We feed zeros so the AdaLN modulation only
        # comes from scale_shift_table itself; this keeps the test
        # bit-deterministic without needing to mock AdaLayerNormSingle.
        v_ts = torch.zeros(B, Lv, 6 * Vd)
        a_ts = torch.zeros(B, La, 6 * Ad)

        # Cross-attn timesteps: 4 scale_shift values + 1 gate value.
        v_cross_ss = torch.zeros(B, Lv, 4 * Vd)
        v_cross_gate = torch.zeros(B, Lv, 1 * Vd)
        a_cross_ss = torch.zeros(B, La, 4 * Ad)
        a_cross_gate = torch.zeros(B, La, 1 * Ad)

        # ------ Path A: no caches, no masks -> full attention everywhere ------
        with torch.no_grad():
            args_v_a = CausalTransformerArgs(
                x=vx.clone(),
                timesteps=v_ts,
                context=v_ctx,
                cross_scale_shift_timestep=v_cross_ss,
                cross_gate_timestep=v_cross_gate,
            )
            args_a_a = CausalTransformerArgs(
                x=ax.clone(),
                timesteps=a_ts,
                context=a_ctx,
                cross_scale_shift_timestep=a_cross_ss,
                cross_gate_timestep=a_cross_gate,
            )
            out_v_a, out_a_a = block(args_v_a, args_a_a)

        # ------ Path B: caches set, empty initially, kv_start=0 ------
        kv_v_self = _alloc_kv_self(B, Lv, Vh, Vdh)
        # A2V cross cache: video Q reads audio K/V (audio_to_video_attn uses
        # audio heads/d_head). Stored on the VIDEO branch.
        kv_v_cross = _alloc_kv_self(B, La, Ah, Adh)
        # V2A cross cache: audio Q reads video K/V (video_to_audio_attn also
        # uses audio heads/d_head per block construction).
        kv_a_self = _alloc_kv_self(B, La, Ah, Adh)
        kv_a_cross = _alloc_kv_self(B, Lv, Ah, Adh)
        v_text = _alloc_text_cache(B, Tctx, Vh, Vdh)
        a_text = _alloc_text_cache(B, Tctx, Ah, Adh)

        with torch.no_grad():
            args_v_b = CausalTransformerArgs(
                x=vx.clone(),
                timesteps=v_ts,
                context=v_ctx,
                cross_scale_shift_timestep=v_cross_ss,
                cross_gate_timestep=v_cross_gate,
                kv_cache_self=kv_v_self,
                kv_cache_cross=kv_v_cross,
                crossattn_cache_text=v_text,
                current_token_start=0,
                cross_token_start=0,
            )
            args_a_b = CausalTransformerArgs(
                x=ax.clone(),
                timesteps=a_ts,
                context=a_ctx,
                cross_scale_shift_timestep=a_cross_ss,
                cross_gate_timestep=a_cross_gate,
                kv_cache_self=kv_a_self,
                kv_cache_cross=kv_a_cross,
                crossattn_cache_text=a_text,
                current_token_start=0,
                cross_token_start=0,
            )
            out_v_b, out_a_b = block(args_v_b, args_a_b)

        # Cache end pointers must reflect what was written.
        assert int(kv_v_self["end"].item()) == Lv
        assert int(kv_a_self["end"].item()) == La
        assert int(kv_v_cross["end"].item()) == La  # A2V wrote audio K/V
        assert int(kv_a_cross["end"].item()) == Lv  # V2A wrote video K/V
        assert v_text["is_init"] is True and v_text["len"] == Tctx
        assert a_text["is_init"] is True and a_text["len"] == Tctx

        # Equivalence (within fp32 roundoff over a deep stack of attentions).
        v_diff = (out_v_a.x - out_v_b.x).abs().max().item()
        a_diff = (out_a_a.x - out_a_b.x).abs().max().item()
        assert v_diff < 1e-3, f"video output diverges: max diff = {v_diff:.2e}"
        assert a_diff < 1e-3, f"audio output diverges: max diff = {a_diff:.2e}"

    def test_sequential_block_writes_advance_end_pointers(self):
        """Two sequential block forwards (block 0 then block 1) both
        write into the same cache and the end pointers advance."""
        B = 1
        Vd, Vh, Vdh = 32, 2, 16
        Ad, Ah, Adh = 24, 2, 12
        Cd = 16
        Lv0, Lv1 = 3, 2
        La0, La1 = 4, 3
        Tctx = 5
        Lv_total = Lv0 + Lv1
        La_total = La0 + La1

        block = _make_block(Vd, Ad, Vh, Ah, Vdh, Adh, Cd)

        kv_v_self = _alloc_kv_self(B, Lv_total, Vh, Vdh)
        kv_v_cross = _alloc_kv_self(B, La_total, Ah, Adh)
        kv_a_self = _alloc_kv_self(B, La_total, Ah, Adh)
        kv_a_cross = _alloc_kv_self(B, Lv_total, Ah, Adh)
        v_text = _alloc_text_cache(B, Tctx, Vh, Vdh)
        a_text = _alloc_text_cache(B, Tctx, Ah, Adh)

        torch.manual_seed(7)
        vx_full = torch.randn(B, Lv_total, Vd)
        ax_full = torch.randn(B, La_total, Ad)
        v_ctx = torch.randn(B, Tctx, Cd)
        a_ctx = torch.randn(B, Tctx, Cd)

        def _zeros_for(L, dim, mult):
            return torch.zeros(B, L, mult * dim)

        # Block 0: write [0, Lv0) and [0, La0).
        with torch.no_grad():
            args_v0 = CausalTransformerArgs(
                x=vx_full[:, :Lv0].clone(),
                timesteps=_zeros_for(Lv0, Vd, 6),
                context=v_ctx,
                cross_scale_shift_timestep=_zeros_for(Lv0, Vd, 4),
                cross_gate_timestep=_zeros_for(Lv0, Vd, 1),
                kv_cache_self=kv_v_self,
                kv_cache_cross=kv_v_cross,
                crossattn_cache_text=v_text,
                current_token_start=0,
                cross_token_start=0,
            )
            args_a0 = CausalTransformerArgs(
                x=ax_full[:, :La0].clone(),
                timesteps=_zeros_for(La0, Ad, 6),
                context=a_ctx,
                cross_scale_shift_timestep=_zeros_for(La0, Ad, 4),
                cross_gate_timestep=_zeros_for(La0, Ad, 1),
                kv_cache_self=kv_a_self,
                kv_cache_cross=kv_a_cross,
                crossattn_cache_text=a_text,
                current_token_start=0,
                cross_token_start=0,
            )
            block(args_v0, args_a0)

        assert int(kv_v_self["end"].item()) == Lv0
        assert int(kv_a_self["end"].item()) == La0
        assert int(kv_v_cross["end"].item()) == La0
        assert int(kv_a_cross["end"].item()) == Lv0

        # Block 1: write [Lv0, Lv0+Lv1) and [La0, La0+La1).
        with torch.no_grad():
            args_v1 = CausalTransformerArgs(
                x=vx_full[:, Lv0:Lv0 + Lv1].clone(),
                timesteps=_zeros_for(Lv1, Vd, 6),
                context=v_ctx,
                cross_scale_shift_timestep=_zeros_for(Lv1, Vd, 4),
                cross_gate_timestep=_zeros_for(Lv1, Vd, 1),
                kv_cache_self=kv_v_self,
                kv_cache_cross=kv_v_cross,
                crossattn_cache_text=v_text,
                current_token_start=Lv0,
                cross_token_start=La0,
            )
            args_a1 = CausalTransformerArgs(
                x=ax_full[:, La0:La0 + La1].clone(),
                timesteps=_zeros_for(La1, Ad, 6),
                context=a_ctx,
                cross_scale_shift_timestep=_zeros_for(La1, Ad, 4),
                cross_gate_timestep=_zeros_for(La1, Ad, 1),
                kv_cache_self=kv_a_self,
                kv_cache_cross=kv_a_cross,
                crossattn_cache_text=a_text,
                current_token_start=La0,
                cross_token_start=Lv0,
            )
            block(args_v1, args_a1)

        assert int(kv_v_self["end"].item()) == Lv_total
        assert int(kv_a_self["end"].item()) == La_total
        assert int(kv_v_cross["end"].item()) == La_total
        assert int(kv_a_cross["end"].item()) == Lv_total
        # Text cache content must NOT have been re-projected on block 1.
        assert v_text["is_init"] is True and v_text["len"] == Tctx
        assert a_text["is_init"] is True and a_text["len"] == Tctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
