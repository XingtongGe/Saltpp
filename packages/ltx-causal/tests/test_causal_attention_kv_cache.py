"""Unit tests for CausalLTXAttention KV-cache codepath.

Phase 1 Hard Gate of the KV-cache migration. These tests verify that
sequentially feeding sub-blocks through the new ``kv_cache`` path
produces the same output as the existing full-sequence path (with the
matching block-causal mask). If any of these tests fail, the cache
implementation has a numerical bug and we must NOT proceed to Phase 2.

Conventions:
- ``B = 1`` to keep tests cheap; logic is batch-agnostic.
- ``CausalLTXAttention`` is constructed in eval() and seeded so weights
  are deterministic.
- We compare on CPU, double-precision-ish via float32 + ``atol=1e-4``.
"""

from __future__ import annotations

import pytest
import torch

from ltx_causal.attention.causal_attention import CausalLTXAttention
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    causal_precompute_freqs_cis,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_attn(query_dim: int, heads: int, dim_head: int) -> CausalLTXAttention:
    torch.manual_seed(0)
    attn = CausalLTXAttention(
        query_dim=query_dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        rope_type=CausalRopeType.INTERLEAVED,
    )
    attn.eval()
    return attn


def _block_causal_mask(block_lens, dtype=torch.bool) -> torch.Tensor:
    """Return [T, T] mask where mask[i, j] = True iff j's block index <= i's.

    All tokens within the same block see each other; tokens in earlier
    blocks see only their own / earlier blocks. This matches the KV-cache
    semantics (each block writes its full K/V, downstream blocks read
    history + own block fully).
    """
    block_id = []
    for b_idx, blen in enumerate(block_lens):
        block_id.extend([b_idx] * blen)
    block_id = torch.tensor(block_id, dtype=torch.long)
    return (block_id.unsqueeze(0) <= block_id.unsqueeze(1)).to(dtype)


def _alloc_kv_cache(B: int, T_max: int, heads: int, dim_head: int):
    return {
        "k": torch.zeros(B, T_max, heads, dim_head),
        "v": torch.zeros(B, T_max, heads, dim_head),
        "end": torch.tensor([0], dtype=torch.long),
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestKVCacheEquivalence:
    """Path A (full + block-causal mask) ≡ Path B (sequential kv_cache)."""

    def test_no_rope_block_causal_match(self):
        torch.manual_seed(42)
        B, D, H, Dh = 1, 64, 4, 16
        block_lens = [4, 3, 3, 2]
        T_total = sum(block_lens)
        attn = _build_attn(D, H, Dh)

        x = torch.randn(B, T_total, D)

        # Path A: full sequence with block-causal mask
        with torch.no_grad():
            mask_bool = _block_causal_mask(block_lens, dtype=torch.bool)
            out_a = attn(x, pe=None, cross_causal_mask=mask_bool)

        # Path B: sequential kv_cache
        kv = _alloc_kv_cache(B, T_total, H, Dh)
        with torch.no_grad():
            chunks = []
            pos = 0
            for blen in block_lens:
                x_block = x[:, pos:pos + blen]
                out_b = attn(x_block, pe=None, kv_cache=kv, kv_start=pos)
                chunks.append(out_b)
                pos += blen
            out_b_full = torch.cat(chunks, dim=1)

        assert out_a.shape == out_b_full.shape
        assert torch.allclose(out_a, out_b_full, atol=1e-4, rtol=1e-4), (
            f"max diff = {(out_a - out_b_full).abs().max().item():.2e}"
        )
        assert int(kv["end"].item()) == T_total

    def test_with_rope_block_causal_match(self):
        """RoPE is applied per-block in the cache path; full RoPE in non-cache.

        We slice the same precomputed RoPE table to feed both paths so the
        rotational frequencies line up token-by-token.
        """
        torch.manual_seed(43)
        B, D, H, Dh = 1, 64, 4, 16
        block_lens = [4, 3, 3, 2]
        T_total = sum(block_lens)
        attn = _build_attn(D, H, Dh)

        x = torch.randn(B, T_total, D)

        # Build a 1D positional grid covering the full T_total tokens.
        grid = torch.tensor([[T_total]], dtype=torch.long)
        pe_full = causal_precompute_freqs_cis(
            grid_sizes=grid,
            dim=H * Dh,
            theta=10000.0,
            max_pos=[T_total + 8],
            start_frame=0,
            rope_type=CausalRopeType.INTERLEAVED,
            device=torch.device("cpu"),
            dtype=torch.float32,
            is_audio=True,  # 1D grid -> use audio path
        )
        # pe_full is a tuple (cos, sin) of shape [1, T_total, inner_dim]

        with torch.no_grad():
            mask_bool = _block_causal_mask(block_lens, dtype=torch.bool)
            out_a = attn(x, pe=pe_full, cross_causal_mask=mask_bool)

        kv = _alloc_kv_cache(B, T_total, H, Dh)
        with torch.no_grad():
            chunks = []
            pos = 0
            for blen in block_lens:
                x_block = x[:, pos:pos + blen]
                pe_block = (
                    pe_full[0][:, pos:pos + blen],
                    pe_full[1][:, pos:pos + blen],
                )
                out_b = attn(x_block, pe=pe_block, kv_cache=kv, kv_start=pos)
                chunks.append(out_b)
                pos += blen
            out_b_full = torch.cat(chunks, dim=1)

        assert torch.allclose(out_a, out_b_full, atol=1e-4, rtol=1e-4), (
            f"max diff = {(out_a - out_b_full).abs().max().item():.2e}"
        )

    def test_kv_start_offset_skips_history(self):
        """When kv_start > 0 with empty history (zeros), the output must
        differ from the no-history case unless the leading slice was
        explicitly written. This guards against off-by-one writes.
        """
        torch.manual_seed(44)
        B, D, H, Dh = 1, 64, 4, 16
        attn = _build_attn(D, H, Dh)

        x = torch.randn(B, 5, D)

        # First call: write [0, 5).
        kv1 = _alloc_kv_cache(B, 16, H, Dh)
        with torch.no_grad():
            out1 = attn(x, pe=None, kv_cache=kv1, kv_start=0)

        # Second call: write [3, 8) on a fresh cache. kv_start=3 means
        # the first 3 slots are still zeros, so attention sees garbage K/V
        # at positions [0, 3) and proper K/V at [3, 8). Output should
        # differ from out1.
        kv2 = _alloc_kv_cache(B, 16, H, Dh)
        with torch.no_grad():
            out2 = attn(x, pe=None, kv_cache=kv2, kv_start=3)

        # Sanity: shapes match.
        assert out1.shape == out2.shape
        # Different kv_start -> different attended history -> different output.
        assert not torch.allclose(out1, out2, atol=1e-3), (
            "Expected outputs to differ when kv_start changes the history slice "
            "the block attends to, but they were identical."
        )


class TestCrossAttnCacheOneShot:
    """Text cross-attention cache: project once, reuse forever."""

    def test_first_and_second_call_match_full_attention(self):
        torch.manual_seed(45)
        B, Dq, Dctx, H, Dh = 1, 64, 96, 4, 16
        T_q, T_ctx = 8, 12

        # Cross attention: context_dim != query_dim
        torch.manual_seed(0)
        attn = CausalLTXAttention(
            query_dim=Dq,
            context_dim=Dctx,
            heads=H,
            dim_head=Dh,
            rope_type=CausalRopeType.INTERLEAVED,
        )
        attn.eval()

        x = torch.randn(B, T_q, Dq)
        ctx = torch.randn(B, T_ctx, Dctx)

        # Reference: full attention, no cache.
        with torch.no_grad():
            out_ref = attn(x, context=ctx, pe=None)

        # First cache call: should initialize and produce same output.
        cache = {
            "k": torch.zeros(B, T_ctx, H * Dh),
            "v": torch.zeros(B, T_ctx, H * Dh),
            "len": 0,
            "is_init": False,
        }
        with torch.no_grad():
            out1 = attn(x, context=ctx, pe=None, crossattn_cache=cache)

        assert cache["is_init"] is True
        assert cache["len"] == T_ctx
        assert torch.allclose(out_ref, out1, atol=1e-4, rtol=1e-4), (
            f"first-call cache output diverges from non-cache reference: "
            f"max diff = {(out_ref - out1).abs().max().item():.2e}"
        )

        # Second cache call: pass a DIFFERENT context to ensure cache is
        # actually being reused (not re-projected). Output should still
        # match the original cached projection.
        ctx_decoy = torch.randn(B, T_ctx, Dctx) * 100.0
        with torch.no_grad():
            out2 = attn(x, context=ctx_decoy, pe=None, crossattn_cache=cache)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Second call should reuse cached K/V regardless of new context."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
