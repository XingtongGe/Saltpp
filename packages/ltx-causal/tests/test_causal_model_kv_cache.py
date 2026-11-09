"""Phase 3 Hard Gate: end-to-end equivalence between _forward_standard
(full-sequence, block_mask) and _forward_with_cache (block-by-block KV cache).

We build a small randomly-initialised CausalLTXModel with num_audio_sink_tokens=16
to lock in the local sink-aware adaptation, partition F_v=7 into block 0 (4 frames)
and block 1 (3 frames) using the standard layout (matches Stage 0/2/3 with
first_block_video_frames=4, num_frame_per_block=3 -> audio split 26/25), then
compare:

  full = model._forward_standard(video, audio, ts, video_ctx, audio_ctx)
  blocks = []
  for blk in [block0, block1]:
      v, a = model(video[blk.v], audio[blk.a], ts[blk.v], video_ctx, audio_ctx,
                   kv_caches=K, current_video_start_frame=blk.v_start,
                   current_audio_start_frame=blk.a_start)
      blocks.append((v, a))
  cached = concat(blocks)

  assert allclose(full, cached, atol=1e-3)
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
# ltx-core lives as a sibling package and is imported by ltx_causal internally.
LTX_CORE_SRC = os.path.normpath(os.path.join(ROOT, "..", "ltx-core", "src"))
for p in (SRC, LTX_CORE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from ltx_causal.transformer.causal_model import (  # noqa: E402
    CausalLTXModel,
    CausalLTXModelConfig,
)
from ltx_causal.attention.mask_builder import compute_av_blocks  # noqa: E402


def _make_small_model(device, dtype=torch.float32):
    """Tiny CausalLTXModel for fast equivalence tests.

    - num_layers=2, video_dim=64 (= 4 heads * 16 d_head), audio_dim=64 (= 4 heads * 16 d_head)
    - in/out_channels=4, video_frame_seqlen=4 (H=W=2), audio_frame_seqlen=1
    - num_audio_sink_tokens=16 (matches local stage 2/3 design)
    - All head_dim >= 16 because FlexAttention requires it.
    """
    cfg = CausalLTXModelConfig(
        num_layers=2,
        video_dim=64,
        audio_dim=64,
        video_heads=4,
        audio_heads=4,
        video_d_head=16,
        audio_d_head=16,
        cross_attention_dim=64,
        audio_cross_attention_dim=64,
        in_channels=4,
        out_channels=4,
        caption_channels=16,
        pe_max_pos=(20, 8, 8),
        audio_pe_max_pos=(20,),
        num_frame_per_block=3,
        first_block_video_frames=4,
        num_audio_sink_tokens=16,
        video_frame_seqlen=4,
        audio_frame_seqlen=1,
    )
    model = CausalLTXModel(cfg).to(device=device, dtype=dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.data.normal_(0.0, 0.02)
        # scale_shift_table parameters are torch.empty -> may contain NaN. Re-init.
        model.scale_shift_table.data.normal_(0.0, 0.02)
        model.audio_scale_shift_table.data.normal_(0.0, 0.02)
    model.eval()
    # Diagnostic gate stats use .item() -> disable for cleanliness.
    for blk in model.transformer_blocks:
        blk._store_gate_stats = False
    return model, cfg


def _build_inputs(cfg, F_v, F_a, B, L_ctx, device, dtype):
    H = W = 2
    video = torch.randn(B, F_v, cfg.in_channels, H, W, device=device, dtype=dtype)
    audio = torch.randn(B, F_a, cfg.in_channels, device=device, dtype=dtype)
    ts = torch.full((B, F_v), 0.5, device=device, dtype=dtype)
    audio_ts = torch.full((B, F_a), 0.5, device=device, dtype=dtype)
    video_ctx = torch.randn(B, L_ctx, cfg.caption_channels, device=device, dtype=dtype)
    audio_ctx = torch.randn(B, L_ctx, cfg.caption_channels, device=device, dtype=dtype)
    return video, audio, ts, audio_ts, video_ctx, audio_ctx


@torch.no_grad()
def test_full_vs_blockwise_cache_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model, cfg = _make_small_model(device, dtype)

    F_v = 7  # block 0: 4, block 1: 3
    blocks = compute_av_blocks(
        total_video_latent_frames=F_v,
        num_frame_per_block=cfg.num_frame_per_block,
        first_block_video_frames=cfg.first_block_video_frames,
    )
    F_a = blocks[-1].audio_end
    assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}: {blocks}"
    assert F_a == 51, f"Expected 51 audio frames, got {F_a}"

    B, L_ctx = 1, 5
    video, audio, ts, audio_ts, vctx, actx = _build_inputs(
        cfg, F_v, F_a, B, L_ctx, device, dtype
    )

    # --- Path A: full-sequence forward ---
    full_v, full_a = model(
        video_latent=video,
        audio_latent=audio,
        timesteps=ts,
        audio_timesteps=audio_ts,
        video_context=vctx,
        audio_context=actx,
    )
    assert full_v.shape == (B, F_v, cfg.out_channels, 2, 2)
    assert full_a.shape == (B, F_a, cfg.out_channels)

    # --- Path B: per-block KV-cache forward ---
    kv = model.init_av_kv_caches(
        batch_size=B,
        max_video_frames=F_v,
        max_audio_frames=F_a,
        text_seq_len=L_ctx,
        device=device,
        dtype=dtype,
    )
    cached_v_pieces = []
    cached_a_pieces = []
    for blk in blocks:
        v_slice = video[:, blk.video_start:blk.video_end]
        a_slice = audio[:, blk.audio_start:blk.audio_end]
        ts_slice = ts[:, blk.video_start:blk.video_end]
        ats_slice = audio_ts[:, blk.audio_start:blk.audio_end]

        v_out, a_out = model(
            video_latent=v_slice,
            audio_latent=a_slice,
            timesteps=ts_slice,
            audio_timesteps=ats_slice,
            video_context=vctx,
            audio_context=actx,
            kv_caches=kv,
            current_video_start_frame=blk.video_start,
            current_audio_start_frame=blk.audio_start,
        )
        assert v_out.shape == (B, blk.video_end - blk.video_start, cfg.out_channels, 2, 2)
        assert a_out.shape == (B, blk.audio_end - blk.audio_start, cfg.out_channels)
        cached_v_pieces.append(v_out)
        cached_a_pieces.append(a_out)

    cached_v = torch.cat(cached_v_pieces, dim=1)
    cached_a = torch.cat(cached_a_pieces, dim=1)

    # --- Sanity on cache end pointers ---
    expected_v_end = F_v * cfg.video_frame_seqlen
    expected_a_end = cfg.num_audio_sink_tokens + F_a * cfg.audio_frame_seqlen
    for layer in kv:
        assert layer["video_self"]["end"].item() == expected_v_end
        assert layer["audio_self"]["end"].item() == expected_a_end
        assert layer["a2v"]["end"].item() == expected_a_end
        assert layer["v2a"]["end"].item() == expected_v_end
        assert layer["video_text"]["is_init"]
        assert layer["audio_text"]["is_init"]
        assert layer["video_text"]["len"] == L_ctx
        assert layer["audio_text"]["len"] == L_ctx

    # --- Compare ---
    v_diff = (full_v - cached_v).abs()
    a_diff = (full_a - cached_a).abs()
    print(f"video |diff| max={v_diff.max().item():.4e} mean={v_diff.mean().item():.4e}")
    print(f"audio |diff| max={a_diff.max().item():.4e} mean={a_diff.mean().item():.4e}")
    torch.testing.assert_close(cached_v, full_v, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(cached_a, full_a, atol=1e-3, rtol=1e-3)


@torch.no_grad()
def test_first_block_only_with_sinks():
    """Block 0 in isolation should match the full forward when F_v == first_block_video_frames."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model, cfg = _make_small_model(device, dtype)

    F_v = cfg.first_block_video_frames  # 4
    blocks = compute_av_blocks(
        F_v, cfg.num_frame_per_block, cfg.first_block_video_frames
    )
    F_a = blocks[-1].audio_end
    assert len(blocks) == 1

    B, L_ctx = 1, 5
    video, audio, ts, audio_ts, vctx, actx = _build_inputs(
        cfg, F_v, F_a, B, L_ctx, device, dtype
    )

    full_v, full_a = model(
        video_latent=video, audio_latent=audio,
        timesteps=ts, audio_timesteps=audio_ts,
        video_context=vctx, audio_context=actx,
    )

    kv = model.init_av_kv_caches(B, F_v, F_a, L_ctx, device, dtype)
    cached_v, cached_a = model(
        video_latent=video, audio_latent=audio,
        timesteps=ts, audio_timesteps=audio_ts,
        video_context=vctx, audio_context=actx,
        kv_caches=kv,
        current_video_start_frame=0,
        current_audio_start_frame=0,
    )

    torch.testing.assert_close(cached_v, full_v, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(cached_a, full_a, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_first_block_only_with_sinks()
    print("[PASS] test_first_block_only_with_sinks")
    test_full_vs_blockwise_cache_equivalence()
    print("[PASS] test_full_vs_blockwise_cache_equivalence")
