"""Phase 4 Gate: wrapper-level equivalence for KV-cache forward.

Verifies that ``CausalLTX2DiffusionWrapper`` returns numerically equivalent
x0 predictions whether it is called as a single full-sequence ``_forward_standard``
pass (with block_mask) or as a series of per-block ``_forward_with_cache``
calls (with KV cache, sink-aware offsets) that share a single cache state.

Same small model config as test_causal_model_kv_cache.py:
  - num_audio_sink_tokens=16 (locks in local sink-aware adaptation)
  - F_v=7 split as block 0 (4 frames) + block 1 (3 frames)
  - audio split via standard layout (26 / 25)

Tolerance: atol=1e-3 / rtol=1e-3 (matches Phase 3 hard gate).
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
LTX_CORE_SRC = os.path.normpath(os.path.join(ROOT, "..", "ltx-core", "src"))
for p in (SRC, LTX_CORE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from ltx_causal.transformer.causal_model import (  # noqa: E402
    CausalLTXModel,
    CausalLTXModelConfig,
)
from ltx_causal.wrapper import CausalLTX2DiffusionWrapper  # noqa: E402
from ltx_causal.attention.mask_builder import compute_av_blocks  # noqa: E402


def _make_wrapper(device, dtype=torch.float32):
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
        model.scale_shift_table.data.normal_(0.0, 0.02)
        model.audio_scale_shift_table.data.normal_(0.0, 0.02)
    model.eval()
    for blk in model.transformer_blocks:
        blk._store_gate_stats = False

    # Wrapper: matched config (video_height/width chosen so that
    # latent_height*latent_width = video_frame_seqlen = 4 -> H=W=2 with vae_spatial_compression=1).
    wrapper = CausalLTX2DiffusionWrapper(
        model=model,
        video_height=2,
        video_width=2,
        vae_spatial_compression=1,
        num_frame_per_block=cfg.num_frame_per_block,
        first_block_video_frames=cfg.first_block_video_frames,
        num_audio_sink_tokens=cfg.num_audio_sink_tokens,
    )
    wrapper.eval()
    return wrapper, cfg


def _build_inputs(cfg, F_v, F_a, B, L_ctx, device, dtype):
    H = W = 2
    video = torch.randn(B, F_v, cfg.in_channels, H, W, device=device, dtype=dtype)
    audio = torch.randn(B, F_a, cfg.in_channels, device=device, dtype=dtype)
    # Use [B, F] timesteps but with uniform value across the sequence so that
    # _process_timestep_causal is a no-op (block-uniform timesteps already).
    ts = torch.full((B, F_v), 0.5, device=device, dtype=dtype)
    audio_ts = torch.full((B, F_a), 0.5, device=device, dtype=dtype)
    cond = {
        "video_context": torch.randn(B, L_ctx, cfg.caption_channels, device=device, dtype=dtype),
        "audio_context": torch.randn(B, L_ctx, cfg.caption_channels, device=device, dtype=dtype),
    }
    return video, audio, ts, audio_ts, cond


@torch.no_grad()
def test_wrapper_full_vs_blockwise_cache_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    wrapper, cfg = _make_wrapper(device, dtype)

    F_v = 7
    blocks = compute_av_blocks(
        F_v, cfg.num_frame_per_block, cfg.first_block_video_frames,
    )
    F_a = blocks[-1].audio_end
    assert len(blocks) == 2 and F_a == 51

    B, L_ctx = 1, 5
    video, audio, ts, audio_ts, cond = _build_inputs(cfg, F_v, F_a, B, L_ctx, device, dtype)

    # Path A: full-sequence wrapper forward (returns x0).
    full_v_x0, full_a_x0 = wrapper(
        noisy_image_or_video=video,
        conditional_dict=cond,
        timestep=ts,
        noisy_audio=audio,
        audio_timestep=audio_ts,
        use_causal_timestep=True,
    )
    assert full_v_x0.shape == (B, F_v, cfg.out_channels, 2, 2)
    assert full_a_x0.shape == (B, F_a, cfg.out_channels)

    # Path B: per-block KV-cache wrapper forward, sharing one cache.
    kv = wrapper.init_av_kv_caches(
        batch_size=B,
        num_video_frames=F_v,
        num_audio_frames=F_a,
        text_seq_len=L_ctx,
        device=device,
        dtype=dtype,
    )

    cached_v_pieces = []
    cached_a_pieces = []
    for blk in blocks:
        v_block = video[:, blk.video_start:blk.video_end]
        a_block = audio[:, blk.audio_start:blk.audio_end]
        ts_block = ts[:, blk.video_start:blk.video_end]
        ats_block = audio_ts[:, blk.audio_start:blk.audio_end]

        v_x0_b, a_x0_b = wrapper(
            _kv_cache_kwargs=dict(
                noisy_video_block=v_block,
                noisy_audio_block=a_block,
                conditional_dict=cond,
                timestep_block=ts_block,
                audio_timestep_block=ats_block,
                kv_caches=kv,
                current_video_start_frame=blk.video_start,
                current_audio_start_frame=blk.audio_start,
            )
        )
        cached_v_pieces.append(v_x0_b)
        cached_a_pieces.append(a_x0_b)

    cached_v_x0 = torch.cat(cached_v_pieces, dim=1)
    cached_a_x0 = torch.cat(cached_a_pieces, dim=1)

    v_diff = (full_v_x0 - cached_v_x0).abs()
    a_diff = (full_a_x0 - cached_a_x0).abs()
    print(f"video x0 |diff| max={v_diff.max().item():.4e} mean={v_diff.mean().item():.4e}")
    print(f"audio x0 |diff| max={a_diff.max().item():.4e} mean={a_diff.mean().item():.4e}")

    torch.testing.assert_close(cached_v_x0, full_v_x0, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(cached_a_x0, full_a_x0, atol=1e-3, rtol=1e-3)


@torch.no_grad()
def test_wrapper_init_caches_shapes():
    """Sanity check on cache buffer shapes from the wrapper-level init."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    wrapper, cfg = _make_wrapper(device, dtype)
    B, F_v, L_ctx = 1, 7, 5
    F_a = 51

    kv = wrapper.init_av_kv_caches(
        batch_size=B,
        num_video_frames=F_v,
        num_audio_frames=F_a,
        text_seq_len=L_ctx,
        device=device,
        dtype=dtype,
    )
    assert len(kv) == cfg.num_layers

    expected_v = F_v * cfg.video_frame_seqlen
    expected_a = cfg.num_audio_sink_tokens + F_a * cfg.audio_frame_seqlen
    for layer in kv:
        assert layer["video_self"]["k"].shape == (B, expected_v, cfg.video_heads, cfg.video_d_head)
        assert layer["audio_self"]["k"].shape == (B, expected_a, cfg.audio_heads, cfg.audio_d_head)
        assert layer["a2v"]["k"].shape == (B, expected_a, cfg.audio_heads, cfg.audio_d_head)
        assert layer["v2a"]["k"].shape == (B, expected_v, cfg.audio_heads, cfg.audio_d_head)
        assert layer["video_text"]["k"].shape == (B, L_ctx, cfg.video_heads * cfg.video_d_head)
        assert layer["audio_text"]["k"].shape == (B, L_ctx, cfg.audio_heads * cfg.audio_d_head)
        assert layer["video_self"]["end"].item() == 0
        assert not layer["video_text"]["is_init"]


if __name__ == "__main__":
    test_wrapper_init_caches_shapes()
    print("[PASS] test_wrapper_init_caches_shapes")
    test_wrapper_full_vs_blockwise_cache_equivalence()
    print("[PASS] test_wrapper_full_vs_blockwise_cache_equivalence")
