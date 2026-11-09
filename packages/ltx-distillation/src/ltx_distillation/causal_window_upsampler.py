"""Block-causal execution wrappers for the official LTX-2 latent upsampler.

The released spatial upsampler is a temporal Conv3d network with GroupNorm.
Rather than changing its parameterization, the causal student runs the same
network on a bounded clean-prefix window and keeps only the current block.  It
therefore sees the complete current AR block and a fixed amount of clean
history, but never a future block.  The official teacher runs the same weights
on the complete clip.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class NormalizedLatentUpsampler(nn.Module):
    """Apply an LTX latent upsampler in normalized VAE-latent coordinates.

    Public tensors use the distillation layout ``[B, F, C, H, W]``.  The
    released upsampler uses ``[B, C, F, H, W]`` and operates on un-normalized
    VAE latents, so the channel statistics are kept as lightweight buffers.
    """

    def __init__(
        self,
        upsampler: nn.Module,
        *,
        channel_mean: torch.Tensor,
        channel_std: torch.Tensor,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.upsampler = upsampler
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.register_buffer(
            "channel_mean",
            channel_mean.detach().float().reshape(1, -1, 1, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "channel_std",
            channel_std.detach().float().reshape(1, -1, 1, 1, 1),
            persistent=True,
        )

    def _run_upsampler(self, latent_bcfhw: torch.Tensor) -> torch.Tensor:
        mean = self.channel_mean.to(latent_bcfhw)
        std = self.channel_std.to(latent_bcfhw)
        unnormalized = latent_bcfhw * std + mean
        if (
            self.gradient_checkpointing
            and torch.is_grad_enabled()
        ):
            upscaled = checkpoint(
                self.upsampler,
                unnormalized,
                use_reentrant=False,
            )
        else:
            upscaled = self.upsampler(unnormalized)
        return (upscaled - mean) / std

    def forward(self, latent_bfchw: torch.Tensor) -> torch.Tensor:
        if latent_bfchw.ndim != 5:
            raise ValueError(
                "latent must have shape [B,F,C,H,W], got "
                f"{tuple(latent_bfchw.shape)}"
            )
        latent_bcfhw = latent_bfchw.permute(0, 2, 1, 3, 4).contiguous()
        upscaled = self._run_upsampler(latent_bcfhw)
        return upscaled.permute(0, 2, 1, 3, 4).contiguous()


class CausalWindowLatentUpsampler(NormalizedLatentUpsampler):
    """Run the official architecture on ``history | current`` and crop current."""

    def __init__(
        self,
        upsampler: nn.Module,
        *,
        channel_mean: torch.Tensor,
        channel_std: torch.Tensor,
        history_frames: int = 18,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            upsampler,
            channel_mean=channel_mean,
            channel_std=channel_std,
            gradient_checkpointing=gradient_checkpointing,
        )
        if int(history_frames) < 0:
            raise ValueError("history_frames must be non-negative")
        self.history_frames = int(history_frames)

    def forward(
        self,
        latent_bfchw: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        total_frames = int(latent_bfchw.shape[1])
        block_start = int(block_start)
        block_end = int(block_end)
        if not 0 <= block_start < block_end <= total_frames:
            raise ValueError(
                f"invalid block [{block_start}:{block_end}) for F={total_frames}"
            )
        window_start = max(0, block_start - self.history_frames)
        window = latent_bfchw[:, window_start:block_end]
        upscaled_window = super().forward(window)
        crop_start = block_start - window_start
        crop_end = crop_start + (block_end - block_start)
        return upscaled_window[:, crop_start:crop_end]


def load_causal_upsampler_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    """Read an exported Stage-3.5 causal-upsampler component checkpoint."""

    path = Path(checkpoint_path)
    try:
        loaded = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError(
            f"Expected a mapping checkpoint at {path}, got "
            f"{type(loaded).__name__}"
        )

    state_dict = loaded.get("causal_upsampler")
    if not isinstance(state_dict, Mapping):
        raise KeyError(
            f"Checkpoint {path} has no mapping 'causal_upsampler' entry"
        )
    metadata = loaded.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"Checkpoint metadata must be a mapping, got "
            f"{type(metadata).__name__}"
        )
    return state_dict, metadata


def build_causal_upsampler_from_checkpoint(
    *,
    checkpoint_path: str,
    spatial_upsampler_path: str,
    causal_upsampler_checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    history_frames: int | None = None,
    gradient_checkpointing: bool = False,
) -> CausalWindowLatentUpsampler:
    """Build one causal upsampler and load exported Stage-3.5 weights.

    Unlike :func:`build_official_and_causal_upsamplers`, this inference/training
    helper does not keep a second official-teacher copy in memory.
    """

    from ltx_pipelines.utils.model_ledger import ModelLedger

    state_dict, metadata = load_causal_upsampler_checkpoint(
        causal_upsampler_checkpoint_path
    )
    saved_history_frames = metadata.get("history_frames")
    if history_frames is None:
        history_frames = int(
            saved_history_frames if saved_history_frames is not None else 18
        )
    elif (
        saved_history_frames is not None
        and int(history_frames) != int(saved_history_frames)
    ):
        raise ValueError(
            "Requested causal upsampler history_frames="
            f"{history_frames}, but checkpoint metadata records "
            f"{saved_history_frames}"
        )

    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=(),
    )
    upsampler_module = ledger.spatial_upsampler()
    video_encoder = ledger.video_encoder()
    stats = video_encoder.per_channel_statistics
    channel_mean = stats.get_buffer("mean-of-means").detach().cpu()
    channel_std = stats.get_buffer("std-of-means").detach().cpu()
    del video_encoder

    causal = CausalWindowLatentUpsampler(
        upsampler_module,
        channel_mean=channel_mean,
        channel_std=channel_std,
        history_frames=int(history_frames),
        gradient_checkpointing=gradient_checkpointing,
    )
    missing, unexpected = causal.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Strict causal upsampler load unexpectedly reported "
            f"missing={missing}, unexpected={unexpected}"
        )
    return causal


def build_official_and_causal_upsamplers(
    *,
    checkpoint_path: str,
    spatial_upsampler_path: str,
    device: torch.device,
    dtype: torch.dtype,
    history_frames: int,
    gradient_checkpointing: bool,
) -> tuple[NormalizedLatentUpsampler, CausalWindowLatentUpsampler]:
    """Load the released upsampler once and clone its initialization."""

    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=(),
    )
    official_module = ledger.spatial_upsampler()
    video_encoder = ledger.video_encoder()
    stats = video_encoder.per_channel_statistics
    channel_mean = stats.get_buffer("mean-of-means").detach().cpu()
    channel_std = stats.get_buffer("std-of-means").detach().cpu()
    del video_encoder

    causal_module = copy.deepcopy(official_module)
    official = NormalizedLatentUpsampler(
        official_module,
        channel_mean=channel_mean,
        channel_std=channel_std,
        gradient_checkpointing=False,
    )
    causal = CausalWindowLatentUpsampler(
        causal_module,
        channel_mean=channel_mean,
        channel_std=channel_std,
        history_frames=history_frames,
        gradient_checkpointing=gradient_checkpointing,
    )
    return official, causal
