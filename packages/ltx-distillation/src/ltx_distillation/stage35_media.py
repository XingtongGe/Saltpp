"""PyAV-only media helpers for Stage-3.5 local logs and benchmarks."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch


@torch.no_grad()
def decode_video_tiled(
    *,
    video_vae: torch.nn.Module,
    latent_bfchw: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Return uint8 frames in ``[T,H,W,C]`` layout."""

    from ltx_core.model.video_vae import TilingConfig, decode_video

    if latent_bfchw.ndim != 5 or latent_bfchw.shape[2] != 128:
        raise ValueError(
            "expected video latent [B,F,128,H,W], got "
            f"{tuple(latent_bfchw.shape)}"
        )
    decoder_latent = latent_bfchw.permute(0, 2, 1, 3, 4).contiguous()
    generator = torch.Generator(device=latent_bfchw.device).manual_seed(int(seed))
    chunks = decode_video(
        decoder_latent,
        video_vae.decoder,
        tiling_config=TilingConfig.default(),
        generator=generator,
    )
    frames = torch.cat([chunk.detach().cpu() for chunk in chunks], dim=0)
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0, 1) * 255.0).to(torch.uint8)
    return frames


def write_video_pyav(
    path: str,
    frames_thwc: torch.Tensor,
    *,
    fps: int,
    audio_array: torch.Tensor | None = None,
    audio_fps: int | None = None,
    audio_codec: str = "aac",
) -> None:
    """Write an H.264 MP4, optionally muxing mono/stereo audio, via PyAV."""

    frames = frames_thwc.detach().cpu()
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0, 1) * 255.0).to(torch.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3], got {tuple(frames.shape)}")
    frames_np = np.ascontiguousarray(frames.numpy())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(frames_np.shape[2])
        stream.height = int(frames_np.shape[1])
        stream.pix_fmt = "yuv420p"

        audio_stream = None
        audio_np = None
        if audio_array is not None:
            if audio_fps is None:
                raise ValueError(
                    "audio_fps is required when audio_array is provided"
                )
            audio = audio_array.detach().cpu().float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
                audio = audio.transpose(0, 1)
            elif audio.ndim != 2:
                raise ValueError(
                    "expected audio [T], [C,T], or [T,C], got "
                    f"{tuple(audio.shape)}"
                )
            if audio.shape[0] > 2:
                audio = audio[:2]
            audio_np = np.ascontiguousarray(
                audio.clamp(-1, 1).numpy().astype(np.float32)
            )
            layout = "mono" if audio_np.shape[0] == 1 else "stereo"
            audio_stream = container.add_stream(
                str(audio_codec), rate=int(audio_fps)
            )
            audio_stream.layout = layout

        for frame_np in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        if audio_stream is not None and audio_np is not None:
            for start in range(0, audio_np.shape[1], 1024):
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_np[:, start : start + 1024],
                    format="fltp",
                    layout=audio_stream.layout.name,
                )
                audio_frame.sample_rate = int(audio_fps)
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)
    finally:
        container.close()


def stitch_side_by_side(
    left_thwc: torch.Tensor,
    right_thwc: torch.Tensor,
) -> torch.Tensor:
    frames = min(int(left_thwc.shape[0]), int(right_thwc.shape[0]))
    height = min(int(left_thwc.shape[1]), int(right_thwc.shape[1]))
    return torch.cat(
        [
            left_thwc[:frames, :height],
            right_thwc[:frames, :height],
        ],
        dim=2,
    )
