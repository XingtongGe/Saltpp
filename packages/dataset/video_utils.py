"""
Video loading utilities for multi-bit-depth video decoding.
Requires: av, cv2, numpy, torch
"""

import io
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

try:
    import av
except ImportError:
    raise ImportError("PyAV is required: pip install av")


def load_video_mixbit(video_bytes: bytes, video_path: str, debug: bool = False):
    """Load video from raw bytes into an av.Container."""
    buf = io.BytesIO(video_bytes)
    container = av.open(buf, metadata_errors="ignore")

    if not container.streams.video:
        raise ValueError(f"No video stream found in {video_path}")

    if debug:
        vs = container.streams.video[0]
        print(f"Video codec={vs.codec_context.name} size={vs.width}x{vs.height} "
              f"fps={vs.average_rate} pix_fmt={vs.codec_context.pix_fmt}")

    return container


def _get_bit_depth(pix_fmt) -> int:
    """Estimate bit depth from pixel format string."""
    if pix_fmt is None:
        return 8
    s = str(pix_fmt).lower()
    if "10" in s and ("p10" in s or "yuv" in s):
        return 10
    if "12" in s and ("p12" in s or "yuv" in s):
        return 12
    if "16" in s and ("p16" in s or "yuv" in s or "rgb" in s):
        return 16
    return 8


def video_to_tensor_mixbit(
    container,
    frame_index: Optional[int] = None,
    indices: Optional[list] = None,
    max_frames: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Decode video container to tensor (C, T, H, W), optionally resized.

    Args:
        container: av.Container
        frame_index: single frame index
        indices: list of frame indices to extract
        max_frames: max frames for sequential decode
        size: (width, height) target resize
        normalize: if True, output range [-1, 1]; else raw uint values

    Returns:
        torch.Tensor (C, T, H, W)
    """
    vs = container.streams.video[0]
    # PyAV/libswscale can produce severely wrong colors for some 10-bit SDR
    # clips when converting directly to rgb48le. Decode to 8-bit RGB instead,
    # matching ffmpeg CLI's default PNG/RGB path used in manual validation.
    out_fmt = "rgb24"
    output_bit_depth = 8

    if indices is not None or frame_index is not None:
        target_indices = [frame_index] if frame_index is not None else list(indices)

        total_frames = []
        max_idx = -1
        for frame in container.decode(vs):
            idx = int(round(frame.pts * vs.time_base * vs.average_rate))
            total_frames.append((idx, frame))
            if idx > max_idx:
                max_idx = idx

        target_indices = [min(x, max_idx) for x in target_indices]
        sorted_targets = sorted(target_indices)

        frames = []
        it = iter(sorted_targets)
        cur = next(it, None)
        if cur is not None:
            for fidx, frame in total_frames:
                if fidx >= cur:
                    arr = frame.to_ndarray(format=out_fmt)
                    frames.append(arr)
                    while fidx >= cur:
                        cur = next(it, None)
                        if cur is None:
                            break
                        if fidx >= cur:
                            frames.append(arr)
                    if cur is None:
                        break
    else:
        frames = []
        count = 0
        for frame in container.decode(vs):
            if max_frames and count >= max_frames:
                break
            frames.append(frame.to_ndarray(format=out_fmt))
            count += 1

    if not frames:
        raise ValueError("No frames extracted from video")

    if size is not None:
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]

    all_frames = np.stack(frames, axis=0)  # (T, H, W, C)
    tensor = torch.from_numpy(all_frames).permute(3, 0, 1, 2)  # (C, T, H, W)

    if normalize:
        if output_bit_depth == 8:
            tensor = tensor.float() / 255.0
        elif output_bit_depth == 10:
            tensor = tensor.float() / (1023 << 6)
        elif output_bit_depth == 12:
            tensor = tensor.float() / (4095 << 4)
        else:
            tensor = tensor.float() / 65535.0
        tensor = tensor * 2.0 - 1.0

    container.close()
    return tensor
