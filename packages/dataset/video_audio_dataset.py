"""
Video+audio+text dataset for LTX-2 real-data training.

Loads raw video frames and audio waveforms from local files or object storage,
ready for online VAE encoding in the trainer. Returns:
  - video:   float32 [C, T, H, W] in [-1, 1]   (T = num_frames, e.g. 121)
  - audio:   float32 [2, samples]  at 16 kHz    (stereo, duration matches video)
  - caption: str
"""

import io
import json
import math
import os
import random
import threading
from typing import List, Optional, Tuple

import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data import Dataset

from oss_client import Client as StorageClient
from video_utils import load_video_mixbit, video_to_tensor_mixbit


AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS = 24
DEFAULT_META_PREFIX = os.environ.get("DATASET_META_PREFIX", "s3://your-bucket/metadata")


def _extract_caption(data_json: dict) -> Optional[str]:
    """Extract the best available caption with audio description."""
    for parts_key in ("segments", "parts", "clips"):
        parts = data_json.get(parts_key)
        if parts and isinstance(parts, list) and len(parts) > 0:
            chosen = parts[0]
            caption = chosen.get("summary_with_audio") or chosen.get("summary")
            if caption:
                return caption
            static = chosen.get("static_caption", "")
            action = chosen.get("action_caption", "")
            if static or action:
                return f"{static} {action}".strip()

    return data_json.get("caption")


def _get_audio_url(data_json: dict) -> Optional[str]:
    """Get the best audio URL (prefer vocal, then full)."""
    return (
        data_json.get("audio_url_vocal")
        or data_json.get("audio_url")
        or data_json.get("audio_url_full")
    )


class VideoAudioDataset(Dataset):
    """Dataset that loads video, audio, and caption for training."""

    def __init__(
        self,
        data_list_paths: List[str],
        oss_conf_path: Optional[str] = None,
        num_frames: int = 121,
        video_height: int = 512,
        video_width: int = 768,
        target_fps: int = VIDEO_FPS,
        audio_sample_rate: int = AUDIO_SAMPLE_RATE,
        require_audio: bool = True,
        max_entries_per_list: int = 0,
        timeout_sec: int = 120,
        preserve_aspect_ratio: bool = False,
        spatial_align: int = 32,
        min_side: int = 128,
        meta_prefix: str = DEFAULT_META_PREFIX,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.video_height = video_height
        self.video_width = video_width
        self.target_fps = target_fps
        self.audio_sample_rate = audio_sample_rate
        self.require_audio = require_audio
        self.timeout_sec = timeout_sec
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.spatial_align = spatial_align
        self.min_side = min_side
        self.meta_prefix = meta_prefix.rstrip("/")

        self.mclient = StorageClient(conf_path=oss_conf_path)
        self.entries = self._load_all_lists(data_list_paths, max_entries_per_list)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                f"[VideoAudioDataset] Loaded {len(self.entries)} entries from "
                f"{len(data_list_paths)} lists"
            )

    def _load_all_lists(
        self, list_paths: List[str], max_per_list: int
    ) -> List[str]:
        entries = []
        for lp in list_paths:
            try:
                if os.path.isfile(lp):
                    with open(lp, "r") as f:
                        raw = f.read()
                else:
                    raw = self.mclient.get(f"{self.meta_prefix}/{lp}").decode()
                lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                if max_per_list > 0:
                    lines = lines[:max_per_list]
                entries.extend(lines)
            except Exception as e:
                print(f"[VideoAudioDataset] WARN: failed to load list {lp}: {e}")
        return entries

    def _load_sample(self, idx: int) -> dict:
        entry = self.entries[idx]
        raw = self.mclient.get(f"{self.meta_prefix}/{entry}.json").decode()
        data_json = json.loads(raw)

        caption = _extract_caption(data_json)
        if not caption:
            raise ValueError(f"No caption for {entry}")

        video_tensor, video_start_sec = self._load_video(data_json)
        audio_waveform = self._load_audio(data_json, video_start_sec)

        return {
            "video": video_tensor,
            "audio_waveform": audio_waveform,
            "caption": caption,
            "entry": entry,
            "clip_url": data_json.get("clip_url", ""),
            "audio_url": _get_audio_url(data_json) or "",
            "fps": data_json.get("fps", ""),
            "f_num": data_json.get("f_num", ""),
        }

    def _load_video(self, data_json: dict) -> Tuple[torch.Tensor, float]:
        clip_url = data_json["clip_url"]
        actual_fps = int(data_json.get("fps", 24))
        video_len = int(data_json.get("f_num", 0))

        video_bytes = self.mclient.get(clip_url)
        container = load_video_mixbit(video_bytes, clip_url)

        indices, start_frame = self._select_frame_indices(video_len, actual_fps)
        start_sec = start_frame / actual_fps if actual_fps > 0 else 0.0

        if self.preserve_aspect_ratio:
            tensor = video_to_tensor_mixbit(
                container, indices=indices, size=None, normalize=True,
            )
            _, _, orig_h, orig_w = tensor.shape
            new_h, new_w = self._compute_area_preserving_size(orig_h, orig_w)
            video_tchw = torch.nn.functional.interpolate(
                tensor.permute(1, 0, 2, 3),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            tensor = video_tchw.permute(1, 0, 2, 3)
        else:
            tensor = video_to_tensor_mixbit(
                container,
                indices=indices,
                size=(self.video_width, self.video_height),
                normalize=True,
            )
        return tensor, start_sec

    def _compute_area_preserving_size(
        self, orig_h: int, orig_w: int,
    ) -> Tuple[int, int]:
        align = self.spatial_align
        target_area = float(self.video_height * self.video_width)
        aspect = max(orig_w, 1) / max(orig_h, 1)

        new_h_raw = math.sqrt(target_area / aspect)
        new_w_raw = new_h_raw * aspect

        new_h = max(self.min_side, int(round(new_h_raw / align)) * align)
        new_w = max(self.min_side, int(round(new_w_raw / align)) * align)
        return new_h, new_w

    def _select_frame_indices(
        self, video_len: int, actual_fps: int
    ) -> Tuple[list, int]:
        target_duration_frames = int(
            actual_fps * (self.num_frames / self.target_fps)
        )
        available = max(video_len, 1)

        if target_duration_frames >= available:
            start = 0
            span = available
        else:
            start = random.randint(0, available - target_duration_frames)
            span = target_duration_frames

        indices = torch.linspace(start, start + span - 1, self.num_frames)
        indices = indices.long().clamp(0, available - 1).tolist()
        return indices, start

    def _load_audio(
        self, data_json: dict, video_start_sec: float = 0.0
    ) -> torch.Tensor:
        audio_url = _get_audio_url(data_json)
        video_duration = self.num_frames / self.target_fps
        target_samples = int(video_duration * self.audio_sample_rate)

        if not audio_url:
            return torch.zeros(2, target_samples, dtype=torch.float32)

        try:
            audio_bytes = self.mclient.get(audio_url)
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

            if sr != self.audio_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.audio_sample_rate
                )

            if waveform.size(0) == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.size(0) > 2:
                waveform = waveform[:2]

            audio_start = int(video_start_sec * self.audio_sample_rate)
            audio_start = max(0, min(audio_start, waveform.size(1) - 1))
            waveform = waveform[:, audio_start:]

            if waveform.size(1) > target_samples:
                waveform = waveform[:, :target_samples]
            elif waveform.size(1) < target_samples:
                pad = target_samples - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad))

            return waveform.float()

        except Exception as e:
            print(f"[VideoAudioDataset] audio load failed ({audio_url}): {e}")
            return torch.zeros(2, target_samples, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        result = {"data": None, "error": None}

        def worker():
            try:
                result["data"] = self._load_sample(idx)
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec)

        if thread.is_alive() or result["error"] is not None:
            if thread.is_alive():
                reason = f"timeout ({self.timeout_sec}s)"
            else:
                reason = str(result["error"])
            print(f"[VideoAudioDataset] idx={idx} failed: {reason}, retrying random")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return result["data"]


def collate_real_data(batch: List[dict]) -> dict:
    """Collate function for VideoAudioDataset."""
    videos = torch.stack([item["video"] for item in batch])
    audios = torch.stack([item["audio_waveform"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {
        "video": videos,
        "audio_waveform": audios,
        "captions": captions,
        "entries": [item.get("entry", "") for item in batch],
        "clip_urls": [item.get("clip_url", "") for item in batch],
        "audio_urls": [item.get("audio_url", "") for item in batch],
        "fps": [item.get("fps", "") for item in batch],
        "f_nums": [item.get("f_num", "") for item in batch],
    }
