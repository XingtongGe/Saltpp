"""
Frame-range video+audio dataset for LTX-2 causal training.

This dataset follows annotated segment frame_range metadata instead of randomly
cutting a fixed-length window from the whole clip:
  - select the configured segment part and its frame_range
  - sample the frame_range at 24 fps
  - snap the resulting pixel frame count down to LTX's 8n+1 requirement
  - trim/pad audio to the same LTX duration, in stereo at 16 kHz
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import threading
from bisect import bisect_right
from typing import Any

import av
import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import Dataset

from oss_client import Client as StorageClient
from video_utils import load_video_mixbit
from video_audio_dataset import AUDIO_SAMPLE_RATE, VIDEO_FPS, _get_audio_url


DEFAULT_META_PREFIX = os.environ.get("DATASET_META_PREFIX", "s3://your-bucket/metadata")

PART_KEY_FALLBACKS = (
    "segments",
    "parts",
    "clips",
)

CAPTION_PRIORITY = (
    "summary_with_audio_timestamp",
    "summary_with_audio_speaktime",
    "summary_with_audio",
    "caption",
)

_PROMPT_ENTITY_TAG_RE = re.compile(r"<(?:ID|OBJ|ENV)_\d+>")
_PROMPT_SPEAKER_SPAN_TAG_RE = re.compile(
    r"<(?:start|end)(?:_invisible)?_ID_\d+>"
)


def filter_prompt_symbols(
    prompt: str,
    *,
    remove_entity_tags: bool = False,
    remove_speaker_span_tags: bool = False,
    cleanup_spacing: bool = True,
) -> str:
    """Remove annotation-only prompt symbols while preserving shot/time text."""
    filtered = prompt
    if remove_entity_tags:
        filtered = _PROMPT_ENTITY_TAG_RE.sub("", filtered)
    if remove_speaker_span_tags:
        filtered = _PROMPT_SPEAKER_SPAN_TAG_RE.sub("", filtered)
    if cleanup_spacing:
        filtered = re.sub(r"\s+([,.;:!?])", r"\1", filtered)
        filtered = re.sub(r"\s+'", "'", filtered)
        filtered = re.sub(r" {2,}", " ", filtered)
    return filtered.strip()


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_lines(
    path: str | None,
    cast: type | None = None,
    n: int | None = None,
    fill: Any = None,
) -> list[Any]:
    if path is None:
        return [] if n is None else [fill for _ in range(n)]
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    if cast is int:
        return [int(x) for x in lines]
    return lines


def _normalize_meta_key(json_file_path: str) -> str:
    key = str(json_file_path).strip()
    for scheme in ("s3://", "oss://"):
        if key.startswith(scheme):
            key = key.split(scheme, 1)[1]
    key = key.lstrip("/")
    if key.endswith(".json"):
        key = key[:-5]
    if key.startswith("ref/"):
        key = key[len("ref/") :]
    return key


def _build_meta_json_path(json_file_path: str, meta_prefix: str) -> str:
    return "{}/{}.json".format(meta_prefix.rstrip("/"), _normalize_meta_key(json_file_path))


def _caption_from_part(
    part: dict[str, Any],
    caption_priority: tuple[str, ...] = CAPTION_PRIORITY,
) -> str:
    caption = ""
    for key in caption_priority:
        caption = part.get(key) or ""
        if caption:
            break
    if not caption:
        static_caption = part.get("static_caption", "")
        action_caption = part.get("action_caption", "")
        caption = " ".join([static_caption, action_caption]).strip()

    bg = part.get("background_audio")
    if bg:
        caption = " ".join(
            [caption, f"<background_audio> {bg} <background_audio>"]
        ).strip()
    return str(caption).replace("\n", "")


def _select_part(
    data_json: dict[str, Any],
    part_key: str | None,
    part_idx: int | None,
    rng: random.Random,
    part_key_priority: tuple[str, ...] = PART_KEY_FALLBACKS,
) -> tuple[str, int, dict[str, Any]]:
    if part_key:
        if part_key not in data_json:
            raise ValueError(f"part_key {part_key!r} not present")
        selected_key = part_key
        parts = data_json[part_key]
    else:
        selected_key = ""
        parts = None
        for key in part_key_priority:
            if key in data_json:
                selected_key = key
                parts = data_json[key]
                break
        if parts is None:
            raise ValueError("no valid parts key found in metadata")

    if not isinstance(parts, list) or not parts:
        raise ValueError(f"{selected_key} is empty or not a list")

    selected_idx = rng.randrange(len(parts)) if part_idx is None else int(part_idx)
    if selected_idx < 0 or selected_idx >= len(parts):
        raise ValueError(
            f"part_idx {selected_idx} out of range for {selected_key} len={len(parts)}"
        )

    part = parts[selected_idx]
    if not isinstance(part, dict) or "frame_range" not in part:
        raise ValueError(f"{selected_key}[{selected_idx}] has no frame_range")
    return selected_key, selected_idx, part


def _snap_to_ltx_frames(raw_frames: int) -> int:
    raw_frames = max(1, int(raw_frames))
    return raw_frames - ((raw_frames - 1) % 8)


def _linspace_indices(
    start: int,
    end: int,
    sample_frames: int,
    video_len: int,
) -> list[int]:
    if end < start:
        raise ValueError(f"empty frame range [{start}, {end}]")
    chosen = list(range(start, end + 1))
    if not chosen:
        raise ValueError("no chosen frame indices")

    if sample_frames > len(chosen):
        last = chosen[-1]
        while len(chosen) < sample_frames and last + 1 < video_len:
            last += 1
            chosen.append(last)
        while len(chosen) < sample_frames:
            chosen.append(chosen[-1])

    lin = torch.floor(torch.linspace(0, len(chosen) - 1, sample_frames))
    lin = lin.long().tolist()
    return [max(0, min(chosen[i], video_len - 1)) for i in lin]


def _video_to_tensor_mixbit_seek(
    container,
    indices: list[int],
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not indices:
        raise ValueError("indices cannot be empty")

    video_stream = container.streams.video[0]
    start_frame = int(indices[0])
    target_set = set(int(x) for x in indices)
    num_frames = len(indices)

    try:
        pts_per_frame_inv = 1.0 / (
            video_stream.average_rate * video_stream.time_base
        )
        seek_pts = int(start_frame * pts_per_frame_inv)
        container.seek(seek_pts, backward=True, any_frame=False, stream=video_stream)
    except Exception:
        container.seek(0)

    frames = []
    for decode_idx, frame in enumerate(container.decode(video_stream)):
        current_frame_idx = int(
            round(frame.pts * video_stream.time_base * video_stream.average_rate)
        )
        if current_frame_idx < start_frame:
            continue
        if len(frames) >= num_frames:
            break
        if current_frame_idx in target_set:
            arr = frame.to_ndarray(format="rgb24")
            if decode_idx == 0 and current_frame_idx > start_frame:
                for _ in range(start_frame, current_frame_idx):
                    frames.append(arr)
                    if len(frames) >= num_frames:
                        break
            if len(frames) < num_frames:
                frames.append(arr)

    if not frames:
        raise ValueError("no frames extracted from video")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    if size is not None:
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]

    all_frames = np.stack(frames, axis=0)
    tensor = torch.from_numpy(all_frames).permute(3, 0, 1, 2).float() / 255.0
    try:
        container.close()
    except Exception:
        pass
    return tensor * 2.0 - 1.0


def _load_audio_with_pyav(audio_bytes: bytes, sample_rate: int) -> tuple[torch.Tensor, int]:
    """Decode audio bytes without torchaudio.load."""
    container = av.open(io.BytesIO(audio_bytes), metadata_errors="ignore")
    try:
        if not container.streams.audio:
            raise ValueError("no audio stream found")

        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=int(sample_rate),
        )
        chunks: list[np.ndarray] = []

        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, list):
                resampled = [resampled]
            for audio_frame in resampled:
                arr = audio_frame.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[0] != 2 and arr.shape[-1] == 2:
                    arr = arr.T
                chunks.append(arr.astype(np.float32, copy=False))

        tail = resampler.resample(None)
        if tail is not None:
            if not isinstance(tail, list):
                tail = [tail]
            for audio_frame in tail:
                arr = audio_frame.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[0] != 2 and arr.shape[-1] == 2:
                    arr = arr.T
                chunks.append(arr.astype(np.float32, copy=False))

        if not chunks:
            raise ValueError("no audio frames decoded")

        waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).float()
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.size(0) > 2:
            waveform = waveform[:2]
        return waveform, int(sample_rate)
    finally:
        container.close()


class FrameRangeVideoAudioDataset(Dataset):
    """Loads segment-annotated config/list rows and returns LTX-ready video/audio samples."""

    def __init__(
        self,
        data_config_path: str,
        oss_conf_path: str | None = None,
        video_height: int = 512,
        video_width: int = 768,
        target_fps: int = VIDEO_FPS,
        audio_sample_rate: int = AUDIO_SAMPLE_RATE,
        require_audio: bool = True,
        require_audio_flag: bool = True,
        caption_priority: list[str] | tuple[str, ...] | None = None,
        part_key_priority: list[str] | tuple[str, ...] | None = None,
        prompt_filter_remove_entity_tags: bool = False,
        prompt_filter_remove_speaker_span_tags: bool = False,
        prompt_filter_cleanup_spacing: bool = True,
        expand_tasks: bool = False,
        max_entries_per_group: int = 0,
        epoch_size: int = 0,
        min_frames: int = 113,
        max_frames: int = 193,
        timeout_sec: int = 120,
        preserve_aspect_ratio: bool = True,
        spatial_align: int = 32,
        min_side: int = 128,
        meta_prefix: str = DEFAULT_META_PREFIX,
        seed: int = 7182,
        audio_range_mode: str = "segment",
        prompt_only: bool = False,
    ):
        super().__init__()
        self.data_config_path = data_config_path
        self.video_height = int(video_height)
        self.video_width = int(video_width)
        self.target_fps = int(target_fps)
        self.audio_sample_rate = int(audio_sample_rate)
        self.require_audio = bool(require_audio)
        self.require_audio_flag = bool(require_audio_flag)
        self.caption_priority = tuple(caption_priority or CAPTION_PRIORITY)
        self.part_key_priority = tuple(part_key_priority or PART_KEY_FALLBACKS)
        self.prompt_filter_remove_entity_tags = bool(prompt_filter_remove_entity_tags)
        self.prompt_filter_remove_speaker_span_tags = bool(prompt_filter_remove_speaker_span_tags)
        self.prompt_filter_cleanup_spacing = bool(prompt_filter_cleanup_spacing)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.timeout_sec = int(timeout_sec)
        self.preserve_aspect_ratio = bool(preserve_aspect_ratio)
        self.spatial_align = int(spatial_align)
        self.min_side = int(min_side)
        self.meta_prefix = meta_prefix
        self.seed = int(seed)
        self.audio_range_mode = str(audio_range_mode)
        self.prompt_only = bool(prompt_only)

        self.groups, self.share_params = self._load_config_groups(
            data_config_path,
            expand_tasks=expand_tasks,
            max_entries_per_group=max_entries_per_group,
        )
        if not self.groups:
            raise ValueError(f"No data groups loaded from {data_config_path}")

        self.group_weights = [max(float(g["effective_weight"]), 0.0) for g in self.groups]
        if sum(self.group_weights) <= 0:
            self.group_weights = [1.0 for _ in self.groups]
        total = 0.0
        self.group_cum_weights = []
        for weight in self.group_weights:
            total += weight
            self.group_cum_weights.append(total)
        self.group_weight_total = total

        if epoch_size > 0:
            self._length = int(epoch_size)
        else:
            self._length = sum(len(g["rows"]) for g in self.groups)

        if oss_conf_path is None:
            oss_conf_path = self.share_params.get("oss_file_path")
        self.mclient = StorageClient(conf_path=oss_conf_path)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                "[FrameRangeDataset] Loaded "
                f"{sum(len(g['rows']) for g in self.groups)} rows from "
                f"{len(self.groups)} groups; epoch_size={self._length}; "
                f"frames=[{self.min_frames},{self.max_frames or 'inf'}]; "
                f"caption_priority={list(self.caption_priority)}; "
                f"part_key_priority={list(self.part_key_priority)}; "
                f"prompt_filter={self._prompt_filter_enabled()}; "
                f"prompt_only={self.prompt_only}"
            )

    def _prompt_filter_enabled(self) -> bool:
        return (
            self.prompt_filter_remove_entity_tags
            or self.prompt_filter_remove_speaker_span_tags
        )

    def _filter_prompt(self, prompt: str) -> str:
        if not self._prompt_filter_enabled():
            return prompt
        return filter_prompt_symbols(
            prompt,
            remove_entity_tags=self.prompt_filter_remove_entity_tags,
            remove_speaker_span_tags=self.prompt_filter_remove_speaker_span_tags,
            cleanup_spacing=self.prompt_filter_cleanup_spacing,
        )

    def _load_config_groups(
        self,
        config_path: str,
        expand_tasks: bool,
        max_entries_per_group: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg["data"]
        share_params = data_cfg.get("share_params", {})
        ratios = data_cfg.get("ratio", [1] * len(data_cfg["dataset"]))
        groups = []

        for dataset_idx, dataset_cfg in enumerate(data_cfg["dataset"]):
            tasks = dataset_cfg.get("params", {}).get("tasks", [])
            if not tasks:
                continue
            task_size_sum = sum(int(task.get("task_size", 1)) for task in tasks)
            selected_tasks = list(enumerate(tasks)) if expand_tasks else [(0, tasks[0])]

            for task_idx, task in selected_tasks:
                rows = self._load_task_rows(task)
                if max_entries_per_group > 0:
                    rows = rows[:max_entries_per_group]
                if not rows:
                    continue
                task_size = int(task.get("task_size", 1))
                ratio = ratios[dataset_idx] if dataset_idx < len(ratios) else 1
                groups.append(
                    {
                        "sample_group_idx": len(groups),
                        "dataset_idx": dataset_idx,
                        "task_idx": task_idx,
                        "ratio": ratio,
                        "task_size": task_size,
                        "task_size_sum": task_size_sum,
                        "effective_weight": float(ratio) * task_size / max(task_size_sum, 1),
                        "task_type": task.get("task_type", "video"),
                        "task_training": task.get("task_training", "random"),
                        "rows": rows,
                    }
                )
        return groups, share_params

    def _load_task_rows(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        json_paths = _ensure_list(task.get("json_list_path"))
        size_paths = _ensure_list(task.get("size_list_path"))
        index_paths = _ensure_list(task.get("index_list_path"))
        key_paths = _ensure_list(task.get("key_list_path"))
        i2v_paths = _ensure_list(task.get("i2v_list_path"))
        sound_paths = _ensure_list(task.get("audio_flag_list_path"))

        for file_idx, json_path in enumerate(json_paths):
            size_path = size_paths[file_idx] if file_idx < len(size_paths) else None
            index_path = index_paths[file_idx] if file_idx < len(index_paths) else None
            key_path = key_paths[file_idx] if file_idx < len(key_paths) else None
            i2v_path = i2v_paths[file_idx] if file_idx < len(i2v_paths) else None
            sound_path = sound_paths[file_idx] if file_idx < len(sound_paths) else None

            data_list = _read_lines(json_path)
            size_list = _read_lines(size_path)
            index_list = _read_lines(index_path, int, len(data_list), None)
            key_list = _read_lines(key_path, None, len(data_list), None)
            i2v_list = _read_lines(i2v_path, int, len(data_list), 1)
            sound_list = _read_lines(sound_path, int, len(data_list), 1)

            if not index_list:
                index_list = [None for _ in data_list]
            if not key_list:
                key_list = [None for _ in data_list]
            if not i2v_list:
                i2v_list = [1 for _ in data_list]
            if not sound_list:
                sound_list = [1 for _ in data_list]

            if not (
                len(data_list)
                == len(size_list)
                == len(index_list)
                == len(key_list)
                == len(i2v_list)
                == len(sound_list)
            ):
                raise ValueError(f"List length mismatch around {json_path}")

            for row_idx, entry in enumerate(data_list):
                w_s, h_s = size_list[row_idx].split()[:2]
                rows.append(
                    {
                        "entry": entry,
                        "source_json_list": json_path,
                        "source_file_idx": file_idx,
                        "row_idx": row_idx,
                        "orig_w": int(w_s),
                        "orig_h": int(h_s),
                        "part_idx": index_list[row_idx],
                        "part_key": key_list[row_idx],
                        "i2v_flag": i2v_list[row_idx],
                        "audio_flag": sound_list[row_idx],
                    }
                )
        return rows

    def __len__(self) -> int:
        return self._length

    def _choose_row(self, idx: int) -> tuple[dict[str, Any], dict[str, Any], random.Random]:
        rng = random.Random(self.seed + int(idx))
        group_idx = bisect_right(
            self.group_cum_weights,
            rng.random() * self.group_weight_total,
        )
        group_idx = min(group_idx, len(self.groups) - 1)
        group = self.groups[group_idx]
        row = group["rows"][rng.randrange(len(group["rows"]))]
        return group, row, rng

    def _build_i2v_plan(
        self,
        data_json: dict[str, Any],
        row: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        video_len = int(data_json["f_num"])
        actual_fps = float(data_json["fps"])
        if video_len <= 0 or actual_fps <= 0:
            raise ValueError(f"bad video metadata f_num={video_len}, fps={actual_fps}")

        part_key, part_idx, part = _select_part(
            data_json,
            row.get("part_key"),
            row.get("part_idx"),
            rng,
            self.part_key_priority,
        )
        frame_range = part["frame_range"]
        start, end = int(frame_range[0]), int(frame_range[1])
        raw_frames = int((end + 1 - start) / actual_fps * self.target_fps)
        sample_frames = _snap_to_ltx_frames(raw_frames)
        indices = _linspace_indices(start, end, sample_frames, video_len)

        caption = _caption_from_part(part, self.caption_priority)

        return {
            "part_key": part_key,
            "part_idx": part_idx,
            "caption": caption,
            "frame_range": [start, end],
            "audio_frame_range": [start, end],
            "indices": indices,
            "sample_frames": sample_frames,
            "raw_frames": raw_frames,
            "actual_fps": actual_fps,
            "video_len": video_len,
        }

    def _build_v2v_plan(self, data_json: dict[str, Any]) -> dict[str, Any]:
        video_len = int(data_json["f_num"])
        actual_fps = float(data_json["fps"])
        if video_len <= 0 or actual_fps <= 0:
            raise ValueError(f"bad video metadata f_num={video_len}, fps={actual_fps}")

        ref_parts = data_json.get("reference_parts")
        if not isinstance(ref_parts, list) or len(ref_parts) < 2:
            raise ValueError("reference_parts missing or too short")
        ref_part, target_part = ref_parts[0], ref_parts[1]

        original_ref = ref_part["frame_range"]
        target_range = target_part["frame_range"]
        max_ref_frames = max(1, int(actual_fps * 2))
        ref_range = (
            max(int(original_ref[0]), int(original_ref[1]) - max_ref_frames + 1),
            int(original_ref[1]),
        )
        frame_start, frame_end = ref_range[0], int(target_range[1])
        audio_start, audio_end = int(original_ref[0]), int(target_range[1])

        raw_frames = int((frame_end + 1 - frame_start) / actual_fps * self.target_fps)
        sample_frames = _snap_to_ltx_frames(raw_frames)
        indices = _linspace_indices(frame_start, frame_end, sample_frames, video_len)

        return {
            "part_key": "reference_parts",
            "part_idx": 1,
            "caption": _caption_from_part(target_part, self.caption_priority),
            "first_frame_caption": _caption_from_part(ref_part, self.caption_priority),
            "frame_range": [frame_start, frame_end],
            "audio_frame_range": [audio_start, audio_end],
            "indices": indices,
            "sample_frames": sample_frames,
            "raw_frames": raw_frames,
            "actual_fps": actual_fps,
            "video_len": video_len,
        }

    def _target_size(self, orig_w: int, orig_h: int) -> tuple[int, int]:
        align = self.spatial_align
        target_area = float(self.video_height * self.video_width)
        aspect = max(orig_w, 1) / max(orig_h, 1)
        h_raw = math.sqrt(target_area / aspect)
        w_raw = h_raw * aspect
        h = max(self.min_side, int(round(h_raw / align)) * align)
        w = max(self.min_side, int(round(w_raw / align)) * align)
        return h, w

    def _load_video(self, data_json: dict[str, Any], row: dict[str, Any], plan: dict[str, Any]) -> torch.Tensor:
        clip_url = data_json.get("clip_url")
        if not clip_url:
            raise ValueError("missing clip_url")
        video_bytes = self.mclient.get(clip_url)
        container = load_video_mixbit(video_bytes, clip_url)
        if self.preserve_aspect_ratio:
            target_h, target_w = self._target_size(row["orig_w"], row["orig_h"])
            size = (target_w, target_h)
        else:
            size = (self.video_width, self.video_height)
        return _video_to_tensor_mixbit_seek(container, indices=plan["indices"], size=size)

    def _load_audio(self, data_json: dict[str, Any], plan: dict[str, Any]) -> torch.Tensor:
        audio_url = _get_audio_url(data_json)
        target_samples = int(plan["sample_frames"] / self.target_fps * self.audio_sample_rate)
        if not audio_url:
            return torch.zeros(2, target_samples, dtype=torch.float32)

        audio_bytes = self.mclient.get(audio_url)
        waveform, _ = _load_audio_with_pyav(audio_bytes, self.audio_sample_rate)

        if self.audio_range_mode == "video":
            start_frame = int(plan["frame_range"][0])
        else:
            start_frame = int(plan["audio_frame_range"][0])
        start_sample = int(start_frame / plan["actual_fps"] * self.audio_sample_rate)
        start_sample = max(0, min(start_sample, max(waveform.size(1) - 1, 0)))
        waveform = waveform[:, start_sample:]

        if waveform.size(1) > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.size(1) < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.size(1)))
        return waveform.float()

    def _load_sample(self, idx: int) -> dict[str, Any]:
        group, row, rng = self._choose_row(idx)
        if self.require_audio_flag and not bool(row["audio_flag"]):
            raise ValueError("audio_flag is 0")

        meta_path = _build_meta_json_path(row["entry"], self.meta_prefix)
        data_json = json.loads(self.mclient.get(meta_path).decode("utf-8"))
        audio_url = _get_audio_url(data_json)
        if self.require_audio and not audio_url:
            raise ValueError("missing audio_url")

        if group["task_type"] == "video_varlen_v2v":
            plan = self._build_v2v_plan(data_json)
        else:
            plan = self._build_i2v_plan(data_json, row, rng)

        if plan["sample_frames"] < self.min_frames:
            raise ValueError(f"sample_frames {plan['sample_frames']} < min_frames {self.min_frames}")
        if self.max_frames > 0 and plan["sample_frames"] > self.max_frames:
            raise ValueError(f"sample_frames {plan['sample_frames']} > max_frames {self.max_frames}")

        caption = self._filter_prompt(plan["caption"])
        if not caption:
            raise ValueError("empty caption")

        if self.prompt_only:
            return {
                "prompt": caption,
                "num_frames": plan["sample_frames"],
                "caption": caption,
                "sample_frames": plan["sample_frames"],
                "entry": row["entry"],
                "sample_group": group["sample_group_idx"],
                "task_type": group["task_type"],
            }

        video = self._load_video(data_json, row, plan)
        audio = self._load_audio(data_json, plan)

        return {
            "video": video,
            "audio_waveform": audio,
            "caption": caption,
            "entry": row["entry"],
            "clip_url": data_json.get("clip_url", ""),
            "audio_url": audio_url or "",
            "fps": data_json.get("fps", ""),
            "f_num": data_json.get("f_num", ""),
            "sample_frames": plan["sample_frames"],
            "raw_frames": plan["raw_frames"],
            "frame_range": plan["frame_range"],
            "audio_frame_range": plan["audio_frame_range"],
            "sample_group": group["sample_group_idx"],
            "task_type": group["task_type"],
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
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
            reason = f"timeout ({self.timeout_sec}s)" if thread.is_alive() else str(result["error"])
            print(f"[FrameRangeDataset] idx={idx} failed: {reason}, retrying random")
            return self.__getitem__(random.randint(0, len(self) - 1))
        return result["data"]
