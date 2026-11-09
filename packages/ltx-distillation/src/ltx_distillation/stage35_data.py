"""Metadata-only prompt stream for scale-wise online rollout training."""

from __future__ import annotations

import json
import os
import random
import sys
from glob import glob
from pathlib import Path
from typing import Any

import torch


class Stage35LocalPromptDataset(torch.utils.data.Dataset):
    """Fixed-shape prompt stream backed entirely by shared local files."""

    def __init__(
        self,
        *,
        prompts: list[str],
        num_frames: int,
        height: int,
        width: int,
        epoch_size: int,
    ) -> None:
        if not prompts:
            raise ValueError("Stage35 local prompt dataset is empty")
        self.prompts = prompts
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.epoch_size = max(int(epoch_size), len(prompts))

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt_index = int(idx) % len(self.prompts)
        return {
            "caption": self.prompts[prompt_index],
            "sample_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "entry": f"local_prompt_{prompt_index:06d}",
        }


def _read_prompt_lines(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                prompt = str(payload.get("prompt", "")).strip()
                if prompt:
                    prompts.append(prompt)
                    continue
            except json.JSONDecodeError:
                pass
        prompts.append(line)
    return prompts


def build_stage35_local_prompt_dataset(config) -> Stage35LocalPromptDataset:
    """Load training prompts without object-store credentials or media I/O."""

    prompts: list[str] = []
    for value in config.get("stage35_local_prompt_files", []):
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"local prompt file does not exist: {path}")
        prompts.extend(_read_prompt_lines(path))

    matched_paths: list[Path] = []
    for pattern in config.get("stage35_local_prompt_globs", []):
        matched_paths.extend(Path(value) for value in sorted(glob(str(pattern))))
    for path in matched_paths:
        if not path.is_file():
            continue
        prompt = path.read_text(encoding="utf-8").strip()
        if prompt:
            prompts.append(prompt)

    prompts = list(dict.fromkeys(prompts))
    return Stage35LocalPromptDataset(
        prompts=prompts,
        num_frames=int(config.get("num_frames", 121)),
        height=int(config.video_height),
        width=int(config.video_width),
        epoch_size=int(config.get("stage35_local_prompt_epoch_size", 12000)),
    )


def _dataset_package_path() -> Path:
    return Path(__file__).resolve().parents[3] / "dataset"


def build_stage35_metadata_dataset(config):
    """Build the same frame-range sampling distribution without media I/O."""

    dataset_path = str(_dataset_package_path())
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)

    from frame_range_dataset import (  # type: ignore
        FrameRangeVideoAudioDataset,
        _build_meta_json_path,
        _get_audio_url,
    )

    max_retries = max(int(config.get("metadata_max_retries", 64)), 1)

    class Stage35MetadataPromptDataset(FrameRangeVideoAudioDataset):
        def _load_sample(self, idx: int) -> dict[str, Any]:
            group, row, rng = self._choose_row(idx)
            if self.require_audio_flag and not bool(row["audio_flag"]):
                raise ValueError("audio_flag is 0")

            meta_path = _build_meta_json_path(row["entry"], self.meta_prefix)
            data_json = json.loads(self.mclient.get(meta_path).decode("utf-8"))
            if self.require_audio and not _get_audio_url(data_json):
                raise ValueError("missing audio_url")

            if group["task_type"] == "video_varlen_v2v":
                plan = self._build_v2v_plan(data_json)
            else:
                plan = self._build_i2v_plan(data_json, row, rng)

            sample_frames = int(plan["sample_frames"])
            if sample_frames < self.min_frames:
                raise ValueError(
                    f"sample_frames {sample_frames} < min_frames {self.min_frames}"
                )
            if self.max_frames > 0 and sample_frames > self.max_frames:
                raise ValueError(
                    f"sample_frames {sample_frames} > max_frames {self.max_frames}"
                )
            caption = self._filter_prompt(plan["caption"])
            if not caption:
                raise ValueError("empty caption")

            if self.preserve_aspect_ratio:
                height, width = self._target_size(row["orig_w"], row["orig_h"])
            else:
                height, width = self.video_height, self.video_width

            return {
                "caption": caption,
                "sample_frames": sample_frames,
                "height": int(height),
                "width": int(width),
                "entry": row["entry"],
                "sample_group": int(group["sample_group_idx"]),
                "task_type": str(group["task_type"]),
            }

        def __getitem__(self, idx: int) -> dict[str, Any]:
            current_idx = int(idx)
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return self._load_sample(current_idx)
                except Exception as exc:
                    last_error = exc
                    if attempt == 1 or attempt == max_retries:
                        print(
                            f"[Stage35Metadata] idx={current_idx} failed "
                            f"({attempt}/{max_retries}): {exc}",
                            flush=True,
                        )
                    current_idx = random.randint(0, len(self) - 1)
            raise RuntimeError(
                "Stage35 metadata retries exhausted; verify object-store "
                f"credentials. Last error: {last_error}"
            ) from last_error

    default_meta = os.environ.get("DATASET_META_PREFIX", "s3://your-bucket/metadata")
    return Stage35MetadataPromptDataset(
        data_config_path=config.data_config_path,
        oss_conf_path=config.get("oss_conf_path"),
        video_height=int(config.video_height),
        video_width=int(config.video_width),
        target_fps=int(config.get("target_fps", 24)),
        audio_sample_rate=int(config.get("audio_sample_rate", 16000)),
        require_audio=bool(config.get("require_audio", True)),
        require_audio_flag=bool(config.get("require_audio_flag", True)),
        caption_priority=config.get("caption_priority"),
        part_key_priority=config.get("part_key_priority"),
        prompt_filter_remove_entity_tags=bool(
            config.get("prompt_filter_remove_entity_tags", False)
        ),
        prompt_filter_remove_speaker_span_tags=bool(
            config.get("prompt_filter_remove_speaker_span_tags", False)
        ),
        prompt_filter_cleanup_spacing=bool(
            config.get("prompt_filter_cleanup_spacing", True)
        ),
        expand_tasks=bool(config.get("expand_tasks", False)),
        max_entries_per_group=int(config.get("max_entries_per_group", 0)),
        epoch_size=int(config.get("epoch_size", 0)),
        min_frames=int(config.get("min_frames", 113)),
        max_frames=int(config.get("max_frames", 193)),
        timeout_sec=int(config.get("data_timeout_sec", 120)),
        preserve_aspect_ratio=bool(config.get("preserve_aspect_ratio", True)),
        spatial_align=int(config.get("spatial_align", 32)),
        min_side=int(config.get("min_side", 128)),
        meta_prefix=str(config.get("meta_prefix", default_meta)),
        seed=int(config.get("seed", 42)),
        audio_range_mode=str(config.get("audio_range_mode", "segment")),
    )


def collate_stage35_prompts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 1:
        raise ValueError("Variable-length training requires batch_size=1")
    sample = samples[0]
    return {
        "prompts": [sample["caption"]],
        "num_frames": int(sample["sample_frames"]),
        "height": int(sample["height"]),
        "width": int(sample["width"]),
        "entries": [sample["entry"]],
    }


def build_stage35_dataloader(config):
    prompt_source = str(
        config.get("stage35_prompt_source", "metadata")
    ).strip().lower()
    if prompt_source == "local_files":
        dataset = build_stage35_local_prompt_dataset(config)
    elif prompt_source == "metadata":
        dataset = build_stage35_metadata_dataset(config)
    else:
        raise ValueError(
            "stage35_prompt_source must be local_files or metadata"
        )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        sampler=sampler,
        collate_fn=collate_stage35_prompts,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=False,
    )
