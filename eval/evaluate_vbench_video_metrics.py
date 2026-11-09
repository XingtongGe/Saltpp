#!/usr/bin/env python3
"""Evaluate the VBench video quality and consistency metrics on generated MP4 videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

VBENCH_ROOT = os.environ.get("VBENCH_ROOT")
if VBENCH_ROOT:
    sys.path.insert(0, str(Path(VBENCH_ROOT).resolve()))

from vbench import VBench  # noqa: E402
from vbench.distributed import barrier, dist_init, get_rank, get_world_size  # noqa: E402


DIMENSIONS = (
    "aesthetic_quality",
    "imaging_quality",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "temporal_flickering",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_results(raw: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for dimension in DIMENSIONS:
        value = raw.get(dimension)
        if not isinstance(value, list) or not value or not isinstance(value[0], (int, float)):
            raise ValueError(f"invalid VBench result for {dimension}: {type(value).__name__}")
        metrics[dimension] = float(value[0])
    return metrics


def write_summary(path: Path, metadata: dict[str, Any], metrics: dict[str, float]) -> None:
    labels = {
        "aesthetic_quality": "Aesthetic Quality",
        "imaging_quality": "Imaging Quality",
        "subject_consistency": "Subject Consistency",
        "background_consistency": "Background Consistency",
        "motion_smoothness": "Motion Smoothness",
        "temporal_flickering": "Temporal Flickering",
    }
    lines = [
        "# VBench Video Metrics",
        "",
        f"- videos: {metadata['num_videos']}",
        f"- source: `{metadata['videos_dir']}`",
        "",
        "| Metric | Direction | Value (x100) |",
        "| --- | --- | ---: |",
    ]
    for key in DIMENSIONS:
        lines.append(f"| {labels[key]} | up | {metrics[key] * 100.0:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt-independent VBench metrics on a directory of MP4 files."
    )
    parser.add_argument("--videos-dir", required=True, type=Path, help="Directory containing MP4 files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write VBench results")
    parser.add_argument(
        "--full-info-json",
        type=Path,
        default=None,
        help="Optional path to VBench_full_info.json (defaults to $VBENCH_ROOT/vbench/VBench_full_info.json)",
    )
    parser.add_argument(
        "--imaging-preprocessing-mode",
        choices=("longer", "shorter", "shorter_centercrop", "None"),
        default="longer",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not videos_dir.is_dir():
        raise ValueError(f"video directory does not exist: {videos_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not args.force:
        print(f"[vbench] complete result already exists: {metrics_path}")
        return 0

    dist_init()
    rank = get_rank()
    world_size = get_world_size()
    raw_dir = output_dir / "raw"

    full_info_path = args.full_info_json
    if full_info_path is None:
        if VBENCH_ROOT:
            full_info_path = Path(VBENCH_ROOT) / "vbench" / "VBench_full_info.json"
        else:
            import vbench as _vb

            full_info_path = Path(_vb.__file__).resolve().parent / "VBench_full_info.json"

    try:
        barrier()
        raw_dir.mkdir(parents=True, exist_ok=True)
        evaluator = VBench(
            torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))),
            str(full_info_path),
            str(raw_dir),
        )
        evaluator.evaluate(
            videos_path=str(videos_dir),
            name="saltpp_vbench_metrics",
            dimension_list=list(DIMENSIONS),
            local=True,
            read_frame=False,
            mode="custom_input",
            imaging_quality_preprocessing_mode=args.imaging_preprocessing_mode,
        )

        if rank == 0:
            raw_results_path = raw_dir / "saltpp_vbench_metrics_eval_results.json"
            with raw_results_path.open("r", encoding="utf-8") as handle:
                raw_results = json.load(handle)
            metrics = normalize_results(raw_results)
            selected = sorted(videos_dir.glob("*.mp4"))
            metadata = {
                "schema_version": 1,
                "dimensions": list(DIMENSIONS),
                "num_videos": len(selected),
                "videos_dir": str(videos_dir),
                "world_size": world_size,
                "imaging_quality_preprocessing_mode": args.imaging_preprocessing_mode,
                "raw_results": str(raw_results_path),
                "raw_results_sha256": sha256_file(raw_results_path),
                "completed_at_utc": utc_now(),
            }
            write_json(metrics_path, {**metadata, "metrics": metrics})
            write_summary(output_dir / "summary.md", metadata, metrics)
            print(f"[vbench] wrote {metrics_path} and {output_dir / 'summary.md'}")
        barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
