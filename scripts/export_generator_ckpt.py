#!/usr/bin/env python3
"""Export a lightweight generator_only.pt checkpoint from a full training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def _summarize(state: dict[str, Any]) -> dict[str, Any]:
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if len(tensors) != len(state):
        non_tensors = sorted(key for key, value in state.items() if not isinstance(value, torch.Tensor))
        raise ValueError(f"state dict contains non-tensor values: {non_tensors[:10]}")
    dtypes: Counter[str] = Counter()
    logical_bytes = 0
    numel = 0
    for tensor in tensors:
        tensor_bytes = tensor.numel() * tensor.element_size()
        logical_bytes += tensor_bytes
        numel += tensor.numel()
        dtypes[str(tensor.dtype)] += tensor_bytes
    schema = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(state.items())
    ]
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "tensor_count": len(tensors),
        "parameter_and_buffer_numel": numel,
        "logical_bytes": logical_bytes,
        "dtypes_by_logical_bytes": dict(sorted(dtypes.items())),
        "state_schema_sha256": schema_sha256,
    }


def _source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    return loaded


def _validate_existing(
    output: Path,
    *,
    expected_step: int,
    expected_state: dict[str, torch.Tensor],
    expected_summary: dict[str, Any],
) -> dict[str, Any]:
    exported = _load(output)
    if set(exported) != {"generator", "step"}:
        raise ValueError(f"unexpected exported keys in {output}: {sorted(exported)}")
    if int(exported["step"]) != expected_step:
        raise ValueError(
            f"exported step mismatch in {output}: {exported['step']} != {expected_step}"
        )
    summary = _summarize(exported["generator"])
    if summary != expected_summary:
        raise ValueError(f"exported tensor summary mismatch in {output}")
    actual_state = exported["generator"]
    if set(actual_state) != set(expected_state):
        raise ValueError(f"exported state-dict keys mismatch in {output}")
    for key, expected in expected_state.items():
        actual = actual_state[key]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f"exported tensor metadata mismatch for {key} in {output}")
        if expected.numel() == 0:
            continue
        for linear_index in sorted({0, expected.numel() // 2, expected.numel() - 1}):
            remainder = linear_index
            coordinates = []
            for size in reversed(expected.shape):
                coordinates.append(remainder % size)
                remainder //= size
            index = tuple(reversed(coordinates))
            if not torch.equal(expected[index], actual[index]):
                raise ValueError(
                    f"exported tensor sample mismatch for {key} at {index} in {output}"
                )
    return summary


def export_generator(source: Path, output: Path, *, expected_step: int | None) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("source and output paths must differ")

    print(f"[load] {source}", flush=True)
    checkpoint = _load(source)
    if "generator" not in checkpoint:
        raise ValueError(f"source checkpoint has no 'generator' key: {source}")
    step = int(checkpoint.get("step", -1))
    if expected_step is not None and step != expected_step:
        raise ValueError(f"source step mismatch: checkpoint={step}, expected={expected_step}")
    if step < 0:
        raise ValueError(f"source checkpoint has no valid step: {source}")

    summary = _summarize(checkpoint["generator"])
    print(
        f"[source] step={step} tensors={summary['tensor_count']:,} "
        f"logical_bytes={summary['logical_bytes']:,}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        _validate_existing(
            output,
            expected_step=step,
            expected_state=checkpoint["generator"],
            expected_summary=summary,
        )
        print(f"[skip] valid export already exists: {output}", flush=True)
    else:
        temp_handle = tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        )
        temp_path = Path(temp_handle.name)
        temp_handle.close()
        try:
            print(f"[write] {output}", flush=True)
            torch.save({"generator": checkpoint["generator"], "step": step}, temp_path)
            _validate_existing(
                temp_path,
                expected_step=step,
                expected_state=checkpoint["generator"],
                expected_summary=summary,
            )
            os.replace(temp_path, output)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": _source_identity(source),
        "source_key": "generator",
        "step": step,
        "export": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            **summary,
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.yaml")
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
    print(f"[done] {output} ({output.stat().st_size:,} bytes)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the non-EMA generator from a training checkpoint."
    )
    parser.add_argument("--source", required=True, type=Path, help="Source model.pt")
    parser.add_argument("--output", required=True, type=Path, help="Destination generator_only.pt")
    parser.add_argument("--expected-step", type=int, default=None)
    args = parser.parse_args()
    try:
        export_generator(args.source, args.output, expected_step=args.expected_step)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
