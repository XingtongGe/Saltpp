#!/usr/bin/env python3
"""
Extract training prompts (captions) + per-clip LTX frame length from a
frame-range data config, without decoding any video/audio.

Output format (JSONL, one object per line):
    {"prompt": "<caption>", "num_frames": <pixel frames, 8n+1>}

Usage:
    python scripts/extract_dataset_prompts.py \
        --stage_config configs/stage1_csf_teacher/csf_ar_teacher_480p.yaml \
        --out eval/prompts/rollout_prompts_example.jsonl \
        --target 100000 \
        --num_workers 16
"""

import argparse
from collections import Counter
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _dataset_package_path() -> Path:
    return Path(__file__).resolve().parents[1] / "packages" / "dataset"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_config", default=None, help="Frame-range data config yaml.")
    p.add_argument(
        "--stage_config",
        default=None,
        help="Training stage config. Its data_config_path, frame range, caption priority, and prompt filters are used as defaults.",
    )
    p.add_argument("--oss_conf", default=None, help="Object storage conf yaml.")
    p.add_argument("--out", required=True, help="Output .jsonl path (prompt + num_frames).")
    p.add_argument("--emit_txt", action="store_true", help="Also write a prompt-per-line .txt next to --out.")
    p.add_argument("--min_frames", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--target_fps", type=int, default=None)
    p.add_argument("--meta_prefix", default=os.environ.get("DATASET_META_PREFIX", "s3://your-bucket/metadata"))
    p.add_argument("--require_audio", action="store_true", default=None)
    p.add_argument("--no_require_audio", dest="require_audio", action="store_false")
    p.add_argument("--require_audio_flag", action="store_true", default=None)
    p.add_argument("--no_require_audio_flag", dest="require_audio_flag", action="store_false")
    p.add_argument("--audio_range_mode", default=None)
    p.add_argument("--max_entries_per_group", type=int, default=None)
    p.add_argument("--expand_tasks", action="store_true", default=None)
    p.add_argument("--no_expand_tasks", dest="expand_tasks", action="store_false")
    p.add_argument("--caption_priority", default="", help="Comma-separated caption keys, highest priority first.")
    p.add_argument("--part_key_priority", default="", help="Comma-separated part keys, highest priority first.")
    p.add_argument("--prompt_filter_remove_entity_tags", action="store_true", default=None)
    p.add_argument("--no_prompt_filter_remove_entity_tags", dest="prompt_filter_remove_entity_tags", action="store_false")
    p.add_argument("--prompt_filter_remove_speaker_span_tags", action="store_true", default=None)
    p.add_argument("--no_prompt_filter_remove_speaker_span_tags", dest="prompt_filter_remove_speaker_span_tags", action="store_false")
    p.add_argument("--prompt_filter_cleanup_spacing", action="store_true", default=None)
    p.add_argument("--no_prompt_filter_cleanup_spacing", dest="prompt_filter_cleanup_spacing", action="store_false")
    p.add_argument("--dedup_caption", action="store_true", help="Drop duplicate captions (keep first).")
    p.add_argument(
        "--caption_keys",
        default="",
        help="Comma-separated part-level caption keys to emit per clip (multi-variant augmentation).",
    )
    p.add_argument("--num_workers", type=int, default=32, help="Threads for metadata fetch.")
    p.add_argument("--connect_timeout", type=float, default=5.0)
    p.add_argument("--readwrite_timeout", type=float, default=10.0)
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--flush_every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--include_source_metadata",
        action="store_true",
        help="Add source/source_dataset_idx/source_group_idx to each record for mix auditing.",
    )
    p.add_argument(
        "--source_ratios",
        default="",
        help="Optional exact accepted-record mix, keyed by source list directory (e.g. split_a=1,split_b=1).",
    )
    p.add_argument("--target", type=int, default=100000, help="Collect up to N valid prompts (sampling mode).")
    p.add_argument("--scan_limit", type=int, default=0, help="Max rows to probe; 0 => target*30.")
    p.add_argument("--scan_all", action="store_true", help="Enumerate ALL rows in listed order.")
    p.add_argument("--weighted", action="store_true", help="Use weighted with-replacement sampling.")
    args = p.parse_args()

    stage_cfg = {}
    if args.stage_config:
        import yaml

        with open(args.stage_config, "r", encoding="utf-8") as f:
            stage_cfg = yaml.safe_load(f) or {}

    def stage_default(arg_name, config_name, fallback):
        value = getattr(args, arg_name)
        if value is None:
            value = stage_cfg.get(config_name, fallback)
            setattr(args, arg_name, value)

    stage_default("data_config", "data_config_path", None)
    stage_default("oss_conf", "oss_conf_path", None)
    stage_default("min_frames", "min_frames", 113)
    stage_default("max_frames", "max_frames", 193)
    stage_default("target_fps", "target_fps", 24)
    stage_default("require_audio", "require_audio", True)
    stage_default("require_audio_flag", "require_audio_flag", True)
    stage_default("audio_range_mode", "audio_range_mode", "segment")
    stage_default("max_entries_per_group", "max_entries_per_group", 0)
    stage_default("expand_tasks", "expand_tasks", False)
    stage_default("prompt_filter_remove_entity_tags", "prompt_filter_remove_entity_tags", False)
    stage_default("prompt_filter_remove_speaker_span_tags", "prompt_filter_remove_speaker_span_tags", False)
    stage_default("prompt_filter_cleanup_spacing", "prompt_filter_cleanup_spacing", True)
    stage_default("seed", "seed", 7182)

    if not args.caption_priority:
        args.caption_priority = stage_cfg.get("caption_priority", [])
    else:
        args.caption_priority = [x.strip() for x in args.caption_priority.split(",") if x.strip()]
    if not args.part_key_priority:
        args.part_key_priority = stage_cfg.get("part_key_priority", [])
    else:
        args.part_key_priority = [x.strip() for x in args.part_key_priority.split(",") if x.strip()]

    if not args.data_config:
        p.error("one of --data_config or --stage_config containing data_config_path is required")
    if args.source_ratios and not args.weighted:
        p.error("--source_ratios requires --weighted to preserve the configured group distribution")
    if args.source_ratios and args.scan_all:
        p.error("--source_ratios cannot be combined with --scan_all")
    return args


def main():
    args = parse_args()

    sys.path.insert(0, str(_dataset_package_path()))
    from oss_client import Client as StorageClient
    from video_audio_dataset import _get_audio_url
    from frame_range_dataset import (
        FrameRangeVideoAudioDataset,
        _build_meta_json_path,
    )

    dataset = FrameRangeVideoAudioDataset(
        data_config_path=args.data_config,
        oss_conf_path=args.oss_conf,
        target_fps=args.target_fps,
        require_audio=args.require_audio,
        require_audio_flag=args.require_audio_flag,
        caption_priority=args.caption_priority or None,
        part_key_priority=args.part_key_priority or None,
        prompt_filter_remove_entity_tags=args.prompt_filter_remove_entity_tags,
        prompt_filter_remove_speaker_span_tags=args.prompt_filter_remove_speaker_span_tags,
        prompt_filter_cleanup_spacing=args.prompt_filter_cleanup_spacing,
        expand_tasks=args.expand_tasks,
        max_entries_per_group=args.max_entries_per_group,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        meta_prefix=args.meta_prefix,
        seed=args.seed,
        audio_range_mode=args.audio_range_mode,
    )

    oss_conf = args.oss_conf or dataset.share_params.get("oss_file_path")
    total_rows = sum(len(g["rows"]) for g in dataset.groups)

    client = StorageClient(
        conf_path=oss_conf,
        connect_timeout=args.connect_timeout,
        readwrite_timeout=args.readwrite_timeout,
        max_attempts=args.max_attempts,
        max_connections=max(args.num_workers + 16, 32),
    )
    client._oss_client()

    variant_keys = tuple(k.strip() for k in args.caption_keys.split(",") if k.strip())

    def _source_name(group):
        rows = group.get("rows", [])
        if rows:
            source_list = rows[0].get("source_json_list", "")
            if source_list:
                return Path(source_list).parent.name
        return f"dataset_{group['dataset_idx']}"

    source_quotas = {}
    if args.source_ratios:
        source_weights = {}
        for item in args.source_ratios.split(","):
            name, separator, weight_text = item.strip().partition("=")
            if not separator or not name or float(weight_text) <= 0:
                raise ValueError(f"Invalid --source_ratios item: {item!r}")
            source_weights[name] = float(weight_text)

        available_sources = {_source_name(group) for group in dataset.groups}
        if set(source_weights) != available_sources:
            raise ValueError(
                "--source_ratios keys must exactly match loaded sources: "
                f"expected={sorted(available_sources)}, got={sorted(source_weights)}"
            )

        weight_total = sum(source_weights.values())
        raw_quotas = {
            source: args.target * weight / weight_total
            for source, weight in source_weights.items()
        }
        source_quotas = {source: int(raw) for source, raw in raw_quotas.items()}
        remainder = args.target - sum(source_quotas.values())
        remainder_order = sorted(
            source_quotas,
            key=lambda source: raw_quotas[source] - source_quotas[source],
            reverse=True,
        )
        for source in remainder_order[:remainder]:
            source_quotas[source] += 1
        print(f"[extract] exact source quotas: {source_quotas}", flush=True)

    def _variant_caption(part, key):
        raw = part.get(key)
        if not (isinstance(raw, str) and raw.strip()):
            return None
        caption = str(raw)
        bg = part.get("background_audio")
        if bg:
            caption = " ".join([caption, f"<background_audio> {bg} <background_audio>"]).strip()
        caption = caption.replace("\n", "").strip()
        return caption or None

    def _extract(item):
        group, row, rng = item
        if args.require_audio_flag and not bool(row["audio_flag"]):
            return []
        try:
            meta_path = _build_meta_json_path(row["entry"], dataset.meta_prefix)
            data_json = json.loads(client.get(meta_path).decode("utf-8"))
            audio_url = _get_audio_url(data_json)
            if args.require_audio and not audio_url:
                return []
            is_v2v = group["task_type"] == "video_varlen_v2v"
            plan = dataset._build_v2v_plan(data_json) if is_v2v else dataset._build_i2v_plan(data_json, row, rng)
            sf = int(plan["sample_frames"])
            if sf < args.min_frames:
                return []
            if args.max_frames > 0 and sf > args.max_frames:
                return []
            if not variant_keys:
                caption = dataset._filter_prompt(str(plan["caption"]).strip())
                return [{"prompt": caption, "num_frames": sf}] if caption else []
            if is_v2v:
                part = data_json["reference_parts"][1]
            else:
                part = data_json[plan["part_key"]][plan["part_idx"]]
            out = []
            local_seen = set()
            for key in variant_keys:
                cap = _variant_caption(part, key)
                cap = dataset._filter_prompt(cap) if cap else None
                if cap and cap not in local_seen:
                    local_seen.add(cap)
                    out.append({"prompt": cap, "num_frames": sf})
            return out
        except Exception:  # noqa: BLE001
            return []

    def _row_rng(row):
        return random.Random(args.seed + (hash(row["entry"]) % (2**31)))

    def _iter_items():
        if args.scan_all:
            for group in dataset.groups:
                for row in group["rows"]:
                    yield group, row, _row_rng(row)
        elif args.weighted:
            scan_limit = args.scan_limit if args.scan_limit > 0 else args.target * 30
            probed_entries = set()
            idx = 0
            emitted = 0
            while emitted < scan_limit:
                group, row, rng = dataset._choose_row(idx)
                idx += 1
                key = row["entry"]
                if key in probed_entries:
                    continue
                probed_entries.add(key)
                emitted += 1
                yield group, row, rng
        else:
            flat = [(group, row) for group in dataset.groups for row in group["rows"]]
            random.Random(args.seed).shuffle(flat)
            limit = args.scan_limit if args.scan_limit > 0 else len(flat)
            for group, row in flat[:limit]:
                yield group, row, _row_rng(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    seen_caption = set()
    n_ok = 0
    n_probed = 0
    frame_sum = 0
    frame_min = None
    frame_max = None
    source_counts = Counter()
    item_gen = _iter_items()
    stop = False
    chunk_size = max(args.num_workers * 8, 256)

    txt_path = os.path.splitext(args.out)[0] + ".txt" if args.emit_txt else None

    resuming = args.resume and os.path.exists(args.out)
    if resuming:
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                cap = rec.get("prompt")
                if not cap:
                    continue
                if args.dedup_caption:
                    seen_caption.add(cap)
                n_ok += 1
                sf = rec.get("num_frames")
                if isinstance(sf, int):
                    frame_sum += sf
                    frame_min = sf if frame_min is None else min(frame_min, sf)
                    frame_max = sf if frame_max is None else max(frame_max, sf)
                source = rec.get("source")
                if isinstance(source, str) and source:
                    source_counts[source] += 1
        print(f"[extract] resume: loaded {n_ok} existing prompts from {args.out}", flush=True)
        stop = not args.scan_all and n_ok >= args.target

    open_mode = "a" if resuming else "w"
    fout = open(args.out, open_mode, encoding="utf-8")
    ftxt = open(txt_path, open_mode, encoding="utf-8") if txt_path else None

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(item_gen))
                    except StopIteration:
                        stop = True
                        break
                if not chunk:
                    break
                futures = {ex.submit(_extract, item): item[0] for item in chunk}
                for fut in as_completed(futures):
                    group = futures[fut]
                    n_probed += 1
                    for rec in fut.result():
                        if args.dedup_caption:
                            if rec["prompt"] in seen_caption:
                                continue
                            seen_caption.add(rec["prompt"])
                        source_idx = int(group["dataset_idx"])
                        source = _source_name(group)
                        if source_quotas and source_counts[source] >= source_quotas[source]:
                            continue
                        source_counts[source] += 1
                        if args.include_source_metadata:
                            rec["source"] = source
                            rec["source_dataset_idx"] = source_idx
                            rec["source_group_idx"] = int(group["sample_group_idx"])
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if ftxt is not None:
                            ftxt.write(rec["prompt"].replace("\n", " ").strip() + "\n")
                        n_ok += 1
                        sf = rec["num_frames"]
                        frame_sum += sf
                        frame_min = sf if frame_min is None else min(frame_min, sf)
                        frame_max = sf if frame_max is None else max(frame_max, sf)
                        if n_ok % args.flush_every == 0:
                            fout.flush()
                            if ftxt is not None:
                                ftxt.flush()
                        if not args.scan_all and n_ok >= args.target:
                            stop = True
                            break
                    if stop:
                        break
                print(
                    f"[extract] probed {n_probed} | collected {n_ok}"
                    + (f"/{args.target}" if not args.scan_all else "")
                    + f" | hit={n_ok / max(1, n_probed):.2f} (total_rows={total_rows})",
                    flush=True,
                )
    finally:
        fout.flush()
        fout.close()
        if ftxt is not None:
            ftxt.flush()
            ftxt.close()

    if n_ok:
        print(
            f"[extract] wrote {n_ok} prompts -> {args.out} | "
            f"num_frames min={frame_min} max={frame_max} "
            f"mean={frame_sum / n_ok:.1f}",
            flush=True,
        )
        if txt_path:
            print(f"[extract] wrote prompt-only txt -> {txt_path}", flush=True)
    else:
        print("[extract] WARNING: no prompts extracted; check config/storage/filters.", flush=True)


if __name__ == "__main__":
    main()
