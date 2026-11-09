#!/usr/bin/env python3
"""1664x960 scale-wise (2LR + 2HR) causal audio-video inference for Salt++ (Stage 3).

Runs 2 low-resolution (480p) calls and 2 high-resolution (960p) calls per
causal block (4 total generator calls) with independent LR/HR KV caches and a
causal x2 latent upsampler.

Examples:
    # Single-prompt 1664x960 generation:
    python scripts/infer_960p_scale_wise.py \
        --config configs/stage3_scale_wise_960p/scale_wise_2lr2hr_960p.yaml \
        --checkpoint ./checkpoints/saltpp_960p/generator_only.pt \
        --prompt "A close-up of a guitarist strumming an acoustic guitar and singing warmly in a sunlit studio." \
        --output-dir ./outputs/demo_960p

    # Batch generation from a prompt file (e.g. JavisBench-mini):
    python scripts/infer_960p_scale_wise.py \
        --config configs/stage3_scale_wise_960p/scale_wise_2lr2hr_960p.yaml \
        --checkpoint ./checkpoints/saltpp_960p/generator_only.pt \
        --prompt-file ./eval/prompts/JavisBench-mini.prompts.txt \
        --output-dir ./outputs/javisbench_960p
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _setup_paths() -> None:
    repo_root = Path(
        os.environ.get(
            "SALTPP_ROOT",
            os.environ.get("OMNIFORCING_LTX2_ROOT", Path(__file__).resolve().parents[1]),
        )
    ).resolve()
    dist_root = repo_root / "packages" / "ltx-distillation"
    if not dist_root.is_dir():
        raise RuntimeError(f"Invalid repository root: {repo_root}")
    for path in (
        dist_root / "src",
        repo_root / "packages" / "ltx-causal" / "src",
        repo_root / "packages" / "ltx-core" / "src",
        repo_root / "packages" / "ltx-pipelines" / "src",
        repo_root / "packages" / "dataset",
    ):
        sys.path.insert(0, str(path))


_setup_paths()

from ltx_causal.attention.mask_builder import compute_aligned_audio_frames  # noqa: E402
from ltx_distillation.causal_window_upsampler import (  # noqa: E402
    build_causal_upsampler_from_checkpoint,
)
from ltx_distillation.models.text_encoder_wrapper import (  # noqa: E402
    create_text_encoder_wrapper,
)
from ltx_distillation.models.vae_wrapper import create_vae_wrappers  # noqa: E402
from ltx_distillation.stage35_media import decode_video_tiled, write_video_pyav  # noqa: E402
from ltx_distillation.stage35_refiner import (  # noqa: E402
    build_causal_wrapper_from_checkpoint,
)
from ltx_distillation.stage4_reference_dpo import Stage4ReferenceDPO  # noqa: E402
from ltx_distillation.train_distillation import (  # noqa: E402
    _load_config_with_base,
    compute_latent_shapes,
)


EXPECTED_SCHEDULE = (1000, 960, 889, 727, 0)


def _read_prompts(
    prompt_path: Optional[Path],
    single_prompt: Optional[str],
    start_index: int,
    limit: int,
) -> list[str]:
    if single_prompt:
        return [single_prompt.strip()]
    if prompt_path is None:
        raise ValueError("One of --prompt or --prompt-file is required")
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if start_index < 0 or start_index >= len(prompts):
        raise ValueError(
            f"invalid start_index={start_index} for total={len(prompts)} prompts"
        )
    prompts = prompts[start_index:]
    if limit > 0:
        prompts = prompts[:limit]
    return prompts


def _sample_paths(output_dir: Path, index: int) -> tuple[Path, Path, Path]:
    stem = f"sample_{index:04d}"
    return (
        output_dir / f"{stem}.mp4",
        output_dir / f"{stem}.wav",
        output_dir / f"{stem}.txt",
    )


def _sample_complete(output_dir: Path, index: int) -> bool:
    mp4_path, wav_path, _ = _sample_paths(output_dir, index)
    return all(path.is_file() and path.stat().st_size > 0 for path in (mp4_path, wav_path))


def _load_trained_upsampler(
    module: torch.nn.Module,
    checkpoint_path: Path,
) -> int:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping: {checkpoint_path}")
    step = int(checkpoint.get("step", 0))
    state = checkpoint.get("causal_upsampler")
    if not isinstance(state, dict):
        print(
            "[load] checkpoint does not contain a separate 'causal_upsampler' key "
            "(e.g. generator_only.pt); keeping causal_upsampler_checkpoint_path weights.",
            flush=True,
        )
        del checkpoint
        return step
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "strict causal upsampler load unexpectedly returned "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    del state, checkpoint
    return step


def _normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    audio = waveform.detach().cpu().float()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
        audio = audio.transpose(0, 1)
    if audio.ndim != 2:
        raise ValueError(f"expected decoded audio [C,T], got {tuple(audio.shape)}")
    return audio


def _save_sample(
    *,
    frames: torch.Tensor,
    waveform: torch.Tensor,
    prompt: str,
    mp4_path: Path,
    wav_path: Path,
    txt_path: Path,
    fps: int,
    audio_sample_rate: int,
) -> None:
    audio = _normalize_audio(waveform)
    mp4_tmp = mp4_path.with_suffix(".part.mp4")
    wav_tmp = wav_path.with_suffix(".part.wav")
    txt_tmp = txt_path.with_suffix(".part.txt")
    for path in (mp4_path, wav_path, txt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torchaudio.save(str(wav_tmp), audio, int(audio_sample_rate))
        write_video_pyav(
            str(mp4_tmp),
            frames,
            fps=int(fps),
            audio_array=audio,
            audio_fps=int(audio_sample_rate),
        )
        txt_tmp.write_text(prompt + "\n", encoding="utf-8")
        os.replace(wav_tmp, wav_path)
        os.replace(mp4_tmp, mp4_path)
        os.replace(txt_tmp, txt_path)
    except BaseException:
        for path in (mp4_tmp, wav_tmp, txt_tmp):
            if path.exists():
                path.unlink()
        raise


def _validate_config(config: Any, args: argparse.Namespace) -> None:
    schedule = tuple(int(value) for value in config.denoising_step_list)
    if schedule != EXPECTED_SCHEDULE:
        raise ValueError(
            f"Scale-wise 2LR+2HR requires schedule {list(EXPECTED_SCHEDULE)}, got {list(schedule)}"
        )
    if not bool(config.get("strict_denoising_sigmas", False)):
        raise ValueError("Scale-wise 2LR+2HR requires strict_denoising_sigmas=true")
    if not bool(config.get("generator_use_causal_wrapper", False)):
        raise ValueError("Scale-wise 2LR+2HR requires generator_use_causal_wrapper=true")
    if (int(args.num_frames) - 1) % 8 != 0:
        raise ValueError("num_frames must be 1 + 8*k")
    if int(args.height) % 32 or int(args.width) % 32:
        raise ValueError("LR height and width must be divisible by 32")


def _build_runtime(
    config: Any,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    load_vaes: bool = True,
) -> tuple[Any, Any, Any, Any, Stage4ReferenceDPO, int]:
    print("[load] causal scale-wise policy...", flush=True)
    policy = build_causal_wrapper_from_checkpoint(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
        dtype=dtype,
        trainable=False,
        gradient_checkpointing=False,
    )

    print("[load] causal x2 latent upsampler...", flush=True)
    causal_upsampler = build_causal_upsampler_from_checkpoint(
        checkpoint_path=str(config.checkpoint_path),
        spatial_upsampler_path=str(config.spatial_upsampler_path),
        causal_upsampler_checkpoint_path=str(config.causal_upsampler_checkpoint_path),
        device=device,
        dtype=dtype,
        history_frames=int(config.get("causal_upsampler_history_frames", 18)),
        gradient_checkpointing=False,
    )
    checkpoint_step = _load_trained_upsampler(causal_upsampler, checkpoint_path)
    causal_upsampler.eval().requires_grad_(False)

    print("[load] text encoder...", flush=True)
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=str(config.checkpoint_path),
        gemma_path=str(config.get("text_encoder_checkpoint", config.gemma_path)),
        device=device,
        dtype=dtype,
    ).eval().requires_grad_(False)

    if load_vaes:
        print("[load] tiled video/audio VAE decoders...", flush=True)
        video_vae, audio_vae = create_vae_wrappers(
            checkpoint_path=str(config.checkpoint_path),
            device=device,
            dtype=dtype,
        )
        video_vae.eval().requires_grad_(False)
        audio_vae.eval().requires_grad_(False)
    else:
        video_vae = None
        audio_vae = None

    rollout = Stage4ReferenceDPO(
        policy=policy,
        reference=policy,
        official_refiner=None,
        causal_upsampler=causal_upsampler,
        device=device,
        dtype=dtype,
        num_frame_per_block=int(config.get("num_frame_per_block", 3)),
        first_block_video_frames=int(config.get("first_block_video_frames", 4)),
        enable_chunk_recycling=bool(config.get("enable_chunk_recycling", False)),
    )
    return policy, causal_upsampler, text_encoder, (video_vae, audio_vae), rollout, checkpoint_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Salt++ 1664x960 scale-wise (2LR+2HR) generator")
    parser.add_argument("--config", required=True, help="Stage 3 scale-wise config yaml")
    parser.add_argument("--checkpoint", required=True, help="Stage 3 model.pt or generator_only.pt")
    parser.add_argument("--prompt", default=None, help="Single text prompt for quick generation")
    parser.add_argument("--prompt-file", default=None, help="Text file with one prompt per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Number of prompts to run (0 = all)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=480, help="LR height (HR output is 2x)")
    parser.add_argument("--width", type=int, default=832, help="LR width (HR output is 2x)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio-sample-rate", type=int, default=24000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Scale-wise 960p generation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = _load_config_with_base(args.config)
    _validate_config(config, args)
    checkpoint_path = Path(args.checkpoint).resolve()
    prompt_path = Path(args.prompt_file).resolve() if args.prompt_file else None
    output_dir = Path(args.output_dir).resolve() / "t2av"
    prompts = _read_prompts(prompt_path, args.prompt, args.start_index, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    policy, upsampler, text_encoder, vaes, rollout, checkpoint_step = _build_runtime(
        config, checkpoint_path, device, dtype
    )
    video_vae, audio_vae = vaes
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    video_shape, audio_shape = compute_latent_shapes(
        num_frames=int(args.num_frames),
        video_height=int(args.height),
        video_width=int(args.width),
        batch_size=1,
    )
    audio_shape[1] = compute_aligned_audio_frames(
        total_video_latent_frames=int(video_shape[1]),
        num_frame_per_block=int(config.get("num_frame_per_block", 3)),
        first_block_video_frames=int(config.get("first_block_video_frames", 4)),
    )
    print(
        f"[ready] load={load_seconds:.2f}s checkpoint_step={checkpoint_step} "
        f"LR_latent={tuple(video_shape)} HR_latent="
        f"{(video_shape[0], video_shape[1], video_shape[2], video_shape[3] * 2, video_shape[4] * 2)} "
        f"output={args.height * 2}x{args.width * 2}",
        flush=True,
    )
    print(
        f"[t2av] generating {len(prompts)} cases from global index {args.start_index}",
        flush=True,
    )

    case_timings: list[dict[str, float | int]] = []
    run_start = time.perf_counter()
    for local_index, prompt in enumerate(prompts):
        global_index = int(args.start_index) + local_index
        mp4_path, wav_path, txt_path = _sample_paths(output_dir, global_index)
        if args.skip_existing and _sample_complete(output_dir, global_index):
            print(f"[skip] global={global_index}", flush=True)
            continue
        seed = int(args.seed) + global_index
        case_start = time.perf_counter()
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            text_start = time.perf_counter()
            conditional_dict = text_encoder(text_prompts=[prompt])
            torch.cuda.synchronize()
            text_seconds = time.perf_counter() - text_start

            generation_start = time.perf_counter()
            sample = rollout.sample_scale_wise(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
            )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_start

            decode_start = time.perf_counter()
            frames = decode_video_tiled(
                video_vae=video_vae,
                latent_bfchw=sample.hr_video,
                seed=seed,
            )
            waveform = audio_vae.decode_to_waveform(sample.hr_audio)
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - decode_start

        save_start = time.perf_counter()
        _save_sample(
            frames=frames,
            waveform=waveform,
            prompt=prompt,
            mp4_path=mp4_path,
            wav_path=wav_path,
            txt_path=txt_path,
            fps=int(args.fps),
            audio_sample_rate=int(args.audio_sample_rate),
        )
        save_seconds = time.perf_counter() - save_start
        total_seconds = time.perf_counter() - case_start
        peak_gib = torch.cuda.max_memory_allocated(device) / float(1024**3)
        case_timings.append(
            {
                "index": global_index,
                "text_seconds": text_seconds,
                "generation_seconds": generation_seconds,
                "decode_seconds": decode_seconds,
                "save_seconds": save_seconds,
                "total_seconds": total_seconds,
                "peak_allocated_gib": peak_gib,
            }
        )
        print(
            f"[t2av] {local_index + 1}/{len(prompts)} global={global_index} "
            f"seed={seed} gen={generation_seconds:.2f}s "
            f"decode={decode_seconds:.2f}s save={save_seconds:.2f}s "
            f"total={total_seconds:.2f}s peak={peak_gib:.2f}GiB",
            flush=True,
        )
        del conditional_dict, sample, frames, waveform
        torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - run_start
    generated = len(case_timings)
    timing = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "load_seconds": load_seconds,
        "wall_seconds": wall_seconds,
        "generated_cases": generated,
        "average_seconds_per_generated_case": (
            sum(float(item["total_seconds"]) for item in case_timings) / generated
            if generated
            else 0.0
        ),
        "cases": case_timings,
    }
    timing_path = output_dir.parent / f"timing_{args.start_index:04d}_{len(prompts):04d}.json"
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(
        f"[done] generated={generated} wall={wall_seconds:.2f}s "
        f"timing={timing_path}",
        flush=True,
    )
    del policy, upsampler, text_encoder, video_vae, audio_vae, rollout


if __name__ == "__main__":
    main()
