"""Shared model/runtime helpers for Stage-3.5 causal refiner distillation."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from safetensors import safe_open


def unwrap_module(module: nn.Module) -> nn.Module:
    current = module
    while hasattr(current, "module"):
        current = current.module
    return current


def checkpoint_generator_state(
    path: str,
    *,
    use_ema: bool = False,
) -> tuple[dict[str, torch.Tensor], str]:
    # Stage3/4 checkpoints may also contain a second 19B critic.  Memory-map
    # the archive so loading only ``generator`` does not eagerly materialize
    # the entire 70+ GiB checkpoint once per distributed rank.
    try:
        loaded = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except (TypeError, RuntimeError):
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        return loaded, "raw"
    if use_ema and "ema_generator" in loaded:
        return loaded["ema_generator"], "ema_generator"
    for key in ("generator", "model", "state_dict"):
        if key in loaded:
            return loaded[key], key
    return loaded, "raw"


def build_causal_wrapper_from_checkpoint(
    *,
    config: Any,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trainable: bool,
    gradient_checkpointing: bool,
    use_ema: bool = False,
) -> nn.Module:
    """Build the same causal wrapper used by Stage 0/3 and load a checkpoint."""

    from ltx_causal import CausalLTXModel
    from ltx_distillation.ar_teacher.ar_teacher import (
        ARTeacherConfig,
        LTX2ARTeacher,
    )

    helper = LTX2ARTeacher(
        ARTeacherConfig(
            checkpoint_path=str(config.get("checkpoint_path", "")),
            num_frame_per_block=int(config.get("num_frame_per_block", 3)),
            first_block_video_frames=int(config.get("first_block_video_frames", 4)),
            gradient_checkpointing=bool(gradient_checkpointing),
            mixed_precision=dtype == torch.bfloat16,
            num_train_timestep=int(config.get("num_train_timestep", 1000)),
            num_audio_sink_tokens=int(config.get("num_audio_sink_tokens", 0)),
            enable_causal_log_rescale=bool(
                config.get("enable_causal_log_rescale", False)
            ),
            enable_chunk_recycling=bool(
                config.get("enable_chunk_recycling", False)
            ),
        ),
        device=device,
    )
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("meta"):
            model = CausalLTXModel(helper._make_model_config())
            wrapper = helper._make_wrapper(model)
    finally:
        torch.set_default_dtype(previous_default_dtype)
    wrapper = wrapper.to_empty(device=device)

    state_dict, source_key = checkpoint_generator_state(
        checkpoint_path,
        use_ema=use_ema,
    )
    state_dict = helper._remap_state_dict_keys(state_dict)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    ignored = ("mask_builder", "causal_gate", "audio_sink_tokens")
    real_missing = [key for key in missing if not any(part in key for part in ignored)]
    print(
        f"[Stage35][causal-load] {checkpoint_path} key={source_key} "
        f"missing={len(real_missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if real_missing:
        print(f"  missing examples: {real_missing[:10]}", flush=True)
    if unexpected:
        print(f"  unexpected examples: {unexpected[:10]}", flush=True)
    del state_dict

    wrapper.requires_grad_(bool(trainable))
    if gradient_checkpointing:
        wrapper.enable_gradient_checkpointing()
    if trainable:
        wrapper.train()
    else:
        wrapper.eval()
    return wrapper


@torch.no_grad()
def merge_lora_inplace(
    model: nn.Module,
    *,
    lora_path: str,
    strength: float,
) -> None:
    """Merge the released refiner LoRA into an already-loaded model."""

    if float(strength) == 0.0:
        return
    parameters = dict(model.named_parameters())
    matched = 0
    unmatched: list[str] = []
    with safe_open(lora_path, framework="pt", device="cpu") as lora_file:
        keys = set(lora_file.keys())
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        for pair_index, a_key in enumerate(a_keys, start=1):
            b_key = a_key[: -len(".lora_A.weight")] + ".lora_B.weight"
            if b_key not in keys:
                raise KeyError(f"LoRA B tensor missing for {a_key}")
            target_key = a_key.removeprefix("diffusion_model.")
            target_key = target_key[: -len(".lora_A.weight")] + ".weight"
            parameter = parameters.get(target_key)
            if parameter is None:
                unmatched.append(target_key)
                continue
            lora_a = lora_file.get_tensor(a_key).to(
                device=parameter.device,
                dtype=torch.bfloat16,
            )
            lora_b = lora_file.get_tensor(b_key).to(
                device=parameter.device,
                dtype=torch.bfloat16,
            )
            delta = torch.matmul(lora_b, lora_a)
            if tuple(delta.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"LoRA shape mismatch for {target_key}: "
                    f"{tuple(delta.shape)} != {tuple(parameter.shape)}"
                )
            parameter.add_(delta.to(parameter.dtype), alpha=float(strength))
            matched += 1
            del lora_a, lora_b, delta
            if pair_index % 100 == 0 or pair_index == len(a_keys):
                print(
                    f"[Stage35][lora] {pair_index}/{len(a_keys)} "
                    f"pairs, matched={matched}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if matched == 0:
        raise RuntimeError("No refiner LoRA tensors matched the model")
    print(
        f"[Stage35][lora] merged={matched}, unmatched={len(unmatched)}, "
        f"strength={strength}",
        flush=True,
    )


def build_official_refiner(
    *,
    base_checkpoint: str,
    lora_path: str,
    lora_strength: float,
    device: torch.device,
    dtype: torch.dtype,
    high_height: int,
    high_width: int,
) -> nn.Module:
    from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=base_checkpoint,
        loras=(),
    )
    transformer = ledger.transformer()
    velocity_model = getattr(transformer, "velocity_model", transformer)
    merge_lora_inplace(
        velocity_model,
        lora_path=lora_path,
        strength=lora_strength,
    )
    wrapper = LTX2DiffusionWrapper(
        model=transformer,
        video_height=int(high_height),
        video_width=int(high_width),
    )
    return wrapper.to(device=device, dtype=dtype).eval().requires_grad_(False)


def full_sigma(
    value: torch.Tensor,
    *,
    batch_size: int,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).expand(batch_size, frames)


def add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(
        sigma,
        device=clean.device,
        dtype=clean.dtype,
    )
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.reshape(*([1] * clean.ndim))
    elif sigma_tensor.ndim == 1:
        sigma_tensor = sigma_tensor.reshape(-1, *([1] * (clean.ndim - 1)))
    elif sigma_tensor.ndim == 2:
        sigma_tensor = sigma_tensor.reshape(
            *sigma_tensor.shape,
            *([1] * (clean.ndim - 2)),
        )
    return ((1.0 - sigma_tensor) * clean + sigma_tensor * noise).to(clean.dtype)


def euler_update(
    sample: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    if float(sigma_next.item()) == 0.0:
        return x0
    dt = sigma_next.float() - sigma.float()
    velocity = (sample.float() - x0.float()) / sigma.float().clamp_min(1e-8)
    return (sample.float() + velocity * dt).to(sample.dtype)


def init_kv_caches(
    model: nn.Module,
    *,
    batch_size: int,
    total_video_frames: int,
    total_audio_frames: int,
    text_seq_len: int,
    video_frame_seqlen: int,
    device: torch.device,
    dtype: torch.dtype,
):
    wrapper = unwrap_module(model)
    return wrapper.init_av_kv_caches(
        batch_size=batch_size,
        num_video_frames=total_video_frames,
        num_audio_frames=total_audio_frames,
        text_seq_len=text_seq_len,
        device=device,
        dtype=dtype,
        video_frame_seqlen=video_frame_seqlen,
    )


def kv_block_forward(
    model: nn.Module,
    *,
    video: torch.Tensor,
    audio: Optional[torch.Tensor],
    video_sigma: torch.Tensor,
    audio_sigma: Optional[torch.Tensor],
    conditional_dict: dict[str, Any],
    kv_caches,
    block,
    ref_video_block: Optional[torch.Tensor] = None,
):
    return model(
        _kv_cache_kwargs=dict(
            noisy_video_block=video,
            noisy_audio_block=audio,
            conditional_dict=conditional_dict,
            timestep_block=video_sigma,
            audio_timestep_block=audio_sigma,
            kv_caches=kv_caches,
            current_video_start_frame=block.video_start,
            current_audio_start_frame=block.audio_start,
            ref_video_block=ref_video_block,
        )
    )


@torch.no_grad()
def refresh_clean_cache(
    model: nn.Module,
    *,
    video_x0: torch.Tensor,
    audio_x0: Optional[torch.Tensor],
    conditional_dict: dict[str, Any],
    kv_caches,
    block,
    sigma_dtype: torch.dtype,
    ref_video_block: Optional[torch.Tensor] = None,
) -> None:
    batch_size = video_x0.shape[0]
    video_sigma = torch.zeros(
        (batch_size, video_x0.shape[1]),
        device=video_x0.device,
        dtype=sigma_dtype,
    )
    audio_sigma = None
    if audio_x0 is not None:
        audio_sigma = torch.zeros(
            (batch_size, audio_x0.shape[1]),
            device=audio_x0.device,
            dtype=sigma_dtype,
        )
    kv_block_forward(
        model,
        video=video_x0.detach(),
        audio=audio_x0.detach() if audio_x0 is not None else None,
        video_sigma=video_sigma,
        audio_sigma=audio_sigma,
        conditional_dict=conditional_dict,
        kv_caches=kv_caches,
        block=block,
        ref_video_block=ref_video_block,
    )


@torch.no_grad()
def official_x0_at_exit(
    model: nn.Module,
    *,
    initial_video: torch.Tensor,
    initial_audio: torch.Tensor,
    conditional_dict: dict[str, Any],
    sigmas: torch.Tensor,
    exit_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    video = initial_video
    audio = initial_audio
    batch_size = video.shape[0]
    for step_index in range(int(exit_index) + 1):
        sigma = sigmas[step_index]
        video_sigma = full_sigma(
            sigma,
            batch_size=batch_size,
            frames=video.shape[1],
            device=video.device,
            dtype=sigmas.dtype,
        )
        audio_sigma = full_sigma(
            sigma,
            batch_size=batch_size,
            frames=audio.shape[1],
            device=audio.device,
            dtype=sigmas.dtype,
        )
        x0_video, x0_audio = model(
            noisy_image_or_video=video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=audio,
            audio_timestep=audio_sigma,
        )
        if step_index == int(exit_index):
            return x0_video, x0_audio
        sigma_next = sigmas[step_index + 1]
        video = euler_update(video, x0_video, sigma, sigma_next)
        audio = euler_update(audio, x0_audio, sigma, sigma_next)
    raise AssertionError("unreachable")


@torch.no_grad()
def official_refine_full(
    model: nn.Module,
    *,
    initial_video: torch.Tensor,
    initial_audio: torch.Tensor,
    conditional_dict: dict[str, Any],
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    video = initial_video
    audio = initial_audio
    batch_size = video.shape[0]
    for step_index, sigma in enumerate(sigmas[:-1]):
        video_sigma = full_sigma(
            sigma,
            batch_size=batch_size,
            frames=video.shape[1],
            device=video.device,
            dtype=sigmas.dtype,
        )
        audio_sigma = full_sigma(
            sigma,
            batch_size=batch_size,
            frames=audio.shape[1],
            device=audio.device,
            dtype=sigmas.dtype,
        )
        x0_video, x0_audio = model(
            noisy_image_or_video=video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=audio,
            audio_timestep=audio_sigma,
        )
        sigma_next = sigmas[step_index + 1]
        video = euler_update(video, x0_video, sigma, sigma_next)
        audio = euler_update(audio, x0_audio, sigma, sigma_next)
    return video, audio


@torch.no_grad()
def causal_refine_full(
    model: nn.Module,
    *,
    initial_video: torch.Tensor,
    initial_audio: torch.Tensor,
    conditional_dict: dict[str, Any],
    sigmas: torch.Tensor,
    blocks,
    enable_chunk_recycling: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ltx_causal.wrapper import build_block_ref_from_prev

    batch_size = initial_video.shape[0]
    output_video = torch.zeros_like(initial_video)
    output_audio = torch.zeros_like(initial_audio)
    kv_caches = init_kv_caches(
        model,
        batch_size=batch_size,
        total_video_frames=initial_video.shape[1],
        total_audio_frames=initial_audio.shape[1],
        text_seq_len=conditional_dict["video_context"].shape[1],
        video_frame_seqlen=initial_video.shape[-2] * initial_video.shape[-1],
        device=initial_video.device,
        dtype=initial_video.dtype,
    )
    previous_video: Optional[torch.Tensor] = None
    for block in blocks:
        current_video = initial_video[:, block.video_start:block.video_end].clone()
        current_audio = initial_audio[:, block.audio_start:block.audio_end].clone()
        ref_video = (
            build_block_ref_from_prev(
                prev_chunk_latent=previous_video,
                current_block_size=block.video_frames,
                template=initial_video,
            )
            if enable_chunk_recycling
            else None
        )
        for step_index, sigma in enumerate(sigmas[:-1]):
            video_sigma = full_sigma(
                sigma,
                batch_size=batch_size,
                frames=current_video.shape[1],
                device=current_video.device,
                dtype=sigmas.dtype,
            )
            audio_sigma = full_sigma(
                sigma,
                batch_size=batch_size,
                frames=current_audio.shape[1],
                device=current_audio.device,
                dtype=sigmas.dtype,
            )
            x0_video, x0_audio = kv_block_forward(
                model,
                video=current_video,
                audio=current_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                kv_caches=kv_caches,
                block=block,
                ref_video_block=ref_video,
            )
            sigma_next = sigmas[step_index + 1]
            current_video = euler_update(
                current_video, x0_video, sigma, sigma_next
            )
            current_audio = euler_update(
                current_audio, x0_audio, sigma, sigma_next
            )
        refresh_clean_cache(
            model,
            video_x0=current_video,
            audio_x0=current_audio,
            conditional_dict=conditional_dict,
            kv_caches=kv_caches,
            block=block,
            sigma_dtype=sigmas.dtype,
            ref_video_block=ref_video,
        )
        output_video[:, block.video_start:block.video_end] = current_video
        output_audio[:, block.audio_start:block.audio_end] = current_audio
        previous_video = current_video.detach()
    return output_video, output_audio
