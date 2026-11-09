#!/usr/bin/env python3
"""Offline 480p causal audio-video inference for Salt++ (Stage 1 / Stage 2 checkpoints).

Supports both full training checkpoints (`model.pt`) and lightweight exported
generator checkpoints (`generator_only.pt` with `--direct-generator-checkpoint-load`).

Examples:
    # Quick single-prompt 4-step generation (Stage 2 Salt++ 480p checkpoint):
    python scripts/infer_480p.py \
        --config configs/stage2_context_aligned_dmd/step2_rollout_ar_dmd_4step.yaml \
        --checkpoint ./checkpoints/saltpp_480p/generator_only.pt \
        --direct-generator-checkpoint-load \
        --denoising-update consistency \
        --denoising-step-list 1000,960,889,727,0 \
        --guidance-scale 1.0 \
        --prompt "A close-up of a guitarist strumming an acoustic guitar and singing warmly in a sunlit studio." \
        --output-dir ./outputs/demo_480p

    # Batch generation from a prompt file (e.g., JavisBench-mini):
    python scripts/infer_480p.py \
        --config configs/stage2_context_aligned_dmd/step2_rollout_ar_dmd_4step.yaml \
        --checkpoint ./checkpoints/saltpp_480p/generator_only.pt \
        --direct-generator-checkpoint-load \
        --denoising-update consistency \
        --denoising-step-list 1000,960,889,727,0 \
        --guidance-scale 1.0 \
        --prompt-file ./eval/prompts/JavisBench-mini.prompts.txt \
        --zero-pad-samples --save-wav \
        --output-dir ./outputs/javisbench_480p
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import av
import numpy as np
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
        raise RuntimeError(
            f"Invalid repository root {repo_root}; expected {dist_root}"
        )
    for path in (
        dist_root / "src",
        repo_root / "packages" / "ltx-causal" / "src",
        repo_root / "packages" / "ltx-core" / "src",
        repo_root / "packages" / "ltx-pipelines" / "src",
        repo_root / "packages" / "dataset",
    ):
        sys.path.insert(0, str(path))


_setup_paths()

from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

from ltx_causal.attention.mask_builder import compute_av_blocks  # noqa: E402
from ltx_causal.wrapper import build_block_ref_from_prev  # noqa: E402
from ltx_distillation.ar_teacher.ar_teacher import ARTeacherConfig, LTX2ARTeacher  # noqa: E402
from ltx_distillation.train_distillation import _load_config_with_base  # noqa: E402


VIDEO_CODEC = "libx264"
VIDEO_CRF = "18"
VIDEO_PRESET = "medium"
VIDEO_PIXEL_FORMAT = "yuv420p"
MP4_AUDIO_CODEC = "aac"
MP4_AUDIO_BIT_RATE = 192_000


def write_video_pyav(
    filename: str,
    video_array: torch.Tensor,
    *,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
) -> None:
    """Write the canonical benchmark MP4 with fully specified PyAV settings."""

    frames = video_array.detach().cpu()
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
    frames_np = np.ascontiguousarray(frames.numpy())
    if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
        raise ValueError(
            f"expected video tensor [T,H,W,3], got {tuple(frames_np.shape)}"
        )

    audio_np = None
    audio_layout = None
    if audio_array is not None:
        if audio_fps is None:
            raise ValueError("audio_fps is required when audio_array is provided")
        audio = audio_array.detach().cpu().float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.transpose(0, 1)
        elif audio.ndim != 2:
            raise ValueError(
                f"expected audio tensor [T], [C,T], or [T,C], got {tuple(audio.shape)}"
            )
        if audio.shape[0] > 2:
            audio = audio[:2]
        audio_np = np.ascontiguousarray(audio.clamp(-1, 1).numpy().astype(np.float32))
        audio_layout = "mono" if audio_np.shape[0] == 1 else "stereo"

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(filename, mode="w")
    try:
        video_stream = container.add_stream(VIDEO_CODEC, rate=int(fps))
        video_stream.width = int(frames_np.shape[2])
        video_stream.height = int(frames_np.shape[1])
        video_stream.pix_fmt = VIDEO_PIXEL_FORMAT
        video_stream.options = {"crf": VIDEO_CRF, "preset": VIDEO_PRESET}

        audio_stream = None
        if audio_np is not None:
            audio_stream = container.add_stream(MP4_AUDIO_CODEC, rate=int(audio_fps))
            audio_stream.layout = str(audio_layout)
            audio_stream.bit_rate = MP4_AUDIO_BIT_RATE

        for frame_np in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in video_stream.encode(frame):
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

        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)
    finally:
        container.close()


def compute_latent_shapes(
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")

    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    return (
        [batch_size, latent_frames, latent_channels, latent_h, latent_w],
        [batch_size, audio_frames, latent_channels],
    )


class CausalAVInferenceRunner:
    def __init__(
        self,
        config: Any,
        checkpoint_path: str,
        *,
        use_ema: bool = False,
        sigma_schedule: str = "ltx",
        denoising_update: str = "euler",
        guidance_scale: Optional[float] = None,
        lora_dir: Optional[str] = None,
        direct_generator_checkpoint_load: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if bool(config.get("mixed_precision", True)) else torch.float32
        self.sigma_schedule = sigma_schedule
        if denoising_update not in {"euler", "consistency"}:
            raise ValueError(f"Unknown denoising_update: {denoising_update}")
        self.denoising_update = denoising_update

        ar_config = ARTeacherConfig(
            checkpoint_path=config.get("checkpoint_path", ""),
            bidirectional_model_checkpoint=config.get("bidirectional_model_checkpoint"),
            causal_model_checkpoint=config.get("causal_model_checkpoint"),
            text_encoder_checkpoint=config.get("text_encoder_checkpoint", config.get("gemma_path", "")),
            num_frame_per_block=config.get("num_frame_per_block", 3),
            first_block_video_frames=config.get("first_block_video_frames", 4),
            gradient_checkpointing=False,
            mixed_precision=config.get("mixed_precision", True),
            num_train_timestep=config.get("num_train_timestep", 1000),
            sigma_shift=config.get("sigma_shift", 1.0),
            sigma_min=config.get("sigma_min", 0.0),
            sigma_max=config.get("sigma_max", 1.0),
            timestep_sampler_type=config.get("timestep_sampler_type", "uniform_shifted"),
            timestep_sampler_std=config.get("timestep_sampler_std", 1.0),
            timestep_sampler_uniform_prob=config.get("timestep_sampler_uniform_prob", 0.1),
            timestep_sampler_eps=config.get("timestep_sampler_eps", 1e-3),
            use_gaussian_weight=config.get("use_gaussian_weight", False),
            noise_augmentation_max_sigma=config.get("noise_augmentation_max_sigma", 0.0),
            video_loss_weight=config.get("video_loss_weight", 1.0),
            audio_loss_weight=config.get("audio_loss_weight", 0.16),
            num_audio_sink_tokens=config.get("num_audio_sink_tokens", 0),
            enable_causal_log_rescale=config.get("enable_causal_log_rescale", False),
            enable_chunk_recycling=config.get("enable_chunk_recycling", False),
            first_frame_conditioning_p=config.get("first_frame_conditioning_p", 0.0),
        )

        print("[load] building causal AR generator modules...", flush=True)
        self.ar_teacher = LTX2ARTeacher(ar_config, device=self.device)
        if direct_generator_checkpoint_load:
            self._build_models_for_direct_generator_load(ar_config)
        else:
            self.ar_teacher._load_models()
        self.ar_teacher._generator.eval().requires_grad_(False)
        self.ar_teacher._text_encoder.eval().requires_grad_(False)

        print(f"[load] loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        key = "ema_generator" if use_ema and "ema_generator" in ckpt else "generator"
        if use_ema and key != "ema_generator":
            print("[load] --use-ema requested but checkpoint has no ema_generator; using generator.", flush=True)
        missing, unexpected = self.ar_teacher._generator.load_state_dict(ckpt[key], strict=False)
        if missing:
            print(f"[load] missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"[load] unexpected keys: {len(unexpected)}", flush=True)
        if direct_generator_checkpoint_load and (missing or unexpected):
            raise RuntimeError(
                "direct generator checkpoint load requires a complete state dict: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        self.ckpt_step = int(ckpt.get("step", 0))
        del ckpt
        self._release_host_memory()

        if lora_dir:
            from peft import PeftModel

            print(f"[load] applying LoRA adapter: {lora_dir}", flush=True)
            self.ar_teacher._generator = PeftModel.from_pretrained(
                self.ar_teacher._generator,
                lora_dir,
                adapter_name="default",
                is_trainable=False,
            )
            self.ar_teacher._generator.set_adapter("default")
            self.ar_teacher._generator.eval().requires_grad_(False)

        self.benchmark_num_inference_steps = int(config.get("benchmark_num_inference_steps", 4))
        self.benchmark_guidance_scale = (
            float(guidance_scale)
            if guidance_scale is not None
            else float(config.get("benchmark_guidance_scale", 1.0))
        )
        self.benchmark_seed = int(config.get("benchmark_seed", 12345))
        self.benchmark_video_fps = int(config.get("benchmark_video_fps", 24))
        self.benchmark_audio_sample_rate = int(config.get("benchmark_audio_sample_rate", 24000))
        self.benchmark_context_noise = int(config.get("benchmark_context_noise", 0))
        self.benchmark_clear_cuda_cache_per_round = bool(
            config.get("benchmark_clear_cuda_cache_per_round", False)
        )
        self.use_config_denoising_step_list = bool(
            config.get("benchmark_use_denoising_step_list", False)
        )

        self._build_vae_encoders()
        self._build_vae_decoders()

    def _build_models_for_direct_generator_load(self, ar_config: ARTeacherConfig) -> None:
        from ltx_causal import CausalLTXModel
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper

        checkpoint_path = getattr(ar_config, "checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError("direct generator load still requires checkpoint_path for text encoding")

        print("[load] direct generator checkpoint mode: skip base generator weights", flush=True)
        self.ar_teacher._text_encoder = create_text_encoder_wrapper(
            checkpoint_path=checkpoint_path,
            gemma_path=ar_config.text_encoder_checkpoint,
            device=self.device,
            dtype=self.dtype,
        )
        self.ar_teacher._text_encoder.requires_grad_(False)
        self.ar_teacher._text_encoder.eval()

        previous_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.dtype)
            with torch.device(self.device):
                model = CausalLTXModel(self.ar_teacher._make_model_config())
        finally:
            torch.set_default_dtype(previous_default_dtype)
        self.ar_teacher._generator = self.ar_teacher._make_wrapper(model).to(
            device=self.device, dtype=self.dtype
        )

    @staticmethod
    def _release_host_memory() -> None:
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass

    @staticmethod
    def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    def _build_vae_encoders(self) -> None:
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
        from ltx_core.model.audio_vae import (
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            AudioEncoderConfigurator,
            AudioProcessor,
        )
        from ltx_pipelines.utils.model_ledger import ModelLedger

        print("[load] building VAE encoders on CPU...", flush=True)
        registry = StateDictRegistry()
        ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=self.config.checkpoint_path,
            registry=registry,
        )
        self.video_encoder = ledger.video_encoder().eval().requires_grad_(False)
        audio_enc_builder = Builder(
            model_path=self.config.checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.audio_encoder = audio_enc_builder.build(device=torch.device("cpu"), dtype=self.dtype).eval()
        self.audio_encoder.requires_grad_(False)
        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins,
            mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
        )

    def _build_vae_decoders(self) -> None:
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        print("[load] building VAE decoders on CPU...", flush=True)
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=self.config.checkpoint_path,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)

    def _encoders_to_device(self) -> None:
        self.video_encoder = self.video_encoder.to(device=self.device, dtype=self.dtype)
        self.audio_encoder = self.audio_encoder.to(device=self.device, dtype=self.dtype)
        self.audio_processor = self.audio_processor.to(device=self.device)

    def _encoders_to_cpu(self) -> None:
        self.video_encoder = self.video_encoder.to(device="cpu")
        self.audio_encoder = self.audio_encoder.to(device="cpu")
        self.audio_processor = self.audio_processor.to(device="cpu")
        torch.cuda.empty_cache()

    def _vae_decoders_to_device(self) -> None:
        self.video_vae = self.video_vae.to(device=self.device)
        self.audio_vae = self.audio_vae.to(device=self.device)

    def _vae_decoders_to_cpu(self) -> None:
        self.video_vae = self.video_vae.to(device="cpu")
        self.audio_vae = self.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    def _compute_sigma_schedule(
        self, num_steps: int, *, height: Optional[int] = None, width: Optional[int] = None
    ) -> torch.Tensor:
        if self.use_config_denoising_step_list:
            step_list = self.config.get(
                "benchmark_denoising_step_list",
                self.config.get("denoising_step_list", None),
            )
            if step_list is None:
                raise ValueError(
                    "benchmark_use_denoising_step_list=true but no "
                    "benchmark_denoising_step_list or denoising_step_list is set"
                )
            return torch.tensor([float(t) / 1000.0 for t in step_list], dtype=torch.float32)

        num_frames = int(self.config.get("num_frames", 121))
        video_h = int(height or self.config.get("video_height", 480))
        video_w = int(width or self.config.get("video_width", 832))

        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        if self.sigma_schedule == "uniform_shifted":
            shift = float(self.config.get("sigma_shift", 1.0))
            if shift != 1.0:
                sigmas = torch.where(
                    sigmas != 0,
                    shift * sigmas / (1.0 + (shift - 1.0) * sigmas),
                    sigmas,
                )
            return sigmas.float()

        if self.sigma_schedule != "ltx":
            raise ValueError(f"Unknown sigma_schedule: {self.sigma_schedule}")

        latent_h = video_h // 32
        latent_w = video_w // 32
        latent_f = (num_frames - 1) // 8 + 1
        tokens = latent_f * latent_h * latent_w
        base_shift, max_shift = 0.95, 2.05
        x1, x2 = 1024, 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
            0,
        )
        non_zero = sigmas != 0
        nz = sigmas[non_zero]
        scale = (1.0 - nz[-1]) / (1.0 - 0.1)
        sigmas[non_zero] = 1.0 - (1.0 - nz) / scale
        return sigmas.float()

    def _init_kv_caches_for_conditional(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        *,
        batch_size: int,
        total_video_frames: int,
        total_audio_frames: int,
        video_frame_seqlen: int,
    ):
        wrapper = self._unwrap_module(self.ar_teacher._generator)
        text_ctx = conditional_dict.get("video_context")
        if text_ctx is None:
            text_ctx = conditional_dict.get("context")
        if text_ctx is None:
            raise ValueError("conditional_dict must provide video_context or context")
        return wrapper.init_av_kv_caches(
            batch_size=batch_size,
            num_video_frames=total_video_frames,
            num_audio_frames=total_audio_frames,
            text_seq_len=int(text_ctx.shape[1]),
            device=self.device,
            dtype=self.dtype,
            video_frame_seqlen=video_frame_seqlen,
        )

    @torch.no_grad()
    def _refresh_kv_cache_block(
        self,
        *,
        current_video: torch.Tensor,
        current_audio: Optional[torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor],
        kv_caches,
        block,
        ref_video_block: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size = current_video.shape[0]
        ctx_t = float(self.benchmark_context_noise) / float(self.config.get("num_train_timestep", 1000))
        video_ctx_sigma = torch.full(
            (batch_size, current_video.shape[1]),
            ctx_t,
            device=current_video.device,
            dtype=self.dtype,
        )
        audio_ctx_sigma = None
        if current_audio is not None:
            audio_ctx_sigma = torch.full(
                (batch_size, current_audio.shape[1]),
                ctx_t,
                device=current_audio.device,
                dtype=self.dtype,
            )

        if ctx_t > 0.0:
            refresh_video = self.ar_teacher._add_noise(
                current_video.detach(),
                video_ctx_sigma.to(dtype=current_video.dtype),
            )
            refresh_audio = None
            if current_audio is not None:
                refresh_audio = self.ar_teacher._add_noise(
                    current_audio.detach(),
                    audio_ctx_sigma.to(dtype=current_audio.dtype),
                )
        else:
            refresh_video = current_video.detach()
            refresh_audio = current_audio.detach() if current_audio is not None else None

        self.ar_teacher._generator(
            _kv_cache_kwargs=dict(
                noisy_video_block=refresh_video,
                noisy_audio_block=refresh_audio,
                conditional_dict=conditional_dict,
                timestep_block=video_ctx_sigma,
                audio_timestep_block=audio_ctx_sigma,
                kv_caches=kv_caches,
                current_video_start_frame=block.video_start,
                current_audio_start_frame=block.audio_start,
                ref_video_block=ref_video_block,
            )
        )

    @torch.no_grad()
    def generate_with_cfg_kv_cache(
        self,
        *,
        pos_conditional_dict: Dict[str, torch.Tensor],
        neg_conditional_dict: Dict[str, torch.Tensor],
        video_shape: tuple,
        audio_shape: tuple,
        seed: int,
        pixel_height: int,
        pixel_width: int,
        first_block_latent: Optional[torch.Tensor] = None,
    ):
        device = self.device
        dtype = self.dtype
        batch_size = video_shape[0]
        total_video_frames = video_shape[1]
        cfg_scale = self.benchmark_guidance_scale
        enable_chunk_recycling = bool(
            getattr(self._unwrap_module(self.ar_teacher._generator), "enable_chunk_recycling", False)
        )

        sigmas = self._compute_sigma_schedule(
            self.benchmark_num_inference_steps,
            height=pixel_height,
            width=pixel_width,
        ).to(device)
        blocks = compute_av_blocks(
            total_video_latent_frames=total_video_frames,
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            first_block_video_frames=self.config.get("first_block_video_frames", 4),
        )

        total_audio_frames = audio_shape[1] if audio_shape is not None else 0
        frame_seqlen = int(video_shape[-2]) * int(video_shape[-1])
        kv_pos = self._init_kv_caches_for_conditional(
            pos_conditional_dict,
            batch_size=batch_size,
            total_video_frames=total_video_frames,
            total_audio_frames=total_audio_frames,
            video_frame_seqlen=frame_seqlen,
        )
        kv_neg = None
        if cfg_scale != 1.0:
            kv_neg = self._init_kv_caches_for_conditional(
                neg_conditional_dict,
                batch_size=batch_size,
                total_video_frames=total_video_frames,
                total_audio_frames=total_audio_frames,
                video_frame_seqlen=frame_seqlen,
            )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = torch.zeros(audio_shape, device=device, dtype=dtype) if audio_shape else None
        prev_chunk_latent: Optional[torch.Tensor] = None

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)

            for block_idx, block in enumerate(blocks):
                ref_video_block = (
                    build_block_ref_from_prev(
                        prev_chunk_latent=prev_chunk_latent,
                        current_block_size=block.video_frames,
                        template=video,
                    )
                    if enable_chunk_recycling
                    else None
                )

                if block_idx == 0 and first_block_latent is not None:
                    cur_video = first_block_latent[:, : block.video_frames].to(device=device, dtype=dtype)
                    video[:, block.video_start:block.video_end] = cur_video
                    cur_audio = None
                    if audio is not None and block.audio_frames > 0:
                        cur_audio = torch.zeros(
                            (batch_size, block.audio_frames, audio_shape[2]),
                            device=device,
                            dtype=dtype,
                        )
                        audio[:, block.audio_start:block.audio_end] = cur_audio
                    self._refresh_kv_cache_block(
                        current_video=cur_video,
                        current_audio=cur_audio,
                        conditional_dict=pos_conditional_dict,
                        kv_caches=kv_pos,
                        block=block,
                        ref_video_block=ref_video_block,
                    )
                    if cfg_scale != 1.0:
                        self._refresh_kv_cache_block(
                            current_video=cur_video,
                            current_audio=cur_audio,
                            conditional_dict=neg_conditional_dict,
                            kv_caches=kv_neg,
                            block=block,
                            ref_video_block=ref_video_block,
                        )
                    prev_chunk_latent = cur_video.detach()
                    continue

                cur_video = torch.randn(
                    (batch_size, block.video_frames, *video_shape[2:]),
                    device=device,
                    dtype=dtype,
                )
                cur_audio = None
                if audio is not None and block.audio_frames > 0:
                    cur_audio = torch.randn(
                        (batch_size, block.audio_frames, audio_shape[2]),
                        device=device,
                        dtype=dtype,
                    )

                for step_idx, sigma in enumerate(sigmas[:-1]):
                    video_sigma = sigma.expand(batch_size, cur_video.shape[1]).to(dtype)
                    audio_sigma = (
                        sigma.expand(batch_size, cur_audio.shape[1]).to(dtype)
                        if cur_audio is not None
                        else None
                    )

                    x0_pos_v, x0_pos_a = self.ar_teacher._generator(
                        _kv_cache_kwargs=dict(
                            noisy_video_block=cur_video,
                            noisy_audio_block=cur_audio,
                            conditional_dict=pos_conditional_dict,
                            timestep_block=video_sigma,
                            audio_timestep_block=audio_sigma,
                            kv_caches=kv_pos,
                            current_video_start_frame=block.video_start,
                            current_audio_start_frame=block.audio_start,
                            ref_video_block=ref_video_block,
                        )
                    )

                    if cfg_scale != 1.0:
                        x0_neg_v, x0_neg_a = self.ar_teacher._generator(
                            _kv_cache_kwargs=dict(
                                noisy_video_block=cur_video,
                                noisy_audio_block=cur_audio,
                                conditional_dict=neg_conditional_dict,
                                timestep_block=video_sigma,
                                audio_timestep_block=audio_sigma,
                                kv_caches=kv_neg,
                                current_video_start_frame=block.video_start,
                                current_audio_start_frame=block.audio_start,
                                ref_video_block=ref_video_block,
                            )
                        )
                        x0_v = x0_neg_v + cfg_scale * (x0_pos_v - x0_neg_v)
                        x0_a = x0_neg_a + cfg_scale * (x0_pos_a - x0_neg_a) if x0_neg_a is not None else x0_pos_a
                    else:
                        x0_v = x0_pos_v
                        x0_a = x0_pos_a

                    sigma_next = sigmas[step_idx + 1]
                    if sigma_next > 0:
                        if self.denoising_update == "consistency":
                            next_video_sigma = sigma_next.expand(batch_size, cur_video.shape[1]).to(dtype)
                            cur_video = self.ar_teacher._add_noise(
                                x0_v,
                                next_video_sigma.to(dtype=x0_v.dtype),
                            )
                        else:
                            dt = sigma_next - sigma
                            cur_video = (
                                cur_video.float()
                                + (cur_video.float() - x0_v.float()) / sigma.float() * dt.float()
                            ).to(dtype)
                        if cur_audio is not None and x0_a is not None:
                            if self.denoising_update == "consistency":
                                next_audio_sigma = sigma_next.expand(batch_size, cur_audio.shape[1]).to(dtype)
                                cur_audio = self.ar_teacher._add_noise(
                                    x0_a,
                                    next_audio_sigma.to(dtype=x0_a.dtype),
                                )
                            else:
                                cur_audio = (
                                    cur_audio.float()
                                    + (cur_audio.float() - x0_a.float()) / sigma.float() * dt.float()
                                ).to(dtype)
                    else:
                        cur_video = x0_v
                        if cur_audio is not None and x0_a is not None:
                            cur_audio = x0_a

                video[:, block.video_start:block.video_end] = cur_video
                if audio is not None and cur_audio is not None:
                    audio[:, block.audio_start:block.audio_end] = cur_audio

                self._refresh_kv_cache_block(
                    current_video=cur_video,
                    current_audio=cur_audio,
                    conditional_dict=pos_conditional_dict,
                    kv_caches=kv_pos,
                    block=block,
                    ref_video_block=ref_video_block,
                )
                if cfg_scale != 1.0:
                    self._refresh_kv_cache_block(
                        current_video=cur_video,
                        current_audio=cur_audio,
                        conditional_dict=neg_conditional_dict,
                        kv_caches=kv_neg,
                        block=block,
                        ref_video_block=ref_video_block,
                    )

                prev_chunk_latent = cur_video.detach()
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

        return video, audio

    @torch.no_grad()
    def encode_first_frame(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        target_h = int(self.config.get("video_height", 480))
        target_w = int(self.config.get("video_width", 832))
        target_area = target_h * target_w

        aspect = orig_w / orig_h
        new_h = int((target_area / aspect) ** 0.5)
        new_w = int(new_h * aspect)
        new_h = max(32, round(new_h / 32) * 32)
        new_w = max(32, round(new_w / 32) * 32)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_tensor = TF.to_tensor(img) * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=self.dtype)

        self._encoders_to_device()
        latent = self.video_encoder(img_tensor)
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        self._encoders_to_cpu()

        first_block_frames = int(self.config.get("first_block_video_frames", 4))
        if latent.shape[1] < first_block_frames:
            latent = latent.repeat(1, first_block_frames, 1, 1, 1)[:, :first_block_frames]
        return latent, new_h, new_w

    @torch.no_grad()
    def decode_and_save(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor],
        out_path: str,
        wav_path: Optional[str] = None,
    ) -> None:
        self._vae_decoders_to_device()
        video_pixel = self.video_vae.decode_to_pixel(video_latent)
        audio_waveform = None
        if audio_latent is not None:
            try:
                audio_waveform = self.audio_vae.decode_to_waveform(audio_latent)
            except Exception as exc:
                print(f"[decode] audio failed for {out_path}: {exc}", flush=True)

        vid = video_pixel[0]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)
        vid = vid.permute(0, 2, 3, 1)
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        wav_for_mp4 = None
        if audio_waveform is not None:
            wav_for_mp4 = audio_waveform[0].detach().cpu().float()
            if wav_path:
                wav_tmp_path = str(Path(wav_path).with_suffix(".part.wav"))
                try:
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                    wav_to_save = wav_for_mp4
                    if wav_to_save.dim() == 1:
                        wav_to_save = wav_to_save.unsqueeze(0)
                    elif wav_to_save.shape[0] > wav_to_save.shape[-1]:
                        wav_to_save = wav_to_save.transpose(0, 1)
                    torchaudio.save(
                        wav_tmp_path, wav_to_save, self.benchmark_audio_sample_rate
                    )
                    os.replace(wav_tmp_path, wav_path)
                except Exception as exc:
                    if os.path.exists(wav_tmp_path):
                        os.remove(wav_tmp_path)
                    raise RuntimeError(f"failed to save WAV {wav_path}") from exc

        mp4_tmp_path = str(Path(out_path).with_suffix(".part.mp4"))
        try:
            write_video_pyav(
                mp4_tmp_path,
                vid,
                fps=self.benchmark_video_fps,
                audio_array=wav_for_mp4,
                audio_fps=(
                    self.benchmark_audio_sample_rate if wav_for_mp4 is not None else None
                ),
            )
            os.replace(mp4_tmp_path, out_path)
        except Exception:
            if os.path.exists(mp4_tmp_path):
                os.remove(mp4_tmp_path)
            raise

        self._vae_decoders_to_cpu()
        del video_pixel, audio_waveform
        torch.cuda.empty_cache()


def _load_t2av_prompts(
    config: Any,
    limit: int,
    prompt_file_override: Optional[str] = None,
    single_prompt: Optional[str] = None,
) -> list[str]:
    if single_prompt:
        return [single_prompt.strip()]
    if prompt_file_override:
        prompt_files = [prompt_file_override]
    else:
        prompt_files = config.get("benchmark_prompt_files", None)
        if prompt_files is None:
            prompt_file = config.get("benchmark_prompt_file")
            prompt_files = [prompt_file] if prompt_file else []
        elif isinstance(prompt_files, str):
            prompt_files = [prompt_files]

    if not prompt_files:
        return []

    prompts = []
    for prompt_file in prompt_files:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())

    if prompt_file_override is None:
        prompts = prompts[: int(config.get("benchmark_num_prompts", len(prompts)))]
    return prompts[:limit] if limit > 0 else prompts


def _load_ti2av_cases(config: Any, limit: int) -> list[dict[str, str]]:
    cases = []
    for case_dir in config.get("benchmark_ti2av_case_dirs", []):
        prompt_path = os.path.join(case_dir, "prompt.txt")
        image_path = os.path.join(case_dir, "first_frame.png")
        if not (os.path.isfile(prompt_path) and os.path.isfile(image_path)):
            print(f"[ti2av] skip missing case: {case_dir}", flush=True)
            continue
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        cases.append({"case_dir": case_dir, "prompt": prompt, "image_path": image_path})
    return cases[:limit] if limit > 0 else cases


def _save_meta(path: str, **items: Any) -> None:
    meta_path = os.path.splitext(path)[0] + ".txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for key, value in items.items():
            f.write(f"{key}: {value}\n")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Salt++ 480p causal audio-video inference")
    parser.add_argument("--config", required=True, help="Training or inference config yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint model.pt or generator_only.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task", choices=["t2av", "ti2av", "both"], default="t2av")
    parser.add_argument("--prompt", default=None, help="Single text prompt for quick generation")
    parser.add_argument("--prompt-file", default=None, help="Override config benchmark_prompt_file")
    parser.add_argument("--limit", type=int, default=0, help="Cap cases per task (0 = all)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for sharded prompt generation")
    parser.add_argument("--use-ema", action="store_true", help="Load ema_generator if present")
    parser.add_argument("--zero-pad-samples", action="store_true", help="Write sample_0000.mp4 naming")
    parser.add_argument("--save-wav", action="store_true", help="Also save a sibling sample_0000.wav file")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose mp4 already exists")
    parser.add_argument(
        "--use-config-denoising-step-list",
        action="store_true",
        help="Use config denoising_step_list as strict t/1000 inference sigmas",
    )
    parser.add_argument(
        "--denoising-step-list",
        default=None,
        help="Explicit comma-separated strict timesteps (e.g. 1000,960,889,727,0)",
    )
    parser.add_argument(
        "--sigma-schedule",
        choices=["ltx", "uniform_shifted"],
        default="ltx",
    )
    parser.add_argument(
        "--denoising-update",
        choices=["euler", "consistency"],
        default="consistency",
        help="Per-step latent update ('consistency' for few-step Salt++, 'euler' for 40-step teacher)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override benchmark_guidance_scale; use 1.0 to disable CFG",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override benchmark_num_inference_steps for schedule-based inference",
    )
    parser.add_argument("--lora-dir", default=None, help="Optional PEFT LoRA adapter directory")
    parser.add_argument(
        "--direct-generator-checkpoint-load",
        action="store_true",
        default=os.environ.get("CAUSAL_AV_DIRECT_GENERATOR_CHECKPOINT_LOAD", "0") == "1",
        help="Construct causal generator directly without loading base LTX-2 generator weights first",
    )
    args = parser.parse_args()

    config = _load_config_with_base(args.config)
    if args.num_inference_steps is not None:
        if args.num_inference_steps <= 0:
            raise ValueError("--num-inference-steps must be positive")
        config.benchmark_num_inference_steps = int(args.num_inference_steps)
    if args.denoising_step_list:
        step_list = [int(item.strip()) for item in args.denoising_step_list.split(",") if item.strip()]
        if len(step_list) < 2 or step_list[0] != 1000 or step_list[-1] != 0:
            raise ValueError(
                "--denoising-step-list must contain at least two entries and run from 1000 to 0"
            )
        if any(left <= right for left, right in zip(step_list, step_list[1:])):
            raise ValueError("--denoising-step-list must be strictly descending")
        config.benchmark_denoising_step_list = step_list
        config.benchmark_use_denoising_step_list = True
    elif args.use_config_denoising_step_list:
        config.benchmark_use_denoising_step_list = True

    out_root = args.output_dir
    if out_root is None:
        ckpt_name = Path(args.checkpoint).parent.name
        out_root = os.path.join(str(config.get("output_path", "./outputs")), "offline_benchmark", ckpt_name)
    os.makedirs(out_root, exist_ok=True)

    runner = CausalAVInferenceRunner(
        config,
        args.checkpoint,
        use_ema=args.use_ema,
        sigma_schedule=args.sigma_schedule,
        denoising_update=args.denoising_update,
        guidance_scale=args.guidance_scale,
        lora_dir=args.lora_dir,
        direct_generator_checkpoint_load=args.direct_generator_checkpoint_load,
    )
    print(
        f"[params] steps={runner.benchmark_num_inference_steps} cfg={runner.benchmark_guidance_scale} "
        f"sigma_schedule={runner.sigma_schedule} denoising_update={runner.denoising_update} "
        f"seed={runner.benchmark_seed} output={out_root}",
        flush=True,
    )
    if runner.use_config_denoising_step_list:
        print(
            f"[params] strict_denoising_step_list="
            f"{list(config.get('benchmark_denoising_step_list', config.get('denoising_step_list')))}",
            flush=True,
        )

    with torch.no_grad():
        neg_cond = runner.ar_teacher.text_encoder(text_prompts=[""])

    if args.task in ("t2av", "both"):
        prompts = _load_t2av_prompts(
            config,
            0,
            prompt_file_override=args.prompt_file,
            single_prompt=args.prompt,
        )
        if args.start_index:
            prompts = prompts[args.start_index :]
        if args.limit > 0:
            prompts = prompts[: args.limit]
        t2av_dir = os.path.join(out_root, "t2av")
        os.makedirs(t2av_dir, exist_ok=True)
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=int(config.get("num_frames", 121)),
            video_height=int(config.get("video_height", 480)),
            video_width=int(config.get("video_width", 832)),
            batch_size=1,
        )
        for local_idx, prompt in enumerate(prompts):
            idx = args.start_index + local_idx
            seed = runner.benchmark_seed + idx
            sample_name = f"sample_{idx:04d}" if args.zero_pad_samples else f"sample_{idx}"
            out_path = os.path.join(t2av_dir, f"{sample_name}.mp4")
            wav_path = os.path.join(t2av_dir, f"{sample_name}.wav") if args.save_wav else None
            output_complete = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            if args.save_wav:
                output_complete = output_complete and bool(
                    wav_path and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
                )
            if args.skip_existing and output_complete:
                print(f"[t2av] skip existing global={idx} -> {out_path}", flush=True)
                continue
            print(f"[t2av] {local_idx + 1}/{len(prompts)} global={idx} seed={seed}", flush=True)
            pos_cond = runner.ar_teacher.text_encoder(text_prompts=[prompt])
            video_latent, audio_latent = runner.generate_with_cfg_kv_cache(
                pos_conditional_dict=pos_cond,
                neg_conditional_dict=neg_cond,
                video_shape=tuple(video_shape),
                audio_shape=tuple(audio_shape),
                seed=seed,
                pixel_height=int(config.get("video_height", 480)),
                pixel_width=int(config.get("video_width", 832)),
            )
            runner.decode_and_save(video_latent, audio_latent, out_path, wav_path=wav_path)
            _save_meta(
                out_path,
                task="t2av",
                index=idx,
                seed=seed,
                sigma_schedule=runner.sigma_schedule,
                prompt=prompt,
            )
            del pos_cond, video_latent, audio_latent
            torch.cuda.empty_cache()

    if args.task in ("ti2av", "both"):
        cases = _load_ti2av_cases(config, args.limit)
        ti2av_dir = os.path.join(out_root, "ti2av")
        os.makedirs(ti2av_dir, exist_ok=True)
        for idx, case in enumerate(cases):
            seed = runner.benchmark_seed + 10000 + idx
            out_path = os.path.join(ti2av_dir, f"sample_{idx}.mp4")
            print(f"[ti2av] {idx + 1}/{len(cases)} seed={seed} case={case['case_dir']}", flush=True)
            pos_cond = runner.ar_teacher.text_encoder(text_prompts=[case["prompt"]])
            first_block_latent, pixel_h, pixel_w = runner.encode_first_frame(case["image_path"])
            video_shape, audio_shape = compute_latent_shapes(
                num_frames=int(config.get("num_frames", 121)),
                video_height=pixel_h,
                video_width=pixel_w,
                batch_size=1,
            )
            video_latent, audio_latent = runner.generate_with_cfg_kv_cache(
                pos_conditional_dict=pos_cond,
                neg_conditional_dict=neg_cond,
                video_shape=tuple(video_shape),
                audio_shape=tuple(audio_shape),
                seed=seed,
                pixel_height=pixel_h,
                pixel_width=pixel_w,
                first_block_latent=first_block_latent,
            )
            runner.decode_and_save(video_latent, audio_latent, out_path)
            _save_meta(
                out_path,
                task="ti2av",
                index=idx,
                seed=seed,
                sigma_schedule=runner.sigma_schedule,
                height=pixel_h,
                width=pixel_w,
                image_path=case["image_path"],
                prompt=case["prompt"],
            )
            del pos_cond, first_block_latent, video_latent, audio_latent
            torch.cuda.empty_cache()

    print(f"[done] saved to {out_root}", flush=True)


if __name__ == "__main__":
    main()
