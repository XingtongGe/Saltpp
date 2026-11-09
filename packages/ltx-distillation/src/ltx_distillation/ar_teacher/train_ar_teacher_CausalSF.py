"""
AR Teacher Training Script with Teacher Forcing (Method A).

Trains a causal AR diffusion model on real video+audio data
using 2x-sequence teacher forcing. Initializes from a bidirectional
LTX-2 checkpoint. The trained AR teacher generates ODE pairs for Stage 2.

Usage:
    torchrun --nproc_per_node=8 -m ltx_distillation.ar_teacher.train_ar_teacher_selfflow \
        --config_path configs/stage0_ar_teacher.yaml
"""

import os
import sys
import time
import gc
import copy
import functools
import argparse
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from omegaconf import OmegaConf, DictConfig
import wandb

from ltx_distillation.ar_teacher.ar_teacher import LTX2ARTeacher, ARTeacherConfig
from ltx_distillation.util import (
    fsdp_state_dict,
    init_logging_folder,
    cycle,
    barrier,
)


# ============================================================================
# Causal Self-Flow projection head
# ============================================================================


class SimpleHead(torch.nn.Module):
    """Lightweight projection head for Causal Self-Flow feature alignment.

    Self-Flow style: Linear -> SiLU -> Linear, same in/out dim. The output
    layer is zero-initialized so that, at the start of training, the projected
    student feature is 0 and no gradient flows into the backbone through the
    head (the head warms up its own weights first).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        mid_dim = hidden_dim * 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mid_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim, hidden_dim),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def launch_distributed_job():
    timeout_sec = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SEC", "3600"))
    pg_timeout = timedelta(seconds=timeout_sec)
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=pg_timeout)
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    elif torch.cuda.is_available():
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:29500',
            world_size=1,
            rank=0,
            timeout=pg_timeout,
        )
        torch.cuda.set_device(0)
    else:
        raise RuntimeError("CUDA is required for training")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_video_compat(
    filename: str,
    video_array: torch.Tensor,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
    audio_codec: Optional[str] = None,
):
    """Write an mp4 across torchvision versions.

    torchvision 0.27 no longer exposes torchvision.io.write_video. Use it when
    present, otherwise fall back to PyAV with optional AAC audio muxing.
    """
    try:
        from torchvision.io import write_video

        kwargs = {}
        if audio_array is not None:
            kwargs.update(
                audio_array=audio_array,
                audio_fps=audio_fps,
                audio_codec=audio_codec,
            )
        return write_video(filename, video_array, fps=fps, **kwargs)
    except ImportError:
        pass

    import av
    import numpy as np

    frames = video_array.detach().cpu()
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
    frames_np = np.ascontiguousarray(frames.numpy())

    if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
        raise ValueError(f"expected video tensor [T, H, W, 3], got {tuple(frames_np.shape)}")

    height, width = frames_np.shape[1], frames_np.shape[2]
    container = av.open(filename, mode="w")
    try:
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        audio_stream = None
        audio_np = None
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
                    f"expected audio tensor [T], [C, T], or [T, C], got {tuple(audio.shape)}"
                )
            if audio.shape[0] > 2:
                audio = audio[:2]
            audio_np = np.ascontiguousarray(audio.clamp(-1, 1).numpy().astype(np.float32))
            layout = "mono" if audio_np.shape[0] == 1 else "stereo"
            audio_stream = container.add_stream(audio_codec or "aac", rate=int(audio_fps))
            audio_stream.layout = layout

        for frame_np in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)

        if audio_stream is not None and audio_np is not None:
            chunk_samples = 1024
            for start in range(0, audio_np.shape[1], chunk_samples):
                audio_chunk = audio_np[:, start : start + chunk_samples]
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_chunk, format="fltp", layout=audio_stream.layout.name,
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


def get_sharding_strategy(name: str) -> ShardingStrategy:
    strategies = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategies.get(name, ShardingStrategy.FULL_SHARD)


def fsdp_wrap(
    module: torch.nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
    wrap_strategy: str = "size",
    min_num_params: int = 1e8,
    transformer_module: Optional[tuple] = None,
) -> FSDP:
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )

    auto_wrap_policy = None
    if wrap_strategy == "transformer" and transformer_module is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_module),
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(min_num_params),
        )

    return FSDP(
        module,
        sharding_strategy=get_sharding_strategy(sharding_strategy),
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )


# ============================================================================
# Trainer
# ============================================================================

class ARTeacherTrainer:
    """
    Trainer for AR Teacher with Teacher Forcing (Method A).

    Combines:
    - Real data pipeline with online VAE encoding
    - FSDP distributed training
    - Flow matching loss via teacher forcing
    """

    def __init__(self, config: DictConfig):
        self.config = config

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0

        # Seed
        seed = config.get("seed", 42)
        if seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            seed = random_seed.item()
        set_seed(seed + self.global_rank)

        # Logging
        self.output_path = config.output_path
        if self.is_main_process:
            self.output_path, _ = init_logging_folder(config)
        else:
            os.makedirs(self.output_path, exist_ok=True)

        # Broadcast output_path to all ranks
        if dist.is_initialized() and self.world_size > 1:
            path_tensor = None
            if self.is_main_process:
                path_bytes = self.output_path.encode('utf-8')
                path_tensor = torch.tensor(
                    list(path_bytes) + [0] * (512 - len(path_bytes)),
                    dtype=torch.uint8, device=self.device,
                )
            else:
                path_tensor = torch.zeros(512, dtype=torch.uint8, device=self.device)
            dist.broadcast(path_tensor, src=0)
            if not self.is_main_process:
                self.output_path = bytes(
                    path_tensor.cpu().tolist()
                ).rstrip(b'\x00').decode('utf-8')
                os.makedirs(self.output_path, exist_ok=True)

        barrier()

        # Self-Flow options (isolated to this trainer).
        self.enable_self_flow = bool(config.get("enable_self_flow", True))
        self.self_flow_mask_ratio = float(config.get("self_flow_mask_ratio", 0.25))
        self.self_flow_repr_weight = float(config.get("self_flow_repr_weight", 0.1))
        self.self_flow_dual_timestep_mode = str(
            config.get("self_flow_dual_timestep_mode", "min_cleaner")
        )
        self.use_ema_teacher = bool(config.get("use_ema_teacher", False))
        self.ema_decay = float(config.get("ema_decay", 0.999))

        if self.is_main_process and self.enable_self_flow:
            teacher_tag = f"ema(decay={self.ema_decay})" if self.use_ema_teacher else "online"
            print(
                "[SelfFlow] enabled | "
                f"mask_ratio={self.self_flow_mask_ratio:.3f} "
                f"repr_weight={self.self_flow_repr_weight:.4f} "
                f"dual_mode={self.self_flow_dual_timestep_mode} "
                f"teacher={teacher_tag}"
            )

        # Causal Self-Flow (CSF) options. The difference vs. the legacy
        # `self_flow` branch is in the CONTEXT: the student sees per-chunk
        # randomly-noised context (sigma ~ U[0, csf_ctx_sigma_max]) while the
        # (EMA) teacher keeps the clean teacher-forcing context; both share the
        # exact same noisy target frames + timestep. They are aligned on DiT
        # intermediate features (student shallow -> MLP -> teacher deep, 1-cos).
        self.enable_causal_self_flow = bool(config.get("enable_causal_self_flow", False))
        self.csf_ctx_sigma_min = float(config.get("csf_ctx_sigma_min", 0.0))
        self.csf_ctx_sigma_max = float(config.get("csf_ctx_sigma_max", 0.5))
        self.csf_repr_weight = float(config.get("csf_repr_weight", 0.005))
        self.csf_clean_tf_ratio = float(config.get("csf_clean_tf_ratio", 0.5))
        self.csf_warmup_steps = int(config.get("csf_warmup_steps", 2000))
        self.csf_align_audio = bool(config.get("csf_align_audio", True))
        # Feature-extraction block indices (default: 0.3 / 0.7 of depth).
        _num_layers = int(config.get("num_layers", 48))
        self.csf_shallow_layer = int(config.get("csf_shallow_layer", round(0.3 * _num_layers)))
        self.csf_deep_layer = int(config.get("csf_deep_layer", round(0.7 * _num_layers)))

        # Build AR Teacher module
        ar_config = ARTeacherConfig(
            checkpoint_path=config.get("checkpoint_path", ""),
            bidirectional_model_checkpoint=config.get("bidirectional_model_checkpoint"),
            causal_model_checkpoint=config.get("causal_model_checkpoint"),
            text_encoder_checkpoint=config.get("text_encoder_checkpoint", config.get("gemma_path", "")),
            num_frame_per_block=config.get("num_frame_per_block", 3),
            first_block_video_frames=config.get("first_block_video_frames", 1),
            gradient_checkpointing=config.gradient_checkpointing,
            mixed_precision=config.mixed_precision,
            num_train_timestep=config.get("num_train_timestep", 1000),
            sigma_shift=config.get("sigma_shift", 1.0),
            sigma_min=config.get("sigma_min", 0.0),
            sigma_max=config.get("sigma_max", 1.0),
            timestep_sampler_type=config.get("timestep_sampler_type", "uniform_shifted"),
            timestep_sampler_std=config.get("timestep_sampler_std", 1.0),
            timestep_sampler_uniform_prob=config.get("timestep_sampler_uniform_prob", 0.1),
            timestep_sampler_eps=config.get("timestep_sampler_eps", 1e-3),
            use_gaussian_weight=config.get("use_gaussian_weight", False),
            denoising_loss_type=str(config.get("denoising_loss_type", "x0")),
            velocity_loss_scale=float(config.get("velocity_loss_scale", 1.0)),
            noise_augmentation_max_sigma=config.get("noise_augmentation_max_sigma", 0.0),
            video_loss_weight=config.get("video_loss_weight", 1.0),
            audio_loss_weight=config.get("audio_loss_weight", 0.16),
            num_audio_sink_tokens=config.get("num_audio_sink_tokens", 0),
            enable_causal_log_rescale=config.get("enable_causal_log_rescale", False),
            enable_chunk_recycling=config.get("enable_chunk_recycling", False),
            first_frame_conditioning_p=config.get("first_frame_conditioning_p", 0.0),
        )
        self.ar_teacher = LTX2ARTeacher(ar_config, device=self.device)
        self.ar_teacher._load_models()
        self._configure_causal_mask_cache()
        if self.is_main_process:
            print(
                "[DenoisingLoss] "
                f"type={self.ar_teacher.config.denoising_loss_type} "
                f"velocity_scale={self.ar_teacher.config.velocity_loss_scale:g} "
                f"gaussian_weight={self.ar_teacher.config.use_gaussian_weight}"
            )

        # Capture model feature dims (before FSDP wrap) for the CSF projectors.
        _model_cfg = getattr(self.ar_teacher._generator, "model", None)
        _model_cfg = getattr(_model_cfg, "config", None)
        self._csf_video_dim = int(getattr(_model_cfg, "video_dim", config.get("video_dim", 4096)))
        self._csf_audio_dim = int(getattr(_model_cfg, "audio_dim", config.get("audio_dim", 2048)))

        # Build VAE encoders for online encoding
        self._build_vae_encoders()

        # EMA teacher: deepcopy generator BEFORE FSDP wrapping
        if self.use_ema_teacher and (self.enable_self_flow or self.enable_causal_self_flow):
            if self.is_main_process:
                print("[ARTeacher] Creating EMA generator copy...")
            self.ema_generator = copy.deepcopy(self.ar_teacher._generator)
            self.ema_generator.requires_grad_(False)
            if self.is_main_process:
                print("[ARTeacher] EMA generator copy created.")
        else:
            self.ema_generator = None

        # FSDP wrap generator
        if self.is_main_process:
            print("[ARTeacher] FSDP wrapping generator...")
        gen_wrap_strategy = config.get("generator_fsdp_wrap_strategy", "size")
        gen_transformer_module = None
        if gen_wrap_strategy == "transformer":
            from ltx_causal.transformer.causal_block import CausalAVTransformerBlock
            gen_transformer_module = (CausalAVTransformerBlock,)

        fsdp_kwargs = dict(
            sharding_strategy=config.get("sharding_strategy", "hybrid_full"),
            mixed_precision=config.mixed_precision,
            wrap_strategy=gen_wrap_strategy,
            transformer_module=gen_transformer_module,
        )

        self.ar_teacher._generator = fsdp_wrap(
            self.ar_teacher._generator, **fsdp_kwargs,
        )
        if self.is_main_process:
            print("[ARTeacher] Generator FSDP wrapped.")

        # FSDP wrap EMA generator with identical sharding
        if self.ema_generator is not None:
            if self.is_main_process:
                print("[ARTeacher] FSDP wrapping EMA generator...")
            self.ema_generator = fsdp_wrap(
                self.ema_generator, **fsdp_kwargs,
            )
            if self.is_main_process:
                print("[ARTeacher] EMA generator FSDP wrapped.")

        if self.is_main_process:
            print("[ARTeacher] FSDP wrapping text encoder...")
        self.ar_teacher._text_encoder = fsdp_wrap(
            self.ar_teacher._text_encoder,
            sharding_strategy=config.get("sharding_strategy", "hybrid_full"),
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.get("text_encoder_fsdp_wrap_strategy", "size"),
        )
        if self.is_main_process:
            print("[ARTeacher] Text encoder FSDP wrapped.")

        # Causal Self-Flow projection heads (student shallow feature -> teacher
        # deep feature). FSDP-wrapped with the same sharding so their gradients
        # are reduced/synchronized across ranks like the generator's.
        self.csf_projector_video = None
        self.csf_projector_audio = None
        if self.enable_causal_self_flow:
            self.csf_projector_video = SimpleHead(self._csf_video_dim).to(
                self.device, self.dtype
            )
            self.csf_projector_video = fsdp_wrap(self.csf_projector_video, **fsdp_kwargs)
            if self.csf_align_audio:
                self.csf_projector_audio = SimpleHead(self._csf_audio_dim).to(
                    self.device, self.dtype
                )
                self.csf_projector_audio = fsdp_wrap(self.csf_projector_audio, **fsdp_kwargs)
            if self.is_main_process:
                print(
                    "[CausalSelfFlow] enabled | "
                    f"ctx_sigma=[{self.csf_ctx_sigma_min:.2f},{self.csf_ctx_sigma_max:.2f}] "
                    f"repr_weight={self.csf_repr_weight:.4f} "
                    f"clean_tf_ratio={self.csf_clean_tf_ratio:.2f} "
                    f"warmup={self.csf_warmup_steps} "
                    f"layers=({self.csf_shallow_layer}->{self.csf_deep_layer}) "
                    f"align_audio={self.csf_align_audio} "
                    f"teacher={'ema' if self.ema_generator is not None else 'online'}"
                )

        # Optimizer (generator + CSF projector params)
        opt_params = [p for p in self.ar_teacher._generator.parameters() if p.requires_grad]
        if self.csf_projector_video is not None:
            opt_params += [p for p in self.csf_projector_video.parameters() if p.requires_grad]
        if self.csf_projector_audio is not None:
            opt_params += [p for p in self.csf_projector_audio.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            opt_params,
            lr=config.lr,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Warmup scheduler
        self.warmup_steps = config.get("warmup_steps", 0)
        self.lr_schedule_type = config.get("lr_schedule_type", "constant")
        self.lr_milestones = list(config.get("lr_milestones", []))
        self.lr_values = list(config.get("lr_values", [config.lr]))
        if self.lr_schedule_type == "piecewise_constant":
            if len(self.lr_values) != len(self.lr_milestones) + 1:
                raise ValueError(
                    "piecewise_constant LR schedule requires len(lr_values) "
                    "to equal len(lr_milestones) + 1"
                )
            self.lr_milestones = [int(step) for step in self.lr_milestones]
            self.lr_values = [float(lr) for lr in self.lr_values]

        # Dataloader
        if self.is_main_process:
            print("[ARTeacher] Initializing dataloader...")
        self._init_dataloader()
        if self.is_main_process:
            print("[ARTeacher] Dataloader initialized.")

        # Training state
        self.step = 0
        self.max_grad_norm = config.get("max_grad_norm", 10.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.previous_time = None

        if self.is_main_process:
            effective_batch = config.batch_size * self.gradient_accumulation_steps * self.world_size
            print(f"[ARTeacher] Effective batch size = "
                  f"{config.batch_size} x {self.gradient_accumulation_steps} x {self.world_size} = "
                  f"{effective_batch}")

        # Benchmark (multi-step Euler + CFG visualization)
        self._init_benchmark()

        # Resume from checkpoint if specified
        resume_path = config.get("resume_checkpoint", None)
        if resume_path:
            self._resume_from_checkpoint(resume_path)

    # ── Resume ──

    def _resume_from_checkpoint(self, resume_path: str):
        if self.is_main_process:
            print(f"[Resume] Loading checkpoint from {resume_path}")

        ckpt = torch.load(resume_path, map_location="cpu")
        self.step = ckpt.get("step", 0)

        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        load_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            self.ar_teacher._generator, StateDictType.FULL_STATE_DICT, load_cfg
        ):
            self.ar_teacher._generator.load_state_dict(ckpt["generator"])

        if self.ema_generator is not None and "ema_generator" in ckpt:
            with FSDP.state_dict_type(
                self.ema_generator, StateDictType.FULL_STATE_DICT, load_cfg
            ):
                self.ema_generator.load_state_dict(ckpt["ema_generator"])
            if self.is_main_process:
                print("[Resume] EMA generator state loaded.")
        elif self.ema_generator is not None:
            if self.is_main_process:
                print("[Resume] No ema_generator in checkpoint, EMA starts from student weights.")
            with FSDP.state_dict_type(
                self.ema_generator, StateDictType.FULL_STATE_DICT, load_cfg
            ):
                self.ema_generator.load_state_dict(ckpt["generator"])

        if self.csf_projector_video is not None and "csf_projector_video" in ckpt:
            with FSDP.state_dict_type(
                self.csf_projector_video, StateDictType.FULL_STATE_DICT, load_cfg
            ):
                self.csf_projector_video.load_state_dict(ckpt["csf_projector_video"])
            if self.is_main_process:
                print("[Resume] CSF video projector state loaded.")
        if self.csf_projector_audio is not None and "csf_projector_audio" in ckpt:
            with FSDP.state_dict_type(
                self.csf_projector_audio, StateDictType.FULL_STATE_DICT, load_cfg
            ):
                self.csf_projector_audio.load_state_dict(ckpt["csf_projector_audio"])
            if self.is_main_process:
                print("[Resume] CSF audio projector state loaded.")

        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
        if self.is_main_process:
            print(f"[Resume] Resumed at step {self.step}")

    # ── VAE encoders ──

    def _build_vae_encoders(self):
        from ltx_pipelines.utils.model_ledger import ModelLedger
        from ltx_core.model.audio_vae import (
            AudioEncoder,
            AudioEncoderConfigurator,
            AudioProcessor,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder

        config = self.config

        if self.is_main_process:
            print("[ARTeacher] Building VAE encoders...")

        registry = StateDictRegistry()
        ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=config.checkpoint_path,
            registry=registry,
        )

        self.video_encoder = ledger.video_encoder()
        self.video_encoder.eval()

        audio_enc_builder = Builder(
            model_path=config.checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.audio_encoder = audio_enc_builder.build(
            device=torch.device("cpu"), dtype=self.dtype
        ).eval()

        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins,
            mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
        )

        if self.is_main_process:
            print(f"[ARTeacher] VAE encoders built "
                  f"(audio sr={self.audio_encoder.sample_rate}, "
                  f"mel_bins={self.audio_encoder.mel_bins})")

    def _encoders_to_device(self):
        self.video_encoder = self.video_encoder.to(device=self.device, dtype=self.dtype)
        self.audio_encoder = self.audio_encoder.to(device=self.device, dtype=self.dtype)
        self.audio_processor = self.audio_processor.to(device=self.device)

    def _encoders_to_cpu(self):
        self.video_encoder = self.video_encoder.to(device="cpu")
        self.audio_encoder = self.audio_encoder.to(device="cpu")
        self.audio_processor = self.audio_processor.to(device="cpu")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_batch(self, batch: dict):
        """Encode raw video and audio to latent space."""
        self._encoders_to_device()

        video = batch["video"].to(device=self.device, dtype=self.dtype)
        audio_wav = batch["audio_waveform"].to(device=self.device, dtype=self.dtype)
        text_prompts = batch["captions"]

        video_latent = self.video_encoder(video)
        video_latent = video_latent.permute(0, 2, 1, 3, 4).contiguous()

        mel = self.audio_processor.waveform_to_mel(
            audio_wav, waveform_sample_rate=self.audio_encoder.sample_rate
        )
        audio_latent = self.audio_encoder(mel)
        B, C_z, T_a, F_mel = audio_latent.shape
        audio_latent = audio_latent.permute(0, 2, 1, 3).reshape(B, T_a, C_z * F_mel)

        self._encoders_to_cpu()

        return video_latent, audio_latent, text_prompts

    # ── Dataloader ──

    def _init_dataloader(self):
        sys.path.insert(0, str(self._dataset_package_path()))

        from video_audio_dataset import VideoAudioDataset, collate_real_data
        from frame_range_dataset import FrameRangeVideoAudioDataset

        config = self.config

        dataset_type = str(config.get("dataset_type", "frame_range"))
        if dataset_type == "frame_range":
            if int(config.batch_size) != 1:
                raise ValueError(
                    "frame_range uses variable T/spatial shapes; keep batch_size=1 "
                    "and increase gradient_accumulation_steps for larger effective batch."
                )
            default_meta = os.environ.get("DATASET_META_PREFIX", "s3://your-bucket/metadata")
            dataset = FrameRangeVideoAudioDataset(
                data_config_path=config.data_config_path,
                oss_conf_path=getattr(config, "oss_conf_path", None),
                video_height=config.video_height,
                video_width=config.video_width,
                target_fps=getattr(config, "target_fps", 24),
                audio_sample_rate=getattr(config, "audio_sample_rate", 16000),
                require_audio=getattr(config, "require_audio", True),
                require_audio_flag=getattr(config, "require_audio_flag", True),
                caption_priority=config.get("caption_priority", None),
                part_key_priority=config.get("part_key_priority", None),
                prompt_filter_remove_entity_tags=bool(
                    config.get("prompt_filter_remove_entity_tags", False)
                ),
                prompt_filter_remove_speaker_span_tags=bool(
                    config.get("prompt_filter_remove_speaker_span_tags", False)
                ),
                prompt_filter_cleanup_spacing=bool(
                    config.get("prompt_filter_cleanup_spacing", True)
                ),
                expand_tasks=getattr(config, "expand_tasks", False),
                max_entries_per_group=getattr(config, "max_entries_per_group", 0),
                epoch_size=getattr(config, "epoch_size", 0),
                min_frames=getattr(config, "min_frames", 113),
                max_frames=getattr(config, "max_frames", 193),
                timeout_sec=getattr(config, "data_timeout_sec", 120),
                preserve_aspect_ratio=getattr(config, "preserve_aspect_ratio", True),
                spatial_align=getattr(config, "spatial_align", 32),
                min_side=getattr(config, "min_side", 128),
                meta_prefix=getattr(config, "meta_prefix", default_meta),
                seed=getattr(config, "seed", 7182),
                audio_range_mode=getattr(config, "audio_range_mode", "segment"),
            )
        elif dataset_type == "video_audio":
            data_list_paths = config.data_list_paths
            if isinstance(data_list_paths, str):
                data_list_paths = [data_list_paths]

            dataset = VideoAudioDataset(
                data_list_paths=list(data_list_paths),
                oss_conf_path=getattr(config, "oss_conf_path", None),
                num_frames=config.num_frames,
                video_height=config.video_height,
                video_width=config.video_width,
                target_fps=getattr(config, "target_fps", 24),
                audio_sample_rate=getattr(config, "audio_sample_rate", 16000),
                require_audio=getattr(config, "require_audio", True),
                max_entries_per_list=getattr(config, "max_entries_per_list", 0),
                timeout_sec=getattr(config, "data_timeout_sec", 120),
                preserve_aspect_ratio=getattr(config, "preserve_aspect_ratio", False),
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_real_data,
            num_workers=getattr(config, "num_workers", 2),
            pin_memory=True,
        )

        self.dataloader = cycle(dataloader)

    @staticmethod
    def _dataset_package_path():
        # __file__ = .../packages/ltx-distillation/src/ltx_distillation/ar_teacher/train_ar_teacher.py
        # parents: [0]=ar_teacher, [1]=ltx_distillation, [2]=src, [3]=ltx-distillation, [4]=packages
        return Path(__file__).resolve().parents[4] / "dataset"

    def _configure_causal_mask_cache(self):
        mask_cache_max = self.config.get("causal_mask_cache_max", None)
        if mask_cache_max is None:
            return
        mask_cache_max = int(mask_cache_max)
        if mask_cache_max <= 0:
            raise ValueError("causal_mask_cache_max must be positive")
        generator = getattr(self.ar_teacher, "_generator", None)
        if generator is None:
            return
        setattr(generator, "_MASK_CACHE_MAX", mask_cache_max)
        if self.is_main_process:
            print(f"[ARTeacher] causal mask cache max = {mask_cache_max}")

    # ── LR warmup ──

    def _get_base_lr(self) -> float:
        if self.lr_schedule_type == "constant":
            return float(self.config.lr)
        if self.lr_schedule_type == "piecewise_constant":
            lr = self.lr_values[0]
            for milestone, value in zip(self.lr_milestones, self.lr_values[1:]):
                if self.step >= milestone:
                    lr = value
                else:
                    break
            return lr
        raise ValueError(f"Unsupported lr_schedule_type: {self.lr_schedule_type}")

    def _get_lr_scale(self) -> float:
        if self.warmup_steps <= 0 or self.step >= self.warmup_steps:
            return 1.0
        return self.step / self.warmup_steps

    def _apply_lr(self):
        scale = self._get_lr_scale()
        base_lr = self._get_base_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = base_lr * scale

    # ── Aligned audio frames ──

    def _align_audio(self, clean_video, clean_audio):
        """Trim or pad audio latent to match block-aligned frame count."""
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        F_v = clean_video.shape[1]
        expected_a = compute_aligned_audio_frames(
            F_v,
            self.config.get("num_frame_per_block", 3),
            self.config.get("first_block_video_frames", 1),
        )
        actual_a = clean_audio.shape[1]

        if actual_a > expected_a:
            clean_audio = clean_audio[:, :expected_a, :]
        elif actual_a < expected_a:
            pad = expected_a - actual_a
            clean_audio = torch.nn.functional.pad(clean_audio, (0, 0, 0, pad))

        return clean_audio

    def _compute_base_losses(
        self,
        pred_x0_video: torch.Tensor,
        pred_x0_audio: torch.Tensor,
        pred_velocity_video: Optional[torch.Tensor],
        pred_velocity_audio: Optional[torch.Tensor],
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        video_noise: Optional[torch.Tensor],
        audio_noise: Optional[torch.Tensor],
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Stage0 base denoising losses on noisy locations."""
        return self.ar_teacher.compute_denoising_losses(
            pred_x0_video=pred_x0_video,
            pred_x0_audio=pred_x0_audio,
            pred_velocity_video=pred_velocity_video,
            pred_velocity_audio=pred_velocity_audio,
            clean_video=clean_video,
            clean_audio=clean_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

    def _scale_base_denoising_loss(
        self,
        video_loss: torch.Tensor,
        audio_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine modalities and scale only native velocity supervision."""
        base_loss_unscaled = (
            self.ar_teacher.config.video_loss_weight * video_loss
            + self.ar_teacher.config.audio_loss_weight * audio_loss
        )
        scale = (
            self.ar_teacher.config.velocity_loss_scale
            if self.ar_teacher.config.denoising_loss_type == "velocity"
            else 1.0
        )
        return scale * base_loss_unscaled, base_loss_unscaled

    def _compute_repr_losses(
        self,
        student_video_x0: torch.Tensor,
        teacher_video_x0: torch.Tensor,
        student_audio_x0: torch.Tensor,
        teacher_audio_x0: torch.Tensor,
        video_mask_high: torch.Tensor,
        audio_mask_high: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-Flow representation alignment via cosine distance."""
        device = student_video_x0.device

        # [B, F, C, H, W] -> [B, F, C*H*W]
        s_video = student_video_x0.float().flatten(2)
        t_video = teacher_video_x0.float().detach().flatten(2)
        video_cos = F.cosine_similarity(s_video, t_video, dim=-1)  # [B, F]
        if video_mask_high.any():
            video_repr = (1.0 - video_cos)[video_mask_high].mean()
        else:
            video_repr = torch.zeros((), device=device, dtype=student_video_x0.dtype)

        # [B, F_a, C] -> [B, F_a, C]
        s_audio = student_audio_x0.float()
        t_audio = teacher_audio_x0.float().detach()
        audio_cos = F.cosine_similarity(s_audio, t_audio, dim=-1)  # [B, F_a]
        if audio_mask_high.any():
            audio_repr = (1.0 - audio_cos)[audio_mask_high].mean()
        else:
            audio_repr = torch.zeros((), device=device, dtype=student_audio_x0.dtype)

        return video_repr, audio_repr

    def _generator_loss_self_flow(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Dual-timestep + cleaner teacher-view alignment loss."""
        if self.self_flow_dual_timestep_mode != "min_cleaner":
            raise ValueError(
                f"Unsupported self_flow_dual_timestep_mode={self.self_flow_dual_timestep_mode}. "
                "Only 'min_cleaner' is supported."
            )

        B, F_v = clean_video.shape[:2]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        # 1) Dual-timestep sampling
        video_sigma_a = self.ar_teacher._sample_per_block_sigma(B, F_v, device)
        video_sigma_b = self.ar_teacher._sample_per_block_sigma(B, F_v, device)
        video_sigma_cleaner = torch.minimum(video_sigma_a, video_sigma_b)
        video_sigma_noisier = torch.maximum(video_sigma_a, video_sigma_b)

        video_mask_high = (torch.rand(B, F_v, device=device) < self.self_flow_mask_ratio)
        video_sigma_student = torch.where(video_mask_high, video_sigma_noisier, video_sigma_cleaner)
        video_sigma_teacher = video_sigma_cleaner

        audio_sigma_student = self.ar_teacher._expand_video_sigma_to_audio(video_sigma_student, F_a)
        audio_sigma_teacher = self.ar_teacher._expand_video_sigma_to_audio(video_sigma_teacher, F_a)
        audio_mask_high = self.ar_teacher._expand_video_sigma_to_audio(
            video_mask_high.float(), F_a
        ) > 0.5

        # 2) Build noisy student/teacher inputs
        video_noise_student = torch.randn_like(clean_video)
        audio_noise_student = torch.randn_like(clean_audio)
        noisy_video_student = self.ar_teacher._add_noise(
            clean_video, video_sigma_student, noise=video_noise_student
        )
        noisy_audio_student = self.ar_teacher._add_noise(
            clean_audio, audio_sigma_student, noise=audio_noise_student
        )
        noisy_video_teacher = self.ar_teacher._add_noise(clean_video, video_sigma_teacher)
        noisy_audio_teacher = self.ar_teacher._add_noise(clean_audio, audio_sigma_teacher)

        # Keep the clean-context branch behavior aligned with Stage0.
        clean_sigma_val = self.ar_teacher.config.noise_augmentation_max_sigma
        if clean_sigma_val > 0:
            clean_video_sigma = torch.rand(B, F_v, device=device) * clean_sigma_val
            clean_audio_sigma = self.ar_teacher._expand_video_sigma_to_audio(clean_video_sigma, F_a)
            clean_video_input = self.ar_teacher._add_noise(clean_video, clean_video_sigma)
            clean_audio_input = self.ar_teacher._add_noise(clean_audio, clean_audio_sigma)
        else:
            clean_video_sigma = torch.zeros(B, F_v, device=device)
            clean_audio_sigma = torch.zeros(B, F_a, device=device)
            clean_video_input = clean_video
            clean_audio_input = clean_audio

        # 3) Student forward (mixed-noise view)
        (
            student_x0_video,
            student_x0_audio,
            student_velocity_video,
            student_velocity_audio,
        ) = self.ar_teacher.generator(
            _tf_kwargs=dict(
                noisy_video=noisy_video_student,
                clean_video=clean_video_input,
                conditional_dict=conditional_dict,
                noisy_timestep=video_sigma_student.to(self.dtype),
                clean_timestep=clean_video_sigma.to(self.dtype),
                noisy_audio=noisy_audio_student,
                clean_audio=clean_audio_input,
                noisy_audio_timestep=audio_sigma_student.to(self.dtype),
                clean_audio_timestep=clean_audio_sigma.to(self.dtype),
                use_causal_timestep=False,
                return_velocity=True,
            )
        )

        # Capture chunk-recycling sanity stats right after the student forward.
        # Must happen before the teacher forward to avoid the EMA-less fallback
        # overwriting the cached GPU scalars.
        ref_sanity_stats = (
            self._capture_ref_sanity_stats()
            if getattr(self.ar_teacher.config, "enable_chunk_recycling", False)
            else {}
        )

        # 4) Teacher forward (cleaner-noise view, detached)
        teacher_model = self.ema_generator if self.ema_generator is not None else self.ar_teacher.generator
        with torch.no_grad():
            teacher_x0_video, teacher_x0_audio = teacher_model(
                _tf_kwargs=dict(
                    noisy_video=noisy_video_teacher,
                    clean_video=clean_video_input,
                    conditional_dict=conditional_dict,
                    noisy_timestep=video_sigma_teacher.to(self.dtype),
                    clean_timestep=clean_video_sigma.to(self.dtype),
                    noisy_audio=noisy_audio_teacher,
                    clean_audio=clean_audio_input,
                    noisy_audio_timestep=audio_sigma_teacher.to(self.dtype),
                    clean_audio_timestep=clean_audio_sigma.to(self.dtype),
                    use_causal_timestep=False,
                )
            )

        # 5) Base denoising objective on student view
        video_loss, audio_loss = self._compute_base_losses(
            pred_x0_video=student_x0_video,
            pred_x0_audio=student_x0_audio,
            pred_velocity_video=student_velocity_video,
            pred_velocity_audio=student_velocity_audio,
            clean_video=clean_video,
            clean_audio=clean_audio,
            video_noise=video_noise_student,
            audio_noise=audio_noise_student,
            video_sigma=video_sigma_student,
            audio_sigma=audio_sigma_student,
        )
        base_loss, base_loss_unscaled = self._scale_base_denoising_loss(
            video_loss, audio_loss
        )

        # 6) Representation alignment objective
        repr_video_loss, repr_audio_loss = self._compute_repr_losses(
            student_video_x0=student_x0_video,
            teacher_video_x0=teacher_x0_video,
            student_audio_x0=student_x0_audio,
            teacher_audio_x0=teacher_x0_audio,
            video_mask_high=video_mask_high,
            audio_mask_high=audio_mask_high,
        )
        repr_loss = (
            self.ar_teacher.config.video_loss_weight * repr_video_loss
            + self.ar_teacher.config.audio_loss_weight * repr_audio_loss
        )

        total_loss = base_loss + self.self_flow_repr_weight * repr_loss
        log_dict = {
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_loss_unscaled": base_loss_unscaled.detach(),
            "repr_video_loss": repr_video_loss.detach(),
            "repr_audio_loss": repr_audio_loss.detach(),
            "repr_loss": repr_loss.detach(),
            "video_sigma_mean": video_sigma_student.mean().detach(),
            "video_sigma_max": video_sigma_student.max().detach(),
            "self_flow_mask_ratio_actual": video_mask_high.float().mean().detach(),
            "pred_x0_video": student_x0_video.detach(),
            "video_sigma": video_sigma_student.detach(),
        }
        if ref_sanity_stats:
            log_dict.update(ref_sanity_stats)
        return total_loss, log_dict

    # ── Causal Self-Flow (context-noise alignment) ──

    def _sample_csf_context_sigma(
        self,
        batch_size: int,
        num_video_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-chunk uniform context noise sigma in [ctx_min, ctx_max].

        Mirrors `_sample_per_block_sigma`'s block layout (one sigma per block,
        broadcast to all frames of the block) but uses a plain uniform draw on
        the CSF context range with no timestep shift.
        """
        from ltx_causal.attention.mask_builder import compute_av_blocks

        blocks = compute_av_blocks(
            num_video_frames,
            self.ar_teacher.num_frame_per_block,
            self.ar_teacher.first_block_video_frames,
        )
        lo, hi = self.csf_ctx_sigma_min, self.csf_ctx_sigma_max
        sigma = torch.zeros(batch_size, num_video_frames, device=device)
        for block in blocks:
            block_sigma = torch.rand(batch_size, 1, device=device) * (hi - lo) + lo
            sigma[:, block.video_start:block.video_end] = block_sigma.expand(
                -1, block.video_end - block.video_start
            )
        return sigma

    def _csf_feature_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        projector: torch.nn.Module,
        num_frames: int,
        skip_frames: int,
    ) -> torch.Tensor:
        """1 - cosine similarity between projected student feature and teacher
        feature, computed per-frame and averaged over frames past block 0.

        student_feat / teacher_feat: [B, num_frames * tokens_per_frame, D].
        """
        B = student_feat.shape[0]
        D = student_feat.shape[-1]
        tokens_per_frame = student_feat.shape[1] // num_frames
        s = projector(student_feat)
        t = teacher_feat.detach()
        s = s.reshape(B, num_frames, tokens_per_frame, D)
        t = t.reshape(B, num_frames, tokens_per_frame, D)
        if skip_frames > 0:
            s = s[:, skip_frames:]
            t = t[:, skip_frames:]
        if s.shape[1] == 0:
            return torch.zeros((), device=student_feat.device, dtype=torch.float32)
        s = s.flatten(2).float()
        t = t.flatten(2).float()
        cos = F.cosine_similarity(s, t, dim=-1)  # [B, num_frames - skip]
        return (1.0 - cos).mean()

    def _generator_loss_causal_self_flow(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Causal Self-Flow: align a noisy-context student to a clean-context
        (EMA) teacher via DiT intermediate features.

        Both branches share the SAME noisy target frames + timestep; only the
        teacher-forcing CONTEXT (clean half) differs: student sees per-chunk
        random noise (sigma ~ U[ctx_min, ctx_max]); teacher keeps clean context.
        """
        from ltx_causal.attention.mask_builder import compute_av_blocks

        B, F_v = clean_video.shape[:2]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        # 1) Target noise (identical for student & teacher).
        video_sigma = self.ar_teacher._sample_per_block_sigma(B, F_v, device)
        audio_sigma = self.ar_teacher._expand_video_sigma_to_audio(video_sigma, F_a)
        video_noise = torch.randn_like(clean_video)
        audio_noise = torch.randn_like(clean_audio)
        noisy_video = self.ar_teacher._add_noise(
            clean_video, video_sigma, noise=video_noise
        )
        noisy_audio = self.ar_teacher._add_noise(
            clean_audio, audio_sigma, noise=audio_noise
        )

        # 2) Teacher context: clean (matching Stage0 TF, optional small aug).
        clean_sigma_val = self.ar_teacher.config.noise_augmentation_max_sigma
        if clean_sigma_val > 0:
            teacher_ctx_v_sigma = torch.rand(B, F_v, device=device) * clean_sigma_val
            teacher_ctx_a_sigma = self.ar_teacher._expand_video_sigma_to_audio(teacher_ctx_v_sigma, F_a)
            teacher_ctx_video = self.ar_teacher._add_noise(clean_video, teacher_ctx_v_sigma)
            teacher_ctx_audio = self.ar_teacher._add_noise(clean_audio, teacher_ctx_a_sigma)
        else:
            teacher_ctx_v_sigma = torch.zeros(B, F_v, device=device)
            teacher_ctx_a_sigma = torch.zeros(B, F_a, device=device)
            teacher_ctx_video = clean_video
            teacher_ctx_audio = clean_audio

        # 3) Student context: per-chunk random noise in [ctx_min, ctx_max].
        student_ctx_v_sigma = self._sample_csf_context_sigma(B, F_v, device)
        student_ctx_a_sigma = self.ar_teacher._expand_video_sigma_to_audio(student_ctx_v_sigma, F_a)
        student_ctx_video = self.ar_teacher._add_noise(clean_video, student_ctx_v_sigma)
        student_ctx_audio = self.ar_teacher._add_noise(clean_audio, student_ctx_a_sigma)

        # 4) Student forward: noisy target + noisy context, features with grad.
        (
            student_x0_video,
            student_x0_audio,
            student_feats,
            student_velocity_video,
            student_velocity_audio,
        ) = self.ar_teacher.generator(
            _tf_kwargs=dict(
                noisy_video=noisy_video,
                clean_video=student_ctx_video,
                conditional_dict=conditional_dict,
                noisy_timestep=video_sigma.to(self.dtype),
                clean_timestep=student_ctx_v_sigma.to(self.dtype),
                noisy_audio=noisy_audio,
                clean_audio=student_ctx_audio,
                noisy_audio_timestep=audio_sigma.to(self.dtype),
                clean_audio_timestep=student_ctx_a_sigma.to(self.dtype),
                use_causal_timestep=False,
                return_features=True,
                feature_layer_shallow=self.csf_shallow_layer,
                feature_layer_deep=self.csf_deep_layer,
                return_velocity=True,
            )
        )

        # 5) Teacher forward: same noisy target + clean context, detached.
        teacher_model = self.ema_generator if self.ema_generator is not None else self.ar_teacher.generator
        with torch.no_grad():
            _, _, teacher_feats = teacher_model(
                _tf_kwargs=dict(
                    noisy_video=noisy_video,
                    clean_video=teacher_ctx_video,
                    conditional_dict=conditional_dict,
                    noisy_timestep=video_sigma.to(self.dtype),
                    clean_timestep=teacher_ctx_v_sigma.to(self.dtype),
                    noisy_audio=noisy_audio,
                    clean_audio=teacher_ctx_audio,
                    noisy_audio_timestep=audio_sigma.to(self.dtype),
                    clean_audio_timestep=teacher_ctx_a_sigma.to(self.dtype),
                    use_causal_timestep=False,
                    return_features=True,
                    feature_layer_shallow=self.csf_shallow_layer,
                    feature_layer_deep=self.csf_deep_layer,
                )
            )

        # 6) Base denoising objective on the (noisy-context) student view.
        video_loss, audio_loss = self._compute_base_losses(
            pred_x0_video=student_x0_video,
            pred_x0_audio=student_x0_audio,
            pred_velocity_video=student_velocity_video,
            pred_velocity_audio=student_velocity_audio,
            clean_video=clean_video,
            clean_audio=clean_audio,
            video_noise=video_noise,
            audio_noise=audio_noise,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )
        base_loss, base_loss_unscaled = self._scale_base_denoising_loss(
            video_loss, audio_loss
        )

        # 7) Feature alignment objective (skip block-0 frames: no history).
        blocks = compute_av_blocks(
            F_v, self.ar_teacher.num_frame_per_block, self.ar_teacher.first_block_video_frames,
        )
        skip_v = blocks[0].video_end  # block-0 video frames
        skip_a = blocks[0].audio_end  # block-0 audio frames
        repr_video_loss = self._csf_feature_loss(
            student_feats["video_shallow"], teacher_feats["video_deep"],
            self.csf_projector_video, F_v, skip_v,
        )
        if self.csf_align_audio and self.csf_projector_audio is not None:
            repr_audio_loss = self._csf_feature_loss(
                student_feats["audio_shallow"], teacher_feats["audio_deep"],
                self.csf_projector_audio, F_a, skip_a,
            )
        else:
            repr_audio_loss = torch.zeros((), device=device, dtype=torch.float32)

        repr_loss = (
            self.ar_teacher.config.video_loss_weight * repr_video_loss
            + self.ar_teacher.config.audio_loss_weight * repr_audio_loss
        )

        total_loss = base_loss + self.csf_repr_weight * repr_loss
        log_dict = {
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_loss_unscaled": base_loss_unscaled.detach(),
            "repr_video_loss": repr_video_loss.detach(),
            "repr_audio_loss": repr_audio_loss.detach(),
            "repr_loss": repr_loss.detach(),
            "video_sigma_mean": video_sigma.mean().detach(),
            "csf_ctx_sigma_mean": student_ctx_v_sigma.mean().detach(),
            "pred_x0_video": student_x0_video.detach(),
            "video_sigma": video_sigma.detach(),
        }
        return total_loss, log_dict

    # ── Training-time visual logging ──

    def _log_training_samples(
        self,
        clean_video: torch.Tensor,
        pred_x0_video: torch.Tensor,
        video_sigma: torch.Tensor,
        sigma_mean: float,
        metadata: Optional[dict] = None,
    ):
        """Decode clean/predicted latents via VAE and save side-by-side mp4 for debugging.

        T2AV:  clean | pred_x0  (2-up)
        TI2AV: clean | pred_x0 | pred_x0_with_clean_first_frame  (3-up)
        """
        if not self.is_main_process:
            return

        if not hasattr(self, "video_vae"):
            self._build_vae_decoders()

        self._vae_decoders_to_device()

        log_dir = os.path.join(self.output_path, "train_vis")
        os.makedirs(log_dir, exist_ok=True)

        batchid = 0
        clean_latent = clean_video[batchid : batchid + 1]
        pred_latent = pred_x0_video[batchid : batchid + 1]
        sample_sigma = video_sigma[batchid]  # [F_v]

        # Detect TI2AV: Block 0 sigma ≈ 0
        from ltx_causal.attention.mask_builder import compute_av_blocks
        blocks = compute_av_blocks(
            clean_latent.shape[1],
            self.config.get("num_frame_per_block", 3),
            self.config.get("first_block_video_frames", 1),
        )
        block0 = blocks[0]
        is_ti2av = (sample_sigma[block0.video_start:block0.video_end].max() < 1e-6).item()

        latents_to_decode = [clean_latent, pred_latent]
        if is_ti2av:
            pred_replaced = pred_latent.clone()
            pred_replaced[:, block0.video_start:block0.video_end] = \
                clean_latent[:, block0.video_start:block0.video_end]
            latents_to_decode.append(pred_replaced)

        decoded = []
        with torch.no_grad():
            for latent in latents_to_decode:
                pixel = self.video_vae.decode_to_pixel(latent.to(self.device))
                vid = pixel[0]  # [C, F, H, W]
                if vid.shape[0] == 3:
                    vid = vid.permute(1, 0, 2, 3)  # [F, C, H, W]
                vid = vid.permute(0, 2, 3, 1)  # [F, H, W, C]
                vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)
                decoded.append(vid)

        try:
            fps = self.config.get("benchmark_video_fps", 24)
            min_frames = min(v.shape[0] for v in decoded)
            side_by_side = torch.cat([v[:min_frames] for v in decoded], dim=2)

            task_tag = "ti2av" if is_ti2av else "t2av"
            filename = f"step{self.step:06d}_sigma{int(sigma_mean * 1000)}_{task_tag}.mp4"
            sample_path = os.path.join(log_dir, filename)
            _write_video_compat(sample_path, side_by_side, fps=fps)

            if metadata:
                meta_path = os.path.splitext(sample_path)[0] + ".txt"
                with open(meta_path, "w", encoding="utf-8") as f:
                    for key in ("entry", "clip_url", "audio_url", "fps", "f_num", "caption"):
                        value = metadata.get(key, "")
                        f.write(f"{key}: {value}\n")

            print(f"[TrainVis] Step {self.step}: saved {filename}")
        except Exception as e:
            print(f"[TrainVis] Step {self.step}: write_video failed: {e}")

        self._vae_decoders_to_cpu()

    # ── Benchmark (multi-step Euler + CFG) ──

    def _init_benchmark(self):
        config = self.config
        self.benchmark_enabled = bool(config.get("benchmark_enabled", False))
        self.benchmark_iters = int(config.get("benchmark_iters", config.get("log_iters", 200)))
        self.benchmark_seed = int(config.get("benchmark_seed", 12345))
        self.benchmark_num_prompts = int(config.get("benchmark_num_prompts", 2))
        self.benchmark_num_inference_steps = int(config.get("benchmark_num_inference_steps", 30))
        self.benchmark_guidance_scale = float(config.get("benchmark_guidance_scale", 4.0))
        self.benchmark_video_fps = int(config.get("benchmark_video_fps", 24))
        self.benchmark_audio_sample_rate = int(config.get("benchmark_audio_sample_rate", 24000))
        self.benchmark_use_kv_cache = bool(config.get("benchmark_use_kv_cache", True))
        self.benchmark_context_noise = int(config.get("benchmark_context_noise", 0))
        self.benchmark_parallel_t2av_round_robin = bool(
            config.get("benchmark_parallel_t2av_round_robin", False)
        )
        self.benchmark_prompts = []
        self.benchmark_t2av_groups = []
        self.benchmark_ti2av_cases = []

        if self.benchmark_iters <= 0:
            self.benchmark_enabled = False

        if not self.benchmark_enabled:
            if self.is_main_process:
                print("[Benchmark] Disabled.")
            return

        # T2AV prompts
        prompt_files = config.get("benchmark_prompt_files", None)
        if prompt_files is None:
            prompt_file = config.get("benchmark_prompt_file")
            prompt_files = [prompt_file] if prompt_file else []
        elif isinstance(prompt_files, str):
            prompt_files = [prompt_files]

        if prompt_files:
            try:
                all_prompts = []
                prompts_per_file = config.get("benchmark_prompts_per_file", None)
                if prompts_per_file is not None:
                    if isinstance(prompts_per_file, int):
                        prompts_per_file = [prompts_per_file] * len(prompt_files)
                    else:
                        prompts_per_file = list(prompts_per_file)
                    if len(prompts_per_file) != len(prompt_files):
                        raise ValueError(
                            "benchmark_prompts_per_file must have the same length "
                            "as benchmark_prompt_files"
                        )

                file_counts = []
                for file_idx, prompt_file in enumerate(prompt_files):
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        file_prompts = [l.strip() for l in f if l.strip()]
                    if prompts_per_file is not None:
                        limit = int(prompts_per_file[file_idx])
                        if limit > 0:
                            file_prompts = file_prompts[:limit]
                    file_counts.append((prompt_file, len(file_prompts)))
                    all_prompts.extend(file_prompts)
                self.benchmark_prompts = all_prompts[: self.benchmark_num_prompts]
                if self.is_main_process:
                    print(
                        f"[Benchmark] Loaded {len(self.benchmark_prompts)} T2AV prompts "
                        f"from {list(prompt_files)}"
                    )
                    for prompt_file, count in file_counts:
                        print(f"  {prompt_file}: {count} prompt(s)")
                    for i, p in enumerate(self.benchmark_prompts):
                        print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")
            except Exception as e:
                if self.is_main_process:
                    print(f"[Benchmark] Failed to load T2AV prompts: {e}")

        t2av_groups = config.get("benchmark_t2av_groups", [])
        if t2av_groups:
            for group in t2av_groups:
                name = str(group.get("name", "t2av"))
                start = int(group.get("start", 0))
                end = int(group.get("end", len(self.benchmark_prompts)))
                start = max(0, min(start, len(self.benchmark_prompts)))
                end = max(start, min(end, len(self.benchmark_prompts)))
                if end > start:
                    self.benchmark_t2av_groups.append({
                        "name": name,
                        "start": start,
                        "end": end,
                    })
        elif self.benchmark_prompts:
            self.benchmark_t2av_groups = [{
                "name": "t2av",
                "start": 0,
                "end": len(self.benchmark_prompts),
            }]

        if self.is_main_process and self.benchmark_t2av_groups:
            print("[Benchmark] T2AV groups:")
            for group in self.benchmark_t2av_groups:
                print(f"  {group['name']}: [{group['start']}, {group['end']})")
            if self.benchmark_parallel_t2av_round_robin:
                print(
                    "[Benchmark] T2AV round-robin parallelism enabled with dummy tail blocks. "
                    "Real chunks keep their prompt-specific frame count; extra dummy chunks "
                    "pad each rank to the max block count and are not saved.",
                    flush=True,
                )

        # TI2AV cases: directories containing prompt.txt + first_frame.png
        ti2av_dirs = config.get("benchmark_ti2av_case_dirs", [])
        for case_dir in ti2av_dirs:
            prompt_path = os.path.join(case_dir, "prompt.txt")
            image_path = os.path.join(case_dir, "first_frame.png")
            if os.path.isfile(prompt_path) and os.path.isfile(image_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                self.benchmark_ti2av_cases.append({
                    "prompt": prompt,
                    "image_path": image_path,
                })

        if self.is_main_process and self.benchmark_ti2av_cases:
            print(f"[Benchmark] Loaded {len(self.benchmark_ti2av_cases)} TI2AV cases")
            for i, c in enumerate(self.benchmark_ti2av_cases):
                print(f"  [{i}] {c['prompt'][:60]}... | {c['image_path']}")

        if not self.benchmark_prompts and not self.benchmark_ti2av_cases:
            if self.is_main_process:
                print("[Benchmark] No prompts or TI2AV cases loaded, benchmark disabled.")
            self.benchmark_enabled = False
            return

        self._build_vae_decoders()

    def _benchmark_t2av_group_name(self, prompt_idx: int) -> str:
        for group in self.benchmark_t2av_groups:
            if group["start"] <= prompt_idx < group["end"]:
                return group["name"]
        return "t2av"

    @staticmethod
    def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    def _capture_ref_sanity_stats(self) -> Dict[str, torch.Tensor]:
        """Read the GPU-scalar caches populated by CausalLTXModel._patchify_video
        for chunk-recycling sanity monitoring. Must be called immediately after
        the student forward (before the teacher forward potentially overwrites
        the cache when ema_generator is None).
        """
        stats: Dict[str, torch.Tensor] = {}
        wrap = self._unwrap_module(self.ar_teacher._generator)
        inner = getattr(wrap, "model", wrap)
        inner = self._unwrap_module(inner)

        ref_n = getattr(inner, "_last_ref_patch_norm", None)
        noisy_n = getattr(inner, "_last_noisy_patch_norm", None)
        weight_n = getattr(inner, "_last_ref_weight_norm", None)

        if ref_n is not None and noisy_n is not None:
            stats["ref_patch_norm"] = ref_n.detach()
            stats["noisy_patch_norm"] = noisy_n.detach()
            stats["ref_over_noisy_ratio"] = (
                ref_n / noisy_n.clamp_min(1e-8)
            ).detach()
        if weight_n is not None:
            stats["ref_weight_norm"] = weight_n.detach()
        return stats

    def _init_kv_caches_for_conditional(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        batch_size: int,
        total_video_frames: int,
        total_audio_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        video_frame_seqlen: Optional[int] = None,
    ):
        wrapper = self._unwrap_module(self.ar_teacher._generator)
        init_fn = getattr(wrapper, "init_av_kv_caches", None)
        if init_fn is None:
            raise RuntimeError(
                "Generator does not expose init_av_kv_caches for KV-cache benchmark."
            )

        text_ctx = conditional_dict.get("video_context")
        if text_ctx is None:
            text_ctx = conditional_dict.get("context")
        if text_ctx is None:
            raise ValueError("conditional_dict must provide video_context or context.")
        text_seq_len = int(text_ctx.shape[1])

        return init_fn(
            batch_size=batch_size,
            num_video_frames=total_video_frames,
            num_audio_frames=total_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
            video_frame_seqlen=video_frame_seqlen,
        )

    @torch.no_grad()
    def _refresh_kv_cache_block(
        self,
        current_video: torch.Tensor,
        current_audio: Optional[torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor],
        kv_caches,
        block,
        ref_video_block: Optional[torch.Tensor] = None,
    ) -> None:
        """Align train/inference semantics by refreshing cache with near-clean block."""
        B = current_video.shape[0]
        sigma_dtype = self.dtype
        ctx_t = float(self.benchmark_context_noise) / float(self.config.get("num_train_timestep", 1000))
        video_ctx_sigma = torch.full(
            (B, current_video.shape[1]),
            ctx_t,
            device=current_video.device,
            dtype=sigma_dtype,
        )
        audio_ctx_sigma = None
        if current_audio is not None:
            audio_ctx_sigma = torch.full(
                (B, current_audio.shape[1]),
                ctx_t,
                device=current_audio.device,
                dtype=sigma_dtype,
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

    def _build_vae_decoders(self):
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        config = self.config
        if self.is_main_process:
            print("[Benchmark] Building VAE decoders...")
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=config.checkpoint_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)
        if self.is_main_process:
            print("[Benchmark] VAE decoders built (on CPU).")

    def _vae_decoders_to_device(self):
        if hasattr(self, "video_vae"):
            self.video_vae = self.video_vae.to(device=self.device)
        if hasattr(self, "audio_vae"):
            self.audio_vae = self.audio_vae.to(device=self.device)

    def _vae_decoders_to_cpu(self):
        if hasattr(self, "video_vae"):
            self.video_vae = self.video_vae.to(device="cpu")
        if hasattr(self, "audio_vae"):
            self.audio_vae = self.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    def _benchmark_num_frames_for_prompt(self, prompt: str) -> int:
        default_frames = int(self.config.get("benchmark_num_frames", self.config.get("num_frames", 121)))
        if not bool(self.config.get("benchmark_duration_from_prompt_timestamps", True)):
            return default_frames

        timestamp_values = [
            float(value)
            for pair in re.findall(r"\[(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s\]", prompt)
            for value in pair
        ]
        if not timestamp_values:
            return default_frames

        fps = float(self.config.get("benchmark_video_fps", 24))
        raw_frames = max(1, int(math.ceil(max(timestamp_values) * fps)))
        num_frames = raw_frames + ((8 - ((raw_frames - 1) % 8)) % 8)

        min_frames = int(self.config.get("benchmark_min_frames", default_frames))
        max_frames = self.config.get("benchmark_max_frames", None)
        num_frames = max(num_frames, min_frames)
        if max_frames is not None:
            num_frames = min(num_frames, int(max_frames))
            num_frames = num_frames - ((num_frames - 1) % 8)
        return int(num_frames)

    def _compute_sigma_schedule(
        self,
        num_steps: int,
        num_frames: Optional[int] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
    ) -> torch.Tensor:
        """LTX-2 sigma schedule with token-count-dependent shift + stretch."""
        import math as _math

        num_frames = int(num_frames or self.config.get("num_frames", 121))
        video_h = int(video_height or self.config.get("video_height", 512))
        video_w = int(video_width or self.config.get("video_width", 768))
        latent_h = video_h // 32
        latent_w = video_w // 32
        latent_f = (num_frames - 1) // 8 + 1
        tokens = latent_f * latent_h * latent_w

        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        base_shift, max_shift = 0.95, 2.05
        x1, x2 = 1024, 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        sigmas = torch.where(
            sigmas != 0,
            _math.exp(sigma_shift) / (_math.exp(sigma_shift) + (1 / sigmas - 1)),
            0,
        )
        # Stretch to terminal=0.1
        non_zero = sigmas != 0
        nz = sigmas[non_zero]
        scale = (1.0 - nz[-1]) / (1.0 - 0.1)
        sigmas[non_zero] = 1.0 - (1.0 - nz) / scale

        return sigmas.float()

    @torch.no_grad()
    def _generate_causal_with_cfg(
        self,
        pos_conditional_dict: Dict[str, torch.Tensor],
        neg_conditional_dict: Dict[str, torch.Tensor],
        video_shape: tuple,
        audio_shape: tuple,
        seed: int,
        first_block_latent: Optional[torch.Tensor] = None,
    ):
        """
        Multi-step Euler sampling with CFG, autoregressive prefix-rerun per block.

        Block 0: denoise from noise (no context), or use first_block_latent if TI2AV.
        Block k>0: prefix-rerun with clean blocks 0..k-1 as context.
        """
        from ltx_causal.attention.mask_builder import (
            compute_aligned_audio_frames,
            compute_av_blocks,
        )

        device = self.device
        dtype = torch.bfloat16
        B = video_shape[0]
        F_v = video_shape[1]
        cfg_scale = self.benchmark_guidance_scale

        sigmas = self._compute_sigma_schedule(
            self.benchmark_num_inference_steps,
            num_frames=(F_v - 1) * 8 + 1,
            video_height=int(video_shape[-2]) * 32,
            video_width=int(video_shape[-1]) * 32,
        ).to(device)

        blocks = compute_av_blocks(
            total_video_latent_frames=F_v,
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            first_block_video_frames=self.config.get("first_block_video_frames", 1),
        )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = torch.zeros(audio_shape, device=device, dtype=dtype) if audio_shape else None

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            for block_idx, block in enumerate(blocks):
                # TI2AV: Block 0 uses clean first-frame latent (sigma=0)
                if block_idx == 0 and first_block_latent is not None:
                    cur_video = first_block_latent[:, :block.video_frames].to(device=device, dtype=dtype)
                    video[:, block.video_start:block.video_end] = cur_video
                    if audio is not None and block.audio_frames > 0:
                        audio[:, block.audio_start:block.audio_end] = torch.zeros(
                            (B, block.audio_frames, audio_shape[2]),
                            device=device, dtype=dtype,
                        )
                    continue

                # Initialize current block with noise
                cur_video = torch.randn(
                    (B, block.video_frames, *video_shape[2:]),
                    device=device, dtype=dtype,
                )
                cur_audio = None
                if audio is not None and block.audio_frames > 0:
                    cur_audio = torch.randn(
                        (B, block.audio_frames, audio_shape[2]),
                        device=device, dtype=dtype,
                    )

                prev_video = video[:, :block.video_start] if block.video_start > 0 else None
                prev_audio = (
                    audio[:, :block.audio_start]
                    if audio is not None and block.audio_start > 0
                    else None
                )

                for step_idx, sigma in enumerate(sigmas[:-1]):
                    # Build full prefix+block input
                    if prev_video is not None:
                        full_video = torch.cat([prev_video, cur_video], dim=1)
                        video_ts = torch.cat([
                            torch.zeros(B, prev_video.shape[1], device=device, dtype=dtype),
                            sigma.expand(B, cur_video.shape[1]).to(dtype),
                        ], dim=1)
                    else:
                        full_video = cur_video
                        video_ts = sigma.expand(B, cur_video.shape[1]).to(dtype)

                    full_audio = None
                    audio_ts = None
                    if cur_audio is not None:
                        if prev_audio is not None:
                            full_audio = torch.cat([prev_audio, cur_audio], dim=1)
                            audio_ts = torch.cat([
                                torch.zeros(B, prev_audio.shape[1], device=device, dtype=dtype),
                                sigma.expand(B, cur_audio.shape[1]).to(dtype),
                            ], dim=1)
                        else:
                            full_audio = cur_audio
                            audio_ts = sigma.expand(B, cur_audio.shape[1]).to(dtype)

                    # Positive pass
                    x0_pos_v, x0_pos_a = self.ar_teacher._generator(
                        noisy_image_or_video=full_video,
                        conditional_dict=pos_conditional_dict,
                        timestep=video_ts,
                        noisy_audio=full_audio,
                        audio_timestep=audio_ts,
                        use_causal_timestep=False,
                    )

                    # Negative pass (for CFG)
                    if cfg_scale != 1.0:
                        x0_neg_v, x0_neg_a = self.ar_teacher._generator(
                            noisy_image_or_video=full_video,
                            conditional_dict=neg_conditional_dict,
                            timestep=video_ts,
                            noisy_audio=full_audio,
                            audio_timestep=audio_ts,
                            use_causal_timestep=False,
                        )
                        x0_v = x0_neg_v + cfg_scale * (x0_pos_v - x0_neg_v)
                        x0_a = x0_neg_a + cfg_scale * (x0_pos_a - x0_neg_a) if x0_neg_a is not None else x0_pos_a
                    else:
                        x0_v = x0_pos_v
                        x0_a = x0_pos_a

                    # Extract current block predictions
                    x0_cur_v = x0_v[:, block.video_start:block.video_end]
                    x0_cur_a = (
                        x0_a[:, block.audio_start:block.audio_end]
                        if x0_a is not None and cur_audio is not None
                        else None
                    )

                    # Euler step: x_{next} = x + (x - x0) / sigma * (sigma_next - sigma)
                    sigma_next = sigmas[step_idx + 1]
                    if sigma_next > 0:
                        dt = sigma_next - sigma
                        cur_video = (
                            cur_video.float()
                            + (cur_video.float() - x0_cur_v.float()) / sigma.float() * dt.float()
                        ).to(dtype)
                        if cur_audio is not None and x0_cur_a is not None:
                            cur_audio = (
                                cur_audio.float()
                                + (cur_audio.float() - x0_cur_a.float()) / sigma.float() * dt.float()
                            ).to(dtype)
                    else:
                        cur_video = x0_cur_v
                        if cur_audio is not None and x0_cur_a is not None:
                            cur_audio = x0_cur_a

                    del x0_pos_v, x0_pos_a, x0_v, x0_a
                    if cfg_scale != 1.0:
                        del x0_neg_v, x0_neg_a

                # Store denoised block
                video[:, block.video_start:block.video_end] = cur_video
                if audio is not None and cur_audio is not None:
                    audio[:, block.audio_start:block.audio_end] = cur_audio

                torch.cuda.empty_cache()

        return video, audio

    @torch.no_grad()
    def _generate_causal_with_cfg_kv_cache(
        self,
        pos_conditional_dict: Dict[str, torch.Tensor],
        neg_conditional_dict: Dict[str, torch.Tensor],
        video_shape: tuple,
        audio_shape: tuple,
        seed: int,
        first_block_latent: Optional[torch.Tensor] = None,
        pad_to_video_shape: Optional[tuple] = None,
        pad_to_audio_shape: Optional[tuple] = None,
    ):
        """KV-cache blockwise benchmark inference with optional CFG."""
        from ltx_causal.attention.mask_builder import compute_av_blocks
        from ltx_causal.wrapper import build_block_ref_from_prev

        device = self.device
        dtype = torch.bfloat16
        B = video_shape[0]
        F_v = video_shape[1]
        rollout_video_shape = pad_to_video_shape or video_shape
        rollout_audio_shape = pad_to_audio_shape or audio_shape
        rollout_F_v = rollout_video_shape[1]
        cfg_scale = self.benchmark_guidance_scale
        # Cached recycling flag from the (possibly FSDP-wrapped) generator.
        enable_chunk_recycling = bool(getattr(
            self._unwrap_module(self.ar_teacher._generator),
            "enable_chunk_recycling",
            False,
        ))

        sigmas = self._compute_sigma_schedule(
            self.benchmark_num_inference_steps,
            num_frames=(F_v - 1) * 8 + 1,
            video_height=int(video_shape[-2]) * 32,
            video_width=int(video_shape[-1]) * 32,
        ).to(device)
        blocks = compute_av_blocks(
            total_video_latent_frames=F_v,
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            first_block_video_frames=self.config.get("first_block_video_frames", 1),
        )
        rollout_blocks = compute_av_blocks(
            total_video_latent_frames=rollout_F_v,
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            first_block_video_frames=self.config.get("first_block_video_frames", 1),
        )

        total_audio_frames = rollout_audio_shape[1] if rollout_audio_shape is not None else 0
        # Derive runtime tokens-per-frame from the latent shape so KV cache
        # buffers are sized to the actual per-case resolution (TI2AV cases
        # have variable aspect ratios under preserve_aspect_ratio=True).
        H_lat, W_lat = rollout_video_shape[-2], rollout_video_shape[-1]
        frame_seqlen = int(H_lat) * int(W_lat)
        kv_pos = self._init_kv_caches_for_conditional(
            conditional_dict=pos_conditional_dict,
            batch_size=B,
            total_video_frames=rollout_F_v,
            total_audio_frames=total_audio_frames,
            device=device,
            dtype=dtype,
            video_frame_seqlen=frame_seqlen,
        )
        kv_neg = None
        if cfg_scale != 1.0:
            kv_neg = self._init_kv_caches_for_conditional(
                conditional_dict=neg_conditional_dict,
                batch_size=B,
                total_video_frames=rollout_F_v,
                total_audio_frames=total_audio_frames,
                device=device,
                dtype=dtype,
                video_frame_seqlen=frame_seqlen,
            )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = torch.zeros(audio_shape, device=device, dtype=dtype) if audio_shape else None

        # Track the previous chunk's final latent for AAPT-style ref injection.
        # Stored as a [B, F_prev, C, H, W] tensor (model x0 from prev block).
        prev_chunk_latent: Optional[torch.Tensor] = None

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            for block_idx, rollout_block in enumerate(rollout_blocks):
                has_real_block = block_idx < len(blocks)
                block = blocks[block_idx] if has_real_block else rollout_block

                # Build ref for current block (always; helper handles "no recycling"
                # by returning zeros which is then ignored downstream).
                if enable_chunk_recycling:
                    ref_video_block = build_block_ref_from_prev(
                        prev_chunk_latent=prev_chunk_latent,
                        current_block_size=block.video_frames,
                        template=video,
                    )
                else:
                    ref_video_block = None

                # TI2AV: Block 0 uses clean first-frame latent (sigma=0)
                if has_real_block and block_idx == 0 and first_block_latent is not None:
                    cur_video = first_block_latent[:, :block.video_frames].to(device=device, dtype=dtype)
                    video[:, block.video_start:block.video_end] = cur_video
                    cur_audio = None
                    if audio is not None and block.audio_frames > 0:
                        cur_audio = torch.zeros(
                            (B, block.audio_frames, audio_shape[2]),
                            device=device, dtype=dtype,
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
                    (B, block.video_frames, *video_shape[2:]),
                    device=device,
                    dtype=dtype,
                )
                cur_audio = None
                if audio is not None and block.audio_frames > 0:
                    cur_audio = torch.randn(
                        (B, block.audio_frames, audio_shape[2]),
                        device=device,
                        dtype=dtype,
                    )

                for step_idx, sigma in enumerate(sigmas[:-1]):
                    video_sigma = sigma.expand(B, cur_video.shape[1]).to(self.dtype)
                    audio_sigma = (
                        sigma.expand(B, cur_audio.shape[1]).to(self.dtype)
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
                        dt = sigma_next - sigma
                        cur_video = (
                            cur_video.float()
                            + (cur_video.float() - x0_v.float()) / sigma.float() * dt.float()
                        ).to(dtype)
                        if cur_audio is not None and x0_a is not None:
                            cur_audio = (
                                cur_audio.float()
                                + (cur_audio.float() - x0_a.float()) / sigma.float() * dt.float()
                            ).to(dtype)
                    else:
                        cur_video = x0_v
                        if cur_audio is not None and x0_a is not None:
                            cur_audio = x0_a

                    del x0_pos_v, x0_pos_a, x0_v, x0_a
                    if cfg_scale != 1.0:
                        del x0_neg_v, x0_neg_a

                if has_real_block:
                    video[:, block.video_start:block.video_end] = cur_video
                if has_real_block and audio is not None and cur_audio is not None:
                    audio[:, block.audio_start:block.audio_end] = cur_audio

                # Refresh cache with final block state.
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

                # Update prev_chunk_latent for the NEXT block's ref construction.
                prev_chunk_latent = cur_video.detach()

                if self.config.get("benchmark_clear_cuda_cache_per_round", False):
                    torch.cuda.empty_cache()

        return video, audio

    def _decode_and_save_sample(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor],
        prompt_idx: int,
        step_dir: str,
        prompt_text: Optional[str] = None,
    ):
        """Decode latents via VAE and save as mp4."""
        video_pixel = self.video_vae.decode_to_pixel(video_latent)

        audio_waveform = None
        if audio_latent is not None:
            try:
                audio_waveform = self.audio_vae.decode_to_waveform(audio_latent)
            except Exception as e:
                print(f"[Benchmark][Rank {self.global_rank}] Audio decode failed: {e}")

        vid = video_pixel[0]  # [C, F, H, W]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)  # [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)  # [F, H, W, C]
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")

        written_ok = False
        if audio_waveform is not None:
            try:
                wav = audio_waveform[0].cpu().float()
                _write_video_compat(
                    sample_path, vid, fps=self.benchmark_video_fps,
                    audio_array=wav, audio_fps=self.benchmark_audio_sample_rate,
                    audio_codec="aac",
                )
                written_ok = True
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] Audio mux failed for "
                    f"{sample_path}; falling back to video-only mp4: {e}",
                    flush=True,
                )

        if not written_ok:
            try:
                _write_video_compat(sample_path, vid, fps=self.benchmark_video_fps)
            except Exception as e:
                print(f"[Benchmark][Rank {self.global_rank}] write_video failed: {e}")
                return

        if prompt_text is not None:
            txt_path = os.path.splitext(sample_path)[0] + ".txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"index: {prompt_idx}\n")
                    f.write(f"group: {os.path.basename(step_dir)}\n")
                    f.write(f"prompt: {prompt_text}\n")
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] Failed to save prompt "
                    f"sidecar for {sample_path}: {e}",
                    flush=True,
                )

        del video_pixel, audio_waveform
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_first_frame(self, image_path: str):
        """
        Load image, resize preserving aspect ratio to target area (aligned to 32px),
        encode with video VAE, and tile to fill Block 0 frames.

        Returns:
            (latent, pixel_h, pixel_w) -- latent and the actual pixel dimensions used.
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        target_h = self.config.get("video_height", 512)
        target_w = self.config.get("video_width", 768)
        target_area = target_h * target_w

        aspect = orig_w / orig_h
        new_h = int((target_area / aspect) ** 0.5)
        new_w = int(new_h * aspect)
        new_h = max(32, round(new_h / 32) * 32)
        new_w = max(32, round(new_w / 32) * 32)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_tensor = TF.to_tensor(img) * 2.0 - 1.0  # [3, H, W], [-1, 1]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
        img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)

        self._encoders_to_device()
        latent = self.video_encoder(img_tensor)
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H_lat, W_lat]
        self._encoders_to_cpu()

        first_block_frames = self.config.get("first_block_video_frames", 1)
        if latent.shape[1] < first_block_frames:
            latent = latent.repeat(1, first_block_frames, 1, 1, 1)[:, :first_block_frames]

        return latent, new_h, new_w

    def _run_benchmark_and_log(self):
        """
        Run multi-step Euler + CFG benchmark: T2AV then TI2AV.

        All ranks participate (FSDP requires collective forward).
        Work distributed across ranks in ceil(N/W) rounds.
        """
        import math as _math

        config = self.config
        torch.cuda.empty_cache()

        num_t2av = len(self.benchmark_prompts)
        num_ti2av = len(self.benchmark_ti2av_cases)
        total = num_t2av + num_ti2av

        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: {num_t2av} T2AV + {num_ti2av} TI2AV samples "
                f"({self.benchmark_num_inference_steps}-step Euler, CFG={self.benchmark_guidance_scale}) "
                f"across {self.world_size} ranks..."
            )

        step_dir = os.path.join(self.output_path, "benchmark", f"step_{self.step:07d}")
        t2av_group_dirs = {
            group["name"]: os.path.join(step_dir, group["name"])
            for group in self.benchmark_t2av_groups
        }
        if not t2av_group_dirs:
            t2av_group_dirs = {"t2av": os.path.join(step_dir, "t2av")}
        ti2av_dir = os.path.join(step_dir, "ti2av")
        for group_dir in t2av_group_dirs.values():
            os.makedirs(group_dir, exist_ok=True)
        os.makedirs(ti2av_dir, exist_ok=True)

        from ltx_distillation.train_distillation import compute_latent_shapes

        with torch.no_grad():
            neg_conditional_dict = self.ar_teacher.text_encoder(text_prompts=[""])

        was_training = self.ar_teacher._generator.training
        self.ar_teacher._generator.eval()
        try:
            self._vae_decoders_to_device()

            benchmark_start = time.time()

            # ── T2AV pass ──
            if num_t2av > 0:
                max_prompt_num_frames = max(
                    self._benchmark_num_frames_for_prompt(prompt)
                    for prompt in self.benchmark_prompts
                )
                max_video_shape, max_audio_shape = compute_latent_shapes(
                    num_frames=max_prompt_num_frames,
                    video_height=config.get("video_height", 512),
                    video_width=config.get("video_width", 768),
                    batch_size=1,
                )

                if self.benchmark_parallel_t2av_round_robin:
                    if self.is_main_process:
                        print(
                            f"[Benchmark] T2AV max rollout frames={max_prompt_num_frames} "
                            f"latent_frames={max_video_shape[1]}",
                            flush=True,
                        )

                    num_t2av_rounds = _math.ceil(num_t2av / self.world_size)
                    t2av_work_items = []
                    for round_idx in range(num_t2av_rounds):
                        prompt_idx = round_idx * self.world_size + self.global_rank
                        has_real = prompt_idx < num_t2av
                        t2av_work_items.append((prompt_idx if has_real else 0, has_real))

                    local_assignments = [
                        (
                            prompt_idx,
                            self._benchmark_num_frames_for_prompt(self.benchmark_prompts[prompt_idx]),
                        )
                        for prompt_idx, has_real in t2av_work_items
                        if has_real
                    ]
                    print(
                        f"[Benchmark][Rank {self.global_rank}] T2AV assigned "
                        f"(prompt_idx, frames): {local_assignments}",
                        flush=True,
                    )
                else:
                    t2av_work_items = [
                        (prompt_idx, True)
                        for prompt_idx in range(num_t2av)
                    ]

                for prompt_idx, has_real_prompt in t2av_work_items:
                    prompt_text = self.benchmark_prompts[prompt_idx]
                    prompt_num_frames = self._benchmark_num_frames_for_prompt(prompt_text)
                    video_shape, audio_shape = compute_latent_shapes(
                        num_frames=prompt_num_frames,
                        video_height=config.get("video_height", 512),
                        video_width=config.get("video_width", 768),
                        batch_size=1,
                    )

                    with torch.no_grad():
                        pos_cond = self.ar_teacher.text_encoder(text_prompts=[prompt_text])

                    seed = self.benchmark_seed + prompt_idx
                    gen_fn = (
                        self._generate_causal_with_cfg_kv_cache
                        if self.benchmark_use_kv_cache
                        else self._generate_causal_with_cfg
                    )
                    gen_kwargs = {}
                    if self.benchmark_parallel_t2av_round_robin and self.benchmark_use_kv_cache:
                        gen_kwargs = dict(
                            pad_to_video_shape=tuple(max_video_shape),
                            pad_to_audio_shape=tuple(max_audio_shape),
                        )
                    video_latent, audio_latent = gen_fn(
                        pos_conditional_dict=pos_cond,
                        neg_conditional_dict=neg_conditional_dict,
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        seed=seed,
                        **gen_kwargs,
                    )

                    if has_real_prompt and (
                        self.is_main_process or self.benchmark_parallel_t2av_round_robin
                    ):
                        group_name = self._benchmark_t2av_group_name(prompt_idx)
                        self._decode_and_save_sample(
                            video_latent=video_latent,
                            audio_latent=audio_latent,
                            prompt_idx=prompt_idx,
                            step_dir=t2av_group_dirs.get(
                                group_name, next(iter(t2av_group_dirs.values())),
                            ),
                            prompt_text=prompt_text,
                        )

                    del video_latent, audio_latent, pos_cond
                    torch.cuda.empty_cache()
                    barrier()

            # ── TI2AV pass ──
            if num_ti2av > 0:
                num_rounds = _math.ceil(num_ti2av / self.world_size)
                for round_idx in range(num_rounds):
                    case_idx = round_idx * self.world_size + self.global_rank
                    has_real = case_idx < num_ti2av
                    case = (
                        self.benchmark_ti2av_cases[case_idx] if has_real
                        else self.benchmark_ti2av_cases[0]
                    )

                    with torch.no_grad():
                        pos_cond = self.ar_teacher.text_encoder(text_prompts=[case["prompt"]])

                    first_block_latent, case_pixel_h, case_pixel_w = self._encode_first_frame(
                        case["image_path"],
                    )

                    # Compute per-case shapes based on first frame's aspect ratio
                    case_video_shape, case_audio_shape = compute_latent_shapes(
                        num_frames=config.get("num_frames", 121),
                        video_height=case_pixel_h,
                        video_width=case_pixel_w,
                        batch_size=1,
                    )

                    seed = self.benchmark_seed + 10000 + case_idx
                    gen_fn = (
                        self._generate_causal_with_cfg_kv_cache
                        if self.benchmark_use_kv_cache
                        else self._generate_causal_with_cfg
                    )
                    video_latent, audio_latent = gen_fn(
                        pos_conditional_dict=pos_cond,
                        neg_conditional_dict=neg_conditional_dict,
                        video_shape=tuple(case_video_shape),
                        audio_shape=tuple(case_audio_shape),
                        seed=seed,
                        first_block_latent=first_block_latent,
                    )

                    if has_real:
                        self._decode_and_save_sample(
                            video_latent=video_latent,
                            audio_latent=audio_latent,
                            prompt_idx=case_idx,
                            step_dir=ti2av_dir,
                            prompt_text=case["prompt"],
                        )

                    del video_latent, audio_latent, pos_cond, first_block_latent
                    torch.cuda.empty_cache()
                    barrier()

        finally:
            if was_training:
                self.ar_teacher._generator.train()

        self._vae_decoders_to_cpu()

        benchmark_elapsed = time.time() - benchmark_start

        barrier()

        if self.is_main_process:
            wandb_dict = {}

            # Log T2AV samples
            t2av_rows_by_group = {name: [] for name in t2av_group_dirs}
            for idx in range(num_t2av):
                group_name = self._benchmark_t2av_group_name(idx)
                group_dir = t2av_group_dirs.get(
                    group_name, next(iter(t2av_group_dirs.values())),
                )
                sample_path = os.path.join(group_dir, f"sample_{idx}.mp4")
                if os.path.exists(sample_path):
                    wandb_dict[f"benchmark/{group_name}/sample_{idx}"] = wandb.Video(
                        sample_path, fps=self.benchmark_video_fps, format="mp4",
                    )
                    t2av_rows_by_group.setdefault(group_name, []).append([
                        idx, self.benchmark_prompts[idx], sample_path,
                    ])
            for group_name, t2av_rows in t2av_rows_by_group.items():
                if t2av_rows:
                    wandb_dict[f"benchmark/{group_name}/prompt_table"] = wandb.Table(
                        columns=["index", "prompt", "local_path"], data=t2av_rows,
                    )

            # Log TI2AV samples
            ti2av_rows = []
            for idx in range(num_ti2av):
                sample_path = os.path.join(ti2av_dir, f"sample_{idx}.mp4")
                if os.path.exists(sample_path):
                    wandb_dict[f"benchmark/ti2av/sample_{idx}"] = wandb.Video(
                        sample_path, fps=self.benchmark_video_fps, format="mp4",
                    )
                    ti2av_rows.append([
                        idx,
                        self.benchmark_ti2av_cases[idx]["prompt"][:80],
                        self.benchmark_ti2av_cases[idx]["image_path"],
                        sample_path,
                    ])
            if ti2av_rows:
                wandb_dict["benchmark/ti2av/case_table"] = wandb.Table(
                    columns=["index", "prompt", "image_path", "local_path"],
                    data=ti2av_rows,
                )

            if wandb_dict:
                wandb.log(wandb_dict, step=self.step)

            print(
                f"[Benchmark] Step {self.step}: {num_t2av} T2AV + {num_ti2av} TI2AV in "
                f"{benchmark_elapsed:.1f}s | saved to {step_dir}",
                flush=True,
            )

        barrier()

    # ── Training step ──

    def train_one_step(self):
        self.ar_teacher._generator.train()
        if self.csf_projector_video is not None:
            self.csf_projector_video.train()
        if self.csf_projector_audio is not None:
            self.csf_projector_audio.train()
        self._apply_lr()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        accum_steps = max(1, int(self.gradient_accumulation_steps))

        # Decide branch for this optimizer step (broadcast from rank 0 so every
        # rank runs the same forward graph -> FSDP collectives stay in sync).
        # CSF activates only after warmup; otherwise we fall back to pure TF.
        use_csf = False
        if self.enable_causal_self_flow:
            flag = torch.zeros(1, device=self.device)
            if self.is_main_process:
                if self.step >= self.csf_warmup_steps and torch.rand(1).item() > self.csf_clean_tf_ratio:
                    flag.fill_(1.0)
            if dist.is_initialized() and self.world_size > 1:
                dist.broadcast(flag, src=0)
            use_csf = flag.item() > 0.5

        # Only retain decoded tensors for visualization on a vis step (saves memory).
        train_vis_iters = self.config.get("train_vis_iters", 0)
        do_vis = (
            train_vis_iters > 0
            and self.step > 0
            and self.step % train_vis_iters == 0
            and self.is_main_process
        )

        # One optimizer update spanning `accum_steps` micro-batches.
        # zero_grad once; each micro-batch backward accumulates into the (sharded)
        # .grad buffers; clip/step/EMA happen once after the accumulation loop.
        self.optimizer.zero_grad()

        accum_log: dict[str, float] | None = None
        vis_clean_video = None
        vis_pred_x0_video = None
        vis_video_sigma = None
        vis_sigma_mean = None
        vis_metadata = None

        for micro_idx in range(accum_steps):
            batch = next(self.dataloader)
            clean_video, clean_audio, text_prompts = self._encode_batch(batch)

            # Align audio frames
            clean_audio = self._align_audio(clean_video, clean_audio)

            # Text encoding
            with torch.no_grad():
                conditional_dict = self.ar_teacher.text_encoder(text_prompts=text_prompts)

            # Flow matching loss with teacher forcing (optionally Self-Flow /
            # Causal-Self-Flow augmented).
            if use_csf:
                loss, log_dict = self._generator_loss_causal_self_flow(
                    clean_video=clean_video,
                    clean_audio=clean_audio,
                    conditional_dict=conditional_dict,
                )
            elif self.enable_self_flow:
                loss, log_dict = self._generator_loss_self_flow(
                    clean_video=clean_video,
                    clean_audio=clean_audio,
                    conditional_dict=conditional_dict,
                )
            else:
                loss, log_dict = self.ar_teacher.generator_loss(
                    clean_video=clean_video,
                    clean_audio=clean_audio,
                    conditional_dict=conditional_dict,
                )

            # Scale by accum_steps so the accumulated gradient equals the mean
            # gradient over the effective (larger) batch.
            (loss / accum_steps).backward()

            # Accumulate scalar logs (mean over micro-batches).
            step_logs = {
                "loss": loss.item(),
                "video_loss": log_dict["video_loss"].item(),
                "audio_loss": log_dict["audio_loss"].item(),
                "base_loss_unscaled": log_dict["base_loss_unscaled"].item(),
                "video_sigma_mean": log_dict["video_sigma_mean"].item(),
            }
            if use_csf:
                step_logs.update({
                    "base_loss": log_dict["base_loss"].item(),
                    "repr_loss": log_dict["repr_loss"].item(),
                    "repr_video_loss": log_dict["repr_video_loss"].item(),
                    "repr_audio_loss": log_dict["repr_audio_loss"].item(),
                    "csf_ctx_sigma_mean": log_dict["csf_ctx_sigma_mean"].item(),
                })
            elif self.enable_self_flow:
                step_logs.update({
                    "base_loss": log_dict["base_loss"].item(),
                    "repr_loss": log_dict["repr_loss"].item(),
                    "repr_video_loss": log_dict["repr_video_loss"].item(),
                    "repr_audio_loss": log_dict["repr_audio_loss"].item(),
                    "self_flow_mask_ratio_actual": log_dict["self_flow_mask_ratio_actual"].item(),
                })
            if accum_log is None:
                accum_log = {k: 0.0 for k in step_logs}
            for k, v in step_logs.items():
                accum_log[k] += v / accum_steps

            # Retain the last micro-batch's tensors for visualization only.
            if do_vis and micro_idx == accum_steps - 1:
                vis_clean_video = clean_video.detach()
                vis_pred_x0_video = log_dict["pred_x0_video"]
                vis_video_sigma = log_dict["video_sigma"]
                vis_sigma_mean = step_logs["video_sigma_mean"]
                vis_metadata = {
                    "entry": batch.get("entries", [""])[0],
                    "clip_url": batch.get("clip_urls", [""])[0],
                    "audio_url": batch.get("audio_urls", [""])[0],
                    "fps": batch.get("fps", [""])[0],
                    "f_num": batch.get("f_nums", [""])[0],
                    "caption": text_prompts[0] if text_prompts else "",
                }

            del loss, log_dict, clean_video, clean_audio, conditional_dict

        # Gradient clipping (over the accumulated gradient)
        if hasattr(self.ar_teacher._generator, "clip_grad_norm_"):
            grad_norm = self.ar_teacher._generator.clip_grad_norm_(self.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ar_teacher._generator.parameters(), self.max_grad_norm
            )

        # Clip CSF projector grads (separate FSDP units -> own clip call).
        # Only on CSF steps; pure-TF steps leave projector grads at None.
        if use_csf:
            for _proj in (self.csf_projector_video, self.csf_projector_audio):
                if _proj is None:
                    continue
                if hasattr(_proj, "clip_grad_norm_"):
                    _proj.clip_grad_norm_(self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(_proj.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # EMA update (shard-local, no all-gather needed)
        if self.ema_generator is not None:
            with torch.no_grad():
                ema_params = dict(self.ema_generator.named_parameters())
                for name, param in self.ar_teacher._generator.named_parameters():
                    ema_params[name].data.mul_(self.ema_decay).add_(
                        param.data, alpha=1 - self.ema_decay
                    )

        # Logging
        if self.is_main_process:
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            log_data = {
                "train/loss": accum_log["loss"],
                "train/video_loss": accum_log["video_loss"],
                "train/audio_loss": accum_log["audio_loss"],
                "train/grad_norm": grad_norm_val,
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/video_sigma_mean": accum_log["video_sigma_mean"],
                "train/base_denoising_loss_unscaled": accum_log[
                    "base_loss_unscaled"
                ],
                "train/velocity_loss_scale": (
                    self.ar_teacher.config.velocity_loss_scale
                    if self.ar_teacher.config.denoising_loss_type == "velocity"
                    else 1.0
                ),
                "train/grad_clip_coef": min(
                    1.0, self.max_grad_norm / (float(grad_norm_val) + 1e-12)
                ),
            }
            if use_csf:
                log_data.update({
                    "train/csf_base_loss": accum_log["base_loss"],
                    "train/csf_base_loss_unscaled": accum_log["base_loss_unscaled"],
                    "train/csf_repr_loss": accum_log["repr_loss"],
                    "train/csf_repr_video_loss": accum_log["repr_video_loss"],
                    "train/csf_repr_audio_loss": accum_log["repr_audio_loss"],
                    "train/csf_ctx_sigma_mean": accum_log["csf_ctx_sigma_mean"],
                })
            elif self.enable_self_flow:
                log_data.update({
                    "train/base_loss": accum_log["base_loss"],
                    "train/base_loss_unscaled": accum_log["base_loss_unscaled"],
                    "train/repr_loss": accum_log["repr_loss"],
                    "train/repr_video_loss": accum_log["repr_video_loss"],
                    "train/repr_audio_loss": accum_log["repr_audio_loss"],
                    "train/self_flow_mask_ratio_actual": accum_log["self_flow_mask_ratio_actual"],
                })
            wandb.log(log_data, step=self.step)

            if self.step < 10 or self.step % 50 == 0:
                if use_csf:
                    print(
                        f"[Step {self.step}] [CSF] loss={accum_log['loss']:.4f} "
                        f"base={accum_log['base_loss']:.4f} "
                        f"repr={accum_log['repr_loss']:.4f} "
                        f"ctx_sig={accum_log['csf_ctx_sigma_mean']:.3f} "
                        f"v_loss={accum_log['video_loss']:.4f} "
                        f"a_loss={accum_log['audio_loss']:.4f} "
                        f"grad={grad_norm_val:.2f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                elif self.enable_self_flow:
                    print(
                        f"[Step {self.step}] loss={accum_log['loss']:.4f} "
                        f"base={accum_log['base_loss']:.4f} "
                        f"repr={accum_log['repr_loss']:.4f} "
                        f"v_loss={accum_log['video_loss']:.4f} "
                        f"a_loss={accum_log['audio_loss']:.4f} "
                        f"grad={grad_norm_val:.2f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                else:
                    print(
                        f"[Step {self.step}] loss={accum_log['loss']:.4f} "
                        f"v_loss={accum_log['video_loss']:.4f} "
                        f"a_loss={accum_log['audio_loss']:.4f} "
                        f"grad={grad_norm_val:.2f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

        # Visual logging: decode and save clean/pred_x0 as mp4 (last micro-batch)
        if do_vis and vis_clean_video is not None:
            self._log_training_samples(
                clean_video=vis_clean_video,
                pred_x0_video=vis_pred_x0_video,
                video_sigma=vis_video_sigma,
                sigma_mean=vis_sigma_mean,
                metadata=vis_metadata,
            )

        del grad_norm, accum_log
        del vis_clean_video, vis_pred_x0_video, vis_video_sigma
        torch.cuda.empty_cache()

    # ── Checkpointing ──

    def save(self):
        print("Gathering distributed model states...")

        generator_state_dict = fsdp_state_dict(self.ar_teacher._generator)

        state_dict = {
            "generator": generator_state_dict,
            "step": self.step,
        }

        if self.ema_generator is not None:
            state_dict["ema_generator"] = fsdp_state_dict(self.ema_generator)

        if self.csf_projector_video is not None:
            state_dict["csf_projector_video"] = fsdp_state_dict(self.csf_projector_video)
        if self.csf_projector_audio is not None:
            state_dict["csf_projector_audio"] = fsdp_state_dict(self.csf_projector_audio)

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path,
                f"checkpoint_{self.step:06d}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

        del generator_state_dict, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ── Main loop ──

    def train(self):
        max_steps = self.config.get("max_steps", float('inf'))
        log_iters = self.config.get("log_iters", 200)

        while self.step < max_steps:
            self.train_one_step()

            if not self.config.get("no_save", False) and self.step % log_iters == 0 and self.step > 0:
                self.save()
                torch.cuda.empty_cache()
                self.ar_teacher._generator.train()

            # Benchmark sampling
            if (
                self.benchmark_enabled
                and (len(self.benchmark_prompts) > 0 or len(self.benchmark_ti2av_cases) > 0)
                and self.step > 0
                and self.step % self.benchmark_iters == 0
            ):
                self._run_benchmark_and_log()
                self.ar_teacher._generator.train()

            barrier()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None:
                    wandb.log(
                        {"per_iteration_time": current_time - self.previous_time},
                        step=self.step,
                    )
                self.previous_time = current_time

            self.step += 1

        if not self.config.get("no_save", False):
            self.save()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AR Teacher Training with Teacher Forcing")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")

    args = parser.parse_args()

    # --- Verify/raise torch.compile recompile limits early ---
    # flex_attention falls back to a buggy eager path when dynamo exhausts
    # recompile_limit (default 8 on torch 2.12), causing dim-mismatch crashes
    # in backward. Print the runtime value, raise it, then print again so the
    # log shows whether the limit was actually in effect.
    import torch._dynamo as _dynamo
    _rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    _dcfg = _dynamo.config
    print(f"[DYNAMO][rank{_rank}] before: recompile_limit="
          f"{getattr(_dcfg, 'recompile_limit', '<missing>')} "
          f"accumulated_recompile_limit="
          f"{getattr(_dcfg, 'accumulated_recompile_limit', '<missing>')}",
          flush=True)
    for _name, _val in (
        ("recompile_limit", 256),
        ("cache_size_limit", 256),
        ("accumulated_recompile_limit", 4096),
        ("accumulated_cache_size_limit", 4096),
    ):
        if hasattr(_dcfg, _name):
            try:
                setattr(_dcfg, _name, max(int(getattr(_dcfg, _name)), _val))
            except Exception:
                pass
    print(f"[DYNAMO][rank{_rank}] after:  recompile_limit="
          f"{getattr(_dcfg, 'recompile_limit', '<missing>')} "
          f"accumulated_recompile_limit="
          f"{getattr(_dcfg, 'accumulated_recompile_limit', '<missing>')}",
          flush=True)

    config = OmegaConf.load(args.config_path)
    base_config_path = config.get("_base_config", None)
    if base_config_path:
        base_config_path = str(base_config_path)
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(
                os.path.dirname(os.path.abspath(args.config_path)),
                base_config_path,
            )
        overrides = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        del overrides["_base_config"]
        config = OmegaConf.merge(OmegaConf.load(base_config_path), overrides)
    if args.no_save:
        config.no_save = True

    trainer = ARTeacherTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
