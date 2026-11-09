"""
DMD Distillation Training Script for LTX-2.

Usage:
    torchrun --nproc_per_node=8 -m ltx_distillation.train_distillation \
        --config_path configs/ltx2_bidirectional_dmd.yaml
"""

import argparse
import math
import os
import re
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.data import (
    ODERegressionLMDBDataset,
    PromptDataset,
    causal_aligned_audio_frame_count,
)
from ltx_distillation.util import (
    launch_distributed_job,
    set_seed,
    init_logging_folder,
    fsdp_wrap,
    fsdp_state_dict,
    barrier,
    cycle,
)


def _dmd_class_from_config(config):
    """Resolve Stage-3/Stage-4 DMD objectives."""
    loss_name = str(getattr(config, "distillation_loss", "dmd")).lower()
    if loss_name == "dmd":
        return LTX2DMD
    if loss_name == "sc_dmd":
        from ltx_distillation.sc_dmd import LTX2ShortcutConsistencyDMD

        return LTX2ShortcutConsistencyDMD
    if loss_name == "self_context_ar_dmd":
        from ltx_distillation.self_context_ar_dmd import LTX2SelfContextARDMD

        return LTX2SelfContextARDMD
    if loss_name == "self_context_ar_sc_dmd":
        from ltx_distillation.self_context_ar_dmd import (
            LTX2SelfContextARShortcutConsistencyDMD,
        )

        return LTX2SelfContextARShortcutConsistencyDMD
    if loss_name == "stage4_scale_wise_dmd":
        from ltx_distillation.stage4_scale_wise_dmd import (
            LTX2ScaleWiseDMD,
        )

        return LTX2ScaleWiseDMD
    raise ValueError(
        f"Unsupported distillation_loss={loss_name!r} for train_distillation"
    )


def _load_config_with_base(config_path: str):
    """Load an optional relative base config, then apply local overrides."""
    config = OmegaConf.load(config_path)
    base_path = config.get("base_config_path", None)
    if not base_path:
        return config
    base_path = str(base_path)
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(config_path), base_path)
    del config["base_config_path"]
    return OmegaConf.merge(_load_config_with_base(base_path), config)


def _write_video_compat(
    filename: str,
    video_array: torch.Tensor,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
    audio_codec: Optional[str] = None,
):
    """Write an mp4 across torchvision versions (kept identical to Stage 2).

    torchvision 0.27 no longer exposes ``torchvision.io.write_video``. Use it
    when present, otherwise fall back to PyAV with optional AAC audio muxing.
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
) -> Tuple[list, list]:
    """
    Compute latent shapes from video frames and resolution.

    Calculation logic matches LTX-2 native implementation (see ltx_core/types.py):
    - Video: frames = (num_frames - 1) // 8 + 1
    - Audio: frames = round(video_duration * audio_latent_fps)
             where audio_latent_fps = sample_rate / hop_length / downsample = 25

    Args:
        num_frames: Number of raw video frames (must satisfy 1 + 8*k constraint)
        video_height: Video height in pixels
        video_width: Video width in pixels
        batch_size: Batch size
        latent_channels: Number of latent channels
        vae_temporal_compression: VAE temporal compression ratio (default 8)
        vae_spatial_compression: VAE spatial compression ratio (default 32)
        video_fps: Video frame rate (default 24.0)
        audio_sample_rate: Audio sample rate (default 16000)
        audio_hop_length: Audio hop length (default 160)
        audio_latent_downsample: Audio latent downsampling factor (default 4)

    Returns:
        (video_shape, audio_shape)
        - video_shape: [B, latent_frames, C, H, W]
        - audio_shape: [B, audio_frames, C]
    """
    # Check frame count constraint
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(
            f"num_frames must be 1 + 8*k, got {num_frames}. "
            f"Valid values: 1, 9, 17, 25, ..., 121, ..., 241, ..."
        )

    # Compute video latent frames (matches LTX types.py:73)
    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression

    # Compute latent spatial dimensions
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    # Compute audio frames (matches LTX types.py:140-156)
    # video_duration = num_frames / video_fps
    # audio_latent_fps = sample_rate / hop_length / downsample = 16000/160/4 = 25
    # audio_frames = round(video_duration * audio_latent_fps)
    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    video_shape = [batch_size, latent_frames, latent_channels, latent_h, latent_w]
    audio_shape = [batch_size, audio_frames, latent_channels]

    return video_shape, audio_shape


class Trainer:
    """
    DMD Distillation Trainer for LTX-2.

    Handles:
    - Distributed training with FSDP
    - Alternating generator and critic training
    - Checkpointing and logging
    """

    def __init__(self, config):
        self.config = config

        # Initialize distributed environment
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        rank, world_size, local_rank = launch_distributed_job()
        self.global_rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0

        # Set seed
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            if world_size > 1:
                dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + self.global_rank)

        # Logging folder (rank 0 creates, broadcast to others). Kept identical to
        # the Stage-2 trainer, whose multi-node distributed init is stable on this
        # cluster: rank 0 resolves the timestamped output_path, then a raw uint8
        # tensor broadcast (NOT a shared FS sentinel, NOT broadcast_object_list)
        # distributes it to every rank.
        self.output_path = config.output_path
        if self.is_main_process:
            self.output_path, _ = init_logging_folder(config)
        else:
            os.makedirs(self.output_path, exist_ok=True)
        if dist.is_initialized() and self.world_size > 1:
            if self.is_main_process:
                path_bytes = self.output_path.encode("utf-8")
                path_tensor = torch.tensor(
                    list(path_bytes) + [0] * (512 - len(path_bytes)),
                    dtype=torch.uint8, device=self.device,
                )
            else:
                path_tensor = torch.zeros(512, dtype=torch.uint8, device=self.device)
            dist.broadcast(path_tensor, src=0)
            if not self.is_main_process:
                self.output_path = (
                    bytes(path_tensor.cpu().tolist()).rstrip(b"\x00").decode("utf-8")
                )
                os.makedirs(self.output_path, exist_ok=True)
        self.wandb_folder = os.path.join(self.output_path, "wandb")
        barrier()

        # Keep the Stage-3 C objective isolated behind an explicit config key.
        dmd_class = _dmd_class_from_config(config)
        self.dmd = dmd_class(config, device=self.device)
        if self.is_main_process:
            dmd_support = self.dmd.score_sigma_support(fake_score=False)
            fake_support = self.dmd.score_sigma_support(fake_score=True)
            print(
                "[Stage3] score setup: "
                f"class={dmd_class.__name__}; "
                f"real={self.dmd.real_task_type}/"
                f"{'causal' if self.dmd.real_score_use_causal_wrapper else 'bidirectional'}; "
                f"fake={self.dmd.fake_task_type}/"
                f"{'causal' if self.dmd.fake_score_use_causal_wrapper else 'bidirectional'}; "
                f"fake_rollout={self.dmd.fake_score_rollout_mode}; "
                "real_cfg="
                f"video[{self.dmd.real_video_guidance_scale_min:.3f},"
                f"{self.dmd.real_video_guidance_scale_max:.3f}] "
                f"audio[{self.dmd.real_audio_guidance_scale_min:.3f},"
                f"{self.dmd.real_audio_guidance_scale_max:.3f}]; "
                f"dmd_sampler={self.dmd.dmd_timestep_sampler_type} "
                f"support=[{dmd_support[0]:.6f},{dmd_support[1]:.6f}]; "
                f"fake_sampler={self.dmd.fake_score_timestep_sampler_type} "
                f"support=[{fake_support[0]:.6f},{fake_support[1]:.6f}]",
                flush=True,
            )

        # Initialize models from checkpoints BEFORE FSDP wrapping
        # Models must exist before we can wrap them with FSDP
        self.dmd.init_models()
        self._validate_preinstalled_bidirectional_delegate()

        # FSDP wrapping
        self._wrap_with_fsdp()

        # Optimizers
        weight_decay = getattr(config, "weight_decay", 0.0)
        generator_lr = getattr(config, "generator_lr", config.lr)
        critic_lr = getattr(config, "critic_lr", config.lr)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.dmd.generator.parameters() if p.requires_grad],
            lr=generator_lr,
            betas=(config.beta1, config.beta2),
            weight_decay=weight_decay,
        )

        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.dmd.fake_score.parameters() if p.requires_grad],
            lr=critic_lr,
            betas=(config.beta1, config.beta2),
            weight_decay=weight_decay,
        )
        self.ladd_discriminator_optimizer = None
        if hasattr(self.dmd, "ladd_discriminator"):
            self.ladd_discriminator_optimizer = torch.optim.AdamW(
                self.dmd.ladd_discriminator.parameters(),
                lr=float(getattr(config, "ladd_discriminator_lr", 1e-6)),
                betas=(
                    float(getattr(config, "ladd_discriminator_beta1", config.beta1)),
                    float(getattr(config, "ladd_discriminator_beta2", config.beta2)),
                ),
                weight_decay=float(
                    getattr(config, "ladd_discriminator_weight_decay", weight_decay)
                ),
            )

        # Learning rate schedulers
        self.generator_scheduler = self._create_lr_scheduler(self.generator_optimizer)
        self.critic_scheduler = self._create_lr_scheduler(self.critic_optimizer)

        self.use_real_data = bool(getattr(config, "use_real_data", False))
        self.use_metadata_prompt_data = bool(
            getattr(config, "use_metadata_prompt_data", False)
        )
        if self.use_real_data and self.use_metadata_prompt_data:
            raise ValueError(
                "use_real_data and use_metadata_prompt_data are mutually exclusive"
            )
        self.real_data_adapter = None
        self.metadata_prompt_adapter = None
        if self.use_real_data:
            from ltx_distillation.stage3_real_data import Stage3RealDataAdapter

            self.real_data_adapter = Stage3RealDataAdapter(
                config=config,
                device=self.device,
                dtype=self.dtype,
                is_main_process=self.is_main_process,
            )
        elif self.use_metadata_prompt_data:
            from ltx_distillation.stage3_real_data import Stage3MetadataPromptAdapter

            self.metadata_prompt_adapter = Stage3MetadataPromptAdapter(
                config=config,
                is_main_process=self.is_main_process,
            )

        # Dataloader
        self._init_dataloader()

        # Benchmark prompts (for periodic inference visualization)
        self._init_benchmark_prompts()

        self.step = 0
        self.max_grad_norm = float(
            getattr(
                config,
                "ladd_max_grad_norm_generator",
                getattr(config, "max_grad_norm", 10.0),
            )
        )
        self.log_iters = int(getattr(config, "log_iters", 0))
        self.layerwise_grad_log_interval = max(
            1, int(getattr(config, "layerwise_grad_log_interval", config.log_iters))
        )
        self.previous_time = None

        # Resume from a causal DMD checkpoint (full state: generator + critic + step)
        resume_ckpt = getattr(config, "resume_checkpoint", None)
        if resume_ckpt:
            if self.is_main_process:
                print(f"[Resume] Loading causal DMD checkpoint from {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location="cpu")
            self.dmd.generator.load_state_dict(ckpt["generator"])
            self.dmd.fake_score.load_state_dict(ckpt["critic"])
            if (
                self.ladd_discriminator_optimizer is not None
                and "ladd_discriminator" in ckpt
            ):
                self.dmd.ladd_discriminator.load_state_dict(
                    ckpt["ladd_discriminator"]
                )
                if "ladd_discriminator_optimizer" in ckpt:
                    self.ladd_discriminator_optimizer.load_state_dict(
                        ckpt["ladd_discriminator_optimizer"]
                    )
            self.step = ckpt.get("step", 0)
            if self.is_main_process:
                print(f"[Resume] Resumed at step {self.step}")

    def _create_lr_scheduler(self, optimizer):
        """Create learning rate scheduler based on config.

        IMPORTANT: The scheduler is NOT stepped per-optimizer-call. Instead,
        both generator and critic schedulers are stepped once per global
        training step (in the training loop), so they stay synchronized
        even though the generator only trains every dfake_gen_update_ratio steps.

        Supported scheduler_type values:
        - None / "constant": No scheduling (constant LR)
        - "cosine_warmup": Linear warmup then cosine decay to min_lr
        """
        scheduler_type = getattr(self.config, "scheduler_type", None)
        if scheduler_type is None or scheduler_type == "constant":
            return None

        warmup_steps = getattr(self.config, "warmup_steps", 1000)
        max_steps = getattr(self.config, "max_steps", 20000)
        min_lr = getattr(self.config, "min_lr", 1e-7)
        base_lr = optimizer.param_groups[0]["lr"]

        if scheduler_type == "cosine_warmup":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                else:
                    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return max(min_lr / base_lr, cosine_decay)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    def _validate_preinstalled_bidirectional_delegate(self) -> None:
        """No-op: Stage-3 now runs pure-causal without bidirectional delegate."""
        return

    def _wrap_with_fsdp(self):
        """Wrap models with FSDP for distributed training."""
        config = self.config

        self.dmd.generator = fsdp_wrap(
            self.dmd.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.dmd.real_score = fsdp_wrap(
            self.dmd.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )

        self.dmd.fake_score = fsdp_wrap(
            self.dmd.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )

        self.dmd.text_encoder = fsdp_wrap(
            self.dmd.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        )

        # Keep VAEs on CPU to save GPU memory during training.
        # They are only needed for periodic visualization and benchmark decoding.
        # Use _vae_to_device() / _vae_to_cpu() to move them on-demand.
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(dtype=self.dtype)
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(dtype=self.dtype)

        # The LADD head is intentionally replicated (it is small compared with
        # the 19B teacher). Start every rank from rank-0's identical parameters;
        # its gradients are explicitly averaged before the optimizer step.
        if hasattr(self.dmd, "ladd_discriminator") and dist.is_initialized():
            for parameter in self.dmd.ladd_discriminator.parameters():
                dist.broadcast(parameter.data, src=0)
            for buffer in self.dmd.ladd_discriminator.buffers():
                dist.broadcast(buffer.data, src=0)

    def _init_dataloader(self):
        """Initialize data loader."""
        from ltx_distillation.data import collate_prompts, collate_ode_data

        config = self.config

        self.backward_simulation = getattr(config, "backward_simulation", True)
        # Variable-length rollout: each iteration rolls out to the current
        # prompt's own clip length (from the prompt jsonl's num_frames), instead
        # of a fixed config.num_frames. Requires batch_size == 1 so a single
        # length applies to the whole batch, and requires
        # self_forcing_min/max_generated_blocks to be null (so the rollout
        # covers all causal blocks of the sampled length).
        self.variable_length_rollout = bool(getattr(config, "variable_length_rollout", False))
        if self.variable_length_rollout and int(getattr(config, "batch_size", 1)) != 1:
            raise ValueError("variable_length_rollout requires batch_size == 1.")

        if self.use_real_data:
            dataloader = self.real_data_adapter.build_dataloader()
            self.dataloader = cycle(dataloader)
            return
        elif self.use_metadata_prompt_data:
            dataloader = self.metadata_prompt_adapter.build_dataloader()
            self.dataloader = cycle(dataloader)
            return
        elif self.backward_simulation:
            dataset = PromptDataset(config.data_path)
            collate_fn = collate_prompts
        else:
            dataset = ODERegressionLMDBDataset(
                config.data_path,
                max_pair=int(1e8),
            )
            collate_fn = collate_ode_data

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )

        self.dataloader = cycle(dataloader)

    def _init_benchmark_prompts(self):
        """
        Load fixed benchmark prompts from the training prompt file.

        Reads the first ``benchmark_num_prompts`` lines from ``config.data_path``
        so that every benchmark run uses exactly the same prompts for comparison.

        **All ranks** load the prompts because FSDP-wrapped models require all
        ranks to participate in forward passes during benchmark inference.
        """
        config = self.config
        self.benchmark_enabled = getattr(config, "benchmark_enabled", True)
        self.benchmark_iters = int(getattr(config, "benchmark_iters", config.log_iters))
        self.benchmark_seed = getattr(config, "benchmark_seed", 12345)
        self.benchmark_num_prompts = getattr(config, "benchmark_num_prompts", 2)
        self.benchmark_video_fps = getattr(config, "benchmark_video_fps", 24)
        self.benchmark_audio_sample_rate = getattr(config, "benchmark_audio_sample_rate", 24000)
        self.benchmark_mode = str(getattr(config, "benchmark_mode", "bidirectional")).lower()
        if self.benchmark_mode not in {"bidirectional", "causal"}:
            if self.is_main_process:
                print(f"[Benchmark] Invalid benchmark_mode={self.benchmark_mode}, falling back to bidirectional.")
            self.benchmark_mode = "bidirectional"
        self.benchmark_num_frame_per_block = int(getattr(config, "benchmark_num_frame_per_block", getattr(config, "num_frame_per_block", 3)))
        self.benchmark_use_kv_cache = bool(getattr(config, "benchmark_use_kv_cache", True))
        self.benchmark_clear_cuda_cache_per_round = bool(getattr(config, "benchmark_clear_cuda_cache_per_round", True))
        self.benchmark_denoising_step_lists = self._parse_benchmark_denoising_step_lists(
            getattr(config, "benchmark_denoising_step_lists", None),
            num_train_timestep=int(getattr(config, "num_train_timestep", 1000)),
        )
        self.benchmark_prompts = []
        self.benchmark_t2av_groups = []

        if self.benchmark_iters <= 0:
            self.benchmark_enabled = False
            if self.is_main_process:
                print("[Benchmark] Disabled because benchmark_iters <= 0.")

        if not self.benchmark_enabled:
            return

        if self.benchmark_mode == "causal" and not self.benchmark_use_kv_cache:
            raise ValueError(
                "benchmark_mode=causal now requires benchmark_use_kv_cache=true "
                "(prefix-rerun inference path has been retired)."
            )

        # Prompt loading, Stage-2 aligned: support multiple prompt files with an
        # optional per-file cap and t2av groups. Falls back to the singular
        # benchmark_prompt_file / data_path when the plural key is absent.
        try:
            prompt_files = getattr(config, "benchmark_prompt_files", None)
            if prompt_files is None:
                single = getattr(config, "benchmark_prompt_file", None) or getattr(config, "data_path", None)
                prompt_files = [single] if single else []
            elif isinstance(prompt_files, str):
                prompt_files = [prompt_files]
            else:
                prompt_files = list(prompt_files)

            prompts_per_file = getattr(config, "benchmark_prompts_per_file", None)
            if prompts_per_file is not None:
                if isinstance(prompts_per_file, int):
                    prompts_per_file = [prompts_per_file] * len(prompt_files)
                else:
                    prompts_per_file = list(prompts_per_file)
                if len(prompts_per_file) != len(prompt_files):
                    raise ValueError(
                        "benchmark_prompts_per_file must match benchmark_prompt_files length"
                    )

            all_prompts = []
            file_counts = []
            for file_idx, pf in enumerate(prompt_files):
                file_prompts = self._read_prompt_lines(pf)
                if prompts_per_file is not None:
                    limit = int(prompts_per_file[file_idx])
                    if limit > 0:
                        file_prompts = file_prompts[:limit]
                file_counts.append((pf, len(file_prompts)))
                all_prompts.extend(file_prompts)
            self.benchmark_prompts = all_prompts[: self.benchmark_num_prompts]

            if self.is_main_process:
                print(f"[Benchmark] Loaded {len(self.benchmark_prompts)} prompts from {list(prompt_files)}")
                for pf, count in file_counts:
                    print(f"  {pf}: {count} prompt(s)")
                print(f"[Benchmark] mode={self.benchmark_mode}, kv_cache={self.benchmark_use_kv_cache}, frames_per_block={self.benchmark_num_frame_per_block}")
                if self.benchmark_denoising_step_lists:
                    step_counts = [len(schedule) - 1 for schedule in self.benchmark_denoising_step_lists]
                    print(f"[Benchmark] inference step counts (in order): {step_counts}")
                for i, p in enumerate(self.benchmark_prompts):
                    print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")
        except Exception as e:
            if self.is_main_process:
                print(f"[Benchmark] Failed to load prompts: {e}")
            self.benchmark_enabled = False

        if not self.benchmark_prompts:
            self.benchmark_enabled = False
            if self.is_main_process:
                print("[Benchmark] No prompts loaded, benchmark disabled.")
            return

        # t2av groups (Stage-2 aligned): named [start, end) index ranges used for
        # grouped output dirs + wandb logging. Default: one group over all prompts.
        raw_groups = getattr(config, "benchmark_t2av_groups", None) or []
        n = len(self.benchmark_prompts)
        for group in raw_groups:
            name = str(group.get("name", "t2av"))
            start = max(0, min(int(group.get("start", 0)), n))
            end = max(start, min(int(group.get("end", n)), n))
            if end > start:
                self.benchmark_t2av_groups.append({"name": name, "start": start, "end": end})
        if not self.benchmark_t2av_groups:
            self.benchmark_t2av_groups = [{"name": "t2av", "start": 0, "end": n}]
        if self.is_main_process:
            print("[Benchmark] T2AV groups:")
            for group in self.benchmark_t2av_groups:
                print(f"  {group['name']}: [{group['start']}, {group['end']})")

    @staticmethod
    def _read_prompt_lines(path: str) -> list:
        """Read prompts from a .txt (one per line) or .jsonl ({"prompt": ...})."""
        import json as _json

        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        obj = _json.loads(line)
                        p = str(obj.get("prompt", "")).strip()
                        if p:
                            prompts.append(p)
                        continue
                    except _json.JSONDecodeError:
                        pass
                prompts.append(line)
        return prompts

    @staticmethod
    def _parse_benchmark_denoising_step_lists(
        raw_step_lists,
        num_train_timestep: int,
    ) -> list[list[float]]:
        """Validate optional benchmark schedules expressed in training timesteps.

        Each schedule is converted to flow-matching sigmas only when its
        benchmark pass starts. Keeping the raw values here makes startup logs
        and config errors easy to interpret (for example, a 4-step schedule has
        five values including terminal zero).
        """
        if raw_step_lists is None:
            return []

        schedules = [list(schedule) for schedule in raw_step_lists]
        if not schedules:
            raise ValueError("benchmark_denoising_step_lists must not be empty")

        normalized = []
        seen_step_counts = set()
        for schedule in schedules:
            values = [float(value) for value in schedule]
            if len(values) < 2:
                raise ValueError(
                    "each benchmark denoising schedule needs at least one step "
                    "plus terminal zero"
                )
            if abs(values[-1]) > 1e-8:
                raise ValueError(
                    f"benchmark denoising schedule must end at 0: {schedule}"
                )
            if any(value < 0 or value > num_train_timestep for value in values):
                raise ValueError(
                    "benchmark denoising schedule is outside "
                    f"[0, {num_train_timestep}]: {schedule}"
                )
            if any(left <= right for left, right in zip(values, values[1:])):
                raise ValueError(
                    f"benchmark denoising schedule must be strictly descending: {schedule}"
                )

            step_count = len(values) - 1
            if step_count in seen_step_counts:
                raise ValueError(
                    f"duplicate benchmark denoising step count: {step_count}"
                )
            seen_step_counts.add(step_count)
            normalized.append(values)
        return normalized

    def _benchmark_num_frames_for_prompt(self, prompt: str) -> int:
        """Per-prompt rollout length (Stage-2 aligned).

        When ``benchmark_duration_from_prompt_timestamps`` is on, parse the
        largest ``[Xs-Ys]`` timestamp in the prompt, convert to frames at the
        benchmark fps, snap to 8n+1 and clamp to [min,max]. Otherwise return the
        fixed default.
        """
        config = self.config
        default_frames = int(
            getattr(config, "benchmark_num_frames", getattr(config, "num_frames", 121))
        )
        if not bool(getattr(config, "benchmark_duration_from_prompt_timestamps", False)):
            return default_frames

        timestamp_values = [
            float(v)
            for pair in re.findall(r"\[(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s\]", prompt)
            for v in pair
        ]
        if not timestamp_values:
            return default_frames

        fps = float(getattr(config, "benchmark_video_fps", 24))
        raw = max(1, int(math.ceil(max(timestamp_values) * fps)))
        # snap up to 8n+1
        num_frames = raw + ((8 - ((raw - 1) % 8)) % 8)
        min_frames = int(getattr(config, "benchmark_min_frames", default_frames))
        num_frames = max(num_frames, min_frames)
        max_frames = getattr(config, "benchmark_max_frames", None)
        if max_frames is not None:
            num_frames = min(num_frames, int(max_frames))
            # snap down to 8n+1
            num_frames = num_frames - ((num_frames - 1) % 8)
        return int(num_frames)

    def _benchmark_t2av_group_name(self, prompt_idx: int) -> str:
        """Return the t2av group name owning ``prompt_idx`` (Stage-2 aligned)."""
        for group in self.benchmark_t2av_groups:
            if group["start"] <= prompt_idx < group["end"]:
                return group["name"]
        return "t2av"

    def _vae_to_device(self):
        """Move VAEs to GPU for decoding (visualization / benchmark)."""
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(device=self.device)
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(device=self.device)

    def _vae_to_cpu(self):
        """Offload VAEs back to CPU to free GPU memory."""
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(device="cpu")
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    def save(self):
        """Save checkpoint."""
        print("Gathering distributed model states...")

        generator_state_dict = fsdp_state_dict(self.dmd.generator)
        critic_state_dict = fsdp_state_dict(self.dmd.fake_score)

        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict,
            "step": self.step,
        }
        if self.ladd_discriminator_optimizer is not None:
            state_dict["ladd_discriminator"] = (
                self.dmd.ladd_discriminator.state_dict()
            )
            state_dict["ladd_discriminator_optimizer"] = (
                self.ladd_discriminator_optimizer.state_dict()
            )

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path,
                f"checkpoint_{self.step:06d}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            save_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, save_path)
            print(f"Checkpoint saved to {save_path}")

    @staticmethod
    def _snap_ltx_frames(num_frames: int) -> int:
        """Snap a pixel-frame count down to the nearest valid LTX 8n+1 value."""
        num_frames = max(1, int(num_frames))
        return num_frames - ((num_frames - 1) % 8)

    @staticmethod
    def _to_scalar(value):
        """Convert tensor-like values to Python scalars for WandB logging."""
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().float().mean().item()
        return value

    def _compute_layerwise_grad_norms(self, module, prefix):
        """
        Compute per-layer gradient L2 norm for monitoring.

        Aggregation strategy:
        - For transformer blocks, log at block granularity: blocks.{idx}
        - For others, log at up-to-2-level module granularity.
        """
        layer_sq_norm = {}
        fsdp_prefix = "_fsdp_wrapped_module."

        for name, param in module.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            normalized_name = name[len(fsdp_prefix):] if name.startswith(fsdp_prefix) else name
            parts = normalized_name.split(".")
            if len(parts) >= 3 and parts[1] == "blocks" and parts[2].isdigit():
                layer_key = f"blocks.{parts[2]}"
            elif len(parts) >= 2:
                layer_key = f"{parts[0]}.{parts[1]}"
            else:
                layer_key = parts[0]

            grad_sq = param.grad.detach().float().pow(2).sum().item()
            layer_sq_norm[layer_key] = layer_sq_norm.get(layer_key, 0.0) + grad_sq

        return {
            f"train/{prefix}_grad_norm/{k}": math.sqrt(v) for k, v in layer_sq_norm.items()
        }

    def train_one_step(self):
        """Execute one training step."""
        # Set all models to eval mode first (disables dropout/batchnorm),
        # then re-enable train mode for generator and fake_score so that
        # gradient checkpointing remains active during their gradient-enabled
        # forward passes. This is critical for the 19B model's memory footprint.
        # The real_score (teacher) stays in eval mode since it's frozen.
        #
        # For backward simulation's @torch.no_grad() forward passes, the
        # generator is temporarily switched to eval() inside
        # _consistency_backward_simulation() to avoid FSDP+checkpoint conflicts.
        self.dmd.eval()
        self.dmd.generator.train()
        self.dmd.fake_score.train()
        if hasattr(self.dmd, "ladd_discriminator"):
            self.dmd.ladd_discriminator.train()

        # Pass current step to DMD for step-dependent loss weighting
        self.dmd.current_step = self.step

        config = self.config
        TRAIN_GENERATOR = self.step % config.dfake_gen_update_ratio == 0
        LOG_LAYERWISE_GRAD = self.step % self.layerwise_grad_log_interval == 0

        # Periodic cache clearing
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Get batch
        if self.use_real_data:
            batch = next(self.dataloader)
            clean_video, clean_audio, text_prompts = (
                self.real_data_adapter.encode_batch(batch)
            )
            clean_audio = self.real_data_adapter.align_audio(
                clean_video, clean_audio
            )
            batch_num_frames = [
                1 + 8 * (int(clean_video.shape[1]) - 1)
            ] * len(text_prompts)
        elif not self.backward_simulation:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            # ODE latent format: [B, T, F, C, H, W], take last timestep (clean)
            clean_video = batch["ode_latent"][:, -1].to(
                device=self.device,
                dtype=self.dtype,
            )
            # Audio ODE latent format: [B, T, F_a, C], take last timestep (clean)
            if "ode_audio_latent" in batch and batch["ode_audio_latent"] is not None:
                clean_audio = batch["ode_audio_latent"][:, -1].to(
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                clean_audio = None
        else:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            batch_num_frames = batch.get("num_frames", [None] * len(text_prompts))
            clean_video = None
            clean_audio = None

        batch_size = len(text_prompts)

        # Per-iteration rollout length. Default: fixed config.num_frames.
        # With variable_length_rollout, use the current prompt's clip length so
        # the self-forcing rollout matches the Stage-2 [min,max] frame range.
        real_num_frames = int(config.num_frames)
        if (
            (self.backward_simulation or self.use_real_data)
            and self.variable_length_rollout
            and batch_num_frames
            and batch_num_frames[0] is not None
        ):
            real_num_frames = self._snap_ltx_frames(int(batch_num_frames[0]))

        # ---- pad_max: cross-rank rollout-length synchronization ----------------
        # Self-forcing rolls out block-by-block; #blocks depends on the sequence
        # length. If ranks used different lengths, they would issue a different
        # number of per-block FSDP all-gathers and deadlock. So every rank rolls
        # out to the GLOBAL MAX length (padded blocks run forward/backward to keep
        # collectives aligned) while the DMD loss is restricted to each rank's own
        # real_num_frames (padded frames contribute zero gradient; see
        # generator_loss real_video_frames/real_audio_frames).
        rollout_num_frames = real_num_frames
        if self.variable_length_rollout and dist.is_initialized() and self.world_size > 1:
            _lmax = torch.tensor([real_num_frames], device=self.device, dtype=torch.long)
            dist.all_reduce(_lmax, op=dist.ReduceOp.MAX)
            rollout_num_frames = int(_lmax.item())

        # Compute latent shapes (rollout uses the padded/global-max length).
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=rollout_num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=batch_size,
        )
        if self.dmd.generator_use_causal_wrapper:
            # ``compute_latent_shapes`` derives audio length from continuous
            # duration, while causal AV attention partitions both modalities
            # into discrete shared blocks. Align the full rollout too (not only
            # a shorter pad_max crop), otherwise a max-length rank can retain a
            # one-frame audio tail that no causal mask owns.
            audio_shape[1] = causal_aligned_audio_frame_count(
                int(video_shape[1]),
                num_frame_per_block=int(
                    getattr(config, "num_frame_per_block", 3)
                ),
                first_block_video_frames=int(
                    getattr(config, "first_block_video_frames", 1)
                ),
            )
        # Real (unpadded) latent frame counts for this rank; drive the DMD loss
        # mask so padded frames/blocks do not contribute to the generator update.
        real_video_frames = None
        real_audio_frames = None
        if self.use_real_data:
            real_video_frames = int(clean_video.shape[1])
            real_audio_frames = int(clean_audio.shape[1])
            video_shape[-2] = int(clean_video.shape[-2])
            video_shape[-1] = int(clean_video.shape[-1])
        elif rollout_num_frames != real_num_frames:
            real_video_shape, _ = compute_latent_shapes(
                num_frames=real_num_frames,
                video_height=config.video_height,
                video_width=config.video_width,
                batch_size=batch_size,
            )
            real_video_frames = int(real_video_shape[1])
            # The generic duration-based latent shape can differ by one audio
            # frame from the causal AV block geometry (for example 18 video
            # frames -> generic 143, causal-aligned 142). Generator and AR
            # score masks share the causal block partition, so crop pad_max
            # audio using that exact boundary. This is also the rule used by
            # Stage 2 and variable-length Stage-3 benchmark decoding.
            real_audio_frames = causal_aligned_audio_frame_count(
                real_video_frames,
                num_frame_per_block=int(
                    getattr(config, "num_frame_per_block", 3)
                ),
                first_block_video_frames=int(
                    getattr(config, "first_block_video_frames", 1)
                ),
            )

        # Encode text
        with torch.no_grad():
            conditional_dict = self.dmd.text_encoder(text_prompts=text_prompts)

            if not hasattr(self, "unconditional_dict"):
                unconditional_dict = self.dmd.text_encoder(
                    text_prompts=[config.negative_prompt] * batch_size
                )
                unconditional_dict = {
                    k: v.detach() for k, v in unconditional_dict.items()
                }
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Train generator
        if TRAIN_GENERATOR:
            self.generator_optimizer.zero_grad()
            generator_loss, generator_log_dict = self.dmd.generator_loss(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )
            if generator_loss.requires_grad:
                generator_loss.backward()
            generator_layerwise_grad_dict = (
                self._compute_layerwise_grad_norms(self.dmd.generator, "generator")
                if LOG_LAYERWISE_GRAD else {}
            )
            # Use FSDP's clip_grad_norm_ if available, otherwise fall back to torch utility
            if hasattr(self.dmd.generator, 'clip_grad_norm_'):
                generator_grad_norm = self.dmd.generator.clip_grad_norm_(self.max_grad_norm)
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.dmd.generator.parameters(), self.max_grad_norm
                )
            self.generator_optimizer.step()

            # ---- Memory cleanup between generator and critic training ----
            # Save scalar metrics before freeing the computation graph.
            # This is critical because step 0 first allocates Adam optimizer
            # states (momentum + variance ≈ 2× param size), and the remaining
            # graph/activation memory must be released before critic training.
            generator_loss_val = generator_loss.item()
            generator_grad_norm_val = generator_grad_norm.item()
            gen_grad_norm_video = generator_log_dict.get("dmdtrain_gradient_norm_video", 0)
            gen_grad_norm_audio = generator_log_dict.get("dmdtrain_gradient_norm_audio", 0)

            del generator_loss, generator_grad_norm
            torch.cuda.empty_cache()
        else:
            generator_log_dict = {}
            generator_loss_val = None
            generator_grad_norm_val = None
            gen_grad_norm_video = 0
            gen_grad_norm_audio = 0
            generator_layerwise_grad_dict = {}

        # Train critic
        self.critic_optimizer.zero_grad()
        if self.ladd_discriminator_optimizer is not None:
            self.ladd_discriminator_optimizer.zero_grad()

        critic_loss, critic_log_dict = self.dmd.critic_loss(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_video=clean_video,
            clean_audio=clean_audio,
            real_video_frames=real_video_frames,
            real_audio_frames=real_audio_frames,
        )

        critic_loss.backward()
        critic_layerwise_grad_dict = (
            self._compute_layerwise_grad_norms(self.dmd.fake_score, "critic")
            if LOG_LAYERWISE_GRAD else {}
        )
        # Use FSDP's clip_grad_norm_ if available, otherwise fall back to torch utility
        if hasattr(self.dmd.fake_score, 'clip_grad_norm_'):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(self.max_grad_norm)
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        ladd_discriminator_grad_norm = None
        ladd_discriminator_active = bool(
            self.ladd_discriminator_optimizer is not None
            and self.step
            >= int(getattr(config, "ladd_discriminator_start_step", 0))
            and self.step
            % int(getattr(config, "ladd_discriminator_update_ratio", 1))
            == 0
        )
        if ladd_discriminator_active:
            if dist.is_initialized() and self.world_size > 1:
                for parameter in self.dmd.ladd_discriminator.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(self.world_size)
            ladd_discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.ladd_discriminator.parameters(),
                float(getattr(config, "ladd_max_grad_norm_discriminator", 1.0)),
            )
            self.ladd_discriminator_optimizer.step()

        # Benchmark: periodic 4-step inference visualization
        # ALL ranks must participate because FSDP forward passes require
        # collective communication across all ranks.
        BENCHMARK = (
            self.benchmark_enabled
            and len(self.benchmark_prompts) > 0
            and self.step % self.benchmark_iters == 0
            and not getattr(config, "no_visualize", False)
        )

        if BENCHMARK:
            self._run_benchmark_and_log()

        # Logging (all scalars, no GPU tensors)
        if self.is_main_process:
            wandb_dict = {
                "train/critic_loss": critic_loss.item(),
                "train/critic_grad_norm": critic_grad_norm.item(),
            }

            # Add per-component critic losses from log_dict
            wandb_dict.update({
                f"train/{k}": self._to_scalar(v) for k, v in critic_log_dict.items()
            })
            wandb_dict.update(critic_layerwise_grad_dict)

            if TRAIN_GENERATOR:
                wandb_dict.update({
                    "train/generator_loss": generator_loss_val,
                    "train/generator_grad_norm": generator_grad_norm_val,
                    "train/dmdtrain_gradient_norm_video": gen_grad_norm_video,
                    "train/dmdtrain_gradient_norm_audio": gen_grad_norm_audio,
                })
                wandb_dict.update(generator_layerwise_grad_dict)
                for gk, gv in generator_log_dict.items():
                    wandb_dict[f"train/{gk}"] = self._to_scalar(gv)

            wandb_dict["train/lr_generator"] = self.generator_optimizer.param_groups[0]["lr"]
            wandb_dict["train/lr_critic"] = self.critic_optimizer.param_groups[0]["lr"]
            if self.ladd_discriminator_optimizer is not None:
                wandb_dict["train/lr_ladd_discriminator"] = (
                    self.ladd_discriminator_optimizer.param_groups[0]["lr"]
                )
            if ladd_discriminator_grad_norm is not None:
                wandb_dict["train/ladd_discriminator_grad_norm"] = (
                    ladd_discriminator_grad_norm.item()
                )

            wandb.log(wandb_dict, step=self.step)

        del critic_loss, critic_grad_norm
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_benchmark_and_log(self):
        """Run configured benchmark schedules sequentially on identical seeds."""
        num_train_timestep = int(getattr(self.config, "num_train_timestep", 1000))
        if self.benchmark_denoising_step_lists:
            schedule_specs = []
            for raw_schedule in self.benchmark_denoising_step_lists:
                step_count = len(raw_schedule) - 1
                denoising_sigmas = torch.tensor(
                    [value / float(num_train_timestep) for value in raw_schedule],
                    device=self.device,
                    dtype=torch.float32,
                )
                schedule_specs.append((f"{step_count}_steps", denoising_sigmas))
        else:
            step_count = len(self.dmd.denoising_sigmas) - 1
            schedule_specs = [(f"{step_count}_steps", self.dmd.denoising_sigmas)]

        namespace_outputs = len(schedule_specs) > 1
        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: schedules="
                f"{[tag for tag, _ in schedule_specs]} (run in this order)",
                flush=True,
            )

        for schedule_tag, denoising_sigmas in schedule_specs:
            self._run_benchmark_schedule_and_log(
                schedule_tag=schedule_tag,
                denoising_sigmas=denoising_sigmas,
                namespace_outputs=namespace_outputs,
            )

    @torch.no_grad()
    def _run_benchmark_schedule_and_log(
        self,
        *,
        schedule_tag: str,
        denoising_sigmas: torch.Tensor,
        namespace_outputs: bool,
    ):
        """
        Run one inference schedule on fixed benchmark prompts, distributing
        work across all ranks for maximum parallelism.

        **All ranks** must call this method because the generator and text
        encoder are FSDP-wrapped and require collective communication.

        Flow (per round, one prompt per rank):
        1. ALL ranks: encode 1 prompt each (FSDP collective, batch_size=1)
        2. ALL ranks: run inference pipeline (FSDP collective, batch_size=1)
        3. ALL ranks: decode video/audio with local VAE, save mp4 to shared FS
        4. Rank 0: collect all saved files, log to WandB

        This distributes N prompts across W ranks in ceil(N/W) rounds,
        reducing per-rank memory vs the old single-rank-decodes-all approach.

        RNG is forked per prompt for reproducibility without affecting training.
        """
        from ltx_distillation.inference.bidirectional_pipeline import (
            BidirectionalAVInferencePipeline,
        )
        from ltx_distillation.inference.causal_pipeline import (
            CausalAVInferencePipeline,
        )
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        config = self.config

        # Free training intermediate memory before benchmark
        torch.cuda.empty_cache()

        num_prompts = len(self.benchmark_prompts)
        num_rounds = math.ceil(num_prompts / self.world_size)

        if self.is_main_process:
            print(
                f"[Benchmark][{schedule_tag}] Step {self.step}: generating "
                f"{num_prompts} samples "
                f"({self.benchmark_mode} mode) across {self.world_size} ranks "
                f"({num_rounds} round(s))..."
            )

        step_dir = os.path.join(
            self.output_path, "benchmark", f"step_{self.step:07d}"
        )
        if namespace_outputs:
            step_dir = os.path.join(step_dir, schedule_tag)
        os.makedirs(step_dir, exist_ok=True)

        # Per-prompt rollout length (Stage-2 aligned; parsed from prompt timestamps
        # when benchmark_duration_from_prompt_timestamps is on, else fixed).
        prompt_num_frames = [
            self._benchmark_num_frames_for_prompt(p) for p in self.benchmark_prompts
        ]
        max_prompt_num_frames = (
            max(prompt_num_frames) if prompt_num_frames else int(config.num_frames)
        )
        # pad_max for benchmark: every rank generates at the GLOBAL-MAX length so
        # all ranks run the same number of causal blocks (aligned FSDP collectives).
        # Each sample is then truncated to its own prompt length before decode;
        # causal generation is prefix-stable so the truncated content matches
        # generating directly at the real length.
        max_video_shape, max_audio_shape = compute_latent_shapes(
            num_frames=max_prompt_num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=1,
        )
        variable_bench = (
            self.benchmark_mode == "causal"
            and bool(getattr(config, "benchmark_duration_from_prompt_timestamps", False))
        )

        # Grouped output dirs (Stage-2 aligned).
        group_dirs = {
            group["name"]: os.path.join(step_dir, group["name"])
            for group in self.benchmark_t2av_groups
        }
        for gd in group_dirs.values():
            os.makedirs(gd, exist_ok=True)

        if self.is_main_process:
            print(
                f"[Benchmark][{schedule_tag}] T2AV rollout: "
                f"max_frames={max_prompt_num_frames} (latent {max_video_shape[1]}), "
                f"variable={variable_bench}, "
                f"sigmas={denoising_sigmas.detach().cpu().tolist()}",
                flush=True,
            )

        # Keep Stage 3 benchmark aligned with the Stage-2 ODE benchmark:
        # temporarily switch the FSDP-wrapped generator to eval() under no_grad,
        # then restore the previous mode afterwards.
        was_training = self.dmd.generator.training
        self.dmd.generator.eval()
        try:
            if self.benchmark_mode == "causal":
                pipeline = CausalAVInferencePipeline(
                    generator=self.dmd.generator,
                    add_noise_fn=self.dmd.add_noise,
                    denoising_sigmas=denoising_sigmas,
                    num_frame_per_block=self.benchmark_num_frame_per_block,
                    use_kv_cache=self.benchmark_use_kv_cache,
                    clear_cuda_cache_per_round=self.benchmark_clear_cuda_cache_per_round,
                    first_block_video_frames=getattr(self.dmd, 'first_block_video_frames', 1),
                    context_noise=getattr(self.dmd, "context_noise", 0),
                    num_train_timestep=getattr(self.config, "num_train_timestep", 1000),
                )
            else:
                pipeline = BidirectionalAVInferencePipeline(
                    generator=self.dmd.generator,
                    add_noise_fn=self.dmd.add_noise,
                    denoising_sigmas=denoising_sigmas,
                )

            self._vae_to_device()

            # Timing: wall-clock for full benchmark, and per-video generation time
            benchmark_wall_start = time.perf_counter()
            my_total_generate_seconds = 0.0

            for round_idx in range(num_rounds):
                prompt_idx = round_idx * self.world_size + self.global_rank
                has_real_prompt = prompt_idx < num_prompts

                actual_idx = prompt_idx if has_real_prompt else 0
                my_prompt = [self.benchmark_prompts[actual_idx]]

                with torch.no_grad():
                    conditional_dict = self.dmd.text_encoder(text_prompts=my_prompt)

                prompt_seed = self.benchmark_seed + prompt_idx
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(prompt_seed)
                    torch.cuda.manual_seed(prompt_seed)

                    gen_start = time.perf_counter()
                    video_latent, audio_latent = pipeline.generate(
                        video_shape=tuple(max_video_shape),
                        audio_shape=tuple(max_audio_shape),
                        conditional_dict=conditional_dict,
                    )
                    gen_elapsed = time.perf_counter() - gen_start
                    my_total_generate_seconds += gen_elapsed

                if has_real_prompt:
                    v_out, a_out = video_latent, audio_latent
                    if variable_bench:
                        real_video_shape, _ = compute_latent_shapes(
                            num_frames=prompt_num_frames[actual_idx],
                            video_height=config.video_height,
                            video_width=config.video_width,
                            batch_size=1,
                        )
                        rvf = int(real_video_shape[1])
                        v_out = video_latent[:, :rvf]
                        if audio_latent is not None:
                            raf = compute_aligned_audio_frames(
                                total_video_latent_frames=rvf,
                                num_frame_per_block=self.benchmark_num_frame_per_block,
                                first_block_video_frames=getattr(self.dmd, "first_block_video_frames", 1),
                            )
                            a_out = audio_latent[:, :raf]
                    group_name = self._benchmark_t2av_group_name(prompt_idx)
                    self._decode_and_save_sample(
                        video_latent=v_out,
                        audio_latent=a_out,
                        prompt_idx=prompt_idx,
                        step_dir=group_dirs.get(group_name, step_dir),
                    )

                del video_latent, audio_latent, conditional_dict
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

                barrier()
        finally:
            if was_training:
                self.dmd.generator.train()

        benchmark_wall_elapsed = time.perf_counter() - benchmark_wall_start

        # Gather total generation time from all ranks (each rank sums its own generate times)
        total_generate_tensor = torch.tensor(
            [my_total_generate_seconds], device=self.device, dtype=torch.float64
        )
        if dist.is_initialized():
            dist.all_reduce(total_generate_tensor, op=dist.ReduceOp.SUM)
        total_generate_seconds = total_generate_tensor.item()

        self._vae_to_cpu()

        barrier()

        # ---- Rank 0: log all samples to WandB and print benchmark timing ----
        if self.is_main_process:
            time_per_video_wall = benchmark_wall_elapsed / max(1, num_prompts)
            time_per_video_generate = total_generate_seconds / max(1, num_prompts)

            benchmark_wandb_dict = {}
            prompt_rows = []
            wandb_prefix = (
                f"benchmark/{schedule_tag}" if namespace_outputs else "benchmark"
            )

            for idx in range(num_prompts):
                group_name = self._benchmark_t2av_group_name(idx)
                sample_path = os.path.join(
                    group_dirs.get(group_name, step_dir), f"sample_{idx}.mp4"
                )
                if os.path.exists(sample_path):
                    benchmark_wandb_dict[
                        f"{wandb_prefix}/{group_name}/sample_{idx}"
                    ] = wandb.Video(
                        sample_path, fps=self.benchmark_video_fps, format="mp4"
                    )
                    prompt_rows.append(
                        [idx, group_name, prompt_num_frames[idx],
                         self.benchmark_prompts[idx], sample_path]
                    )

            if prompt_rows:
                table = wandb.Table(
                    columns=["index", "group", "num_frames", "prompt", "local_path"],
                    data=prompt_rows,
                )
                benchmark_wandb_dict[f"{wandb_prefix}/prompt_table"] = table

            benchmark_wandb_dict[f"{wandb_prefix}/wall_seconds"] = benchmark_wall_elapsed
            benchmark_wandb_dict[
                f"{wandb_prefix}/generate_seconds"
            ] = total_generate_seconds
            benchmark_wandb_dict[
                f"{wandb_prefix}/wall_seconds_per_video"
            ] = time_per_video_wall
            benchmark_wandb_dict[
                f"{wandb_prefix}/generate_seconds_per_video"
            ] = time_per_video_generate

            if benchmark_wandb_dict:
                # Both schedules share one global training step. Accumulate
                # their disjoint keys into that row; the regular train log at
                # the end of train_one_step commits it.
                wandb.log(benchmark_wandb_dict, step=self.step, commit=False)

            # One line: timing + save path (flush so it always appears in logs)
            print(
                f"[Benchmark][{schedule_tag}] Step {self.step}: "
                f"{num_prompts} video(s) | "
                f"wall {benchmark_wall_elapsed:.2f}s ({time_per_video_wall:.2f}s/video) | "
                f"generate {total_generate_seconds:.2f}s ({time_per_video_generate:.2f}s/video) | "
                f"saved to {step_dir}",
                flush=True,
            )

        barrier()

    def _decode_and_save_sample(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        prompt_idx: int,
        step_dir: str,
    ):
        """
        Decode one (video, audio) latent pair and save as mp4 with audio.

        Called by every rank that owns a real benchmark prompt.  VAEs must
        already be on GPU (via ``_vae_to_device``) before calling this.
        """
        # Decode video → pixel  [1, C, F, H, W]  →  [0, 1]
        video_pixel = self.dmd.video_vae.decode_to_pixel(video_latent)

        # Decode audio → waveform  [1, 1, samples]
        audio_waveform = None
        try:
            audio_waveform = self.dmd.audio_vae.decode_to_waveform(audio_latent)
        except Exception as e:
            print(
                f"[Benchmark][Rank {self.global_rank}] Audio decode failed "
                f"for prompt {prompt_idx}: {e}"
            )

        # Prepare video tensor: -> uint8 [F, H, W, C]
        vid = video_pixel[0]  # [C, F, H, W]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)  # -> [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)  # -> [F, H, W, C]
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")

        # Write mp4 via the torchvision-version-agnostic helper (falls back to
        # PyAV when torchvision.io.write_video is unavailable), matching Stage 2.
        written_with_audio = False
        if audio_waveform is not None:
            try:
                wav_float = audio_waveform[0].cpu().float()  # [1, samples]
                _write_video_compat(
                    sample_path,
                    vid,
                    fps=self.benchmark_video_fps,
                    audio_array=wav_float,
                    audio_fps=self.benchmark_audio_sample_rate,
                    audio_codec="aac",
                )
                written_with_audio = True
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video with "
                    f"audio failed for prompt {prompt_idx}; falling back to "
                    f"video-only mp4: {e}"
                )

        # Fallback: silent video + separate wav
        if not written_with_audio:
            try:
                _write_video_compat(sample_path, vid, fps=self.benchmark_video_fps)
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video (silent) "
                    f"failed for prompt {prompt_idx}: {e}"
                )
                return

            if audio_waveform is not None:
                try:
                    import torchaudio

                    wav = audio_waveform[0].cpu().float()
                    wav_path = os.path.join(
                        step_dir, f"sample_{prompt_idx}.wav"
                    )
                    torchaudio.save(
                        wav_path, wav, self.benchmark_audio_sample_rate
                    )
                except Exception as e:
                    print(
                        f"[Benchmark][Rank {self.global_rank}] torchaudio.save "
                        f"failed for prompt {prompt_idx}: {e}"
                    )

        # Free decoded tensors
        del video_pixel, audio_waveform
        torch.cuda.empty_cache()

    def train(self):
        """Main training loop."""
        while True:
            self.train_one_step()

            # Save checkpoint
            if (
                not getattr(self.config, "no_save", False)
                and self.log_iters > 0
                and self.step % self.log_iters == 0
            ):
                self.save()
                torch.cuda.empty_cache()

            barrier()

            # Timing
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None:
                    wandb.log(
                        {"per_iteration_time": current_time - self.previous_time},
                        step=self.step,
                    )
                self.previous_time = current_time

            self.step += 1

            # Step LR schedulers based on global step (both stay synchronized)
            if self.generator_scheduler is not None:
                self.generator_scheduler.step(self.step)
            if self.critic_scheduler is not None:
                self.critic_scheduler.step(self.step)

            # Optional: max steps limit
            max_steps = getattr(self.config, "max_steps", None)
            if max_steps and self.step >= max_steps:
                break

        # FSDP full-state gathering is collective, so every rank must enter
        # save(); only rank 0 performs the filesystem write inside save().
        if not getattr(self.config, "no_save", False):
            self.save()
        barrier()
        if self.is_main_process:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = _load_config_with_base(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
