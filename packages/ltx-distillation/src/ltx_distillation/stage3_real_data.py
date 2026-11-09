"""Metadata-only and real-data adapters for Stage-3 rollout training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


class Stage3MetadataPromptAdapter:
    """Frame-range metadata sampling without video/audio download or VAE encoding."""

    def __init__(self, config, is_main_process: bool) -> None:
        self.config = config
        self.is_main_process = is_main_process

    @staticmethod
    def dataset_package_path() -> Path:
        return Path(__file__).resolve().parents[3] / "dataset"

    def build_dataloader(self):
        dataset_path = str(self.dataset_package_path())
        if dataset_path not in sys.path:
            sys.path.insert(0, dataset_path)
        from frame_range_dataset import FrameRangeVideoAudioDataset
        from ltx_distillation.data import collate_metadata_prompts

        config = self.config
        if str(getattr(config, "dataset_type", "frame_range")) != "frame_range":
            raise ValueError(
                "Metadata prompt-only mode requires dataset_type=frame_range"
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
            prompt_only=True,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )
        if self.is_main_process:
            print(
                "[Stage3] Metadata-only prompt loader enabled; "
                "video/audio download and VAE encoding are disabled.",
                flush=True,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_metadata_prompts,
            num_workers=getattr(config, "num_workers", 2),
            pin_memory=False,
        )


class Stage3RealDataAdapter:
    def __init__(self, config, device, dtype, is_main_process: bool) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.is_main_process = is_main_process
        self._build_vae_encoders()

    @staticmethod
    def dataset_package_path() -> Path:
        return Path(__file__).resolve().parents[3] / "dataset"

    def _build_vae_encoders(self) -> None:
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_core.loader.single_gpu_model_builder import (
            SingleGPUModelBuilder as Builder,
        )
        from ltx_core.model.audio_vae import (
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            AudioEncoderConfigurator,
            AudioProcessor,
        )
        from ltx_pipelines.utils.model_ledger import ModelLedger

        if self.is_main_process:
            print("[Stage3] Building real-data VAE encoders...", flush=True)
        registry = StateDictRegistry()
        ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=self.config.checkpoint_path,
            registry=registry,
        )
        self.video_encoder = ledger.video_encoder().eval()
        audio_builder = Builder(
            model_path=self.config.checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.audio_encoder = audio_builder.build(
            device=torch.device("cpu"), dtype=self.dtype
        ).eval()
        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins,
            mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
        )

    def build_dataloader(self):
        dataset_path = str(self.dataset_package_path())
        if dataset_path not in sys.path:
            sys.path.insert(0, dataset_path)
        from frame_range_dataset import FrameRangeVideoAudioDataset
        from video_audio_dataset import collate_real_data

        config = self.config
        if str(getattr(config, "dataset_type", "frame_range")) != "frame_range":
            raise ValueError("Stage3 real-data mode supports dataset_type=frame_range")
        if int(config.batch_size) != 1:
            raise ValueError("frame_range real-data training requires batch_size=1")

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
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_real_data,
            num_workers=getattr(config, "num_workers", 2),
            pin_memory=True,
        )

    def _encoders_to_device(self) -> None:
        self.video_encoder = self.video_encoder.to(
            device=self.device, dtype=self.dtype
        )
        self.audio_encoder = self.audio_encoder.to(
            device=self.device, dtype=self.dtype
        )
        self.audio_processor = self.audio_processor.to(device=self.device)

    def _encoders_to_cpu(self) -> None:
        self.video_encoder = self.video_encoder.to(device="cpu")
        self.audio_encoder = self.audio_encoder.to(device="cpu")
        self.audio_processor = self.audio_processor.to(device="cpu")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_batch(self, batch: dict):
        self._encoders_to_device()
        try:
            video = batch["video"].to(device=self.device, dtype=self.dtype)
            waveform = batch["audio_waveform"].to(
                device=self.device, dtype=self.dtype
            )
            video_latent = self.video_encoder(video)
            video_latent = video_latent.permute(0, 2, 1, 3, 4).contiguous()

            mel = self.audio_processor.waveform_to_mel(
                waveform, waveform_sample_rate=self.audio_encoder.sample_rate
            )
            audio_latent = self.audio_encoder(mel)
            batch_size, channels, frames, mel_bins = audio_latent.shape
            audio_latent = audio_latent.permute(0, 2, 1, 3).reshape(
                batch_size, frames, channels * mel_bins
            )
            return video_latent, audio_latent, batch["captions"]
        finally:
            self._encoders_to_cpu()

    def align_audio(
        self, clean_video: torch.Tensor, clean_audio: torch.Tensor
    ) -> torch.Tensor:
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        expected = compute_aligned_audio_frames(
            clean_video.shape[1],
            self.config.get("num_frame_per_block", 3),
            self.config.get("first_block_video_frames", 4),
        )
        actual = clean_audio.shape[1]
        if actual > expected:
            return clean_audio[:, :expected]
        if actual < expected:
            return torch.nn.functional.pad(clean_audio, (0, 0, 0, expected - actual))
        return clean_audio
