"""
AR Teacher Training Module with Teacher Forcing (Method A).

Trains a causal autoregressive diffusion model on real data using
2x-sequence teacher forcing, initializing from a bidirectional LTX-2
checkpoint. The resulting AR teacher is used to generate ODE pairs for
subsequent causal ODE distillation (Stage 2).

Training Process:
1. Load clean video + audio latents (online VAE encoding)
2. Sample independent per-block noise levels
3. Build noisy latents via flow matching interpolation
4. Forward pass with 2x sequence [clean | noisy] and teacher forcing masks
5. Compute flow matching MSE loss on velocity predictions

Reference: Causal Forcing (https://arxiv.org/abs/2602.02214)
"""

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ltx_causal.attention.mask_builder import compute_av_blocks, compute_aligned_audio_frames


@dataclass
class ARTeacherConfig:
    """Configuration for AR Teacher training."""

    # Model paths
    checkpoint_path: str = ""
    bidirectional_model_checkpoint: Optional[str] = None
    causal_model_checkpoint: Optional[str] = None
    text_encoder_checkpoint: str = ""

    # Causal generation settings
    num_frame_per_block: int = 3
    first_block_video_frames: int = 1

    # Model settings
    gradient_checkpointing: bool = True
    mixed_precision: bool = True

    # Noise schedule
    num_train_timestep: int = 1000
    sigma_shift: float = 1.0       # Causal Forcing uses 3.0 (Wan default); 1.0 = no shift
    sigma_min: float = 0.0         # Causal Forcing uses ~0.003; 0.0 = no clamp
    sigma_max: float = 1.0         # Causal Forcing uses ~1.0
    # Sampler options: "uniform_shifted" (legacy) or "shifted_logit_normal"
    timestep_sampler_type: str = "uniform_shifted"
    timestep_sampler_std: float = 1.0
    timestep_sampler_uniform_prob: float = 0.1
    timestep_sampler_eps: float = 1e-3

    # Loss weighting
    use_gaussian_weight: bool = False  # Causal Forcing uses Gaussian centered at mid-sigma
    denoising_loss_type: str = "x0"  # "x0" (legacy) or native "velocity"
    velocity_loss_scale: float = 1.0

    # Optional small noise on clean context (0 = perfectly clean)
    noise_augmentation_max_sigma: float = 0.0

    # Loss weights
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 0.16

    # Number of learnable sink tokens prepended to audio
    num_audio_sink_tokens: int = 0

    # Log-ratio rescaling (off by default for teacher forcing)
    enable_causal_log_rescale: bool = False

    # AAPT-style channel-wise concat of the previous chunk as ref signal.
    # Adds a zero-init Linear (ref_patchify_proj) inside CausalLTXModel whose
    # output is added to patchify_proj's output. Off by default so existing
    # checkpoints load unchanged.
    enable_chunk_recycling: bool = False

    # TI2AV mix: probability of setting Block 0 sigma=0 (image conditioning)
    # 0.0 = pure T2AV, 0.5 = 50/50 T2AV/TI2AV, 1.0 = pure TI2AV
    first_frame_conditioning_p: float = 0.0


class LTX2ARTeacher(nn.Module):
    """
    AR Teacher module using Teacher Forcing (Method A: 2x sequence).

    Trains a causal LTX-2 model to predict velocity from noisy inputs
    while conditioning on clean ground-truth preceding blocks via the
    2x-sequence teacher forcing mechanism.
    """

    # Reuse state dict key prefixes from ODE regression for filtering
    _NON_TRANSFORMER_PREFIXES = (
        "vae.", "audio_vae.", "vocoder.",
        "model.vae.", "model.audio_vae.", "model.vocoder.",
    )
    _NON_TRANSFORMER_REMAPPED_PREFIXES = (
        "model.audio_embeddings_connector.",
        "model.video_embeddings_connector.",
    )

    def __init__(
        self,
        config: ARTeacherConfig,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        if config.denoising_loss_type not in {"x0", "velocity"}:
            raise ValueError(
                "denoising_loss_type must be 'x0' or 'velocity', got "
                f"{config.denoising_loss_type!r}"
            )
        if config.velocity_loss_scale <= 0:
            raise ValueError(
                "velocity_loss_scale must be positive, got "
                f"{config.velocity_loss_scale}"
            )

        self.num_frame_per_block = config.num_frame_per_block
        self.first_block_video_frames = config.first_block_video_frames
        self._is_main = not dist.is_initialized() or dist.get_rank() == 0

        # Lazy model loading
        self._generator = None
        self._text_encoder = None

    def _print(self, *args, **kwargs):
        if self._is_main:
            print(*args, **kwargs)

    def _make_model_config(self):
        from ltx_causal import CausalLTXModelConfig
        return CausalLTXModelConfig(
            num_frame_per_block=self.config.num_frame_per_block,
            first_block_video_frames=self.config.first_block_video_frames,
            enable_causal_log_rescale=self.config.enable_causal_log_rescale,
            num_audio_sink_tokens=self.config.num_audio_sink_tokens,
            enable_chunk_recycling=self.config.enable_chunk_recycling,
        )

    def _make_wrapper(self, model):
        from ltx_causal import CausalLTX2DiffusionWrapper
        return CausalLTX2DiffusionWrapper(
            model=model,
            num_frame_per_block=self.config.num_frame_per_block,
            disable_causal_mask=False,
            num_audio_sink_tokens=self.config.num_audio_sink_tokens,
            first_block_video_frames=self.config.first_block_video_frames,
        )

    def _load_models(self):
        """Load generator and text encoder models."""
        if self._generator is not None:
            return

        self._print("Loading models for AR Teacher training...")

        from ltx_causal import (
            CausalLTX2DiffusionWrapper,
            CausalLTXModel,
            CausalLTXModelConfig,
        )
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        gemma_path = self.config.text_encoder_checkpoint

        if not checkpoint_path:
            raise ValueError("ARTeacherConfig requires 'checkpoint_path' for text encoder loading")

        self._text_encoder = create_text_encoder_wrapper(
            checkpoint_path=checkpoint_path,
            gemma_path=gemma_path,
            device=self.device,
            dtype=self.dtype,
        )
        self._text_encoder.requires_grad_(False)
        self._text_encoder.eval()

        if self.config.causal_model_checkpoint:
            self._print(f"Loading causal model from: {self.config.causal_model_checkpoint}")
            self._generator = CausalLTX2DiffusionWrapper.from_pretrained(
                self.config.causal_model_checkpoint,
                num_frame_per_block=self.config.num_frame_per_block,
                first_block_video_frames=self.config.first_block_video_frames,
                enable_causal_log_rescale=self.config.enable_causal_log_rescale,
                num_audio_sink_tokens=self.config.num_audio_sink_tokens,
                enable_chunk_recycling=self.config.enable_chunk_recycling,
            ).to(self.device)

        elif self.config.bidirectional_model_checkpoint:
            self._print(f"Initializing from bidirectional model: {self.config.bidirectional_model_checkpoint}")

            model = CausalLTXModel(self._make_model_config())
            self._generator = self._make_wrapper(model).to(self.device, self.dtype)

            bidirectional_state_dict = self._load_checkpoint_state_dict(
                self.config.bidirectional_model_checkpoint
            )
            state_dict = self._remap_state_dict_keys(bidirectional_state_dict)

            missing, unexpected = self._generator.load_state_dict(state_dict, strict=False)
            self._print(f"Loaded bidirectional weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            real_missing = [k for k in missing if 'mask_builder' not in k and 'causal_gate' not in k and 'audio_sink_tokens' not in k]
            if real_missing:
                self._print(f"  Missing keys: {real_missing[:10]}...")
            if unexpected:
                self._print(f"  Unexpected keys: {unexpected[:10]}...")

        elif self.config.checkpoint_path:
            self._print(f"Initializing from original LTX-2 checkpoint: {self.config.checkpoint_path}")

            model = CausalLTXModel(self._make_model_config())
            self._generator = self._make_wrapper(model).to(self.device, self.dtype)

            state_dict = self._load_checkpoint_state_dict(self.config.checkpoint_path)
            state_dict = self._remap_state_dict_keys(state_dict)

            missing, unexpected = self._generator.load_state_dict(state_dict, strict=False)
            self._print(f"Loaded original LTX-2 weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            real_missing = [k for k in missing if 'mask_builder' not in k and 'causal_gate' not in k and 'audio_sink_tokens' not in k]
            if real_missing:
                self._print(f"  Missing keys: {real_missing[:10]}...")
        else:
            raise ValueError(
                "ARTeacherConfig requires one of: causal_model_checkpoint, "
                "bidirectional_model_checkpoint, or checkpoint_path."
            )

        self._generator.requires_grad_(True)
        if self.config.gradient_checkpointing:
            self._generator.enable_gradient_checkpointing()

        self._print("Models loaded for AR Teacher training.")

    @staticmethod
    def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(checkpoint_path)
        else:
            loaded = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(loaded, dict) and 'generator' in loaded:
                return loaded['generator']
            elif isinstance(loaded, dict) and 'model' in loaded:
                return loaded['model']
            elif isinstance(loaded, dict) and 'state_dict' in loaded:
                return loaded['state_dict']
            return loaded

    @classmethod
    def _remap_state_dict_keys(cls, state_dict: dict) -> dict:
        """Remap state dict keys to CausalLTX2DiffusionWrapper format.

        Handles four source formats (same logic as ODE regression):
        1. Raw LTXModel keys → add "model." prefix
        2. Bidirectional wrapper → remap "model.velocity_model." to "model."
        3. Causal wrapper → no change
        4. Full checkpoint → filter to "model.diffusion_model.*" and remap
        """
        if not state_dict:
            return state_dict

        sample_keys = list(state_dict.keys())[:20]
        has_diffusion_model = any(
            k.startswith("model.diffusion_model.") for k in sample_keys
        )
        if not has_diffusion_model:
            has_diffusion_model = any(
                k.startswith("model.diffusion_model.") for k in state_dict
            )

        if has_diffusion_model:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("model.diffusion_model."):
                    new_key = "model." + k[len("model.diffusion_model."):]
                    if any(new_key.startswith(p) for p in cls._NON_TRANSFORMER_REMAPPED_PREFIXES):
                        continue
                    remapped[new_key] = v
            return remapped

        first_key = next(iter(state_dict))

        if first_key.startswith("model.velocity_model."):
            return {
                "model." + k[len("model.velocity_model."):]: v
                for k, v in state_dict.items()
                if k.startswith("model.velocity_model.")
            }
        elif first_key.startswith("model."):
            has_non_transformer = any(
                k.startswith(p) for k in state_dict
                for p in cls._NON_TRANSFORMER_PREFIXES
            )
            if has_non_transformer:
                return {
                    k: v for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in cls._NON_TRANSFORMER_PREFIXES)
                }
            return state_dict
        else:
            return {
                "model." + k: v for k, v in state_dict.items()
                if not any(k.startswith(p) for p in cls._NON_TRANSFORMER_PREFIXES)
            }

    @property
    def generator(self) -> nn.Module:
        self._load_models()
        return self._generator

    @property
    def text_encoder(self) -> nn.Module:
        self._load_models()
        return self._text_encoder

    def enable_gradient_checkpointing(self) -> None:
        self.generator.enable_gradient_checkpointing()

    def _apply_sigma_shift(self, sigma: torch.Tensor) -> torch.Tensor:
        """Apply Causal-Forcing-style sigma shift: s*σ / (1 + (s-1)*σ).

        When shift=1.0, this is identity. When shift>1, pushes sigma towards
        higher values (more noise), matching Wan's training schedule.
        """
        s = self.config.sigma_shift
        if s == 1.0:
            return sigma
        return s * sigma / (1.0 + (s - 1.0) * sigma)

    def _compute_gaussian_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Gaussian loss weight centered at sigma=0.5, matching Causal Forcing.

        w(σ) = exp(-2 * ((σ - 0.5) / 1.0)²)
        Emphasizes intermediate noise levels, down-weights extremes.
        """
        return torch.exp(-2.0 * ((sigma - 0.5) ** 2))

    def _sample_shifted_logit_normal_sigma(
        self,
        batch_size: int,
        device: torch.device,
        seq_length: int,
    ) -> torch.Tensor:
        """Shifted Logit-Normal sigma sampler aligned with official LTX-2 trainer.

        The shift (mu) is derived from seq_length via linear interpolation and
        embedded as the mean of the logit-normal. No additional sigma_shift needed.
        """
        std = float(self.config.timestep_sampler_std)
        uniform_prob = float(self.config.timestep_sampler_uniform_prob)
        eps = float(self.config.timestep_sampler_eps)

        # Linear interpolation for mu (same formula as official trainer)
        mu = 0.95 + (2.05 - 0.95) / (4096 - 1024) * (float(seq_length) - 1024)

        # Sample from shifted logit-normal
        normal_samples = torch.randn(batch_size, 1, device=device) * std + mu
        logitnormal_samples = torch.sigmoid(normal_samples)

        # Percentile stretching to [0,1]
        p_hi = torch.sigmoid(torch.tensor(mu + 3.0902 * std, device=device))
        p_lo = torch.sigmoid(torch.tensor(mu - 2.5758 * std, device=device))
        stretched = ((logitnormal_samples - p_lo) / (p_hi - p_lo).clamp(min=1e-8)).clamp(0.0, 1.0)

        # Reflect below eps for numerical stability
        stretched = torch.where(stretched >= eps, stretched, 2.0 * eps - stretched).clamp(0.0, 1.0)

        # Uniform fallback
        uniform = (1.0 - eps) * torch.rand(batch_size, 1, device=device) + eps
        prob = torch.rand(batch_size, 1, device=device)
        return torch.where(prob > uniform_prob, stretched, uniform)

    def _infer_video_seq_length(self, num_video_frames: int) -> int:
        """Infer token sequence length for timestep shift fit."""
        try:
            module = self.generator
            while hasattr(module, "module"):
                module = module.module
            frame_seqlen = int(module.model.config.video_frame_seqlen)
            return int(num_video_frames * frame_seqlen)
        except Exception:
            # Conservative fallback: keep schedule functional if internals differ.
            return int(num_video_frames * 384)

    def _sample_per_block_sigma(
        self,
        batch_size: int,
        num_video_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample independent sigma per block, uniform within each block.

        Returns:
            [B, F_v] sigma values, block-uniform
        """
        blocks = compute_av_blocks(
            num_video_frames, self.num_frame_per_block, self.first_block_video_frames,
        )

        sigma = torch.zeros(batch_size, num_video_frames, device=device)
        sampler_type = str(getattr(self.config, "timestep_sampler_type", "uniform_shifted"))
        seq_length = self._infer_video_seq_length(num_video_frames)

        # TI2AV mix: per-sample decision whether Block 0 is conditioned (sigma=0)
        p = self.config.first_frame_conditioning_p
        if p > 0:
            block0_cond_mask = (torch.rand(batch_size, 1, device=device) < p)
        else:
            block0_cond_mask = None

        for block in blocks:
            if sampler_type == "shifted_logit_normal":
                block_sigma = self._sample_shifted_logit_normal_sigma(
                    batch_size=batch_size, device=device, seq_length=seq_length,
                )
            else:
                raw = torch.rand(batch_size, 1, device=device)
                raw = self.config.sigma_min + (self.config.sigma_max - self.config.sigma_min) * raw
                block_sigma = self._apply_sigma_shift(raw)

            if block.is_global_prefix and block0_cond_mask is not None:
                block_sigma = torch.where(block0_cond_mask, torch.zeros_like(block_sigma), block_sigma)

            sigma[:, block.video_start:block.video_end] = block_sigma.expand(
                -1, block.video_end - block.video_start
            )
        return sigma

    def _expand_video_sigma_to_audio(
        self,
        video_sigma: torch.Tensor,
        num_audio_frames: int,
    ) -> torch.Tensor:
        """Expand per-block video sigma to aligned audio frames.

        Each audio frame uses the sigma of its corresponding video block.
        """
        B, F_v = video_sigma.shape
        blocks = compute_av_blocks(F_v, self.num_frame_per_block, self.first_block_video_frames)

        audio_sigma = torch.zeros(B, num_audio_frames, device=video_sigma.device)
        for block in blocks:
            if block.audio_end <= block.audio_start:
                continue
            # Use first video frame's sigma for this block's audio
            block_sigma = video_sigma[:, block.video_start:block.video_start + 1]
            audio_sigma[:, block.audio_start:block.audio_end] = block_sigma.expand(
                -1, block.audio_end - block.audio_start
            )
        return audio_sigma

    def _add_noise(
        self,
        clean: torch.Tensor,
        sigma: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flow-matching noise injection: x_t = (1 - sigma) * x_0 + sigma * noise."""
        if noise is None:
            noise = torch.randn_like(clean)
        sigma_bc = sigma.view(sigma.shape[0], sigma.shape[1], *([1] * (clean.ndim - 2)))
        sigma_bc = sigma_bc.to(dtype=clean.dtype)
        return ((1 - sigma_bc) * clean + sigma_bc * noise).to(dtype=clean.dtype)

    def compute_denoising_losses(
        self,
        *,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized Gaussian-weighted x0 or velocity MSE.

        The Gaussian formula and its normalization are deliberately identical
        for both parameterizations so an x0-vs-velocity experiment changes only
        the prediction target. ``velocity_loss_scale`` is applied later to the
        combined video/audio base loss, not to either modality independently.
        """
        video_mask = video_sigma > 1e-6
        audio_mask = audio_sigma > 1e-6

        if self.config.denoising_loss_type == "velocity":
            if (
                pred_velocity_video is None
                or pred_velocity_audio is None
                or video_noise is None
                or audio_noise is None
            ):
                raise ValueError(
                    "velocity denoising loss requires predicted velocities and "
                    "the exact noises used to construct x_t"
                )
            video_target = video_noise.float() - clean_video.float()
            audio_target = audio_noise.float() - clean_audio.float()
            per_frame_mse = (
                pred_velocity_video.float() - video_target
            ).pow(2).mean(dim=(2, 3, 4))
            per_frame_mse_a = (
                pred_velocity_audio.float() - audio_target
            ).pow(2).mean(dim=2)
        else:
            # Preserve the legacy x0-loss arithmetic for existing configs.
            per_frame_mse = (
                pred_x0_video - clean_video
            ).pow(2).mean(dim=(2, 3, 4))
            per_frame_mse_a = (
                pred_x0_audio - clean_audio
            ).pow(2).mean(dim=2)

        if video_mask.any():
            if self.config.use_gaussian_weight:
                w = self._compute_gaussian_weight(video_sigma)
                weighted_mask = w * video_mask.float()
                video_loss = (
                    per_frame_mse * weighted_mask
                ).sum() / weighted_mask.sum().clamp(min=1e-8)
            else:
                video_loss = per_frame_mse[video_mask].mean()
        else:
            video_loss = torch.zeros(
                (), device=clean_video.device, dtype=per_frame_mse.dtype
            )

        if audio_mask.any():
            if self.config.use_gaussian_weight:
                w_a = self._compute_gaussian_weight(audio_sigma)
                weighted_mask_a = w_a * audio_mask.float()
                audio_loss = (
                    per_frame_mse_a * weighted_mask_a
                ).sum() / weighted_mask_a.sum().clamp(min=1e-8)
            else:
                audio_loss = per_frame_mse_a[audio_mask].mean()
        else:
            audio_loss = torch.zeros(
                (), device=clean_audio.device, dtype=per_frame_mse_a.dtype
            )

        return video_loss, audio_loss

    def generator_loss(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute flow matching loss with teacher forcing.

        Args:
            clean_video: [B, F_v, C, H, W] clean video latents
            clean_audio: [B, F_a, C] clean audio latents
            conditional_dict: Text conditioning

        Returns:
            (loss, log_dict)
        """
        B, F_v = clean_video.shape[:2]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        # Sample per-block sigma
        video_sigma = self._sample_per_block_sigma(B, F_v, device)  # [B, F_v]
        audio_sigma = self._expand_video_sigma_to_audio(video_sigma, F_a)  # [B, F_a]

        # Create noisy latents while retaining the exact flow-matching targets.
        video_noise = torch.randn_like(clean_video)
        audio_noise = torch.randn_like(clean_audio)
        noisy_video = self._add_noise(clean_video, video_sigma, noise=video_noise)
        noisy_audio = self._add_noise(clean_audio, audio_sigma, noise=audio_noise)

        # Clean context sigma (0 or small noise augmentation)
        clean_sigma_val = self.config.noise_augmentation_max_sigma
        if clean_sigma_val > 0:
            clean_video_sigma = torch.rand(B, F_v, device=device) * clean_sigma_val
            clean_audio_sigma = self._expand_video_sigma_to_audio(clean_video_sigma, F_a)
            clean_video_input = self._add_noise(clean_video, clean_video_sigma)
            clean_audio_input = self._add_noise(clean_audio, clean_audio_sigma)
        else:
            clean_video_sigma = torch.zeros(B, F_v, device=device)
            clean_audio_sigma = torch.zeros(B, F_a, device=device)
            clean_video_input = clean_video
            clean_audio_input = clean_audio

        # Route through forward() so FSDP hooks fire correctly
        generator_output = self.generator(
            _tf_kwargs=dict(
                noisy_video=noisy_video,
                clean_video=clean_video_input,
                conditional_dict=conditional_dict,
                noisy_timestep=video_sigma.to(self.dtype),
                clean_timestep=clean_video_sigma.to(self.dtype),
                noisy_audio=noisy_audio,
                clean_audio=clean_audio_input,
                noisy_audio_timestep=audio_sigma.to(self.dtype),
                clean_audio_timestep=clean_audio_sigma.to(self.dtype),
                use_causal_timestep=False,
                return_velocity=self.config.denoising_loss_type == "velocity",
            )
        )
        if self.config.denoising_loss_type == "velocity":
            (
                pred_x0_video,
                pred_x0_audio,
                pred_velocity_video,
                pred_velocity_audio,
            ) = generator_output
        else:
            pred_x0_video, pred_x0_audio = generator_output
            pred_velocity_video = pred_velocity_audio = None

        video_loss, audio_loss = self.compute_denoising_losses(
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

        base_loss_unscaled = (
            self.config.video_loss_weight * video_loss
            + self.config.audio_loss_weight * audio_loss
        )
        denoising_loss_scale = (
            self.config.velocity_loss_scale
            if self.config.denoising_loss_type == "velocity"
            else 1.0
        )
        loss = denoising_loss_scale * base_loss_unscaled

        log_dict = {
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "base_loss_unscaled": base_loss_unscaled.detach(),
            "video_sigma_mean": video_sigma.mean().detach(),
            "video_sigma_max": video_sigma.max().detach(),
            "video_sigma": video_sigma.detach(),
            "pred_x0_video": pred_x0_video.detach(),
        }

        return loss, log_dict
