"""Causal benchmark inference pipeline for LTX-2 AV generation.

Two causal-only runtime paths are supported:
True KV-cache runtime only (use_kv_cache must be true): O(N) blockwise
generation matching Stage-3 self-forcing KV-cache training semantics.
No bidirectional delegate is used.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
    compute_av_blocks,
)


class CausalAVInferencePipeline:
    """Causal AV autoregressive benchmark pipeline."""

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        num_frame_per_block: int = 3,
        use_kv_cache: bool = False,
        clear_cuda_cache_per_round: bool = True,
        first_block_video_frames: int = 1,
        context_noise: int = 0,
        num_train_timestep: int = 1000,
    ):
        if denoising_sigmas.ndim != 1 or denoising_sigmas.numel() < 2:
            raise ValueError(
                "denoising_sigmas must be a 1D tensor with at least 2 entries"
            )

        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.first_block_video_frames = max(1, int(first_block_video_frames))
        if not bool(use_kv_cache):
            raise ValueError(
                "CausalAVInferencePipeline now requires use_kv_cache=true. "
                "Legacy prefix-rerun inference is retired."
            )
        self.clear_cuda_cache_per_round = bool(clear_cuda_cache_per_round)
        self.context_noise = int(context_noise)
        self.num_train_timestep = int(num_train_timestep)
        if self.num_train_timestep <= 0:
            raise ValueError("num_train_timestep must be > 0")
        if not (0 <= self.context_noise <= self.num_train_timestep):
            raise ValueError(
                f"context_noise must be in [0, {self.num_train_timestep}], got {self.context_noise}"
            )

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    @staticmethod
    def _zeros_sigma(
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, frames), device=device, dtype=dtype)

    @staticmethod
    def _full_sigma(
        sigma: torch.Tensor,
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sigma_value = sigma.to(device=device, dtype=dtype)
        return sigma_value.expand(batch_size, frames)

    def _renoise_block(self, clean_block: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
        if clean_block is None:
            return None

        batch_size = clean_block.shape[0]
        num_frames = clean_block.shape[1]
        sigma = self._full_sigma(
            next_sigma,
            batch_size=batch_size,
            frames=num_frames,
            device=clean_block.device,
            dtype=clean_block.dtype,
        )
        return self.add_noise_fn(
            clean_block,
            torch.randn_like(clean_block),
            sigma,
        )

    def _init_kv_caches(
        self,
        *,
        batch_size: int,
        total_video_frames: int,
        total_audio_frames: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        video_frame_seqlen: Optional[int] = None,
    ):
        wrapper = self._unwrap_module(self.generator)
        init_fn = getattr(wrapper, "init_av_kv_caches", None)
        if init_fn is None:
            raise RuntimeError(
                "Generator wrapper does not expose init_av_kv_caches; "
                "please use CausalLTX2DiffusionWrapper with KV-cache support."
            )
        return init_fn(
            batch_size=batch_size,
            num_video_frames=total_video_frames,
            num_audio_frames=total_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
            video_frame_seqlen=video_frame_seqlen,
        )

    def _refresh_kv_cache_block(
        self,
        *,
        current_video: torch.Tensor,
        current_audio: Optional[torch.Tensor],
        kv_caches,
        block,
        conditional_dict: Dict[str, Any],
        device: torch.device,
        sigma_dtype: torch.dtype,
        ref_video_block: Optional[torch.Tensor] = None,
    ) -> None:
        """Always run one no_grad refresh forward after each block."""
        batch_size = current_video.shape[0]
        ctx_t = float(self.context_noise) / float(self.num_train_timestep)
        video_ctx_sigma = torch.full(
            (batch_size, current_video.shape[1]),
            ctx_t,
            device=device,
            dtype=sigma_dtype,
        )
        audio_ctx_sigma = None
        if current_audio is not None:
            audio_ctx_sigma = torch.full(
                (batch_size, current_audio.shape[1]),
                ctx_t,
                device=device,
                dtype=sigma_dtype,
            )

        if ctx_t > 0.0:
            refresh_video = self.add_noise_fn(
                current_video.detach(),
                torch.randn_like(current_video),
                video_ctx_sigma.to(dtype=current_video.dtype),
            )
            refresh_audio = None
            if current_audio is not None:
                refresh_audio = self.add_noise_fn(
                    current_audio.detach(),
                    torch.randn_like(current_audio),
                    audio_ctx_sigma.to(dtype=current_audio.dtype),
                )
        else:
            refresh_video = current_video.detach()
            refresh_audio = current_audio.detach() if current_audio is not None else None

        self.generator(
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

    def _generate_with_kv_cache(
        self,
        *,
        video: torch.Tensor,
        audio: Optional[torch.Tensor],
        video_shape: Tuple[int, ...],
        audio_shape: Optional[Tuple[int, ...]],
        blocks,
        conditional_dict: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from ltx_causal.wrapper import build_block_ref_from_prev

        batch_size = video_shape[0]
        sigma_dtype = self.denoising_sigmas.dtype
        total_video_frames = video_shape[1]
        total_audio_frames = audio_shape[1] if audio_shape is not None else 0

        text_ctx = conditional_dict.get("video_context")
        if text_ctx is None:
            text_ctx = conditional_dict.get("context")
        if text_ctx is None:
            raise ValueError("conditional_dict must provide video_context or context")
        text_seq_len = text_ctx.shape[1]

        # AAPT-style chunk recycling flag (cached from the wrapper).
        gen = self.generator
        while hasattr(gen, "module"):
            gen = gen.module
        enable_chunk_recycling = bool(
            getattr(gen, "enable_chunk_recycling", False)
        )

        kv_caches = self._init_kv_caches(
            batch_size=batch_size,
            total_video_frames=total_video_frames,
            total_audio_frames=total_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
            video_frame_seqlen=int(video_shape[-2]) * int(video_shape[-1]),
        )

        # Track previous chunk's final latent for ref construction.
        prev_chunk_latent: Optional[torch.Tensor] = None

        for block in blocks:
            if enable_chunk_recycling:
                ref_video_block = build_block_ref_from_prev(
                    prev_chunk_latent=prev_chunk_latent,
                    current_block_size=block.video_frames,
                    template=video,
                )
            else:
                ref_video_block = None

            current_video = torch.randn(
                (batch_size, block.video_frames, *video_shape[2:]),
                device=device,
                dtype=dtype,
            )
            current_audio = None
            if audio is not None:
                current_audio = torch.randn(
                    (batch_size, block.audio_frames, audio_shape[2]),
                    device=device,
                    dtype=dtype,
                )

            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                video_sigma = self._full_sigma(
                    sigma,
                    batch_size=batch_size,
                    frames=current_video.shape[1],
                    device=device,
                    dtype=sigma_dtype,
                )
                audio_sigma = None
                if current_audio is not None:
                    audio_sigma = self._full_sigma(
                        sigma,
                        batch_size=batch_size,
                        frames=current_audio.shape[1],
                        device=device,
                        dtype=sigma_dtype,
                    )

                pred_video, pred_audio = self.generator(
                    _kv_cache_kwargs=dict(
                        noisy_video_block=current_video,
                        noisy_audio_block=current_audio,
                        conditional_dict=conditional_dict,
                        timestep_block=video_sigma,
                        audio_timestep_block=audio_sigma,
                        kv_caches=kv_caches,
                        current_video_start_frame=block.video_start,
                        current_audio_start_frame=block.audio_start,
                        ref_video_block=ref_video_block,
                    )
                )

                next_sigma = self.denoising_sigmas[sigma_idx + 1]
                if float(next_sigma.item()) > 0.0:
                    current_video = self._renoise_block(pred_video, next_sigma)
                    if current_audio is not None and pred_audio is not None:
                        current_audio = self._renoise_block(pred_audio, next_sigma)
                else:
                    current_video = pred_video
                    if current_audio is not None:
                        if pred_audio is None:
                            raise RuntimeError(
                                "Generator returned no audio prediction for audio benchmark inference"
                            )
                        current_audio = pred_audio

                if self.clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

            self._refresh_kv_cache_block(
                current_video=current_video,
                current_audio=current_audio,
                kv_caches=kv_caches,
                block=block,
                conditional_dict=conditional_dict,
                device=device,
                sigma_dtype=sigma_dtype,
                ref_video_block=ref_video_block,
            )

            video[:, block.video_start:block.video_end] = current_video
            if audio is not None and current_audio is not None:
                audio[:, block.audio_start:block.audio_end] = current_audio

            # Update prev_chunk_latent for the NEXT block's ref construction.
            prev_chunk_latent = current_video.detach()

        return video, audio

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Optional[Tuple[int, ...]],
        conditional_dict: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if len(video_shape) != 5:
            raise ValueError(f"Expected video_shape=[B,F,C,H,W], got {video_shape}")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        device, dtype = self._module_device_dtype(self.generator)

        batch_size = video_shape[0]
        total_video_frames = video_shape[1]
        blocks = compute_av_blocks(
            total_video_latent_frames=total_video_frames,
            num_frame_per_block=self.num_frame_per_block,
            first_block_video_frames=self.first_block_video_frames,
        )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = None

        if audio_shape is not None:
            if len(audio_shape) != 3:
                raise ValueError(f"Expected audio_shape=[B,F,C], got {audio_shape}")
            expected_audio_frames = compute_aligned_audio_frames(
                total_video_latent_frames=total_video_frames,
                num_frame_per_block=self.num_frame_per_block,
                first_block_video_frames=self.first_block_video_frames,
            )
            if audio_shape[1] != expected_audio_frames:
                raise ValueError(
                    "audio_shape does not match causal block alignment: "
                    f"got F_a={audio_shape[1]}, expected {expected_audio_frames}"
                )
            audio = torch.zeros(audio_shape, device=device, dtype=dtype)

        return self._generate_with_kv_cache(
            video=video,
            audio=audio,
            video_shape=video_shape,
            audio_shape=audio_shape,
            blocks=blocks,
            conditional_dict=conditional_dict,
            device=device,
            dtype=dtype,
        )
