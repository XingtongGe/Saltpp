"""Stage-2 DMD with GT-noised current chunks and generated-prefix context.

The bidirectional real/fake score models and critic objective are inherited
unchanged from :mod:`ltx_distillation.train_stage2`.  Only the causal generator
path changes:

* pass 1 rolls out chunk-by-chunk under ``torch.no_grad``;
* each current chunk is a noised real-data chunk;
* prior chunks in the KV cache are detached one-step generator x0 predictions;
* bidirectional real/fake scores compute a full-clip DMD gradient;
* pass 2 replays each chunk with grad and backpropagates its DMD-gradient slice.

Pass-2 cache refreshes always use pass-1 outputs.  Consequently the replay is
conditioned on the exact same detached generated prefix and never builds a
cross-chunk autograd graph.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from ltx_distillation.stage35_media import decode_video_tiled, write_video_pyav
from ltx_distillation.stage2_self_context_utils import replay_dmd_surrogate_loss
from ltx_distillation.train_stage2 import Stage2Trainer


class Stage2SelfContextDMDTrainer(Stage2Trainer):
    """Two-pass self-context generator training with bidirectional DMD scores."""

    training_mode = "self_context_tf_dmd"

    def __init__(self, config: DictConfig):
        configured_mode = str(config.get("training_mode", self.training_mode))
        if configured_mode != self.training_mode:
            raise ValueError(
                f"{type(self).__name__} requires training_mode={self.training_mode}, "
                f"got {configured_mode}"
            )
        if not bool(config.get("self_context_replay_enabled", True)):
            raise ValueError("self-context Stage 2 currently requires replay")
        if not bool(config.get("same_step_across_blocks", True)):
            raise ValueError(
                "self-context Stage 2 requires same_step_across_blocks=true"
            )

        super().__init__(config)
        self.replay_check_tolerance = float(
            config.get("self_context_replay_check_tolerance", 5e-3)
        )
        self.strict_replay_check = bool(
            config.get("self_context_strict_replay_check", False)
        )
        self.local_visualization_iters = int(
            config.get("local_visualization_iters", 0)
        )
        self.local_video_fps = int(
            config.get(
                "local_video_fps",
                config.get("benchmark_video_fps", 24),
            )
        )
        if (
            self.local_visualization_iters > 0
            and self.is_main_process
            and not hasattr(self, "video_vae")
        ):
            # Benchmark setup normally constructs the CPU VAE wrappers. Keep
            # local training logs usable even when benchmark_enabled=false.
            self._build_vae_decoders()

        if self.is_main_process:
            print(
                "[Stage2 self-context DMD] ready | "
                "generator=GT-noised chunk + generated-prefix KV cache | "
                "real/fake=bidirectional | two_pass_replay=true | "
                f"SVI={self.svi_enabled} | "
                f"local_video_every={self.local_visualization_iters}"
            )

    # ------------------------------------------------------------------ #
    # Generator self-context rollout helpers                              #
    # ------------------------------------------------------------------ #
    def _causal_blocks(self, num_video_frames: int):
        from ltx_causal.attention.mask_builder import compute_av_blocks

        return compute_av_blocks(
            total_video_latent_frames=num_video_frames,
            num_frame_per_block=int(self.config.get("num_frame_per_block", 3)),
            first_block_video_frames=int(
                self.config.get("first_block_video_frames", 4)
            ),
        )

    def _synchronized_rollout_blocks(self, real_video_frames: int):
        """Append dummy full blocks so every rank issues the same FSDP calls.

        We deliberately preserve the local final partial block instead of
        recomputing blocks after padding.  Otherwise padding tokens would enter
        the same intra-block attention window as real tokens on shorter ranks.
        """
        from ltx_causal.attention.mask_builder import (
            AUDIO_FRAMES_PER_BLOCK,
            AVBlock,
        )

        blocks = list(self._causal_blocks(real_video_frames))
        local_count = len(blocks)
        max_count = local_count
        if dist.is_initialized() and self.world_size > 1:
            count_tensor = torch.tensor(
                [local_count], device=self.device, dtype=torch.long
            )
            dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
            max_count = int(count_tensor.item())

        frames_per_block = int(self.config.get("num_frame_per_block", 3))
        while len(blocks) < max_count:
            previous = blocks[-1]
            blocks.append(
                AVBlock(
                    block_idx=len(blocks),
                    video_start=previous.video_end,
                    video_end=previous.video_end + frames_per_block,
                    audio_start=previous.audio_end,
                    audio_end=previous.audio_end + AUDIO_FRAMES_PER_BLOCK,
                )
            )
        return blocks, local_count

    @staticmethod
    def _pad_frames(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
        if tensor.shape[1] > target_frames:
            raise ValueError(
                f"cannot pad {tensor.shape[1]} frames to shorter target {target_frames}"
            )
        if tensor.shape[1] == target_frames:
            return tensor
        padded = tensor.new_zeros(
            (tensor.shape[0], target_frames, *tensor.shape[2:])
        )
        padded[:, : tensor.shape[1]] = tensor
        return padded

    def _generator_kv_block_forward(
        self,
        *,
        noisy_video_block: torch.Tensor,
        noisy_audio_block: Optional[torch.Tensor],
        video_sigma_block: torch.Tensor,
        audio_sigma_block: Optional[torch.Tensor],
        conditional_dict: Dict[str, Any],
        kv_caches,
        block,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.ar_teacher._generator(
            _kv_cache_kwargs=dict(
                noisy_video_block=noisy_video_block,
                noisy_audio_block=noisy_audio_block,
                conditional_dict=conditional_dict,
                timestep_block=video_sigma_block.to(self.dtype),
                audio_timestep_block=(
                    audio_sigma_block.to(self.dtype)
                    if audio_sigma_block is not None
                    else None
                ),
                kv_caches=kv_caches,
                current_video_start_frame=block.video_start,
                current_audio_start_frame=block.audio_start,
                ref_video_block=None,
            )
        )

    @staticmethod
    def _context_residual(
        context_source: torch.Tensor,
        clean_source: torch.Tensor,
        *,
        enabled: bool,
    ) -> Optional[torch.Tensor]:
        if not enabled:
            return None
        return (context_source - clean_source).detach()

    @staticmethod
    def _add_context_residual(
        prediction: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        start: int,
        end: int,
    ) -> Optional[torch.Tensor]:
        if prediction is None:
            return None
        prediction = prediction.detach()
        if residual is None:
            return prediction
        return prediction + residual[:, start:end].to(
            device=prediction.device, dtype=prediction.dtype
        )

    def _refresh_generated_context(
        self,
        *,
        pass1_video_block: torch.Tensor,
        pass1_audio_block: Optional[torch.Tensor],
        video_context_residual: Optional[torch.Tensor],
        audio_context_residual: Optional[torch.Tensor],
        conditional_dict: Dict[str, Any],
        kv_caches,
        block,
    ) -> None:
        """Write detached pass-1 x0 (plus optional SVI error) as clean context."""
        context_video = self._add_context_residual(
            pass1_video_block,
            video_context_residual,
            block.video_start,
            block.video_end,
        )
        context_audio = self._add_context_residual(
            pass1_audio_block,
            audio_context_residual,
            block.audio_start,
            block.audio_end,
        )
        with torch.no_grad():
            self._refresh_kv_cache_block(
                current_video=context_video,
                current_audio=context_audio,
                conditional_dict=conditional_dict,
                kv_caches=kv_caches,
                block=block,
            )

    def _build_noisy_gt_inputs(
        self,
        *,
        generator_video_source: torch.Tensor,
        generator_audio_source: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Video/audio noise is independent, while sigma is shared across every
        # block and both modalities for each batch item.
        noisy_video = self.ar_teacher._add_noise(
            generator_video_source,
            video_sigma,
            noise=torch.randn_like(generator_video_source),
        )
        noisy_audio = self.ar_teacher._add_noise(
            generator_audio_source,
            audio_sigma,
            noise=torch.randn_like(generator_audio_source),
        )
        return noisy_video, noisy_audio

    @torch.no_grad()
    def _run_detached_self_context_rollout(
        self,
        *,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        video_context_residual: Optional[torch.Tensor],
        audio_context_residual: Optional[torch.Tensor],
        blocks: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        kv_caches = self._init_kv_caches_for_conditional(
            conditional_dict=conditional_dict,
            batch_size=noisy_video.shape[0],
            total_video_frames=noisy_video.shape[1],
            total_audio_frames=noisy_audio.shape[1],
            device=noisy_video.device,
            dtype=noisy_video.dtype,
            video_frame_seqlen=int(noisy_video.shape[-2]) * int(noisy_video.shape[-1]),
        )
        rollout_video = torch.empty_like(noisy_video)
        rollout_audio = torch.empty_like(noisy_audio)

        for block in blocks:
            pred_video, pred_audio = self._generator_kv_block_forward(
                noisy_video_block=noisy_video[:, block.video_start : block.video_end],
                noisy_audio_block=noisy_audio[:, block.audio_start : block.audio_end],
                video_sigma_block=video_sigma[:, block.video_start : block.video_end],
                audio_sigma_block=audio_sigma[:, block.audio_start : block.audio_end],
                conditional_dict=conditional_dict,
                kv_caches=kv_caches,
                block=block,
            )
            rollout_video[:, block.video_start : block.video_end] = pred_video.detach()
            rollout_audio[:, block.audio_start : block.audio_end] = pred_audio.detach()
            self._refresh_generated_context(
                pass1_video_block=pred_video,
                pass1_audio_block=pred_audio,
                video_context_residual=video_context_residual,
                audio_context_residual=audio_context_residual,
                conditional_dict=conditional_dict,
                kv_caches=kv_caches,
                block=block,
            )

        del kv_caches
        return rollout_video, rollout_audio, blocks

    @torch.no_grad()
    def _compute_detached_dmd_gradient(
        self,
        *,
        rollout_video: torch.Tensor,
        rollout_audio: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        real_video_frames: int,
        real_audio_frames: int,
        capture_video_predictions: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
        Optional[Dict[str, Any]],
    ]:
        """Run the unchanged bidirectional Stage-2 real/fake score objective."""
        batch_size = rollout_video.shape[0]
        real_video = rollout_video[:, :real_video_frames]
        real_audio = rollout_audio[:, :real_audio_frames]
        score_sigma = self.dmd.sample_dmd_score_sigma(batch_size)
        video_sigma, audio_sigma = self.dmd.expand_sigma_to_frames(
            score_sigma,
            num_video_frames=real_video_frames,
            num_audio_frames=real_audio_frames,
        )

        noisy_video = self.dmd.add_noise(
            real_video.flatten(0, 1),
            torch.randn_like(real_video).flatten(0, 1),
            video_sigma.flatten(0, 1),
        ).unflatten(0, (batch_size, real_video_frames))
        noisy_audio = self.dmd.add_noise(
            real_audio,
            torch.randn_like(real_audio),
            audio_sigma,
        )
        grad_video, grad_audio, log_dict = self.dmd._compute_kl_grad(
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            clean_video=real_video,
            clean_audio=real_audio,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            video_loss_mask=None,
            audio_loss_mask=None,
            return_video_predictions=capture_video_predictions,
        )
        visualization_context = None
        if capture_video_predictions:
            pred_real_video = log_dict.pop("_pred_real_video_x0")
            pred_fake_video = log_dict.pop("_pred_fake_video_x0")
            visualization_context = {
                # Only rank 0 enables capture. Move the real-length tensors to
                # CPU before replay backward and critic training reach their
                # memory peaks. No padded suffix is retained or decoded.
                "generator_video": real_video[:1].detach().cpu(),
                "real_score_video_x0": pred_real_video[:1].detach().cpu(),
                "fake_score_video_x0": pred_fake_video[:1].detach().cpu(),
                "video_sigma_mean": float(video_sigma.float().mean().item()),
                "real_video_guidance_scale": float(
                    log_dict["real_video_guidance_scale"]
                ),
            }
        log_dict.update(
            {
                "alignment/video_sigma_mean": video_sigma.float().mean().item(),
                "alignment/audio_sigma_mean": audio_sigma.float().mean().item(),
            }
        )
        # Dummy suffix blocks exist only to synchronize per-block FSDP calls.
        # Scatter the real-clip gradient into zero-initialized rollout buffers.
        grad_video_full = torch.zeros_like(rollout_video)
        grad_audio_full = torch.zeros_like(rollout_audio)
        grad_video_full[:, :real_video_frames] = grad_video
        grad_audio_full[:, :real_audio_frames] = grad_audio
        return (
            grad_video_full,
            grad_audio_full,
            log_dict,
            visualization_context,
        )

    def _replay_dmd_and_backward(
        self,
        *,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        pass1_video: torch.Tensor,
        pass1_audio: torch.Tensor,
        dmd_grad_video: torch.Tensor,
        dmd_grad_audio: torch.Tensor,
        blocks: list,
        video_context_residual: Optional[torch.Tensor],
        audio_context_residual: Optional[torch.Tensor],
        real_video_frames: int,
        real_audio_frames: int,
    ) -> Dict[str, float]:
        """Replay all chunks, backward one chunk at a time, and keep context fixed."""
        batch_size = noisy_video.shape[0]
        video_frames = noisy_video.shape[1]
        audio_frames = noisy_audio.shape[1]
        video_weights = torch.zeros(
            video_frames, device=self.device, dtype=torch.float64
        )
        video_weights[:real_video_frames] = self.dmd._compute_block_weights(
            real_video_frames
        )
        audio_weights = torch.zeros(
            audio_frames, device=self.device, dtype=torch.float64
        )
        audio_weights[:real_audio_frames] = self.dmd._compute_block_weights(
            real_audio_frames, is_audio=True
        )
        video_weight, audio_weight = self.dmd.get_loss_weights()

        kv_caches = self._init_kv_caches_for_conditional(
            conditional_dict=conditional_dict,
            batch_size=batch_size,
            total_video_frames=video_frames,
            total_audio_frames=audio_frames,
            device=noisy_video.device,
            dtype=noisy_video.dtype,
            video_frame_seqlen=int(noisy_video.shape[-2]) * int(noisy_video.shape[-1]),
        )
        fsdp_no_sync = getattr(self.ar_teacher._generator, "no_sync", None)
        total_loss = 0.0
        total_video_loss = 0.0
        total_audio_loss = 0.0
        replay_max_abs = 0.0
        replay_abs_sum = 0.0
        replay_numel = 0

        for block_index, block in enumerate(blocks):
            is_last = block_index == len(blocks) - 1
            sync_context = (
                fsdp_no_sync()
                if (not is_last and callable(fsdp_no_sync))
                else nullcontext()
            )
            with sync_context:
                video_slice = slice(block.video_start, block.video_end)
                audio_slice = slice(block.audio_start, block.audio_end)
                pred_video, pred_audio = self._generator_kv_block_forward(
                    noisy_video_block=noisy_video[:, video_slice],
                    noisy_audio_block=noisy_audio[:, audio_slice],
                    video_sigma_block=video_sigma[:, video_slice],
                    audio_sigma_block=audio_sigma[:, audio_slice],
                    conditional_dict=conditional_dict,
                    kv_caches=kv_caches,
                    block=block,
                )

                with torch.no_grad():
                    video_diff = (pred_video - pass1_video[:, video_slice]).abs()
                    audio_diff = (pred_audio - pass1_audio[:, audio_slice]).abs()
                    replay_max_abs = max(
                        replay_max_abs,
                        float(video_diff.max().item()),
                        float(audio_diff.max().item()),
                    )
                    replay_abs_sum += float(video_diff.float().sum().item())
                    replay_abs_sum += float(audio_diff.float().sum().item())
                    replay_numel += video_diff.numel() + audio_diff.numel()

                video_loss = replay_dmd_surrogate_loss(
                    pred_video,
                    dmd_grad_video[:, video_slice],
                    video_weights[video_slice],
                    denominator=batch_size * real_video_frames,
                )
                audio_loss = replay_dmd_surrogate_loss(
                    pred_audio,
                    dmd_grad_audio[:, audio_slice],
                    audio_weights[audio_slice],
                    denominator=batch_size * real_audio_frames,
                )
                block_loss = video_weight * video_loss + audio_weight * audio_loss
                block_loss.backward()

                total_loss += float(block_loss.detach().item())
                total_video_loss += float(video_loss.detach().item())
                total_audio_loss += float(audio_loss.detach().item())

                # Never use the grad-enabled replay output as downstream context.
                # Pass-1 x0 makes the cache state identical and truncates BPTT.
                self._refresh_generated_context(
                    pass1_video_block=pass1_video[:, video_slice],
                    pass1_audio_block=pass1_audio[:, audio_slice],
                    video_context_residual=video_context_residual,
                    audio_context_residual=audio_context_residual,
                    conditional_dict=conditional_dict,
                    kv_caches=kv_caches,
                    block=block,
                )

                del pred_video, pred_audio, video_loss, audio_loss, block_loss

        del kv_caches
        if dist.is_initialized() and self.world_size > 1:
            max_tensor = torch.tensor(
                [replay_max_abs], device=self.device, dtype=torch.float32
            )
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            replay_max_abs = float(max_tensor.item())
            sum_count = torch.tensor(
                [replay_abs_sum, float(replay_numel)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(sum_count, op=dist.ReduceOp.SUM)
            replay_abs_sum = float(sum_count[0].item())
            replay_numel = int(sum_count[1].item())
        replay_mean_abs = replay_abs_sum / max(1, replay_numel)
        if self.strict_replay_check and replay_max_abs > self.replay_check_tolerance:
            raise RuntimeError(
                "self-context replay diverged from pass 1: "
                f"max_abs={replay_max_abs:.6g} > "
                f"tolerance={self.replay_check_tolerance:.6g}"
            )
        if (
            self.is_main_process
            and replay_max_abs > self.replay_check_tolerance
            and not self.strict_replay_check
        ):
            print(
                "[Stage2 self-context DMD][warning] replay mismatch: "
                f"max_abs={replay_max_abs:.6g} > "
                f"tolerance={self.replay_check_tolerance:.6g}"
            )
        return {
            "loss": total_loss,
            "video_loss": total_video_loss,
            "audio_loss": total_audio_loss,
            "replay_max_abs": replay_max_abs,
            "replay_mean_abs": replay_mean_abs,
            "num_blocks": float(len(blocks)),
        }

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)

    @torch.no_grad()
    def _run_local_visualization(
        self,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Save generator | CFG real-score x0 | fake-score x0 with PyAV."""

        failure = ""
        if self.is_main_process:
            if context is None:
                failure = "rank 0 did not capture Stage-2 visualization tensors"
            elif not hasattr(self, "video_vae"):
                failure = "Stage-2 local visualization has no video VAE"
            else:
                try:
                    self.video_vae.to(device=self.device, dtype=self.dtype)
                    seed = int(self.config.get("seed", 42)) + int(context["step"])
                    decoded = []
                    for key in (
                        "generator_video",
                        "real_score_video_x0",
                        "fake_score_video_x0",
                    ):
                        latent = context[key].to(
                            device=self.device,
                            dtype=self.dtype,
                        )
                        decoded.append(
                            decode_video_tiled(
                                video_vae=self.video_vae,
                                latent_bfchw=latent,
                                seed=seed,
                            )
                        )
                        del latent
                        torch.cuda.empty_cache()

                    frames = min(int(item.shape[0]) for item in decoded)
                    height = min(int(item.shape[1]) for item in decoded)
                    comparison = torch.cat(
                        [item[:frames, :height] for item in decoded],
                        dim=2,
                    )
                    log_dir = Path(self.output_path) / "local_logs"
                    output_path = (
                        log_dir
                        / (
                            f"step_{int(context['step']):06d}_"
                            "generator_real_fake.mp4"
                        )
                    )
                    write_video_pyav(
                        str(output_path),
                        comparison,
                        fps=self.local_video_fps,
                    )
                    metadata_path = output_path.with_suffix(".txt")
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_text(
                        "layout: self_context_generator_x0 | "
                        "cfg_real_score_pred_x0 | fake_score_pred_x0\n"
                        f"step: {int(context['step'])}\n"
                        f"real_video_latent_frames: "
                        f"{int(context['real_video_frames'])}\n"
                        f"dmd_video_sigma_mean: "
                        f"{float(context['video_sigma_mean']):.8f}\n"
                        f"real_video_guidance_scale: "
                        f"{float(context['real_video_guidance_scale']):.8f}\n"
                        f"prompt: {context['prompt']}\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[Stage2 self-context DMD][local-log] wrote "
                        f"{output_path}",
                        flush=True,
                    )
                    del decoded, comparison
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                finally:
                    self.video_vae.to(device="cpu")
                    torch.cuda.empty_cache()

        failed = torch.tensor(
            [1 if failure else 0],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.broadcast(failed, src=0)
        if int(failed.item()) != 0:
            if self.is_main_process:
                raise RuntimeError(
                    f"Stage-2 local visualization failed: {failure}"
                )
            raise RuntimeError("Stage-2 local visualization failed on rank 0")

    def train_one_step(self) -> None:
        self.ar_teacher._generator.train()
        self.dmd.fake_score.train()
        self.dmd.real_score.eval()
        self._apply_lr()
        self.dmd.current_step = self.step

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        batch = next(self.dataloader)
        clean_video, clean_audio, text_prompts = self._encode_batch(batch)
        clean_audio = self._align_audio(clean_video, clean_audio)
        with torch.no_grad():
            conditional_dict = self.ar_teacher.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self._get_unconditional_dict(len(text_prompts))

        sigma_batch = self.dmd.sample_few_step_sigma(clean_video.shape[0])
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(sigma_batch, src=0)
        real_video_frames = int(clean_video.shape[1])
        real_audio_frames = int(clean_audio.shape[1])

        # Keep the existing SVI data contract. With SVI disabled these are all
        # the original clean latents. When enabled later, ``context_*`` supplies
        # a residual added to generated cache context, while ``generator_*`` is
        # the current-chunk source before GT-noise injection.
        (
            context_video,
            context_audio,
            generator_video,
            generator_audio,
            svi_log,
        ) = self._svi_make_iteration_inputs(
            clean_video=clean_video,
            clean_audio=clean_audio,
            sigma_batch=sigma_batch,
        )
        video_context_residual = self._context_residual(
            context_video, clean_video, enabled=self.svi_enabled
        )
        audio_context_residual = self._context_residual(
            context_audio, clean_audio, enabled=self.svi_enabled
        )

        # Variable-length clips across ranks: synchronize the
        # number of per-block FSDP calls without expanding a real partial block.
        blocks, real_block_count = self._synchronized_rollout_blocks(
            real_video_frames
        )
        rollout_video_frames = blocks[-1].video_end
        rollout_audio_frames = blocks[-1].audio_end
        generator_video_padded = self._pad_frames(
            generator_video, rollout_video_frames
        )
        generator_audio_padded = self._pad_frames(
            generator_audio, rollout_audio_frames
        )
        if video_context_residual is not None:
            video_context_residual = self._pad_frames(
                video_context_residual, rollout_video_frames
            )
        if audio_context_residual is not None:
            audio_context_residual = self._pad_frames(
                audio_context_residual, rollout_audio_frames
            )
        video_sigma = sigma_batch[:, None].expand(
            clean_video.shape[0], rollout_video_frames
        ).contiguous()
        audio_sigma = sigma_batch[:, None].expand(
            clean_video.shape[0], rollout_audio_frames
        ).contiguous()

        train_generator = self.step % self.dfake_gen_update_ratio == 0
        run_local_visualization = bool(
            train_generator
            and self.local_visualization_iters > 0
            and self.step > 0
            and self.step % self.local_visualization_iters == 0
            and not bool(self.config.get("no_visualize", False))
        )
        generator_metrics: Dict[str, float] = {
            "loss": 0.0,
            "video_loss": 0.0,
            "audio_loss": 0.0,
            "replay_max_abs": 0.0,
            "replay_mean_abs": 0.0,
            "num_blocks": 0.0,
        }
        generator_log: Dict[str, Any] = {}
        generator_grad_norm: Any = 0.0
        local_visualization_context: Optional[Dict[str, Any]] = None

        if train_generator:
            # Drop the previous critic step's gradient buffers before the
            # memory-heavy generator rollout/replay phase.
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.dmd.fake_score.requires_grad_(False)
            self.dmd.real_score.requires_grad_(False)
            self.generator_optimizer.zero_grad(set_to_none=True)

            noisy_video, noisy_audio = self._build_noisy_gt_inputs(
                generator_video_source=generator_video_padded,
                generator_audio_source=generator_audio_padded,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )
            pass1_video, pass1_audio, blocks = (
                self._run_detached_self_context_rollout(
                    noisy_video=noisy_video,
                    noisy_audio=noisy_audio,
                    video_sigma=video_sigma,
                    audio_sigma=audio_sigma,
                    conditional_dict=conditional_dict,
                    video_context_residual=video_context_residual,
                    audio_context_residual=audio_context_residual,
                    blocks=blocks,
                )
            )
            (
                dmd_grad_video,
                dmd_grad_audio,
                generator_log,
                local_visualization_context,
            ) = self._compute_detached_dmd_gradient(
                rollout_video=pass1_video,
                rollout_audio=pass1_audio,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
                capture_video_predictions=(
                    run_local_visualization and self.is_main_process
                ),
            )
            if local_visualization_context is not None:
                local_visualization_context.update(
                    {
                        "prompt": text_prompts[0],
                        "step": int(self.step),
                        "real_video_frames": int(real_video_frames),
                    }
                )
            generator_metrics = self._replay_dmd_and_backward(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                pass1_video=pass1_video,
                pass1_audio=pass1_audio,
                dmd_grad_video=dmd_grad_video,
                dmd_grad_audio=dmd_grad_audio,
                blocks=blocks,
                video_context_residual=video_context_residual,
                audio_context_residual=audio_context_residual,
                real_video_frames=real_video_frames,
                real_audio_frames=real_audio_frames,
            )

            if hasattr(self.ar_teacher._generator, "clip_grad_norm_"):
                generator_grad_norm = self.ar_teacher._generator.clip_grad_norm_(
                    self.max_grad_norm
                )
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ar_teacher._generator.parameters(), self.max_grad_norm
                )
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            self._svi_update_bank(
                pred_x0_video=pass1_video[:, :real_video_frames],
                pred_x0_audio=pass1_audio[:, :real_audio_frames],
                clean_video=clean_video,
                clean_audio=clean_audio,
                sigma_batch=sigma_batch,
            )
            del (
                noisy_video,
                noisy_audio,
                pass1_video,
                pass1_audio,
                dmd_grad_video,
                dmd_grad_audio,
            )
            torch.cuda.empty_cache()

        # The fake score remains the unchanged bidirectional Stage-2 critic. It
        # receives a fresh detached sample from the new generator rollout.
        self.dmd.fake_score.requires_grad_(True)
        self.dmd.real_score.requires_grad_(False)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            critic_noisy_video, critic_noisy_audio = self._build_noisy_gt_inputs(
                generator_video_source=generator_video_padded,
                generator_audio_source=generator_audio_padded,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )
            critic_video, critic_audio, _ = self._run_detached_self_context_rollout(
                noisy_video=critic_noisy_video,
                noisy_audio=critic_noisy_audio,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                video_context_residual=video_context_residual,
                audio_context_residual=audio_context_residual,
                blocks=blocks,
            )

        critic_loss, critic_log = self.dmd.critic_loss(
            pred_x0_video_det=critic_video[:, :real_video_frames].detach(),
            pred_x0_audio_det=critic_audio[:, :real_audio_frames].detach(),
            conditional_dict=conditional_dict,
            video_loss_mask=None,
            audio_loss_mask=None,
        )
        critic_loss.backward()
        if hasattr(self.dmd.fake_score, "clip_grad_norm_"):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(self.max_grad_norm)
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        critic_loss_value = float(critic_loss.detach().item())

        # Decode only after both optimizer phases. The three captured latents
        # have already been moved to CPU, so logging does not extend either
        # training graph or retain tensors across the peak-memory sections.
        if run_local_visualization:
            # These gradients have already been consumed by optimizer.step().
            # Release their shards before moving the VAE decoder to the GPU.
            self.critic_optimizer.zero_grad(set_to_none=True)
            self._run_local_visualization(local_visualization_context)
            del local_visualization_context
            torch.cuda.empty_cache()

        if self.is_main_process:
            log_data = {
                "train/critic_loss": critic_loss_value,
                "train/critic_video_loss": critic_log.get("critic_video_loss", 0.0),
                "train/critic_audio_loss": critic_log.get("critic_audio_loss", 0.0),
                "train/critic_grad_norm": self._scalar(critic_grad_norm),
                "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                "train/generator_sigma_mean": float(sigma_batch.float().mean().item()),
                "train/self_context_enabled": 1.0,
                "train/self_context_real_blocks": float(real_block_count),
                "train/self_context_synchronized_blocks": float(len(blocks)),
                "train/self_context_real_video_frames": float(real_video_frames),
                "train/self_context_rollout_video_frames": float(
                    rollout_video_frames
                ),
                "train/svi_enabled": svi_log["enabled"],
                "train/svi_force_clean": svi_log["force_clean"],
                "train/svi_img_injected": svi_log["img_injected"],
                "train/svi_vid_injected": svi_log["vid_injected"],
                "train/svi_video_bank_size": float(self.svi_bank.total_size()),
                "train/svi_audio_bank_size": float(self.svi_audio_bank.total_size()),
            }
            if train_generator:
                log_data.update(
                    {
                        "train/generator_loss": generator_metrics["loss"],
                        "train/generator_video_dmd_loss": generator_metrics[
                            "video_loss"
                        ],
                        "train/generator_audio_dmd_loss": generator_metrics[
                            "audio_loss"
                        ],
                        "train/generator_grad_norm": self._scalar(
                            generator_grad_norm
                        ),
                        "train/generator_lr": self.generator_optimizer.param_groups[0][
                            "lr"
                        ],
                        "train/self_context_num_blocks": generator_metrics[
                            "num_blocks"
                        ],
                        "train/self_context_replay_max_abs": generator_metrics[
                            "replay_max_abs"
                        ],
                        "train/self_context_replay_mean_abs": generator_metrics[
                            "replay_mean_abs"
                        ],
                        "train/dmdtrain_gradient_norm_video": generator_log.get(
                            "dmdtrain_gradient_norm_video", 0.0
                        ),
                        "train/dmdtrain_gradient_norm_audio": generator_log.get(
                            "dmdtrain_gradient_norm_audio", 0.0
                        ),
                        "train/dmd_sigma_mean": generator_log.get(
                            "alignment/video_sigma_mean", 0.0
                        ),
                        "train/real_video_guidance_scale": generator_log.get(
                            "real_video_guidance_scale", 0.0
                        ),
                    }
                )
            wandb.log(log_data, step=self.step)
            if self.step < 10 or self.step % 50 == 0:
                branch = "G+C" if train_generator else "C"
                print(
                    f"[Stage2 self-context DMD][{self.step}] {branch} | "
                    f"gen={generator_metrics['loss']:.5f} "
                    f"critic={critic_loss_value:.5f} "
                    f"sigma={sigma_batch.float().mean().item():.4f} "
                    f"replay_max={generator_metrics['replay_max_abs']:.3e}"
                )

        del (
            clean_video,
            clean_audio,
            conditional_dict,
            unconditional_dict,
            context_video,
            context_audio,
            generator_video,
            generator_audio,
            generator_video_padded,
            generator_audio_padded,
            blocks,
            critic_noisy_video,
            critic_noisy_audio,
            critic_video,
            critic_audio,
            critic_loss,
        )
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: self-context GT-input DMD with replay"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True
    trainer = Stage2SelfContextDMDTrainer(config)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
