"""
Salt++: Context-Aligned Post-Training for Few-Step Streaming Multimodal Generation.

Key Modules:
- ar_teacher: Stage 1 Causal Self-Flow (CSF) AR Teacher
- causal_tf_dmd / train_stage2_*: Stage 2 Context-Aligned AR DMD (clean-prefix) & TF-dCM
- dmd / self_context_ar_dmd / sc_dmd: Stage 2 Context-Aligned AR DMD (on-policy rollout)
- stage4_scale_wise_dmd: Scale-wise 2LR+2HR high-resolution post-training
- models: Model wrappers for LTX-2
- inference: Inference pipelines
"""

from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.loss import get_denoising_loss

__version__ = "0.1.0"

__all__ = [
    "LTX2DMD",
    "get_denoising_loss",
]
