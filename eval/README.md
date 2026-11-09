# Evaluation Guide (JavisBench-mini & VBench)

This directory contains benchmark prompt lists and evaluation helpers used in the **Salt++** paper.

## 1. Prompt Files (`eval/prompts/`)

- `JavisBench-mini.prompts.txt`: The 1,000 official JavisBench-mini evaluation prompts.
- `JavisBench-mini.official-ltx2-seed42.prompts.txt`: The LTX-2 prompt-enhanced version of JavisBench-mini.
- `cinematic_shortshot16.prompts.txt`, `javis16_plus_sft_first_cutshot16.prompts.txt`, `youtube_av_dialogue16.prompts.txt`: The 48-prompt internal validation suite used during training.

## 2. Generating JavisBench-mini Samples

### 480p (4-step Salt++)

```bash
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
```

### 1664×960 Scale-Wise (2LR + 2HR, 4 calls total)

```bash
python scripts/infer_960p_scale_wise.py \
  --config configs/stage3_scale_wise_960p/scale_wise_2lr2hr_960p.yaml \
  --checkpoint ./checkpoints/saltpp_960p/generator_only.pt \
  --prompt-file ./eval/prompts/JavisBench-mini.prompts.txt \
  --output-dir ./outputs/javisbench_960p
```

## 3. Running Evaluations

- **JavisBench**: Follow the official [JavisDiT / JavisBench evaluation pipeline](https://github.com/JavisVerse/JavisDiT) on the generated `t2av/` directory (`sample_0000.mp4` + `sample_0000.wav`).
- **VBench**: Install [VBench](https://github.com/Vchitect/VBench) and run:

```bash
torchrun --nproc_per_node=1 eval/evaluate_vbench_video_metrics.py \
  --videos-dir ./outputs/javisbench_480p/t2av \
  --output-dir ./outputs/javisbench_480p/vbench_eval
```
