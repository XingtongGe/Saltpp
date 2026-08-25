# 🧂 Salt++: Context-Aligned Post-Training for Few-Step Streaming Multimodal Generation

A successor to Salt, extending few-step distribution matching from video distillation to causal joint audio–video generation.

Stay tuned 🚀

## Quantitative Results on JavisBench


### 480p Audio–Video Generation

| Model | Causal | Steps | VQ ↑ | MQ ↑ | AQ ↑ | CLIP ↑ | IB-AV ↑ | Javis ↑ |
|:--|:--:|:--:|--:|--:|--:|--:|--:|--:|
| LTX-2 Base | ✗ | 40 | 1.884 | 0.566 | 4.986 | 0.311 | 0.239 | 0.200 |
| AR Teacher (CSF) | ✓ | 40 | 2.304 | 0.857 | 4.565 | 0.316 | 0.202 | 0.162 |
| OmniForcing | ✓ | 4 | 1.807 | 0.699 | 4.718 | 0.303 | 0.163 | 0.124 |
| Salt++ (TF-dCM route) | ✓ | 4 | <u>2.013</u> | <u>0.826</u> | <u>4.976</u> | <u>0.313</u> | **0.229** | **0.185** |
| **Salt++** | ✓ | 4 | **2.838** | **1.010** | **4.991** | **0.316** | <u>0.184</u> | <u>0.146</u> |

### 480p Video Fidelity and Temporal Consistency

| Model | Causal | Steps | Aesthetic ↑ | Imaging ↑ | Subject Consistency ↑ | Background Consistency ↑ |
|:--|:--:|:--:|--:|--:|--:|--:|
| LTX-2 Base | ✗ | 40 | 53.89 | 67.06 | 95.94 | 95.63 |
| OmniForcing | ✓ | 4 | **56.74** | <u>68.41</u> | <u>96.62</u> | 95.32 |
| Salt++ (TF-dCM route) | ✓ | 4 | 53.65 | 62.87 | 96.58 | <u>96.05</u> |
| **Salt++** | ✓ | 4 | <u>55.43</u> | **69.98** | **97.01** | **96.24** |

### 960p Audio–Video Generation

| Model | Causal | Steps | VQ ↑ | MQ ↑ | AQ ↑ | CLIP ↑ | IB-AV ↑ | Javis ↑ |
|:--|:--:|:--:|--:|--:|--:|--:|--:|--:|
| LTX-2 Base | ✗ | 40 + 3 | 2.231 | 0.607 | 4.867 | 0.311 | 0.169 | 0.145 |
| OmniForcing | ✓ | 4 | 2.298 | 0.832 | 4.721 | 0.293 | 0.165 | 0.128 |
| **Salt++** | ✓ | 4 | **2.730** | **0.957** | **5.113** | **0.318** | **0.193** | **0.157** |
