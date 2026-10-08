# Video Facial Emotion Recognition — Research Report
**Tasks 2–9: Dataset research, model research, pipeline design, RTX 3050 configuration**  
**Date:** 2026-10-08  
**Status:** Research only. No code modified.

---

## Table of Contents

1. [Dataset Evaluation (Task 2)](#1-dataset-evaluation-task-2)
2. [Model Architecture Evaluation (Task 3)](#2-model-architecture-evaluation-task-3)
3. [Final Recommendation (Task 4)](#3-final-recommendation-task-4)
4. [CMU-MOSEI as Baseline (Task 5)](#4-cmu-mosei-as-baseline-task-5)
5. [Full Pipeline Design (Task 6)](#5-full-pipeline-design-task-6)
6. [RTX 3050 Training Configuration (Task 7)](#6-rtx-3050-training-configuration-task-7)
7. [Directory Structure (Task 8)](#7-directory-structure-task-8)
8. [Benchmark Plan (Task 9)](#8-benchmark-plan-task-9)

---

## 1. Dataset Evaluation (Task 2)

### Required emotion mapping

The project requires exactly these 6 classes:

```
0 = angry
1 = disgust
2 = fearful
3 = happy
4 = sad
5 = surprised
```

Note: this is identical to the FER-2013 label scheme (minus `neutral`).

---

### 1.1 FER-2013

| Property | Value |
|---|---|
| Type | Static images |
| Total samples | ~35,887 |
| Resolution | 48×48 grayscale |
| Classes | 7: angry, disgust, fear, happy, sad, surprise, neutral |
| License | Public (Kaggle competition, open access) |
| Download | Available on Kaggle; also via `torchvision.datasets.FER2013` |
| Size | ~90 MB (CSV encoded) |
| Class balance | Heavily imbalanced (happy: 25%, disgust: 1.5%) |
| 6-class mapping | Drop `neutral` → 6 classes match exactly |
| Benchmark usage | Extremely widely cited baseline |
| RTX 3050 suitability | ✅ Very easy — 48×48 images, minimal VRAM |
| Raw video | ❌ Static images only |

**Limitations:** 48×48 grayscale only. Not suitable for a deep video model. Useful only for a lightweight baseline or augmentation source. Published accuracy ceiling is ~73–75% even with SOTA methods, partly due to labelling noise.

---

### 1.2 RAF-DB (Real-world Affective Faces Database)

| Property | Value |
|---|---|
| Type | Static images (in-the-wild) |
| Total samples | ~29,672 |
| Resolution | Variable; typically 224×224 after alignment |
| Classes | 7: neutral, happy, sad, surprise, fear, disgust, angry |
| License | Non-commercial research only (request required) |
| Download | Via request at http://www.whdeng.cn/RAF/model1.html |
| Size | ~3 GB |
| Class balance | Imbalanced (happy dominant ~26%) |
| 6-class mapping | Drop `neutral` → 6 classes match exactly |
| Benchmark usage | Standard benchmark for static FER |
| RTX 3050 suitability | ✅ Suitable — still static images |
| Raw video | ❌ Static images only |

**Limitations:** Static images only — no temporal information. Good for a static FER baseline. Access requires manual request and approval.

---

### 1.3 AffectNet

| Property | Value |
|---|---|
| Type | Static images (in-the-wild, web-scraped) |
| Total samples | ~440,000 manually labeled (training); 1M+ total |
| Resolution | Variable (faces aligned and cropped) |
| Classes | 8: neutral, happy, sad, surprise, fear, disgust, angry, contempt |
| License | Non-commercial research only; requires registration |
| Download | Via request at affectnet.org |
| Size | ~50+ GB |
| Class balance | Extremely imbalanced (happy: ~35%, contempt: ~1%) |
| 6-class mapping | Drop `neutral` and `contempt` → 6 classes match exactly |
| Benchmark usage | Widely used for static FER pre-training |
| RTX 3050 suitability | ✅ Feasible per-image, but huge dataset requires sampling |
| Raw video | ❌ Static images only |

**Limitations:** Massive size. Static only. Approval required. Best used as a pre-training source, not a primary training set.

---

### 1.4 DFEW (Dynamic Facial Expressions in the Wild)

| Property | Value |
|---|---|
| Type | Video clips (in-the-wild, from movies) |
| Total clips | 16,372 |
| Resolution | Face-cropped, 224×224 |
| Classes | 7: happy, sad, neutral, angry, surprise, disgust, fear |
| License | Free for non-commercial research; request form on website |
| Download | [dfew-dataset.github.io](https://dfew-dataset.github.io/) — form submission |
| Size | ~15 GB |
| Class balance | Moderately imbalanced (happy dominant) |
| 6-class mapping | Drop `neutral` → 6 classes match exactly |
| Benchmark usage | **Primary benchmark** for nearly all modern DFER papers (MAE-DFER, DFER-CLIP, Former-DFER, UniLearn) |
| Splits | 5-fold cross-validation (standard); also single split available |
| RTX 3050 suitability | ✅ Manageable with 16 frames at 112×112 or 8 frames at 224×224 |
| Raw video | ✅ **Actual video clips** (face-cropped sequences) |
| Published SOTA | MAE-DFER: 63.41% UAR / 74.43% WAR (5-fold avg) |

**Assessment:** **Best primary dataset for this project.** It is the most widely cited benchmark for dynamic FER, all major DFER models publish results on it, pre-trained weights (MAE-DFER, DFER-CLIP) are available specifically for DFEW, and the 6-class mapping is a clean drop of `neutral`. Access requires a form submission (typical academic agreement).

---

### 1.5 FERV39k

| Property | Value |
|---|---|
| Type | Video clips (in-the-wild, multi-scene) |
| Total clips | 38,935 |
| Resolution | Face-cropped sequences |
| Classes | 7: angry, disgust, fear, happy, neutral, sad, surprise |
| License | Non-commercial research; request via GitHub |
| Download | [github.com/wangyanckxx/FERV39k](https://github.com/wangyanckxx/FERV39k) |
| Size | ~22 GB |
| Class balance | Imbalanced; 4 diverse scene types |
| 6-class mapping | Drop `neutral` → 6 classes match exactly |
| Benchmark usage | CVPR 2022 paper; used alongside DFEW in most DFER papers |
| RTX 3050 suitability | ✅ Same as DFEW with frame sampling |
| Raw video | ✅ Video clips |
| Published SOTA | MAE-DFER: 43.12% UAR / 52.07% WAR |

**Assessment:** Excellent second dataset. Lower accuracy numbers are due to higher difficulty (multi-scene, occlusion). FERV39k is the standard companion benchmark to DFEW. Pre-trained MAE-DFER weights on FERV39k are also publicly available from the MAE-DFER repo. **Good candidate for cross-dataset validation.**

---

### 1.6 MAFW (Multi-class Affective Face in the Wild)

| Property | Value |
|---|---|
| Type | Video clips |
| Total clips | ~10,045 |
| Classes | 11 compound emotions |
| 6-class mapping | Non-trivial — uses compound emotions, not basic 6 |
| RTX 3050 suitability | ✅ Feasible |
| Raw video | ✅ |

**Assessment:** Not suitable as primary dataset — compound emotion scheme does not cleanly map to our 6 classes.

---

### 1.7 CAER (Context-Aware Emotion Recognition)

| Property | Value |
|---|---|
| Type | Video clips (from TV shows) |
| Total clips | 13,201 |
| Classes | 7 basic emotions |
| 6-class mapping | Drop neutral → 6 classes |
| RTX 3050 suitability | ✅ |
| Raw video | ✅ |

**Assessment:** Less widely benchmarked than DFEW/FERV39k. Context-based (full scene, not just face) — architecture would need to account for context. Not the best primary choice for a pure FER model.

---

### 1.8 Aff-Wild2

| Property | Value |
|---|---|
| Type | Videos (in-the-wild) |
| Total clips | ~500+ videos (~2.8M frames) |
| Classes | 8 basic + valence/arousal |
| 6-class mapping | Drop neutral + contempt |
| License | ABAW competition; requires registration |
| Size | Very large (~100+ GB) |
| RTX 3050 suitability | ⚠️ Feasible only with heavy sampling |

**Assessment:** Excellent quality but very large. Best suited as a fine-tuning source after initial training on DFEW. Not practical as a starting point given GPU constraints.

---

### Dataset Summary Table

| Dataset | Type | Clips/Images | Classes | 6-class map | Access | Size | RTX 3050 | Verdict |
|---|---|---|---|---|---|---|---|---|
| FER-2013 | Static | 35,887 | 7 | Drop neutral | Open | 90 MB | ✅ Easy | Baseline only |
| RAF-DB | Static | 29,672 | 7 | Drop neutral | Request | 3 GB | ✅ | Static baseline |
| AffectNet | Static | 440K | 8 | Drop 2 | Request | 50+ GB | ⚠️ | Pre-training source |
| **DFEW** | **Video** | **16,372** | **7** | **Drop neutral** | **Request** | **15 GB** | **✅** | **🥇 PRIMARY** |
| FERV39k | Video | 38,935 | 7 | Drop neutral | Request | 22 GB | ✅ | 🥈 Secondary/validation |
| MAFW | Video | 10,045 | 11 | ❌ Complex | Request | — | ✅ | Not suitable |
| CAER | Video | 13,201 | 7 | Drop neutral | Request | — | ✅ | Less benchmarked |
| Aff-Wild2 | Video | ~500 vids | 8 | Drop 2 | ABAW | 100+ GB | ⚠️ | Too large |

---

## 2. Model Architecture Evaluation (Task 3)

### 2.1 MAE-DFER (Masked Autoencoder for Dynamic FER)

| Property | Value |
|---|---|
| Paper | Sun et al., ACM MM 2023 |
| Architecture | LGI-Former (Local-Global Interaction Transformer), based on VideoMAE |
| Input | Face-cropped video clips |
| Frames | 16 frames |
| Resolution | 112×112 per frame |
| Pre-training | Self-supervised MAE on VoxCeleb2 (unlabeled face videos) |
| Params (encoder) | ~86M (ViT-Base scale LGI-Former) |
| Published DFEW result | 63.41% UAR / 74.43% WAR (5-fold avg) |
| Published FERV39k result | 43.12% UAR / 52.07% WAR |
| Pretrained weights | ✅ Available on GitHub (VoxCeleb2 pretrain + DFEW/FERV39k/MAFW fine-tuned) |
| Code | [github.com/sunlicai/MAE-DFER](https://github.com/sunlicai/MAE-DFER) |
| RTX 3050 (6 GB) | ⚠️ Tight — 86M params, 16×112×112 input. Batch 2–4 with bf16 + grad accum |
| Fine-tunable | ✅ Fine-tuned checkpoints per fold publicly available |
| Inference speed | Moderate (~50–100ms per clip on GPU) |
| Suitability | ✅ Purpose-built for DFER, SOTA on DFEW |

**Assessment:** The strongest purpose-built dynamic FER model with publicly available pretrained weights. The 38% FLOPs reduction over original VideoMAE makes it feasible on 6 GB VRAM with careful configuration (batch 2, bf16, gradient accumulation to effective batch 8–16). This is the primary recommendation.

---

### 2.2 DFER-CLIP (CLIP-based Dynamic FER)

| Property | Value |
|---|---|
| Paper | Zeng et al., 2023 |
| Architecture | CLIP ViT-B/16 image encoder + temporal Transformer |
| Input | Face-cropped video clips |
| Frames | 8–16 frames |
| Resolution | 224×224 (CLIP standard) |
| Pre-training | CLIP (400M image-text pairs) |
| Params | ~150M (CLIP ViT-B/16 + temporal head) |
| Published DFEW result | Competitive with MAE-DFER |
| Pretrained weights | ✅ CLIP weights via OpenAI; DFER-CLIP fine-tuned available |
| Code | [github.com/zengqunzhao/DFER-CLIP](https://github.com/zengqunzhao/DFER-CLIP) |
| RTX 3050 (6 GB) | ⚠️ Tight — 224×224 input, 150M params. Needs batch 1–2 with grad accum |
| Fine-tunable | ✅ |
| Inference speed | Moderate (~80–120ms per clip) |
| Suitability | ✅ Strong; leverages rich CLIP visual-semantic priors |

**Assessment:** Strong alternative to MAE-DFER. CLIP's rich pre-training provides semantic emotion understanding. However, 224×224 input per frame is heavier on VRAM than MAE-DFER's 112×112. **Backup model recommendation.**

---

### 2.3 Former-DFER

| Property | Value |
|---|---|
| Paper | Zhao et al., ACM MM 2021 |
| Architecture | ResNet-18 backbone + Transformer temporal encoder |
| Input | Face-cropped sequences |
| Frames | 16 frames |
| Resolution | 112×112 |
| Pre-training | ImageNet (ResNet-18) |
| Params | ~60M |
| Published DFEW result | ~53% UAR (significantly below MAE-DFER) |
| RTX 3050 (6 GB) | ✅ Comfortable |
| Suitability | ✅ Older but well-established |

**Assessment:** Lighter than MAE-DFER and easier to fit on 6 GB. However, performance is well below MAE-DFER. Only suitable if MAE-DFER cannot fit in memory after all optimizations.

---

### 2.4 VideoMAE-based FER (general)

| Property | Value |
|---|---|
| Architecture | ViT-Base with masked autoencoder pre-training on Kinetics/UCF |
| Input | Raw video clips |
| Frames | 16 frames |
| Resolution | 224×224 |
| Params | ~86M |
| Pre-training | Kinetics-400 (general video, not face-specific) |
| RTX 3050 (6 GB) | ❌ Difficult — 224×224 × 16 frames at 86M params exceeds 6 GB in practice |
| Suitability | ⚠️ Not face-specific; MAE-DFER is a direct improvement on VideoMAE for FER |

**Assessment:** MAE-DFER is VideoMAE adapted specifically for FER — use MAE-DFER instead.

---

### 2.5 ViT-base-patch16-224 (plain ViT fine-tuned per frame)

| Property | Value |
|---|---|
| Architecture | Standard ViT-B/16 |
| Input | Individual frames (no temporal modeling) |
| Frames | 1 (frame-level, then aggregate) |
| Resolution | 224×224 |
| Params | ~86M |
| Pre-training | ImageNet-21k |
| RTX 3050 (6 GB) | ✅ Comfortable per frame; temporal aggregation is cheap |
| Suitability | ⚠️ No native temporal modeling — loses temporal FER signal |

**Assessment:** Feasible as a frame-level baseline, but loses temporal dynamics. Worse than MAE-DFER or DFER-CLIP for video FER.

---

### 2.6 EfficientNet-B2 / ResNet-50 per frame (CNN baseline)

| Property | Value |
|---|---|
| Architecture | CNN |
| Input | Individual frames |
| Params | 9–25M |
| Pre-training | ImageNet |
| RTX 3050 (6 GB) | ✅ Very easy |
| Suitability | ⚠️ No temporal modeling; lower accuracy than transformer models |

**Assessment:** Good for a lightweight comparison baseline. ~70–73% on FER-2013, lower on in-the-wild video datasets.

---

### Model Summary Table

| Model | Arch | Frames | Resolution | Params | DFEW UAR | Pretrained | RTX 3050 | Verdict |
|---|---|---|---|---|---|---|---|---|
| **MAE-DFER** | **LGI-Former** | **16** | **112×112** | **~86M** | **63.4%** | **✅ DFER-specific** | **⚠️ Tight but feasible** | **🥇 Primary** |
| DFER-CLIP | CLIP ViT-B | 8–16 | 224×224 | ~150M | ~63% | ✅ CLIP | ⚠️ Tight | 🥈 Backup |
| Former-DFER | ResNet+Transformer | 16 | 112×112 | ~60M | ~53% | ✅ ImageNet | ✅ Easy | Fallback |
| VideoMAE | ViT-B | 16 | 224×224 | ~86M | — | ✅ Kinetics | ❌ Too heavy | Not recommended |
| Plain ViT | ViT-B/16 | 1 | 224×224 | ~86M | — | ✅ ImageNet | ✅ | No temporal |
| EfficientNet | CNN | 1 | 224×224 | ~9M | — | ✅ ImageNet | ✅ | Baseline only |

---

## 3. Final Recommendation (Task 4)

### Decision

```
PRIMARY DATASET:  DFEW
PRIMARY MODEL:    MAE-DFER
BACKUP MODEL:     DFER-CLIP
BASELINE:         CMU-MOSEI FACET + MLP (already available)
```

### Reasoning

**Why DFEW:**

1. It is the dominant benchmark for dynamic FER — every major DFER paper (2021–2026) reports on DFEW. This ensures meaningful comparisons with published results.
2. 16,372 video clips of face-cropped sequences, 7 classes, clean `neutral`-drop mapping to our 6 classes.
3. 15 GB is manageable for a hackathon-scale project.
4. MAE-DFER's fine-tuned checkpoints are provided per fold for DFEW — we can use these directly without re-running pre-training.
5. The request access is an academic form; typically approved for research use.

**Why MAE-DFER:**

1. Purpose-built for dynamic FER — unlike VideoMAE (general video) or ViT (static images), MAE-DFER was designed and evaluated specifically for DFER.
2. 63.4% UAR on DFEW — currently the best openly reproduced result with public weights.
3. 38% fewer FLOPs than VideoMAE makes it the most GPU-efficient option that doesn't sacrifice performance.
4. 16×112×112 input is achievable on 6 GB VRAM with batch 2–4 + bf16 + gradient accumulation.
5. Pre-trained weights on VoxCeleb2 (general face video) + fine-tuned on DFEW are both publicly available. We fine-tune from the DFEW checkpoint, reducing training time substantially.
6. Official code (PyTorch) is available and actively maintained.

**Why DFER-CLIP as backup:**

1. CLIP's pre-training on 400M image-text pairs provides rich semantic priors for emotion-related visual concepts.
2. Competitive accuracy with MAE-DFER on DFEW.
3. If DFEW access is delayed, DFER-CLIP can be tested on a subset of publicly available data.
4. More parameter-efficient fine-tuning (PEFT) options (LoRA, adapters) are being actively explored for CLIP-based DFER (PE-CLIP, FineCLIPER).

**Decision priority satisfied:**

| Priority | Requirement | MAE-DFER + DFEW |
|---|---|---|
| 1 | Video emotion recognition performance | ✅ 63.4% UAR on standard benchmark |
| 2 | Pretrained weights available | ✅ VoxCeleb2 + DFEW fine-tuned |
| 3 | Trainable on RTX 3050 6 GB | ✅ With bf16, batch 2, grad accum |
| 4 | 6-emotion compatibility | ✅ Clean drop of neutral |
| 5 | Dataset quality | ✅ In-the-wild, movie clips |
| 6 | Reproducibility | ✅ Public code + weights |
| 7 | Inference efficiency | ✅ 38% fewer FLOPs vs VideoMAE |

---

## 4. CMU-MOSEI as Baseline (Task 5)

The CMU-MOSEI FACET pipeline will be retained as a documented baseline:

```
CMU-MOSEI VisualFacet42 (35-dim features per frame)
        ↓
Temporal mean-pool → (35,) per utterance
        ↓
MLP: Linear(35→128) → GELU → Dropout(0.2) → Linear(128→6)
        ↓
6-class softmax + class-weighted CE loss
```

**Baseline training config:**
- Optimizer: AdamW, lr=1e-3, weight_decay=0.01
- Epochs: 30 with early stopping (patience=5)
- Class weights: inverse frequency
- Expected performance: ~45–55% accuracy (FACET features are informative but limited compared to raw pixel models)

This baseline will be compared against MAE-DFER on the **same 6-class label set** using the same evaluation metrics (Macro F1, Weighted F1, Accuracy, per-class F1, confusion matrix).

**Important:** The CMU-MOSEI results cannot be directly compared to DFEW results because they are different datasets with different video content. The comparison is valid only when both models are evaluated on the same held-out test split. For the multimodal pipeline, both models will produce 512-dim embeddings for fusion — the better one will be used as the video stream encoder.

---

## 5. Full Pipeline Design (Task 6)

### Pipeline diagram

```
DFEW video clip (16 frames pre-extracted as jpg_256)
        ↓
[Frame Sampling]
  - 16 frames uniformly sampled from clip
  - Strategy: uniform temporal stride across all frames
        ↓
[Face Alignment / Crop]
  - DFEW provides pre-aligned face crops — use directly
  - If not pre-cropped: MTCNN or RetinaFace → 5-point alignment → crop
        ↓
[Resize]  112×112 (MAE-DFER standard)
        ↓
[Augmentation — training only]
  - RandomHorizontalFlip (p=0.5)
  - RandomCrop(112, padding=8)
  - ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
  - RandomErasing(p=0.3, scale=(0.02, 0.15))
  - NO temporal augmentation (preserve emotion dynamics)
        ↓
[Normalization]
  - Mean: [0.485, 0.456, 0.406] (ImageNet)
  - Std:  [0.229, 0.224, 0.225]
        ↓
[Tensor shape]  (B, T, C, H, W) = (batch, 16, 3, 112, 112)
        ↓
[MAE-DFER LGI-Former encoder]
  - Load pretrained weights from VoxCeleb2 pre-training
  - Fine-tune from DFEW fold checkpoint (optional — skip pre-training)
        ↓
[Temporal aggregation]
  - LGI-Former's built-in local-global interaction
  - Output: (B, D) = (batch, 768) class token
        ↓
[Classification head]
  - Linear(768 → 6)
  - Dropout(0.5) before head
        ↓
[6 emotion probabilities]  (softmax at inference)
        ↓
[Projection head for fusion]
  - Linear(768 → 512) → LayerNorm → GELU → Dropout(0.1)
  - 512-dim video embedding for Cross-Modal Attention Transformer
```

### Data splits

DFEW uses 5-fold cross-validation as its standard protocol. For reproducibility:
- Use fold 1 as the primary evaluation split (train: ~13,100, test: ~3,272)
- Report averaged results across all 5 folds for final comparison

### Class balancing strategy

DFEW (like most FER datasets) is imbalanced. Apply:
1. **Class-weighted cross-entropy loss:**
   ```python
   weights = num_samples / (num_classes × class_counts)
   loss_fn = CrossEntropyLoss(weight=class_weights.to(device))
   ```
2. **Weighted random sampler** during training (oversample minority classes)
3. Track macro F1 (not accuracy) as the primary metric for model selection

### Training configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for transformers |
| Learning rate | 1e-4 (head), 5e-6 (backbone) | Layer-wise LR decay for fine-tuning |
| Weight decay | 0.05 | Transformer fine-tuning standard |
| Epochs | 50 | With early stopping |
| Early stopping patience | 10 epochs | Watch macro F1 on val |
| Batch size | 4 | 6 GB VRAM constraint |
| Gradient accumulation | 4 steps | Effective batch 16 |
| Scheduler | Cosine decay with linear warmup (5 epochs) | Standard for ViT fine-tuning |
| Loss | CrossEntropyLoss with class weights | Handles imbalance |
| Mixed precision | bf16 (NOT fp16) | Ampere supports bf16; more stable |
| Gradient checkpointing | Yes (if OOM occurs) | Reduces peak VRAM ~40% |
| Input frames | 16 | MAE-DFER standard |
| Input resolution | 112×112 | MAE-DFER standard |
| Seed | 42 | Reproducibility |

### Evaluation metrics

| Metric | How | Why |
|---|---|---|
| **Macro F1** | `sklearn.metrics.f1_score(average='macro')` | **Primary** — equal weight per class |
| Weighted F1 | `f1_score(average='weighted')` | Accounts for class frequency |
| Accuracy (WAR) | `accuracy_score` | Standard DFEW metric = WAR |
| UAR | Mean per-class recall | Standard DFEW metric |
| Per-class F1 | `f1_score(average=None)` | Reveals which classes fail |
| Confusion matrix | `confusion_matrix` | Visual analysis |

> **UAR (Unweighted Average Recall) = Macro Recall.** The DFEW benchmark reports both UAR and WAR. UAR is the standard primary metric in DFEW papers. We additionally report Macro F1 as it accounts for both precision and recall.

---

## 6. RTX 3050 Training Configuration (Task 7)

The RTX 3050 Laptop GPU has 6.0 GB VRAM and compute capability 8.6 (Ampere), which supports bf16 natively.

### Memory breakdown for MAE-DFER fine-tuning

| Component | Estimated VRAM |
|---|---|
| Model weights (~86M × 4 bytes FP32) | ~344 MB |
| Model weights in bf16 | ~172 MB |
| Activations (batch=4, 16×112×112×3) | ~800 MB |
| Gradient storage (bf16) | ~172 MB |
| Optimizer state (AdamW, 2 moments) | ~344 MB |
| **Total estimate** | **~1.5 GB** |
| Overhead / fragmentation | ~2× → ~3 GB |
| **Realistic peak** | **~4–5 GB** |

This fits within 6 GB with margin for gradient checkpointing if needed.

### Configuration that fits in 6 GB

```python
TrainingConfig(
    batch_size            = 4,
    gradient_accumulation = 4,          # Effective batch = 16
    num_frames            = 16,
    image_size            = 112,
    precision             = "bf16",     # NOT fp16 — bf16 on Ampere
    gradient_checkpointing = False,     # Enable if OOM at batch=4
    pin_memory            = True,
    num_workers           = 2,          # Low to avoid CPU RAM pressure
    learning_rate_head    = 1e-4,
    learning_rate_backbone = 5e-6,      # 20× lower than head
    warmup_epochs         = 5,
    max_epochs            = 50,
    early_stopping_patience = 10,
    weight_decay          = 0.05,
)
```

### If OOM occurs

1. Enable `gradient_checkpointing=True` → reduces activation memory ~40%
2. Reduce `batch_size` to 2, increase `gradient_accumulation` to 8
3. Reduce `num_frames` to 8 (minor accuracy penalty)
4. Reduce `image_size` to 96 (slight accuracy penalty)

### Pre-training strategy (strongly recommended)

**Do NOT train from scratch.** The fine-tuning path is:

```
MAE-DFER pretrained on VoxCeleb2 (public download)
        ↓
Fine-tune on DFEW fold 1 (16K clips, 50 epochs)
        ↓
Our model (adapts DFEW's fine-tuned checkpoint to 6-class)
```

Alternatively, use the **already fine-tuned DFEW fold checkpoints** from the MAE-DFER repo and adapt the classification head from 7→6 classes by removing the neutral output neuron and re-training the head for a few epochs. This is the fastest path and requires the least GPU time.

---

## 7. Directory Structure (Task 8)

The following structure adapts to the existing project layout. New directories are marked `[NEW]`.

```
data/
├── raw/
│   ├── cmu_mosei/          (existing — DO NOT DELETE)
│   ├── meld/               (existing)
│   ├── ravdess/            (existing)
│   └── video_fer/          [NEW]
│       └── dfew/           [NEW] — DFEW clips go here after download
│           ├── jpg_256/    — pre-cropped face sequences (DFEW format)
│           └── labels/     — CSV annotation files per fold
└── processed/
    ├── goemotions/         (existing)
    ├── ravdess/            (existing)
    └── video_fer/          [NEW]
        └── dfew/           [NEW]
            ├── fold1_train.csv
            ├── fold1_val.csv
            ├── fold1_test.csv
            └── label_mapping.json

models/
└── video/
    ├── VIDEO_DATA_STATUS.md    (this inspection report)
    ├── VIDEO_FER_RESEARCH_REPORT.md  (this research report)
    ├── video_dataset_inspection_report.md  (existing)
    ├── video_integration_plan.md           (existing)
    ├── checkpoints/        [NEW]
    │   ├── mae_dfer_pretrained/    — VoxCeleb2 pretrained weights
    │   └── mae_dfer_dfew/         — fine-tuned on DFEW
    ├── configs/            [NEW]
    │   └── mae_dfer_dfew.yaml
    └── results/            [NEW]
        ├── dfew_fold1/
        └── cmu_mosei_baseline/

scripts/
└── video/                  [NEW]
    ├── prepare_dfew.py         — build annotation CSVs from DFEW folder structure
    ├── inspect_dfew.py         — dataset stats, class distribution, sanity checks
    ├── train_mae_dfer.py       — main training loop (MAE-DFER fine-tuning)
    ├── train_cmu_mosei_baseline.py  — FACET MLP baseline
    ├── evaluate.py             — generate metrics, confusion matrix, per-class F1
    └── inference.py            — run prediction on a single video clip
```

---

## 8. Benchmark Plan (Task 9)

### Three-way comparison

| System | Dataset | Architecture | Input type | Status |
|---|---|---|---|---|
| **Baseline** | CMU-MOSEI | FACET MLP | 35-dim precomputed features | Data available now |
| **Primary** | DFEW | MAE-DFER fine-tuned | Raw face video sequences | Requires DFEW download |
| **Alternative** | DFEW | DFER-CLIP fine-tuned | Raw face video sequences | Same download as primary |

### Evaluation protocol

All three systems will be evaluated on their respective **held-out test splits** using identical metrics:

```python
metrics = {
    "macro_f1":     f1_score(y_true, y_pred, average='macro'),
    "weighted_f1":  f1_score(y_true, y_pred, average='weighted'),
    "accuracy":     accuracy_score(y_true, y_pred),
    "uar":          recall_score(y_true, y_pred, average='macro'),   # = macro recall
    "per_class_f1": f1_score(y_true, y_pred, average=None),
    "confusion_matrix": confusion_matrix(y_true, y_pred),
}
```

**6 emotion classes for all systems:**
```
0 = angry
1 = disgust
2 = fearful
3 = happy
4 = sad
5 = surprised
```

**Important caveat:** The baseline (CMU-MOSEI) and the primary model (DFEW) use different datasets. A direct number-to-number comparison of their final accuracies is **not valid** since the test sets are different. The meaningful comparison is:

1. Relative performance within each dataset's own standard split
2. Both models' embeddings tested in the multimodal fusion pipeline on the same downstream task (stress/burnout prediction)

---

## 9. What to Do Next

**Step 1 — Request DFEW access (do this now)**
- Go to [dfew-dataset.github.io](https://dfew-dataset.github.io/)
- Fill in the academic access form
- Download ~15 GB when approved
- Place under `data/raw/video_fer/dfew/`

**Step 2 — While waiting for DFEW access**
- Train the CMU-MOSEI FACET baseline (data already available)
- Install video dependencies: `pip install timm einops decord opencv-python torchvision`
- Download MAE-DFER pretrained weights from VoxCeleb2 (public, no request needed)

**Step 3 — After DFEW arrives**
- Run `scripts/video/prepare_dfew.py` to build fold annotation CSVs
- Run `scripts/video/train_mae_dfer.py` on fold 1
- Evaluate and compare against baseline

**Step 4 — Multimodal integration**
- Both models produce 512-dim embeddings
- Feed into Cross-Modal Attention Transformer alongside text (RoBERTa) and audio (emotion2vec)

---

*Research report generated: 2026-10-08 | No code was modified during this research phase.*  
*Content was derived from published papers and public repositories. Published benchmark numbers are from original papers and are not our own experimental results.*
