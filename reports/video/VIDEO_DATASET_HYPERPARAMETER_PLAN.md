# Video FER — Dataset Analysis & Hyperparameter Planning
**Date:** 2026-10-08  
**Purpose:** Detailed dataset profile to guide hyperparameter decisions for both the CMU-MOSEI baseline and the MAE-DFER model on DFEW.

---

## Table of Contents

1. [Dataset 1 — CMU-MOSEI FACET 4.2 (Available Now)](#1-dataset-1--cmu-mosei-facet-42-available-now)
2. [Dataset 2 — DFEW (Pending Download)](#2-dataset-2--dfew-pending-download)
3. [Class Imbalance Analysis](#3-class-imbalance-analysis)
4. [Hyperparameter Plan — CMU-MOSEI MLP Baseline](#4-hyperparameter-plan--cmu-mosei-mlp-baseline)
5. [Hyperparameter Plan — MAE-DFER on DFEW](#5-hyperparameter-plan--mae-dfer-on-dfew)
6. [Hyperparameter Comparison Table](#6-hyperparameter-comparison-table)
7. [What the Baseline Results Tell Us](#7-what-the-baseline-results-tell-us)

---

## 1. Dataset 1 — CMU-MOSEI FACET 4.2 (Available Now)

### Source files

| File | Size | Status |
|---|---|---|
| `data/raw/cmu_mosei/CMU_MOSEI_VisualFacet42.csd` | 1,656 MB | ✅ Readable |
| `data/raw/cmu_mosei/CMU_MOSEI_Labels.csd` | 23 MB | ✅ Readable |

### Processed output

| File | Description |
|---|---|
| `data/processed/video_fer/cmu_mosei/features.npy` | Shape (3293, 35) float32 |
| `data/processed/video_fer/cmu_mosei/labels.npy` | Shape (3293,) int64 |
| `data/processed/video_fer/cmu_mosei/metadata.json` | Class counts, label mapping |

### Dataset statistics (actual measured values)

| Property | Value |
|---|---|
| Total videos with labels | 3,293 |
| Feature dimension per sample | 35 (FACET 4.2 Action Units, mean-pooled across frames) |
| Feature range | −25.7 to +22.3 |
| Feature mean | −0.845 |
| Feature std | 1.889 |
| Min frames per video | 344 |
| Max frames per video | 33,550 |
| Mean frames per video | 3,276 |
| Median frames per video | 2,459 |
| Train / Val / Test split | 2,305 / 494 / 494 (70/15/15, stratified) |

### FACET 4.2 feature description

The 35-dim feature vector contains pre-extracted facial action unit intensities per frame, mean-pooled to a single video-level vector:

| Index | Feature |
|---|---|
| 0–17 | Action Unit intensities (AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU18, AU20, AU23, AU24, AU25, AU26) |
| 18–23 | Head pose (pitch, yaw, roll + velocity) |
| 24–27 | Eye gaze direction (left/right x, y) |
| 28–30 | 2D face location (x, y, scale) |
| 31–33 | 3D head rotation (Euler angles) |
| 34 | Face confidence score |

### Class distribution (CMU-MOSEI)

| Class | ID | Count | % of total | Train count |
|---|---|---|---|---|
| angry | 0 | 350 | 10.6% | 245 |
| disgust | 1 | 133 | 4.0% | 93 |
| fearful | 2 | 75 | 2.3% | 52 |
| **happy** | 3 | **2,111** | **64.1%** | **1,478** |
| sad | 4 | 594 | 18.0% | 416 |
| surprised | 5 | 30 | 0.9% | 21 |

**Imbalance ratio (max/min): 70.4×**  
`happy` (2,111) vs `surprised` (30) — extreme skew.

### Class weight calculation (inverse frequency)

Used in `CrossEntropyLoss(weight=...)` to penalise majority class predictions:

```
weight[c] = N_total / (N_classes × N_c)
```

| Class | Count | Weight |
|---|---|---|
| angry | 350 | **0.94** |
| disgust | 133 | **2.47** |
| fearful | 75 | **4.38** |
| happy | 2,111 | **0.16** |
| sad | 594 | **0.55** |
| surprised | 30 | **10.98** |

`surprised` gets ~69× more gradient signal than `happy` per sample.

---

## 2. Dataset 2 — DFEW (Pending Download)

### Access

- Website: [dfew-dataset.github.io](https://dfew-dataset.github.io/)
- License: Non-commercial academic research
- Process: Fill in request form → receive download link (typically 1–3 days)

### Expected data layout after download

```
data/raw/video_fer/dfew/
├── Clip/
│   └── jpg_256/
│       ├── 00001/   ← pre-aligned face frame sequences (jpg_256 = 256px)
│       ├── 00002/
│       └── ...      (16,372 clip folders total)
└── EmoLabel_DataSplit/
    ├── train(single-labeled)/
    │   ├── set_1.csv   ← columns: video_name, label (1-indexed, 1-7)
    │   ├── set_2.csv
    │   ├── set_3.csv
    │   ├── set_4.csv
    │   └── set_5.csv
    └── test(single-labeled)/
        ├── set_1.csv
        └── ...
```

### Expected dataset statistics (from published paper)

| Property | Value |
|---|---|
| Total clips | 16,372 |
| Source | In-the-wild movie clips |
| Face detection | Pre-cropped and aligned by the dataset authors |
| Resolution | 256×256 per frame (pre-cropped) → resize to 160×160 for MAE-DFER |
| Original classes | 7: happy, sad, neutral, angry, surprise, disgust, fear |
| Our classes | 6: drop `neutral` → ~2,750 neutral samples removed |
| Evaluation protocol | 5-fold cross-validation (standard) |
| Frames per clip | Varies; typically 8–80 frames |
| Published SOTA (MAE-DFER) | 63.41% UAR / 74.43% WAR (5-fold avg, 7 classes) |

### DFEW label mapping (DFEW → our 6-class)

| DFEW label | DFEW name | Our ID | Our name |
|---|---|---|---|
| 1 | happy | 3 | happy |
| 2 | sad | 4 | sad |
| 3 | neutral | — | **DROPPED** |
| 4 | angry | 0 | angry |
| 5 | surprise | 5 | surprised |
| 6 | disgust | 1 | disgust |
| 7 | fear | 2 | fearful |

### Expected DFEW class distribution (from paper, approximate)

| Class | Approx. % |
|---|---|
| happy | ~28% |
| sad | ~18% |
| angry | ~18% |
| surprise | ~12% |
| disgust | ~12% |
| fear | ~12% |

DFEW is **significantly more balanced** than CMU-MOSEI (~2.3× max/min imbalance vs 70× in CMU-MOSEI).

---

## 3. Class Imbalance Analysis

### Why this matters for hyperparameter planning

| Dataset | Imbalance ratio | Primary issue |
|---|---|---|
| CMU-MOSEI | **70.4×** | `surprised` (30 samples) vs `happy` (2,111) — too few minority samples to learn reliably |
| DFEW | **~2.3×** | Mild — happy slightly dominant, other classes reasonable |

### Strategies to handle imbalance

| Strategy | CMU-MOSEI | DFEW |
|---|---|---|
| Class-weighted loss | ✅ Essential | ✅ Recommended |
| Weighted random sampler | ✅ Essential | ✅ Recommended |
| Label smoothing | 0.0 (already struggling) | 0.1 (regularisation) |
| Focal loss | Consider for CMU-MOSEI | Optional |
| Oversample minority | ✅ Already in train script | Not needed |
| Primary metric | **Macro F1** (not accuracy) | **Macro F1** + UAR |

### What happens without class weighting (demonstrated by baseline)

The baseline trained **with** class weighting and weighted sampler, yet still achieved only 0.07 Macro F1. Without weighting, the model would collapse entirely to predicting `happy` → ~64% accuracy but 0.0 Macro F1 for 5 out of 6 classes.

---

## 4. Hyperparameter Plan — CMU-MOSEI MLP Baseline

### Architecture decision rationale

| Choice | Why |
|---|---|
| Input dim: 35 | Fixed by FACET 4.2 features |
| Hidden: 256 → 128 | Large enough to learn non-linear combinations; small enough given 2,305 training samples |
| LayerNorm (not BatchNorm) | Works with any batch size, including last batch size 1 |
| GELU activation | Smooth gradient flow; better than ReLU for small feature spaces |
| Dropout 0.3/0.2 | Regularise heavily — only 2,305 samples, 35 features → high overfitting risk |
| Output: 6 | Six emotion classes |

### Hyperparameter grid — what to try

| Hyperparameter | Baseline used | Grid to explore | Rationale |
|---|---|---|---|
| Learning rate | 1e-3 | 1e-4, **1e-3**, 3e-3 | 1e-3 converged fast; lower may improve minority classes |
| Weight decay | 0.01 | 0.001, **0.01**, 0.1 | Prevent overfitting on tiny dataset |
| Dropout (layer 1) | 0.3 | 0.2, **0.3**, 0.5 | More dropout helps generalisation |
| Dropout (layer 2) | 0.2 | 0.1, **0.2**, 0.3 | |
| Hidden sizes | 256→128 | 128→64, **256→128**, 512→256 | 35 features don't need huge capacity |
| Batch size | 64 | 32, **64**, 128 | Small dataset — smaller batch = more gradient noise = more regularisation |
| Epochs | 30 (ES@5) | 50 (ES@10) | More epochs with lower LR may help minority classes |
| Scheduler | CosineAnnealing | **Cosine**, StepLR (γ=0.5 every 10) | Cosine prevents premature convergence |
| Loss | CE + class weights | **CE+weights**, Focal(γ=2) | Focal loss penalises easy negatives more |
| Sampler | Weighted random | **Weighted**, Standard | Already shows sampler is needed |

### Recommended next run for CMU-MOSEI

```python
# Key changes from baseline:
learning_rate  = 3e-4      # Lower LR for better minority class learning
weight_decay   = 0.05      # Stronger regularisation
dropout_1      = 0.5       # Higher to force generalisation
hidden_dims    = [512, 256] # Wider — more capacity for feature interactions
epochs         = 50
early_stopping = 10
loss           = FocalLoss(gamma=2.0, class_weights=True)
```

### Expected CMU-MOSEI ceiling

Given only 35 mean-pooled features and extreme imbalance (30 `surprised` samples), the **realistic ceiling for this approach is ~20–30% Macro F1**. The FACET features simply lack temporal dynamics. This is the hard ceiling — no amount of hyperparameter tuning will overcome the fundamental data limitation.

---

## 5. Hyperparameter Plan — MAE-DFER on DFEW

### GPU memory budget (RTX 3050 6 GB)

| Component | Estimated VRAM |
|---|---|
| MAE-DFER model weights (bf16) | ~172 MB |
| Activations (batch=4, 16×160×160×3) | ~900 MB |
| Gradients (bf16) | ~172 MB |
| Optimizer state (AdamW, 2 moments) | ~344 MB |
| Overhead / fragmentation | ~1.5 GB |
| **Total estimate** | **~3.1 GB** |
| Safety margin remaining | **~2.9 GB** |

This is safe at batch=4. Gradient checkpointing is available if needed.

### Hyperparameter plan — MAE-DFER fine-tuning

| Hyperparameter | Recommended | Range to try | Rationale |
|---|---|---|---|
| **Input size** | **160×160** | 112, **160** | 160 is MAE-DFER's published size; 112 saves VRAM |
| **Num frames** | **16** | 8, **16** | 16 captures more temporal dynamics; 8 if OOM |
| **Sampling rate** | **4** | 2, **4**, 8 | stride-4 = covers 64 frames at 24fps ≈ 2.7s |
| **Batch size** | **4** | 2, **4** | 4 fits in 6 GB with bf16 |
| **Gradient accumulation** | **4** | 4, 8 | Effective batch 16 or 32 |
| **Effective batch size** | **16** | 16, 32 | 16 matches MAE-DFER paper |
| **LR (head)** | **1e-4** | 5e-5, **1e-4**, 5e-4 | Head is randomly initialised → needs higher LR |
| **LR (backbone)** | **5e-6** | 1e-6, **5e-6**, 1e-5 | Pre-trained → low LR prevents catastrophic forgetting |
| **LR ratio (head/backbone)** | **20×** | 10×, **20×**, 50× | Standard for fine-tuning with frozen-then-unfreeze |
| **Weight decay** | **0.05** | 0.01, **0.05**, 0.1 | Transformer fine-tuning standard |
| **Warmup epochs** | **5** | 3, **5**, 10 | ~10% of total epochs; prevents early instability |
| **Scheduler** | **Cosine + warmup** | Cosine, OneCycleLR | Cosine with warmup is MAE-DFER default |
| **Epochs** | **50** | 30, **50**, 100 | Published MAE-DFER used 100; 50 is practical |
| **Early stopping** | **10** | 5, **10** | DFEW can oscillate — wide patience needed |
| **Label smoothing** | **0.1** | 0.0, **0.1**, 0.2 | Regularisation; helps with in-the-wild noise |
| **Random erasing** | **0.25** | 0.0, **0.25** | Occlusion augmentation — important for in-the-wild |
| **Dropout (head)** | **0.5** | 0.3, **0.5** | Prevents head overfitting |
| **Drop path rate** | **0.1** | 0.0, **0.1**, 0.2 | Stochastic depth for transformer |
| **Loss** | **CE + class weights** | CE+weights, Focal | DFEW mildly imbalanced → weighted CE sufficient |
| **Precision** | **bf16** | bf16 | fp16 can be unstable on DFEW; bf16 on Ampere |

### Layer-wise learning rate decay

MAE-DFER's backbone has 16 transformer blocks. Using a decay factor of 0.8:

```
head         → lr = 1e-4
block 16     → lr = 1e-4 × 0.8^1  = 8.0e-5
block 15     → lr = 1e-4 × 0.8^2  = 6.4e-5
...
block 1      → lr = 1e-4 × 0.8^16 = 2.8e-6
patch embed  → lr = 1e-4 × 0.8^17 = 2.3e-6
```

This is the most principled approach for large transformer fine-tuning.  
**Start with simple 2-group LR (head vs backbone), then switch to layer-wise if results are suboptimal.**

### Recommended training phases

**Phase 1 — Head only (epochs 1–10):**
- Freeze backbone
- Train only the 6-class head
- LR = 1e-3, no warmup
- Purpose: stabilise head before unfreezing backbone

**Phase 2 — Full fine-tuning (epochs 11–50):**
- Unfreeze backbone
- LR (head) = 1e-4, LR (backbone) = 5e-6
- Warmup 5 epochs, then cosine decay
- Early stopping on val Macro F1

This two-phase approach significantly reduces the risk of destroying the pretrained features.

### Augmentation plan

| Augmentation | Train | Val/Test | Parameter |
|---|---|---|---|
| RandomHorizontalFlip | ✅ | ❌ | p=0.5 |
| RandomCrop | ✅ | ❌ | crop=160, pad=8 |
| ColorJitter | ✅ | ❌ | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 |
| RandomErasing | ✅ | ❌ | p=0.25, scale=(0.02, 0.15) |
| AutoAugment (RandAugment) | ✅ | ❌ | m=7, n=4 (MAE-DFER default) |
| CenterCrop | ❌ | ✅ | crop=160 |
| Normalize | ✅ | ✅ | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

**No temporal augmentation** — flipping the temporal order would destroy emotion dynamics (anger builds differently than it fades).

---

## 6. Hyperparameter Comparison Table

| Hyperparameter | CMU-MOSEI MLP | MAE-DFER DFEW |
|---|---|---|
| Architecture | MLP (35→256→128→6) | ViT-based LGI-Former (16 blocks) |
| Input | 35-dim tabular | 16 frames × 160×160 RGB |
| Parameters | ~50K | ~86M |
| Batch size | 64 | 4 |
| Effective batch | 64 | 16 (grad accum ×4) |
| Learning rate | 1e-3 | 1e-4 (head) / 5e-6 (backbone) |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.05 |
| Scheduler | CosineAnnealing | Cosine + linear warmup |
| Epochs | 30 (ES@5) | 50 (ES@10) |
| Mixed precision | CPU (no GPU needed) | bf16 |
| Grad accumulation | None | ×4 |
| Class weights | ✅ Required | ✅ Recommended |
| Weighted sampler | ✅ Required | ✅ Recommended |
| Label smoothing | 0.0 | 0.1 |
| Augmentation | None (tabular) | Flip, Crop, Jitter, Erase |
| Primary metric | **Macro F1** | **Macro F1** + UAR |
| Expected ceiling | ~20–30% Macro F1 | ~60–65% Macro F1 |
| Training time | ~15 sec/run | ~2–3 hrs (50 epochs, RTX 3050) |

---

## 7. What the Baseline Results Tell Us

### Actual baseline test results (CMU-MOSEI, already trained)

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | 0.0628 | Model ignores majority class; lower than random on accuracy |
| **Macro F1** | **0.0679** | Very poor — confirms FACET features alone are insufficient |
| Weighted F1 | 0.0325 | Even worse on frequency-weighted basis |
| UAR | 0.2205 | Above random (0.167) — model learns *something* |
| `angry` F1 | 0.1761 | Best class — 245 training samples, moderate |
| `disgust` F1 | 0.0889 | Weak |
| `fearful` F1 | 0.0993 | Weak |
| **`happy` F1** | **0.0000** | Model never predicts happy despite it being 64% of data — class weights overcorrect |
| `sad` F1 | 0.0430 | Weak |
| **`surprised` F1** | **0.0000** | Only 21 training samples — impossible to learn |
| Epochs to converge | 17 (best at epoch 12) | Fast — tabular model, no complex temporal learning |

### Key lessons for MAE-DFER hyperparameter decisions

1. **`happy` and `surprised` are the critical failure points.** Even with class weighting and sampler, the model couldn't learn them. For DFEW (more balanced), this should improve substantially.

2. **UAR (0.22) > accuracy (0.06)** — the class-weighted training is working in the right direction. It is hurting accuracy (the model avoids predicting `happy`) but improving per-class recall. This is the correct trade-off for Macro F1 optimisation.

3. **The 70× imbalance ratio is the fundamental blocker.** 30 `surprised` samples is below the minimum for reliable learning regardless of model architecture. DFEW's ~2.3× imbalance is incomparably better.

4. **For MAE-DFER:** The rich pretrained features mean the model starts from a much better representation. Even the randomly initialised head will outperform the MLP after one epoch of training on real video data. The primary risks are:
   - **Head LR too high** → destroys backbone features early
   - **No warmup** → unstable early training
   - **fp16 instead of bf16** → gradient instability (same pattern seen in DeBERTa)
   - **Batch too large** → OOM on 6 GB VRAM

5. **Start from the DFEW fold-1 fine-tuned checkpoint** (available in MAE-DFER GitHub) rather than the VoxCeleb2 pretrained checkpoint. This requires only adapting the 7-class head to 6-class (remove `neutral` neuron) and fine-tuning for ~10–20 epochs — far less training time than starting from the general pretrain.

---

## 8. Immediate Next Steps

| Step | Command | Prerequisite |
|---|---|---|
| **Run now** | `python scripts/video/train_cmu_mosei_baseline.py` | ✅ Data ready |
| Request DFEW | Visit [dfew-dataset.github.io](https://dfew-dataset.github.io/) | Manual step |
| Download MAE-DFER pretrained ckpt | See MAE-DFER GitHub releases | Internet access |
| After DFEW arrives | `python scripts/video/prepare_dfew.py --fold 1` | DFEW download |
| Inspect DFEW | `python scripts/video/inspect_dfew.py --fold 1` | prepare_dfew done |
| Train MAE-DFER | `python scripts/video/train_mae_dfer.py --fold 1 --pretrain_ckpt <path>` | All above |

---

*Report generated: 2026-10-08 | Based on actual measured dataset statistics and baseline training results.*
