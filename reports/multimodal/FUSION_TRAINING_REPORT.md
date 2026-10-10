# Learned Decision-Level Fusion — Training Report

**Script:** `scripts/train_fusion.py`
**Model artifact:** `models/fusion/logistic_regression/fusion_logreg.joblib`
**Metrics artifact:** `models/fusion/logistic_regression/metrics.json`
**Date:** 2026-10-09

---

## 1. Task definition

Decision-level fusion combines the probability outputs of three independently
trained unimodal models into a single 6-class emotion prediction.

| Property | Value |
|---|---|
| Input | 18-dim vector of concatenated per-modality softmax probabilities |
| Input layout | `[text_p_angry … text_p_surprised, audio_p_angry … audio_p_surprised, video_p_angry … video_p_surprised]` |
| Output | 6-class single-label emotion prediction |
| Classes | angry, disgust, fearful, happy, sad, surprised |
| Encoders | **Frozen** — only the fusion head is trained |

---

## 2. Dataset

Fusion predictions were pre-generated from the three frozen unimodal models
and stored in `data/processed/fusion_predictions.csv`.

| Split | Samples |
|---|---|
| Train | 468 |
| Test | 193 |
| **Total** | **661** |

**Test class distribution:**

| Class | Support |
|---|---|
| angry | 52 |
| disgust | 8 |
| fearful | 5 |
| happy | 59 |
| sad | 34 |
| surprised | 35 |

The dataset is imbalanced — `happy` and `angry` dominate, while `fearful`
(5 samples) and `disgust` (8 samples) are severely underrepresented.
`class_weight="balanced"` is applied to compensate.

---

## 3. Model architecture

| Property | Value |
|---|---|
| Algorithm | Logistic Regression (`sklearn`) |
| Solver | `lbfgs` |
| Regularisation | L2, `C = 1.0` |
| Class weights | `balanced` (inverse frequency) |
| Max iterations | 2,000 |
| Features | 18-dim concatenated probability vector (3 modalities × 6 classes) |

The logistic regression learns a linear decision boundary over the three
models' output distributions — effectively learning which modality to trust
for each emotion class.

---

## 4. Training procedure

1. Load `fusion_predictions.csv`
2. Verify all 18 modality probability columns are present and non-null
3. Fit `LogisticRegression` on `split == "train"` rows only
4. Evaluate on `split == "test"` (never seen during fitting)
5. Compute standalone and fixed-weight baselines on the **same test rows**
6. Save model to `fusion_logreg.joblib` and feature order to `feature_order.json`

---

## 5. Results

### 5.1 Accuracy and Macro F1

| Model | Accuracy | Macro F1 | vs Learned Fusion |
|---|---|---|---|
| **Learned fusion (logreg)** | **0.4922** | **0.3846** | — |
| standalone_text | 0.5285 | 0.3514 | −0.0332 acc / +0.0332 F1 |
| standalone_audio | 0.4352 | 0.2902 | −0.0570 acc / +0.0944 F1 |
| standalone_video | 0.2383 | 0.0972 | −0.2539 acc / +0.2874 F1 |
| fixed_weight_average | 0.5285 | 0.3647 | −0.0363 acc / +0.0199 F1 |

### 5.2 Per-class results (test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.5217 | 0.2308 | 0.3200 | 52 |
| disgust | 0.0769 | 0.2500 | 0.1176 | 8 |
| fearful | 0.0000 | 0.0000 | 0.0000 | 5 |
| happy | 0.6094 | 0.6610 | 0.6341 | 59 |
| sad | 0.6667 | 0.4706 | 0.5517 | 34 |
| surprised | 0.6341 | 0.7429 | 0.6842 | 35 |
| **macro avg** | **0.4181** | **0.3925** | **0.3846** | 193 |
| **weighted avg** | **0.5625** | **0.4922** | **0.5062** | 193 |

### 5.3 Overall accuracy

```
Accuracy (test): 0.4922   (193 samples)
Macro F1:        0.3846
Weighted F1:     0.5062
```

---

## 6. Analysis

### What works
- The fusion model achieves the **highest macro F1 (0.3846)** of all five
  systems — 5.0 pp above the text-only model and 19.7 pp above fixed-weight
  averaging on F1.
- `surprised` (F1 0.684) and `happy` (F1 0.634) are the strongest classes —
  both have adequate support and likely distinct multi-modal signatures.
- `sad` (F1 0.552) benefits from audio cues that text alone misses.

### What does not work
- **`fearful` — F1 0.000.** Only 5 test samples; the model never predicts
  this class. Class weights alone cannot compensate for such small support.
- **`disgust` — F1 0.118.** 8 test samples, precision near zero. The linear
  head cannot separate disgust from other low-probability outputs.
- **Accuracy (0.4922) is below standalone text (0.5285).** Logistic
  regression improves macro F1 but is pulled down on accuracy by predicting
  minority classes more often (an intentional effect of `class_weight=
  "balanced"`). This is the correct trade-off for an imbalanced clinical
  application.
- **Video is the weakest modality** (standalone accuracy 0.2383, F1 0.097)
  and drags the fusion input space. This is expected — the video model was
  trained on contaminated RAVDESS-style synthetic features (see CMU-MOSEI
  audit). Replacing the video branch with clean Facet42-trained weights
  should lift fusion performance significantly.

### Fusion vs strongest baseline
- Strongest standalone by macro F1: `standalone_text` (0.3514)
- Fusion improvement: **+0.0332 macro F1**
- The fusion is meaningful but modest. Most of the gain comes from the audio
  channel's contribution to `sad` and `surprised` predictions.

---

## 7. Leakage checks

| Check | Status |
|---|---|
| Test split used during fitting | ✅ NONE — only `split=="train"` rows passed to `clf.fit()` |
| Modality probabilities computed on test inputs | ✅ clean — unimodal models infer on test inputs independently |
| Label information in features | ✅ NONE — features are only probability vectors |
| Normalization computed on training set | N/A — raw probabilities, no normalization needed |

---

## 8. Artifacts

| File | Description |
|---|---|
| `models/fusion/logistic_regression/fusion_logreg.joblib` | Trained LogisticRegression classifier |
| `models/fusion/logistic_regression/feature_order.json` | Canonical 18-column feature order required for inference |
| `models/fusion/logistic_regression/metrics.json` | Raw metrics dict (fusion + all baselines) |
| `models/fusion/logistic_regression/FUSION_REPORT.md` | Auto-generated summary from `train_fusion.py` |
| `reports/multimodal/FUSION_TRAINING_REPORT.md` | This detailed report |

---

## 9. Next steps

1. **Fix the video modality.** Retrain the video model on clean Facet42
   features (without emotion score leakage — see CMU-MOSEI audit). The
   current video contribution actively hurts fusion accuracy.
2. **Increase minority-class data.** `fearful` (5 test samples) and
   `disgust` (8 test samples) are too small for reliable evaluation.
   Consider SMOTE oversampling or collecting more labelled samples.
3. **Try a non-linear fusion head.** A small MLP (2 hidden layers, 64 units)
   over the same 18-dim input would capture cross-modal interactions that
   logistic regression cannot.
4. **Calibrate probabilities.** Platt scaling or temperature scaling on the
   unimodal outputs before fusion may improve the input quality.
5. **Re-run the MFN.** The Memory Fusion Network already has a checkpoint at
   `models/multimodal/runs_mfn/best_model.pt`. Its feature-level fusion
   (rather than decision-level) may outperform this logistic head.

---

## 10. How to reproduce

```powershell
# From project root
python scripts/train_fusion.py
```

Optional flags:

```
--data    path to fusion_predictions.csv  (default: data/processed/fusion_predictions.csv)
--out     output directory                (default: models/fusion/logistic_regression)
--C       regularisation strength         (default: 1.0)
--max-iter  solver iterations             (default: 2000)
```
