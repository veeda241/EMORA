# Audio Emotion Recognition — Full Training Report

**Date:** 2026-10-08  
**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
**Task:** 6-class utterance-level speech emotion classification  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Model Architectures](#2-model-architectures)
   - 2.1 [Wav2Vec2](#21-wav2vec2--facebookwav2vec2-base)
   - 2.2 [WavLM](#22-wavlm--microsoftwavlm-base)
   - 2.3 [emotion2vec](#23-emotion2vec--iicemotion2vec_base_finetuned)
3. [Training Configuration](#3-training-configuration)
4. [Per-Model Training Results](#4-per-model-training-results)
   - 4.1 [Wav2Vec2 Results](#41-wav2vec2-results)
   - 4.2 [WavLM Results](#42-wavlm-results)
   - 4.3 [emotion2vec Results](#43-emotion2vec-results)
5. [Model Comparison](#5-model-comparison)
6. [Per-Class Analysis](#6-per-class-analysis)
7. [Key Observations](#7-key-observations)
8. [Output Artifacts](#8-output-artifacts)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Dataset | RAVDESS |
| Total samples | 1,152 |
| Train split | 768 samples |
| Validation split | 192 samples |
| Test split | 192 samples |
| Classes | 6 |
| Samples per class (test) | 32 (balanced) |
| Audio format | WAV, 16 kHz mono |

**Emotion classes:** `angry`, `disgust`, `fearful`, `happy`, `sad`, `surprised`

> Note: RAVDESS originally includes *neutral* and *calm* emotions. These were excluded to form a 6-class balanced benchmark consistent with prior work on the dataset.

---

## 2. Model Architectures

### 2.1 Wav2Vec2 — `facebook/wav2vec2-base`

**Origin:** Meta AI (Facebook), 2020  
**Paper:** [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)  

**Architecture overview:**

Wav2Vec2 is a self-supervised speech representation model that operates directly on raw audio waveforms. It consists of two main components:

- **Feature encoder:** A multi-layer 1D CNN that maps raw waveform samples into latent speech representations at a stride of 20ms per frame.
- **Transformer encoder:** A 12-layer transformer (BERT-style) that contextualises the CNN features across the full utterance. The base model has 94M parameters with a hidden size of 768.

**Pre-training:** The model is pre-trained using a contrastive task — it learns to identify the correct quantized speech unit for masked positions from a set of distractors. This forces it to learn rich acoustic representations without any labels.

**Fine-tuning for classification:** A sequence classification head (`Wav2Vec2ForSequenceClassification`) is added on top. The final hidden state is mean-pooled over all time steps, passed through a projector layer, and then a linear classifier. The classifier and projector weights are randomly initialized and trained from scratch; the transformer backbone is fine-tuned end-to-end.

| Parameter | Value |
|---|---|
| Total parameters | ~94M |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| CNN feature extractor layers | 7 |
| Input | Raw waveform (16 kHz) |
| Output (classification) | Mean-pooled hidden state → projector → classifier |

---

### 2.2 WavLM — `microsoft/wavlm-base`

**Origin:** Microsoft Research, 2021  
**Paper:** [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)  

**Architecture overview:**

WavLM shares the same overall architecture as Wav2Vec2 (CNN feature extractor + Transformer encoder) but introduces two key improvements during pre-training:

- **Masked speech prediction:** Like Wav2Vec2, WavLM predicts masked speech frames. However, it uses a gated relative position bias in the transformer to better capture temporal structure.
- **Denoising pre-training:** WavLM is uniquely pre-trained with a *denoising* objective — it receives speech corrupted with additive noise or overlapping speech and must predict the original clean speech frames. This makes it significantly more robust to speaker variation, background noise, and paralinguistic cues like emotion.

This denoising objective is why WavLM consistently outperforms Wav2Vec2 on non-ASR tasks such as emotion recognition, speaker verification, and sound event detection.

**Fine-tuning for classification:** Same head as Wav2Vec2 — `WavLMForSequenceClassification` with a mean-pooled projector + linear classifier on top.

| Parameter | Value |
|---|---|
| Total parameters | ~94M |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| CNN feature extractor layers | 7 |
| Input | Raw waveform (16 kHz) |
| Key advantage over Wav2Vec2 | Denoising pre-training objective |

---

### 2.3 emotion2vec — `iic/emotion2vec_base_finetuned`

**Origin:** DAMO Academy (Alibaba) / Shanghai Jiao Tong University, 2023  
**Paper:** [emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation](https://arxiv.org/abs/2312.15185)  
**Hub:** [ModelScope — iic/emotion2vec_base_finetuned](https://modelscope.cn/models/iic/emotion2vec_base_finetuned)  

**Architecture overview:**

emotion2vec is purpose-built for speech emotion representation. It adapts the data2vec self-supervised framework specifically for emotional speech by combining:

- **Frame-level loss:** The model predicts masked frame representations, forcing it to learn fine-grained acoustic features that carry emotional content (pitch, energy, voice quality).
- **Utterance-level loss:** A global utterance-level contrastive objective ensures the overall embedding captures the holistic emotional state of the entire utterance.

The backbone produces a 768-dimensional embedding per time frame. For utterance-level classification, frame embeddings are mean-pooled into a single fixed-size vector.

**Fine-tuning setup used in this project:**

Because emotion2vec does not natively expose a `ForSequenceClassification` head compatible with HuggingFace's training API, a custom `Emotion2VecClassifier` wrapper was used:

```
emotion2vec backbone (frozen during embedding extraction)
        ↓
  Mean-pool over time frames → (768,)
        ↓
  Linear(768 → 256)
  LayerNorm(256)
  GELU activation
  Dropout(0.1)
  Linear(256 → 6)
        ↓
  CrossEntropyLoss
```

The backbone runs on CPU (FunASR interface), while the classification head trains on GPU. Only the head parameters are updated by the optimizer — the backbone embeddings are used as frozen features. This explains why emotion2vec trains faster than the other two models despite being the strongest performer.

| Parameter | Value |
|---|---|
| Backbone embedding dim | 768 |
| Classification head | Linear(768→256) → LN → GELU → Dropout → Linear(256→6) |
| Backbone trainable | No (frozen, CPU inference) |
| Head trainable | Yes (GPU) |
| Input | Raw waveform (16 kHz) via FunASR |
| Pre-training data | Large-scale emotion speech corpus (self-supervised) |
| Key advantage | Domain-specific emotional pre-training |

---

## 3. Training Configuration

All three models share the same training hyperparameters for a fair comparison.

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Gradient clip norm | 1.0 |
| Mixed precision (AMP) | Yes (CUDA) |
| Random seed | 42 |
| Best model selection | Lowest validation loss |
| Log interval | Every 20 steps |
| Dataset | RAVDESS (6-class) |
| Train / Val / Test | 768 / 192 / 192 |

---

## 4. Per-Model Training Results

### 4.1 Wav2Vec2 Results

**Checkpoint:** `facebook/wav2vec2-base`  
**Training time:** 273.5 s (~4.6 min)

#### Per-epoch metrics

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro F1 |
|---|---|---|---|---|
| 1 | 1.6972 | 1.5667 | 0.4010 | 0.3593 |
| 2 | 1.3752 | 1.4217 | 0.4583 | 0.4445 |
| 3 | 1.0227 | 1.2252 | 0.5833 | 0.5667 |
| 4 | 0.6715 | 1.0689 | 0.6354 | 0.6340 |
| 5 | 0.3848 | 1.0128 | 0.6927 | 0.7014 |
| 6 | 0.2383 | 1.2508 | 0.6667 | 0.6710 |
| 7 | 0.1346 | 1.3014 | 0.6927 | 0.6832 |
| 8 | 0.1084 | 1.1705 | 0.7448 | 0.7494 |
| 9 | 0.0947 | 1.1557 | 0.7656 | 0.7672 |
| **10** | **0.0757** | **1.1006** | **0.7812** | **0.7830** |

> Best model saved at **epoch 10** (lowest dev loss = 1.1006)

#### Test set results

| Metric | Value |
|---|---|
| Accuracy | **68.23%** |
| Macro F1 | **0.6688** |
| Weighted F1 | 0.6688 |

#### Per-class performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.658 | 0.781 | 0.714 | 32 |
| disgust | 1.000 | 0.375 | 0.545 | 32 |
| fearful | 0.882 | 0.469 | 0.612 | 32 |
| happy | 0.525 | 0.656 | 0.583 | 32 |
| sad | 0.714 | 0.938 | 0.811 | 32 |
| surprised | 0.651 | 0.875 | 0.747 | 32 |

**Observation:** Wav2Vec2 struggles with `disgust` (recall only 0.375) and `fearful` (recall 0.469), suggesting its general-purpose acoustic features conflate these more subtle emotions with others. The training loss drops sharply but dev loss increases after epoch 5, indicating overfitting.

---

### 4.2 WavLM Results

**Checkpoint:** `microsoft/wavlm-base`  
**Training time:** 345.7 s (~5.8 min)

#### Per-epoch metrics

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro F1 |
|---|---|---|---|---|
| 1 | 1.7171 | 1.6026 | 0.3750 | 0.3060 |
| 2 | 1.3781 | 1.3415 | 0.5573 | 0.5200 |
| 3 | 1.0002 | 1.0630 | 0.6719 | 0.6751 |
| 4 | 0.6679 | 1.1542 | 0.6198 | 0.6162 |
| 5 | 0.3975 | **0.8967** | **0.7552** | **0.7594** |
| 6 | 0.2766 | 1.0399 | 0.7344 | 0.7334 |
| 7 | 0.2244 | 1.3471 | 0.6823 | 0.6809 |
| 8 | 0.1492 | 1.9368 | 0.6094 | 0.6082 |
| 9 | 0.1466 | 1.5757 | 0.7135 | 0.7098 |
| 10 | 0.0703 | 1.3883 | 0.7500 | 0.7509 |

> Best model saved at **epoch 5** (lowest dev loss = 0.8967)

#### Test set results

| Metric | Value |
|---|---|
| Accuracy | **72.40%** |
| Macro F1 | **0.7111** |
| Weighted F1 | 0.7111 |

#### Per-class performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.625 | 0.938 | 0.750 | 32 |
| disgust | 0.789 | 0.469 | 0.588 | 32 |
| fearful | 0.882 | 0.469 | 0.612 | 32 |
| happy | 0.688 | 0.688 | 0.688 | 32 |
| sad | 0.757 | 0.875 | 0.812 | 32 |
| surprised | 0.744 | 0.906 | 0.817 | 32 |

**Observation:** WavLM's denoising pre-training provides a clear improvement over Wav2Vec2 (+4.2% accuracy). However, it still struggles with `disgust` and `fearful` similarly. The best validation checkpoint is at epoch 5 — after that, the model overfits noticeably (dev loss spikes to 1.94 at epoch 8).

---

### 4.3 emotion2vec Results

**Checkpoint:** `iic/emotion2vec_base_finetuned`  
**Training time:** 232.1 s (~3.9 min)

#### Per-epoch metrics

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro F1 |
|---|---|---|---|---|
| **1** | 1.0373 | **0.6778** | **0.8438** | **0.8454** |
| 2 | 0.3477 | 0.5393 | 0.8490 | 0.8496 |
| 3 | 0.2283 | 0.5617 | 0.8385 | 0.8393 |
| 4 | 0.1954 | 0.5860 | 0.8333 | 0.8341 |
| 5 | 0.1864 | 0.6077 | 0.8385 | 0.8391 |
| 6 | 0.1739 | 0.6242 | 0.8333 | 0.8342 |
| 7 | 0.1650 | 0.6373 | 0.8281 | 0.8297 |
| 8 | 0.1558 | 0.6436 | 0.8333 | 0.8348 |
| 9 | 0.1664 | 0.6564 | 0.8385 | 0.8404 |
| 10 | 0.1613 | 0.6747 | 0.8281 | 0.8306 |

> Best model saved at **epoch 2** (lowest dev loss = 0.5393)

#### Test set results

| Metric | Value |
|---|---|
| Accuracy | **92.71%** |
| Macro F1 | **0.9271** |
| Weighted F1 | 0.9271 |

#### Per-class performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 1.000 | 1.000 | **1.000** | 32 |
| disgust | 0.889 | 1.000 | **0.941** | 32 |
| fearful | 0.929 | 0.813 | **0.867** | 32 |
| happy | 1.000 | 0.938 | **0.968** | 32 |
| sad | 0.794 | 0.844 | **0.818** | 32 |
| surprised | 0.969 | 0.969 | **0.969** | 32 |

**Observation:** emotion2vec achieves 84% accuracy from epoch 1 alone — before any classifier training has converged — demonstrating the power of its emotion-specific pre-training. The best checkpoint is at epoch 2 (dev loss 0.5393); subsequent epochs show gentle overfitting as the head memorizes training data. Notably, it is also the **fastest** model to train despite producing by far the best results, because only the lightweight classification head is updated.

---

## 5. Model Comparison

### Overall performance

| Model | Checkpoint | Test Accuracy | Macro F1 | Weighted F1 | Train Time | Best Epoch |
|---|---|---|---|---|---|---|
| Wav2Vec2 | `facebook/wav2vec2-base` | 68.23% | 0.669 | 0.669 | 273.5 s | 10 |
| WavLM | `microsoft/wavlm-base` | 72.40% | 0.711 | 0.711 | 345.7 s | 5 |
| **emotion2vec** | `iic/emotion2vec_base_finetuned` | **92.71%** | **0.927** | **0.927** | **232.1 s** | 2 |

### Accuracy improvement over baseline

| Comparison | Accuracy gain |
|---|---|
| WavLM vs Wav2Vec2 | +4.2 pp |
| emotion2vec vs Wav2Vec2 | **+24.5 pp** |
| emotion2vec vs WavLM | **+20.3 pp** |

### Training dynamics summary

| Model | Overfitting pattern | Generalisation |
|---|---|---|
| Wav2Vec2 | Train loss → 0.076, Dev loss → 1.10 (gap = 1.02) | Significant overfitting after epoch 5 |
| WavLM | Train loss → 0.070, Dev loss → 1.39 (gap = 1.32) | Stronger overfitting; best at epoch 5 |
| emotion2vec | Train loss → 0.161, Dev loss → 0.675 (gap = 0.51) | Minimal overfitting; stable dev accuracy throughout |

---

## 6. Per-Class Analysis

### F1 score by class across all models

| Class | Wav2Vec2 F1 | WavLM F1 | emotion2vec F1 | Best model |
|---|---|---|---|---|
| angry | 0.714 | 0.750 | **1.000** | emotion2vec |
| disgust | 0.545 | 0.588 | **0.941** | emotion2vec |
| fearful | 0.612 | 0.612 | **0.867** | emotion2vec |
| happy | 0.583 | 0.688 | **0.968** | emotion2vec |
| sad | 0.811 | 0.812 | **0.818** | emotion2vec (marginal) |
| surprised | 0.747 | 0.817 | **0.969** | emotion2vec |

**Hardest class:** `sad` — all three models score similarly here (0.81–0.82 F1), suggesting `sad` shares acoustic overlap with other low-energy emotions regardless of the model used.

**Most improved by emotion2vec:** `disgust` (+0.353 F1 vs Wav2Vec2) and `fearful` (+0.255 F1), which are the two classes where general speech models consistently fail.

---

## 7. Key Observations

1. **Domain-specific pre-training dominates.** emotion2vec's 24.5 percentage point advantage over Wav2Vec2 demonstrates that pre-training on emotion-labelled or emotion-relevant data is far more effective than adapting a general ASR backbone with a classification head.

2. **emotion2vec reaches 84% accuracy at epoch 1.** The frozen backbone already encodes rich enough emotion features that even a randomly initialized head achieves strong performance after one pass. This contrasts sharply with Wav2Vec2 and WavLM, which need several epochs to reach comparable accuracy.

3. **WavLM's denoising helps but is not enough.** WavLM's denoising objective provides a meaningful 4.2 pp improvement over Wav2Vec2, confirming that noise-robustness pre-training helps with paralinguistic tasks. However, it does not close the gap to a purpose-built emotion model.

4. **Overfitting is more severe for general models.** Both Wav2Vec2 and WavLM show large train/dev loss gaps (>1.0), indicating that the full-model fine-tuning approach overfits the small RAVDESS training set (768 samples). emotion2vec's frozen-backbone approach naturally regularises this.

5. **emotion2vec is the recommended model** for downstream use in the conversational agent pipeline. At 92.7% accuracy on 6-class RAVDESS with only 232 seconds of training, it is both the most accurate and the most efficient option.

---

## 8. Output Artifacts

| File | Description |
|---|---|
| `models/audio/runs_wav2vec2/` | Wav2Vec2 fine-tuned weights + processor config |
| `models/audio/runs_wavlm/` | WavLM fine-tuned weights + processor config |
| `models/audio/runs_emotion2vec/model_state_dict.pt` | emotion2vec classification head weights |
| `models/audio/wav2vec2_confusion_matrix.png` | Wav2Vec2 test confusion matrix |
| `models/audio/wavlm_confusion_matrix.png` | WavLM test confusion matrix |
| `models/audio/emotion2vec_confusion_matrix.png` | emotion2vec test confusion matrix |
| `models/audio/wav2vec2_classification_report.csv` | Wav2Vec2 per-class metrics |
| `models/audio/wavlm_classification_report.csv` | WavLM per-class metrics |
| `models/audio/emotion2vec_classification_report.csv` | emotion2vec per-class metrics |
| `models/audio/wav2vec2_training_report.md` | Wav2Vec2 individual training report |
| `models/audio/wavlm_training_report.md` | WavLM individual training report |
| `models/audio/emotion2vec_training_report.md` | emotion2vec individual training report |
| `models/audio/AUDIO_MODELS_TRAINING_REPORT.md` | This consolidated report |

---

*Report generated: 2026-10-08 | Training script: `models/train_all_audio_models.py`*
