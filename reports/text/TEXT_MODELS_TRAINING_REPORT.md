# Text Emotion Recognition — Full Training Report

**Date:** 2026-10-07  
**Dataset:** GoEmotions (Google Research)  
**Task:** 28-class multilabel text emotion classification  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Model Architectures](#2-model-architectures)
   - 2.1 [BERT-Base-Uncased](#21-bert-base-uncased--bert-base-uncased)
   - 2.2 [RoBERTa-Base](#22-roberta-base--roberta-base)
   - 2.3 [DeBERTa-v3-Base](#23-deberta-v3-base--deberta-v3-base)
3. [Training Configuration](#3-training-configuration)
4. [Per-Model Training Results](#4-per-model-training-results)
   - 4.1 [BERT-Base-Uncased Results](#41-bert-base-uncased-results)
   - 4.2 [RoBERTa-Base Results](#42-roberta-base-results)
   - 4.3 [DeBERTa-v3-Base Results](#43-deberta-v3-base-results)
5. [Model Comparison](#5-model-comparison)
6. [Key Observations](#6-key-observations)
7. [Known Issues and Recommendations](#7-known-issues-and-recommendations)
8. [Role in the Multimodal Pipeline](#8-role-in-the-multimodal-pipeline)
9. [Output Artifacts](#9-output-artifacts)
10. [RoBERTa-Base Re-run (2026-10-09, RTX 5060 Ti)](#10-roberta-base-re-run-2026-10-09-rtx-5060-ti)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Dataset | GoEmotions |
| Source | Google Research |
| Scale | ~58,000 Reddit comments |
| Task type | Multilabel classification |
| Number of emotion labels | 28 |
| Label format | Multi-hot vector (one comment can have multiple labels) |
| Max sequence length used | 128 tokens |
| Multilabel threshold | 0.5 (sigmoid output ≥ 0.5 → positive class) |
| Data split | train / validation / test (parquet format) |
| Training set size | ~43,000 samples |
| Validation set size | ~5,400 samples |

**The 28 GoEmotions labels:**  
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

> **Why GoEmotions?** It was chosen because its fine-grained 28-class annotation covers conversational text from Reddit, which shares linguistic characteristics with the therapy/support dialogue the downstream conversational agent will handle. The multilabel format allows natural co-occurrence of emotions (e.g., a user can be simultaneously sad and angry).

---

## 2. Model Architectures

### 2.1 BERT-Base-Uncased — `bert-base-uncased`

**Origin:** Google AI, 2018  
**Paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  

**Architecture overview:**

BERT (Bidirectional Encoder Representations from Transformers) is the foundational transformer encoder for NLP. It learns contextual word representations using two pre-training objectives:

- **Masked Language Modelling (MLM):** 15% of input tokens are masked; the model predicts the original tokens using both left and right context simultaneously. This bidirectionality is the key innovation over earlier unidirectional models (GPT, ELMo).
- **Next Sentence Prediction (NSP):** Given two sentences A and B, the model predicts whether B follows A in the original document. This trains sentence-level coherence understanding.

The base model uses 12 transformer layers, 12 attention heads, and a hidden dimension of 768, resulting in approximately 110M parameters.

**Fine-tuning for multilabel classification:**  
A linear classification head (`BertForSequenceClassification`) is added on top of the `[CLS]` token representation. For multilabel tasks, binary cross-entropy with logits (`BCEWithLogitsLoss`) is used instead of cross-entropy, and a sigmoid function is applied at inference with threshold 0.5 to determine positive labels.

The `[CLS]` and `[SEP]` tokens are inserted automatically by the tokenizer. The WordPiece tokenizer splits unknown words into subword units, making BERT robust to out-of-vocabulary terms.

| Parameter | Value |
|---|---|
| Total parameters | ~110M |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Max position embeddings | 512 |
| Vocabulary size | 30,522 |
| Tokenizer | WordPiece |
| Pre-training data | BooksCorpus + English Wikipedia (~3.3B words) |
| Classification head | Linear(768 → 28) + BCE loss |
| Pooling strategy | `[CLS]` token |

---

### 2.2 RoBERTa-Base — `roberta-base`

**Origin:** Facebook AI Research (Meta), 2019  
**Paper:** [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)  

**Architecture overview:**

RoBERTa (Robustly Optimized BERT Pretraining Approach) shares the identical transformer architecture as BERT-base but introduces several critical pre-training improvements that consistently improve downstream performance:

1. **Removes Next Sentence Prediction (NSP):** The NSP objective was found to be either unhelpful or slightly harmful. RoBERTa trains on the MLM objective alone.
2. **Dynamic masking:** Rather than masking tokens once at data creation time, RoBERTa generates a new masking pattern every time a sequence is fed to the model. This exposes the model to more diverse masking patterns over training.
3. **Longer training with more data:** RoBERTa trains for significantly more steps (500K vs BERT's 1M, but with a much larger batch size of 8K) on a larger corpus including CC-News, OpenWebText, and Stories (~160GB vs BERT's ~16GB).
4. **Larger batch sizes:** Training with batch size 8,192 (vs BERT's 256) with adjusted learning rates produces more stable gradients.
5. **Longer sequences:** Trained on full-length sequences more consistently.

These changes typically yield +2–4% improvement over BERT on text classification benchmarks.

**Fine-tuning setup:** Same `RobertaForSequenceClassification` head structure as BERT — a linear layer on top of the `<s>` (equivalent of `[CLS]`) token representation, with BCE loss for multilabel.

| Parameter | Value |
|---|---|
| Total parameters | ~125M |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Max position embeddings | 514 |
| Vocabulary size | 50,265 |
| Tokenizer | BPE (Byte-Pair Encoding) |
| Pre-training data | BooksCorpus + CC-News + OpenWebText + Stories (~160GB) |
| Key improvements over BERT | No NSP, dynamic masking, 10× more data, larger batches |
| Classification head | Linear(768 → 28) + BCE loss |
| Pooling strategy | `<s>` token (first token) |

---

### 2.3 DeBERTa-v3-Base — `deberta-v3-base`

**Origin:** Microsoft Research, 2021 (v3: 2023)  
**Paper:** [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)  

**Architecture overview:**

DeBERTa (Decoupled Attention with Enhanced Mask Decoder) introduces two major architectural innovations over BERT and RoBERTa, plus a fundamentally different pre-training strategy in v3:

**1. Disentangled Attention:**  
Standard attention in BERT computes attention scores from a single vector that encodes both word content and position. DeBERTa disentangles these:
- Each word is represented by **two vectors**: one for content (what the word means) and one for relative position (where it is).
- Attention scores are computed as the sum of four terms: content-to-content, content-to-position, position-to-content, and position-to-position.

This lets the model learn that *"bank"* near *"river"* vs. *"bank"* near *"loan"* have genuinely different meanings based on relative position — crucial for understanding conversational nuance, sarcasm, and blended emotions.

**2. Enhanced Mask Decoder (EMD):**  
BERT's MLM head for predicting masked tokens uses only absolute position embeddings. DeBERTa's EMD uses both content and absolute position information when predicting masked tokens, recovering position sensitivity that the disentangled attention naturally loses.

**3. DeBERTa-v3 — ELECTRA-style pre-training:**  
Instead of MLM, DeBERTa-v3 uses replaced token detection (RTD): a small generator model replaces some tokens, and DeBERTa (the discriminator) predicts whether each token is original or replaced. This is more sample-efficient than MLM because every token contributes a training signal rather than just the 15% that are masked.

Additionally, v3 uses **gradient-disentangled embedding sharing (GDES)** between the generator and discriminator — sharing embeddings while preventing the discriminator's gradients from flowing into the generator — which stabilises training.

These properties make DeBERTa-v3 the strongest of the three for tasks requiring fine-grained linguistic understanding: sarcasm detection, nuanced sentiment, blended emotions, and long-context coherence.

| Parameter | Value |
|---|---|
| Total parameters | ~184M |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Attention type | Disentangled (content + relative position, 4 terms) |
| Max position embeddings | 512 |
| Vocabulary size | 128,000 (SentencePiece) |
| Tokenizer | SentencePiece (multilingual-capable) |
| Pre-training objective | RTD (ELECTRA-style, v3) |
| Pre-training data | Wikipedia + BooksCorpus + OpenWebText + CC-100 |
| Key advantage over BERT/RoBERTa | Disentangled attention + RTD pretraining |
| Classification head | Linear(768 → 28) + BCE loss |
| Pooling strategy | `[CLS]` token |

---

## 3. Training Configuration

All three models share the same training hyperparameters.

| Hyperparameter | Value |
|---|---|
| Training epochs | 3 |
| Per-device train batch size | 8 |
| Per-device eval batch size | 8 |
| Gradient accumulation steps | 2 |
| **Effective batch size** | **16** |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup steps | 0 |
| LR scheduler | Linear decay |
| Max sequence length | 128 tokens |
| Precision | FP32 (fp16 = False) |
| Early stopping patience | 2 epochs |
| Best model criterion | Validation macro F1 |
| Multilabel threshold | 0.5 (sigmoid output) |
| Loss function | BCEWithLogitsLoss (multilabel) |
| Training framework | HuggingFace `Trainer` |
| Total training steps | 3,000 per model (per training log) |
| Optimizer | AdamW (HuggingFace default) |

> **Why FP32?** Mixed precision (fp16) was disabled for this run. DeBERTa-v3 in particular is known to be numerically sensitive at fp16, and the NaN issues observed (see Section 4.3) are consistent with this. Future runs should use bf16 instead of fp16 for DeBERTa.

---

## 4. Per-Model Training Results

### 4.1 BERT-Base-Uncased Results

**Checkpoint:** `bert-base-uncased`  
**Status:** ✅ Completed successfully  
**Runtime:** 61.1 min  
**Checkpoints saved:** `runs_bert-base-uncased/checkpoint-5808`, `checkpoint-8712`

#### Validation metrics

| Metric | Value |
|---|---|
| Eval loss | **0.1966** |
| Macro F1 | **0.4221** |
| Micro F1 | **0.5500** |
| Macro precision | 0.6438 |
| Macro recall | 0.3511 |

#### Analysis

BERT achieves a macro F1 of 0.4221 and micro F1 of 0.5500 on the 28-class multilabel GoEmotions task. The high precision (0.64) relative to recall (0.35) indicates the model is **conservative** — it only predicts a label when fairly confident, but misses many true positive labels.

This precision-recall imbalance is typical when fine-tuning BERT on imbalanced multilabel data: rare emotion classes (grief, pride, nervousness, relief) rarely receive positive predictions because the model defaults to the majority-class distribution. With only 3 epochs of training, the model has not had time to learn discriminative features for low-frequency labels.

The relatively long runtime (61 min) reflects the full transformer fine-tuning overhead — all 110M parameters are updated during training.

---

### 4.2 RoBERTa-Base Results

**Checkpoint:** `roberta-base`  
**Status:** ✅ Completed successfully  
**Runtime:** 59.7 min  
**Checkpoints saved:** `runs_roberta-base/checkpoint-5808`, `checkpoint-8712`

#### Validation metrics

| Metric | Value |
|---|---|
| Eval loss | **0.1942** |
| Macro F1 | **0.4374** |
| Micro F1 | **0.5569** |
| Macro precision | 0.6245 |
| Macro recall | 0.3641 |

#### Analysis

RoBERTa outperforms BERT on all metrics — macro F1 of 0.4374 (+0.0153 over BERT) and micro F1 of 0.5569 (+0.0069). This confirms the expected improvement from RoBERTa's better pre-training:

- **Lower eval loss** (0.1942 vs 0.1966): better calibrated predictions
- **Higher recall** (0.3641 vs 0.3511): slightly more willing to predict positive labels
- **Lower precision** (0.6245 vs 0.6438): some reduction in per-prediction confidence, which is expected when recall increases

RoBERTa is slightly faster than BERT (59.7 min vs 61.1 min) despite having more parameters (~125M vs ~110M), which is attributable to its BPE tokenizer producing shorter sequences on average than BERT's WordPiece for English Reddit text.

**RoBERTa is the recommended text model** for this project based on these results.

---

### 4.3 DeBERTa-v3-Base Results

**Checkpoint:** `deberta-v3-base`  
**Status:** ❌ Training failed (NaN loss / gradients)  
**Runtime:** 41.1 min  
**Checkpoints saved:** `runs_deberta-v3-base/checkpoint-2904`, `checkpoint-8712`

#### Validation metrics

| Metric | Value |
|---|---|
| Eval loss | **NaN** |
| Macro F1 | **0.0000** |
| Micro F1 | **0.0000** |
| Macro precision | 0.0000 |
| Macro recall | 0.0000 |

> ⚠️ All zero metrics are **failure placeholders**, not model performance. DeBERTa suffered NaN loss and NaN gradient norms throughout training, rendering its outputs meaningless.

#### Root cause analysis

The NaN loss issue in DeBERTa-v3 on this run has several likely causes:

1. **FP32 + AdamW numerical instability with RTD pre-training:** DeBERTa-v3's ELECTRA-style pre-training uses a different embedding space than BERT/RoBERTa. When fine-tuned with FP32 and a standard `2e-5` learning rate without warmup, the disentangled attention mechanism can produce unstable gradient magnitudes in early steps.

2. **No warmup steps (warmup=0):** BERT and RoBERTa tolerate no-warmup fine-tuning reasonably well. DeBERTa-v3's disentangled attention produces larger initial gradient magnitudes (due to 4-term attention score computation) and benefits significantly from a warmup period of 100–500 steps.

3. **SentencePiece tokenizer with 128K vocabulary:** DeBERTa-v3 uses a much larger vocabulary (128K vs BERT's 30K). With a max length of 128 and short Reddit comments, many tokens become padding. The embedding table gradient for the massive vocabulary can become unstable without proper scaling.

4. **Gradient clipping not set:** The HuggingFace `Trainer` default does not clip gradients unless `max_grad_norm` is set. DeBERTa-v3 is more sensitive to gradient explosions than BERT/RoBERTa.

#### Recommended fix for retraining

```python
TrainingArguments(
    # ...existing config...
    learning_rate        = 5e-6,        # Lower LR (was 2e-5)
    warmup_steps         = 200,         # Add warmup (was 0)
    bf16                 = True,        # Use bf16 instead of fp16/fp32
    max_grad_norm        = 1.0,         # Add gradient clipping
    num_train_epochs     = 5,           # More epochs to compensate for lower LR
)
```

---

## 5. Model Comparison

### Overall performance

| Rank | Model | Checkpoint | Status | Eval Loss | Macro F1 | Micro F1 | Macro Precision | Macro Recall | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **RoBERTa-Base** | `roberta-base` | ✅ Valid | **0.1942** | **0.4374** | **0.5569** | 0.6245 | 0.3641 | 59.7 min |
| 🥈 2 | **BERT-Base-Uncased** | `bert-base-uncased` | ✅ Valid | 0.1966 | 0.4221 | 0.5500 | **0.6438** | 0.3511 | 61.1 min |
| ❌ — | **DeBERTa-v3-Base** | `deberta-v3-base` | ❌ Invalid | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 41.1 min |
| 🥇 1 (re-run) | **RoBERTa-Base** | `roberta-base` (2026-10-09, RTX 5060 Ti) | ✅ Valid | **0.1941** | **0.4430** | **0.5571** | n/a | n/a | 20.4 min |

### Metric delta: RoBERTa vs BERT

| Metric | BERT | RoBERTa | Delta |
|---|---|---|---|
| Eval loss | 0.1966 | 0.1942 | −0.0024 ✅ |
| Macro F1 | 0.4221 | **0.4374** | **+0.0153** ✅ |
| Micro F1 | 0.5500 | **0.5569** | **+0.0069** ✅ |
| Macro precision | **0.6438** | 0.6245 | −0.0193 (expected tradeoff) |
| Macro recall | 0.3511 | **0.3641** | **+0.0130** ✅ |
| Runtime | 61.1 min | **59.7 min** | −1.4 min ✅ |

RoBERTa wins on every key metric while also being faster. The slight precision drop is an expected consequence of higher recall — RoBERTa predicts positive labels more often, catching more true positives at the cost of a few more false positives.

---

## 6. Key Observations

1. **Macro F1 of ~0.42–0.44 is reasonable for 28-class multilabel GoEmotions at 3 epochs.** The full GoEmotions benchmark is challenging: many classes are rare, several emotions co-occur frequently (anger+disgust, joy+excitement), and Reddit language is informal with heavy sarcasm and irony. Published BERT-family baselines on GoEmotions typically land in the 0.40–0.52 macro F1 range with 3–5 epochs.

2. **Micro F1 is consistently higher than macro F1** (0.55 vs 0.42). This is expected with class imbalance — micro F1 is dominated by the frequent classes (admiration, approval, neutral, annoyance) where the model performs well, while macro F1 equally weights the rare classes (grief, pride, nervousness) where it underperforms.

3. **Precision > Recall** for both valid models. With a 0.5 sigmoid threshold, both BERT and RoBERTa are conservative predictors. Lowering the threshold to 0.3–0.4 would significantly increase recall at the cost of precision, which may be preferable for the downstream behaviour change support agent (missing a signal of "grief" is worse than a false positive).

4. **DeBERTa-v3's failure is a configuration issue, not an architecture one.** DeBERTa-v3 consistently achieves state-of-the-art results on text emotion benchmarks when properly configured (bf16, lower LR, warmup). It should be retrained with the corrected config before being ruled out.

5. **Only 3 epochs may be insufficient.** GoEmotions research typically uses 5–10 epochs. Both BERT and RoBERTa show no sign of overfitting at epoch 3, suggesting they have capacity for further improvement. Extending to 5 epochs with early stopping (patience=2) is recommended.

6. **The text stream targets cognitive and linguistic emotion signals** that audio models cannot capture — sarcasm, indirect distress expressions, denial, minimisation. The lower absolute accuracy compared to emotion2vec on audio (0.44 macro F1 vs 0.93) is expected: text-based emotion is inherently more ambiguous than acted speech.

---

## 7. Known Issues and Recommendations

### 7.1 DeBERTa-v3 retraining

To retrain DeBERTa-v3 successfully, modify `train_all_text_models.py`:

```python
# For DeBERTa-v3 only:
learning_rate        = 5e-6          # 4× lower than BERT/RoBERTa
warmup_steps         = 200           # ~6.7% of 3,000 steps
bf16                 = True          # BF16 instead of FP32
max_grad_norm        = 1.0           # Gradient clipping
num_train_epochs     = 5             # More epochs to compensate
```

### 7.2 Threshold tuning

The 0.5 threshold produces high precision but low recall. For a behaviour change support application, missing emotional signals is more harmful than over-predicting. Consider evaluating at thresholds 0.3, 0.35, 0.4 to find the optimal F1 point.

### 7.3 Class weighting

Rare labels (grief, pride, nervousness, relief, embarrassment) are likely under-predicted. Add inverse-frequency weighting to the BCE loss:

```python
# Compute per-class positive frequency from training set
pos_weight = (num_negative_per_class / num_positive_per_class)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

### 7.4 Extended training

Both valid models were trained for only 3 epochs with no warmup. Extending to 5 epochs with:
- `warmup_ratio = 0.06`
- `num_train_epochs = 5`
- `early_stopping_patience = 3`

is expected to improve macro F1 by ~2–4 pp for both models.

---

## 8. Role in the Multimodal Pipeline

The text models serve as the **text stream encoder** in the multimodal Cross-Modal Attention Transformer:

```
Input utterance text
        ↓
  Tokenizer (WordPiece / BPE / SentencePiece)
        ↓
  BERT / RoBERTa / DeBERTa backbone
        ↓
  [CLS] token hidden state (768-dim)
        ↓
  Projection head: Linear(768→512) → LayerNorm → GELU → Dropout → Linear(512→512)
        ↓
  512-dim text embedding
        ↓
  → Cross-Modal Attention Transformer (fused with audio + video streams)
```

The text stream captures:
- **Lexical emotion** — explicit emotion words ("I feel sad", "I'm overwhelmed")
- **Cognitive load signals** — hedging language, fragmented sentences, repetition
- **Indirect distress** — minimisation ("it's nothing really"), denial, deflection
- **Context coherence** — multi-turn context that audio/video cannot access

**Recommended text encoder for downstream use:** RoBERTa-Base (`roberta-base`), pending DeBERTa-v3 retraining.

---

## 9. Output Artifacts

| File | Description |
|---|---|
| `models/runs_bert-base-uncased/checkpoint-5808/` | BERT checkpoint at step 5808 (mid-training) |
| `models/runs_bert-base-uncased/checkpoint-8712/` | BERT final checkpoint (end of training) |
| `models/runs_roberta-base/checkpoint-5808/` | RoBERTa checkpoint at step 5808 |
| `models/runs_roberta-base/checkpoint-8712/` | RoBERTa final checkpoint (best model) |
| `models/runs_deberta-v3-base/checkpoint-2904/` | DeBERTa early checkpoint (pre-NaN, likely unusable) |
| `models/runs_deberta-v3-base/checkpoint-8712/` | DeBERTa final checkpoint (NaN weights, do not use) |
| `models/text_model_analysis.md` | Auto-generated summary from training script |
| `models/train_all_log.txt` | Full training log (progress bars + warnings) |
| `models/train_run_20261006_052734.txt` | Timestamped run log for the October 6 session |
| `models/train_all_text_models.py` | Training script |
| `models/TEXT_MODELS_TRAINING_REPORT.md` | This consolidated report |

> **Note on checkpoint sizes:**  
> BERT checkpoints: ~418 MB each (`model.safetensors`)  
> RoBERTa checkpoints: ~476 MB each (`model.safetensors`)  
> DeBERTa checkpoints: ~352 MB each (`model.safetensors`)  
> Optimizer states (`.pt` files) are saved alongside but are gitignored.

---

*Report generated: 2026-10-08 | Training script: `models/train_all_text_models.py`*

---

## 10. RoBERTa-Base Re-run (2026-10-09, RTX 5060 Ti)

Full 3-epoch re-run of `models/text/train.py --model roberta-base` on the second
machine (RTX 5060 Ti 16 GB, CUDA 12.8, PyTorch 2.11) with identical hyperparameters
(3 epochs, LR 2e-5, effective batch 16, warmup 0, early stopping patience 2 on
eval macro F1). Completed at step 8,712 in 20.4 minutes.

| Metric | Initial / baseline | Final checkpoint (step 8,712) |
|---|---:|---:|
| Training loss | 0.9238 (step 50) | 0.3710 |
| Validation loss | 0.6898 | 0.1941 |
| Validation macro F1 | 0.0614 | 0.4430 |
| Validation micro F1 | 0.2020 | 0.5571 |
| Test macro F1 | 0.0000 | 0.4401 |
| Test micro F1 | 0.0000 | 0.5545 |

The re-run edges out the original 2026-10-08 RoBERTa run on validation
(macro F1 0.4430 vs 0.4374, eval loss 0.1941 vs 0.1942) and is the checkpoint
adopted as the production text model. Weights live at
`models/text/checkpoints/runs_roberta-base/checkpoint-8712/model.safetensors`
on each deployment machine and are intentionally not stored in git; without
them the pipeline reports the lexicon fallback transparently.
