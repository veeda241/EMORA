# LLM-Based Multimodal Behaviour-Change Support

## Slide 1 — Title

**LLM-Based Multimodal Conversational Agents for Behaviour-Change Support**

**Subtitle:** Emotion-aware text, speech, and video modelling for stress and
behavioural support

Presenter: `<name>`  
Institution: `<institution>`  
Date: `<date>`

---

## Slide 2 — Topic and base papers

### Topic

Build a multimodal emotion-analysis pipeline that combines:

1. **Text** from conversational messages
2. **Audio** from speech
3. **Video/face** signals from facial-expression clips

The long-term goal is to support an LLM-based conversational agent that can
adapt behaviour-change feedback to the user's emotional and stress state.

### Main base papers and datasets

- **GoEmotions:** Demszky et al., *GoEmotions: A Dataset of Fine-Grained
  Emotions*, ACL 2020.  
  https://arxiv.org/abs/2005.00547
- **MELD:** Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion
  Recognition in Conversations*, ACL 2019.  
  https://aclanthology.org/P19-1050/
- **RAVDESS:** Livingstone and Russo, *The Ryerson Audio-Visual Database of
  Emotional Speech and Song*, PLOS ONE 2018.  
  https://doi.org/10.1371/journal.pone.0196391
- **BERT:** Devlin et al., *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding*, NAACL 2019.
- **RoBERTa:** Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining
  Approach*, 2019.
- **Wav2Vec 2.0:** Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised
  Learning of Speech Representations*, NeurIPS 2020.

The repository uses the datasets and pretrained encoders as engineering
foundations; it does not claim that the final behaviour-change intervention
has already been clinically validated.

---

## Slide 3 — Problem statement

- Human emotion is expressed through multiple channels.
- Text alone misses vocal tone, pauses, intensity, and facial cues.
- Audio alone cannot interpret conversational meaning.
- Video signals can be unavailable or affected by lighting, pose, and privacy.
- A single model may be unreliable when one modality is noisy or missing.

**Research question:** Can separately trained text, audio, and video encoders
produce aligned representations that can later be fused for robust,
emotion-aware behaviour-change support?

---

## Slide 4 — Objectives

### Primary objective

Develop and compare modality-specific encoders for conversational emotion
recognition as the perception layer of a future LLM behaviour-change agent.

### Specific objectives

1. Prepare reproducible GoEmotions, MELD, RAVDESS, and CMU-MOSEI data assets.
2. Compare three text encoders on multilabel GoEmotions.
3. Prepare and evaluate speech emotion recognition using RAVDESS.
4. Establish Wav2Vec2 as the first working audio baseline.
5. Select comparison candidates: emotion2vec, WavLM, and Wav2Vec2.
6. Define a future video comparison: MAE-DFER, EST, and Hybrid CNN + ViT.
7. Project each modality to a common **512-dimensional** vector for fusion.
8. Evaluate robustness, class balance, speaker independence, and modality
   dropout before integrating the LLM agent.

---

## Slide 5 — End-to-end system architecture

```text
Text transcript ──> BERT / RoBERTa / DeBERTa ──> text embedding (768)
                                                        │
Speech waveform ──> emotion2vec / WavLM / Wav2Vec2 ─> audio embedding (768)
                                                        │
Face/video clip ──> MAE-DFER / EST / CNN+ViT ────────> face embedding (512)
                                                        │
                         Linear + LayerNorm + GELU
                                                        │
              three projected 512-dimensional streams
                                                        │
                 Cross-Modal Attention Transformer
                                                        │
               emotion / stress / behaviour-support state
                                                        │
                         LLM response policy
```

The fusion transformer is planned. Current work focuses on data preparation,
encoder comparison, and modality-specific baselines.

---

## Slide 6 — Text models: selection and architecture

### Models

| Model | Role | Why selected |
|---|---|---|
| BERT-Base-Uncased | Baseline | Strong general-purpose bidirectional encoder |
| RoBERTa-Base | Best valid current result | Robust BERT pretraining and strong text representations |
| DeBERTa-v3-Base | Planned high-capacity candidate | Disentangled attention for richer contextual modelling |

### Architecture

```text
Token IDs
  → token/position embeddings
  → multi-layer bidirectional Transformer encoder
  → pooled contextual representation
  → multilabel classification head
  → 28 sigmoid emotion probabilities
```

### Task formulation

- GoEmotions is treated as **multilabel classification**.
- Output size: **28 emotion labels**.
- Loss: multilabel binary cross-entropy supplied by the Hugging Face
  sequence-classification model.
- Prediction threshold: **0.5**.

---

## Slide 7 — Text hyperparameters and model parameters

### Training hyperparameters

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Train batch size | 8 |
| Evaluation batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup steps | 0 |
| Scheduler | Linear |
| Maximum sequence length | 128 |
| Early-stopping patience | 2 |
| Selection metric | Validation macro F1 |
| Precision | FP32 (`fp16=False`) |

### Model parameters

- BERT-Base and RoBERTa-Base: approximately **110M parameters** each.
- DeBERTa-v3-Base: approximately **184M parameters**.
- Exact parameter count can vary slightly with the classification head and
  Transformers implementation.
- The classification head is newly initialized for the 28-label task.

---

## Slide 8 — Text evaluation parameters

Because the task is multilabel, accuracy alone is insufficient.

- **Macro F1:** gives equal weight to every emotion class.
- **Micro F1:** aggregates all label decisions and reflects overall volume.
- **Macro precision:** how precise predictions are across classes.
- **Macro recall:** how many true class labels are recovered.
- **Validation loss:** used with the model-selection configuration.
- **Threshold:** sigmoid probability ≥ 0.5 becomes a positive label.

The final test set must be reported separately from validation selection to
avoid optimistic model-selection bias.

---

## Slide 9 — Text results

### Completed validation results

| Rank | Model | Eval loss | Macro F1 | Micro F1 | Macro precision | Macro recall | Time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | RoBERTa-Base | 0.1942 | **0.4374** | **0.5569** | 0.6245 | 0.3641 | 59.7 min |
| 2 | BERT-Base-Uncased | 0.1966 | 0.4221 | 0.5500 | **0.6438** | 0.3511 | 61.1 min |
| — | DeBERTa-v3-Base | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 41.1 min |

### Interpretation

- **RoBERTa is the current best valid text baseline** by macro and micro F1.
- BERT is competitive and has the highest macro precision.
- DeBERTa completed the scheduled loop but became numerically unstable; its
  zero metrics are failure placeholders, not evidence of poor capability.

---

## Slide 10 — Audio dataset and preprocessing

### RAVDESS

- 1,152 validated WAV speech clips
- Six classes: angry, disgust, fearful, happy, sad, surprised
- Actor-independent split:
  - Train: actors 1–16, 768 clips
  - Validation: actors 17–20, 192 clips
  - Test: actors 21–24, 192 clips
- Equal class counts: 192 clips per emotion

### Preprocessing

```text
WAV file
  → mono waveform
  → resample to 16 kHz
  → Wav2Vec2 processor
  → dynamic batch padding
  → frozen feature encoder + classification head
```

The actor-independent split is important because random utterance splits can
leak speaker identity into the test set.

---

## Slide 11 — Audio models and architecture

### Comparison set

| Role | Model | Status |
|---|---|---|
| Primary | emotion2vec | Selected; checkpoint/loader integration planned |
| Comparison | WavLM-Base | Selected; model-specific training integration planned |
| Baseline | Wav2Vec2-Base | Implemented and piloted |

### Implemented Wav2Vec2 architecture

```text
16 kHz waveform
  → seven-layer convolutional feature encoder
  → Transformer context network
  → mean/sequence representation
  → six-class linear classification head
  → softmax emotion prediction
```

Engineering choices:

- Frozen convolutional feature encoder
- Gradient checkpointing
- AdamW optimizer
- Gradient clipping
- CUDA execution
- Best checkpoint selected by validation loss

---

## Slide 12 — Audio hyperparameters and results

### Hyperparameters used in the pilot

| Parameter | Value |
|---|---:|
| Model | `facebook/wav2vec2-base` |
| Number of classes | 6 |
| Epochs | 1 |
| Batch size | 2 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Gradient clipping | Norm 1.0 |
| Feature encoder | Frozen |
| Gradient checkpointing | Enabled |
| Mixed precision | Enabled |
| Device | NVIDIA RTX 3050 6 GB |
| Seed | 42 |

### Pilot test results

| Accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall |
|---:|---:|---:|---:|---:|
| 0.1667 | 0.0476 | 0.0476 | 0.0278 | 0.1667 |

The model predicted **happy** for every test example in this short one-epoch
pilot. These results are a baseline diagnostic, not a final performance claim.

---

## Slide 13 — Video/face models

### Selected comparison

| Rank/role | Model | Rationale | Status |
|---|---|---|---|
| Primary | MAE-DFER | Masked-video pretraining for dynamic facial expressions | Planned |
| Alternative | EST | Strong spatiotemporal expression modelling | Planned |
| Third | Hybrid CNN + ViT | Local texture plus global temporal/patch context | Planned |

### Proposed architectures

- **MAE-DFER:** masked video autoencoder pretraining followed by a temporal
  expression-recognition head.
- **EST:** spatiotemporal transformer over face crops or frame tokens.
- **Hybrid CNN + ViT:** CNN extracts local facial features; ViT models global
  relationships; fusion head produces a 512-dimensional representation.

No video model has been trained in the current verified run, so no video
accuracy should be presented as an achieved result.

---

## Slide 14 — Planned hyperparameter tuning

### Text search space

- Learning rate: `1e-5`, `2e-5`, `3e-5`
- Maximum length: `128`, `256`
- Weight decay: `0.0`, `0.01`, `0.05`
- Classification threshold: tune from `0.30` to `0.70`
- Warmup ratio: `0.0`, `0.05`, `0.10`
- Gradient accumulation: `1`, `2`, `4`

### Audio search space

- Learning rate: `5e-6`, `1e-5`, `2e-5`
- Batch size: `2`, `4` subject to VRAM
- Epochs: `5`, `10`, with early stopping
- Frozen vs partially unfrozen feature encoder
- Gradient clipping: `0.5`, `1.0`
- Audio duration cap and padding strategy
- emotion2vec/WavLM/Wav2Vec2 encoder comparison

### Video search space

- Face crop size: `112`, `224`
- Clip length: `8`, `16`, `32` frames
- Learning rate: `1e-5` to `3e-4`
- Temporal pooling: mean, attention, transformer pooling
- CNN/ViT fusion: concatenation vs gated fusion

Use a fixed validation split and report the untouched test set only once.

---

## Slide 15 — Planned improvements and reasons

1. **Stabilize DeBERTa training**
   - Lower learning rate
   - Explicit gradient clipping
   - Check non-finite logits/gradients
   - Compare FP32 and safe mixed precision
   - Reason: current DeBERTa run became NaN.

2. **Train the selected audio comparison models**
   - Integrate emotion2vec and WavLM loaders
   - Keep the actor-independent RAVDESS protocol
   - Reason: current audio evidence is only a Wav2Vec2 pilot.

3. **Improve audio supervision**
   - Train for more than one epoch
   - Add class-balanced sampling or class-weighted loss
   - Tune waveform length and learning rate
   - Reason: the pilot collapsed to one predicted class.

4. **Implement the video stream**
   - Add face detection/cropping and temporal clip construction
   - Train MAE-DFER, EST, and Hybrid CNN + ViT
   - Reason: video models are currently registry entries only.

5. **Build true multimodal fusion**
   - Align text, waveform, and video by utterance/timestamp
   - Project each stream to 512 dimensions
   - Add cross-modal attention and modality dropout
   - Reason: robustness is needed when a sensor is missing or noisy.

6. **Improve evaluation validity**
   - Speaker-independent splits for audio/video
   - Conversation-level splits for MELD
   - Per-class confusion matrices and confidence calibration
   - Reason: avoid identity leakage and overclaiming generalization.

---

## Slide 16 — Limitations and responsible-use considerations

- RAVDESS is acted emotional speech and may not represent natural distress.
- GoEmotions is Reddit-derived and contains demographic and topical bias.
- MELD source splits show utterance-ID overlap that must be reviewed before
  leakage-free claims.
- Emotion recognition is not clinical diagnosis.
- The system should support—not replace—human judgement and user consent.
- Audio/video processing has privacy implications; minimize retention and obtain
  explicit consent.
- Current fusion and LLM response policy are future work, not validated
  behaviour-change intervention outcomes.

---

## Slide 17 — Conclusion

- A reproducible multimodal preparation pipeline is in place.
- Text comparison is complete for BERT and RoBERTa; RoBERTa is the best valid
  current result with macro F1 **0.4374**.
- RAVDESS audio data is preprocessed with speaker-independent splits.
- Wav2Vec2 provides the first audio baseline, but the one-epoch pilot needs
  tuning and longer training.
- emotion2vec, WavLM, MAE-DFER, EST, and Hybrid CNN + ViT are the selected
  next comparison models.
- The target system is a 512-dimensional per-modality representation followed
  by cross-modal attention and an adaptive LLM support policy.

**Key message:** the project has a validated foundation and clear next steps,
but planned model comparisons and multimodal fusion must be completed before
claiming a deployable behaviour-change support system.

---

## Slide 18 — Repository evidence

- Text report: `models/text_model_analysis.md`
- Audio report: `models/audio/audio_training_report.md`
- RAVDESS preprocessing: `scripts/prepare_ravdess.py`
- RAVDESS summary: `data/processed/ravdess/preprocessing_summary.json`
- Model registry: `models/comparison_grid.py`
- Generated comparison: `models/comparison_table.md`
- Training plan: `models/training_plan.md`

### Reproducibility commands

```powershell
python scripts\prepare_ravdess.py
python models\train_all_text_models.py
python models\train_all_audio_models.py --dataset ravdess --audio-root data\raw\ravdess
python models\comparison_grid.py
```
