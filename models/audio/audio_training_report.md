# Audio Model Training Report - Wav2Vec2 (RAVDESS)

## Dataset
- Source: `data/raw/ravdess/`
- Metadata: `data/raw/ravdess/ravdess_metadata.csv`
- Emotion set: angry, disgust, fearful, happy, sad, surprised
- Excluded: neutral, calm
- Train samples: 768
- Validation samples: 192
- Test samples: 192
- Total samples: 1152

## Speaker-independent split
- Split by Actor ID, not by individual utterances.
- Train actors: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16
- Validation actors: 17, 18, 19, 20
- Test actors: 21, 22, 23, 24

## RAVDESS label mapping
- label2id: {"angry": 0, "disgust": 1, "fearful": 2, "happy": 3, "sad": 4, "surprised": 5}
- id2label: {"0": "angry", "1": "disgust", "2": "fearful", "3": "happy", "4": "sad", "5": "surprised"}

## Classes
- Number of classes: 6
- Class names: angry, disgust, fearful, happy, sad, surprised
- Label mapping saved to: `models/audio/label_mapping.json`

## Audio source
- Configured audio root: WindowsPath('C:/Users/ADMIN/Downloads/spiderboy projects/LLM-based-conversational-agents-for-behaviour-change-support/data/raw/ravdess')
- Audio found: yes
- Preprocessing: load -> mono -> resample to 16 kHz -> Wav2Vec2 processor -> dynamic padding

## Wav2Vec2 configuration
- Pretrained checkpoint: `facebook/wav2vec2-base`
- num_labels: 6
- label2id: {"angry": 0, "disgust": 1, "fearful": 2, "happy": 3, "sad": 4, "surprised": 5}
- id2label: {"0": "angry", "1": "disgust", "2": "fearful", "3": "happy", "4": "sad", "5": "surprised"}
- Feature encoder: frozen during training
- Gradient checkpointing: enabled

## Training configuration
- Epochs: 10
- Learning rate: 2e-05
- Batch size: 8
- Optimizer: AdamW (weight_decay=0.01)
- Gradient clipping: norm 1.0
- Mixed precision: yes
- Random seed: 42
- Device: cuda
- Train batch steps per epoch: 96

## Training metrics per epoch

| Epoch | Train loss | Dev loss | LR | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| 1 | 1.6968 | 1.5764 | 2.00e-05 | 0.3802 | 0.3227 | 0.3227 |
| 2 | 1.3533 | 1.4157 | 2.00e-05 | 0.4375 | 0.4179 | 0.4179 |
| 3 | 0.9075 | 1.1026 | 2.00e-05 | 0.6406 | 0.6388 | 0.6388 |
| 4 | 0.5804 | 1.0734 | 2.00e-05 | 0.6458 | 0.6428 | 0.6428 |
| 5 | 0.3719 | 1.0195 | 2.00e-05 | 0.7188 | 0.7107 | 0.7107 |
| 6 | 0.1964 | 1.0363 | 2.00e-05 | 0.7396 | 0.7375 | 0.7375 |
| 7 | 0.1223 | 0.9670 | 2.00e-05 | 0.8021 | 0.8030 | 0.8030 |
| 8 | 0.0627 | 1.1782 | 2.00e-05 | 0.7708 | 0.7787 | 0.7787 |
| 9 | 0.0573 | 1.5480 | 2.00e-05 | 0.7188 | 0.7127 | 0.7127 |
| 10 | 0.0956 | 1.7710 | 2.00e-05 | 0.6927 | 0.6924 | 0.6924 |

## Final results (test)

- Accuracy: 0.7240
- Macro F1: 0.7182
- Weighted F1: 0.7182
- Macro precision: 0.7592
- Macro recall: 0.7240

### Per-class results (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.6818 | 0.9375 | 0.7895 | 32 |
| disgust | 0.8571 | 0.5625 | 0.6792 | 32 |
| fearful | 0.9412 | 0.5000 | 0.6531 | 32 |
| happy | 0.5500 | 0.6875 | 0.6111 | 32 |
| sad | 0.7317 | 0.9375 | 0.8219 | 32 |
| surprised | 0.7931 | 0.7188 | 0.7541 | 32 |

### Confusion matrix (test)

![confusion_matrix](confusion_matrix.png)

### Classification report CSV

Saved to `models/audio/classification_report.csv`.

## Training time

- Total wall-clock training time: 107.11 seconds

## GPU memory considerations

- GPU memory usage depends heavily on waveform length and batch size.
- Wav2Vec2 feature encoder was frozen and gradient checkpointing was enabled to reduce memory pressure.
- For ~6 GB VRAM GPUs, reduce `batch_size`, shorten long utterances during preprocessing, or increase gradient accumulation if needed.
- No claim is made here about which audio model is best; only Wav2Vec2 was trained in this stage.

## Output artifacts

- Model: `models/audio/runs_wav2vec2/`
- Processor: `models/audio/runs_wav2vec2/`
- Report: `models/audio/audio_training_report.md`
- Log: `models/audio/audio_training_log.txt`
- Confusion matrix: `models/audio/confusion_matrix.png` or `models/audio/ravdess_confusion_matrix.png`
- Classification report: `models/audio/classification_report.csv`
- Label mapping: `models/audio/label_mapping.json`

## Ready for multimodal use?

This model can serve as the **audio branch** of a future multimodal emotion-recognition system once:

- For MELD: the MELD audio directory is available and aligned to the MELD CSVs, and
- For RAVDESS: the prepared `data/raw/ravdess/ravdess_metadata.csv` and extracted WAV files are in place, and
- Train/dev/test splits are made disjoint at the speaker level before any serious evaluation is trusted.

At that point, this Wav2Vec2 branch can emit a fixed audio representation that can be fused with the text branch (already trained to 87.32% accuracy with DeBERTa-v3-base) and a future face branch.