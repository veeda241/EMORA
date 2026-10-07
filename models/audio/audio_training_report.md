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
- Configured audio root: WindowsPath('data/raw/ravdess')
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
- Epochs: 1
- Learning rate: 2e-05
- Batch size: 2
- Optimizer: AdamW (weight_decay=0.01)
- Gradient clipping: norm 1.0
- Mixed precision: yes
- Random seed: 42
- Device: cuda
- Train batch steps per epoch: 4

## Training metrics per epoch

| Epoch | Train loss | Dev loss | LR | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| 1 | 1.7141 | 1.7979 | 2.00e-05 | 0.1667 | 0.0476 | 0.0476 |

## Final results (test)

- Accuracy: 0.1667
- Macro F1: 0.0476
- Weighted F1: 0.0476
- Macro precision: 0.0278
- Macro recall: 0.1667

### Per-class results (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.0000 | 0.0000 | 0.0000 | 32 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 32 |
| fearful | 0.0000 | 0.0000 | 0.0000 | 32 |
| happy | 0.1667 | 1.0000 | 0.2857 | 32 |
| sad | 0.0000 | 0.0000 | 0.0000 | 32 |
| surprised | 0.0000 | 0.0000 | 0.0000 | 32 |

### Confusion matrix (test)

![confusion_matrix](confusion_matrix.png)

### Classification report CSV

Saved to `models/audio/classification_report.csv`.

## Training time

- Total wall-clock training time: 8.69 seconds

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