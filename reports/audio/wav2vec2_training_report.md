# Audio Training Report — wav2vec2 (RAVDESS)

## Model
- Key: `wav2vec2`
- Checkpoint: `facebook/wav2vec2-base`

## Dataset
- Name: RAVDESS
- Train: 768, Dev: 192, Test: 192
- Emotion classes (6): angry, disgust, fearful, happy, sad, surprised

## Training config
- Epochs: 10
- Batch size: 8
- Learning rate: 2e-05
- Optimizer: AdamW (weight_decay=0.01)
- Grad clip norm: 1.0
- Mixed precision: yes
- Seed: 42
- Device: cuda

## Per-epoch metrics

| Epoch | Train Loss | Dev Loss | Accuracy | Macro F1 |
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
| 10 | 0.0757 | 1.1006 | 0.7812 | 0.7830 |

## Test results
- Accuracy:   0.6823
- Macro F1:   0.6688
- Weighted F1:0.6688

### Per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.6579 | 0.7812 | 0.7143 | 32 |
| disgust | 1.0000 | 0.3750 | 0.5455 | 32 |
| fearful | 0.8824 | 0.4688 | 0.6122 | 32 |
| happy | 0.5250 | 0.6562 | 0.5833 | 32 |
| sad | 0.7143 | 0.9375 | 0.8108 | 32 |
| surprised | 0.6512 | 0.8750 | 0.7467 | 32 |

## Training time: 273.5 s

## Output
- Weights: `C:\hackathon\Gemini_CLI\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\runs_wav2vec2`
- Confusion matrix: `C:\hackathon\Gemini_CLI\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wav2vec2_confusion_matrix.png`
- Classification report: `C:\hackathon\Gemini_CLI\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wav2vec2_classification_report.csv`