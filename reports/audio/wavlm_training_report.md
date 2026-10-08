# Audio Training Report — wavlm (RAVDESS)

## Model
- Key: `wavlm`
- Checkpoint: `microsoft/wavlm-base`

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
| 1 | 1.7171 | 1.6026 | 0.3750 | 0.3060 |
| 2 | 1.3781 | 1.3415 | 0.5573 | 0.5200 |
| 3 | 1.0002 | 1.0630 | 0.6719 | 0.6751 |
| 4 | 0.6679 | 1.1542 | 0.6198 | 0.6162 |
| 5 | 0.3975 | 0.8967 | 0.7552 | 0.7594 |
| 6 | 0.2766 | 1.0399 | 0.7344 | 0.7334 |
| 7 | 0.2244 | 1.3471 | 0.6823 | 0.6809 |
| 8 | 0.1492 | 1.9368 | 0.6094 | 0.6082 |
| 9 | 0.1466 | 1.5757 | 0.7135 | 0.7098 |
| 10 | 0.0703 | 1.3883 | 0.7500 | 0.7509 |

## Test results
- Accuracy:   0.7240
- Macro F1:   0.7111
- Weighted F1:0.7111

### Per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.6250 | 0.9375 | 0.7500 | 32 |
| disgust | 0.7895 | 0.4688 | 0.5882 | 32 |
| fearful | 0.8824 | 0.4688 | 0.6122 | 32 |
| happy | 0.6875 | 0.6875 | 0.6875 | 32 |
| sad | 0.7568 | 0.8750 | 0.8116 | 32 |
| surprised | 0.7436 | 0.9062 | 0.8169 | 32 |

## Training time: 345.7 s

## Output
- Weights: `C:\hackathon\Gemini_CLI\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\runs_wavlm`
- Confusion matrix: `C:\hackathon\Gemini_CLI\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wavlm_confusion_matrix.png`
- Classification report: `C:\hackathon\Gemini_CLI\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wavlm_classification_report.csv`