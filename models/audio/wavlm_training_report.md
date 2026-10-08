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
| 1 | 1.7210 | 1.6210 | 0.3698 | 0.2976 |
| 2 | 1.3996 | 1.4031 | 0.5312 | 0.4935 |
| 3 | 1.0104 | 1.0785 | 0.6823 | 0.6802 |
| 4 | 0.6534 | 0.9417 | 0.7240 | 0.7231 |
| 5 | 0.4146 | 0.8908 | 0.7604 | 0.7615 |
| 6 | 0.2772 | 0.9946 | 0.7552 | 0.7545 |
| 7 | 0.1913 | 1.4858 | 0.6823 | 0.6863 |
| 8 | 0.1195 | 1.2735 | 0.7448 | 0.7475 |
| 9 | 0.1536 | 1.4482 | 0.7188 | 0.7189 |
| 10 | 0.1333 | 1.3797 | 0.7500 | 0.7518 |

## Test results
- Accuracy:   0.7552
- Macro F1:   0.7407
- Weighted F1:0.7407

### Per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.6154 | 1.0000 | 0.7619 | 32 |
| disgust | 0.7500 | 0.3750 | 0.5000 | 32 |
| fearful | 0.9000 | 0.5625 | 0.6923 | 32 |
| happy | 0.7143 | 0.7812 | 0.7463 | 32 |
| sad | 0.8235 | 0.8750 | 0.8485 | 32 |
| surprised | 0.8571 | 0.9375 | 0.8955 | 32 |

## Training time: 122.7 s

## Output
- Weights: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\runs_wavlm`
- Confusion matrix: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wavlm_confusion_matrix.png`
- Classification report: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wavlm_classification_report.csv`