# Audio Training Report — wavlm (RAVDESS)

## Model
- Key: `wavlm`
- Checkpoint: `microsoft/wavlm-base`

## Dataset
- Name: RAVDESS
- Train: 768, Dev: 192, Test: 192
- Emotion classes (6): angry, disgust, fearful, happy, sad, surprised

## Training config
- Epochs: 1
- Batch size: 4
- Learning rate: 2e-05
- Optimizer: AdamW (weight_decay=0.01)
- Grad clip norm: 1.0
- Mixed precision: yes
- Seed: 42
- Device: cuda

## Per-epoch metrics

| Epoch | Train Loss | Dev Loss | Accuracy | Macro F1 |
|---|---|---|---|---|
| 1 | 1.7805 | 1.7938 | 0.1667 | 0.0476 |

## Test results
- Accuracy:   0.1667
- Macro F1:   0.0476
- Weighted F1:0.0476

### Per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.0000 | 0.0000 | 0.0000 | 32 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 32 |
| fearful | 0.0000 | 0.0000 | 0.0000 | 32 |
| happy | 0.1667 | 1.0000 | 0.2857 | 32 |
| sad | 0.0000 | 0.0000 | 0.0000 | 32 |
| surprised | 0.0000 | 0.0000 | 0.0000 | 32 |

## Training time: 2.5 s

## Output
- Weights: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\runs_wavlm`
- Confusion matrix: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wavlm_confusion_matrix.png`
- Classification report: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wavlm_classification_report.csv`