# Audio Training Report — emotion2vec (RAVDESS)

## Model
- Key: `emotion2vec`
- Checkpoint: `iic/emotion2vec_base_finetuned`

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
| 1 | 1.0373 | 0.6778 | 0.8438 | 0.8454 |
| 2 | 0.3477 | 0.5394 | 0.8490 | 0.8496 |
| 3 | 0.2283 | 0.5618 | 0.8385 | 0.8393 |
| 4 | 0.1954 | 0.5861 | 0.8333 | 0.8341 |
| 5 | 0.1863 | 0.6078 | 0.8385 | 0.8391 |
| 6 | 0.1739 | 0.6243 | 0.8333 | 0.8342 |
| 7 | 0.1650 | 0.6373 | 0.8281 | 0.8297 |
| 8 | 0.1558 | 0.6436 | 0.8333 | 0.8348 |
| 9 | 0.1664 | 0.6565 | 0.8385 | 0.8404 |
| 10 | 0.1614 | 0.6747 | 0.8281 | 0.8306 |

## Test results
- Accuracy:   0.9271
- Macro F1:   0.9271
- Weighted F1:0.9271

### Per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 1.0000 | 1.0000 | 1.0000 | 32 |
| disgust | 0.8889 | 1.0000 | 0.9412 | 32 |
| fearful | 0.9286 | 0.8125 | 0.8667 | 32 |
| happy | 1.0000 | 0.9375 | 0.9677 | 32 |
| sad | 0.7941 | 0.8438 | 0.8182 | 32 |
| surprised | 0.9688 | 0.9688 | 0.9688 | 32 |

## Training time: 157.5 s

## Output
- Weights: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\runs_emotion2vec`
- Confusion matrix: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\emotion2vec_confusion_matrix.png`
- Classification report: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\emotion2vec_classification_report.csv`