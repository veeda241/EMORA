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
| 1 | 1.6968 | 1.5764 | 0.3802 | 0.3227 |
| 2 | 1.3533 | 1.4157 | 0.4375 | 0.4179 |
| 3 | 0.9075 | 1.1026 | 0.6406 | 0.6388 |
| 4 | 0.5804 | 1.0734 | 0.6458 | 0.6428 |
| 5 | 0.3719 | 1.0195 | 0.7188 | 0.7107 |
| 6 | 0.1964 | 1.0363 | 0.7396 | 0.7375 |
| 7 | 0.1223 | 0.9670 | 0.8021 | 0.8030 |
| 8 | 0.0627 | 1.1782 | 0.7708 | 0.7787 |
| 9 | 0.0573 | 1.5480 | 0.7188 | 0.7127 |
| 10 | 0.0956 | 1.7710 | 0.6927 | 0.6924 |

## Test results
- Accuracy:   0.7240
- Macro F1:   0.7182
- Weighted F1:0.7182

### Per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.6818 | 0.9375 | 0.7895 | 32 |
| disgust | 0.8571 | 0.5625 | 0.6792 | 32 |
| fearful | 0.9412 | 0.5000 | 0.6531 | 32 |
| happy | 0.5500 | 0.6875 | 0.6111 | 32 |
| sad | 0.7317 | 0.9375 | 0.8219 | 32 |
| surprised | 0.7931 | 0.7188 | 0.7541 | 32 |

## Training time: 100.3 s

## Output
- Weights: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\runs_wav2vec2`
- Confusion matrix: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wav2vec2_confusion_matrix.png`
- Classification report: `C:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support\models\audio\wav2vec2_classification_report.csv`