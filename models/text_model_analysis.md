# GoEmotions Text Model Training Analysis

Generated: 2026-10-09 05:50:46
Labels: 28 | Multilabel threshold: 0.5

## Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Train/eval batch | 8 / 8 |
| Gradient accumulation | 2 |
| Effective batch | 16 |
| Learning rate | 2e-05 |
| Weight decay | 0.01 |
| Warmup steps | 0 |
| Max sequence length | 128 |
| Scheduler | Linear |
| Best metric | Macro F1 |

## Comparison

| Model | Validation loss | Validation macro F1 | Test macro F1 | Test micro F1 | Time (min) |
|---|---:|---:|---:|---:|---:|
| roberta-base (RoBERTa-Base) | 0.1941 | 0.4430 | 0.4401 | 0.5545 | 20.4 |