# GoEmotions Text Model Training Analysis

Generated: 2026-10-08 12:33:50
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

| Model | Eval loss | Macro F1 | Micro F1 | Time (min) |
|---|---:|---:|---:|---:|
| roberta-base (RoBERTa-Base) | 0.1949 | 0.4337 | 0.5526 | 20.4 |
| bert-base-uncased (BERT-Base-Uncased) | 0.1965 | 0.4336 | 0.5561 | 20.0 |
| deberta-v3-base (DeBERTa-v3-Base) | inf | 0.0000 | 0.0000 | 0.0 |