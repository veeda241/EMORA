# GoEmotions Text Model Training Report

**Generated:** 2026-10-07 00:19:02  
**Dataset:** GoEmotions  
**Labels:** 28  
**Training epochs:** 3  
**Best-model criterion:** Validation macro F1  
**Multilabel threshold:** 0.5

## Run status

The training run completed for all three registered text models. BERT and
RoBERTa completed successfully and produced valid validation metrics. DeBERTa
completed its scheduled epochs, but its loss and gradients became `NaN`; its
metrics are therefore invalid and it should not be selected as the best model.

| Model | Status | Runtime | Notes |
|---|---|---:|---|
| RoBERTa-Base | Completed | 59.7 min | Best valid macro F1 |
| BERT-Base-Uncased | Completed | 61.1 min | Valid result |
| DeBERTa-v3-Base | Invalid | 41.1 min | NaN loss/gradients throughout evaluation |

## Hyperparameters

| Parameter | Value |
|---|---:|
| Train/eval batch size | 8 / 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup steps | 0 |
| Maximum sequence length | 128 |
| Scheduler | Linear |
| Precision | FP32 (`fp16=False`) |

## Validation comparison

| Rank | Model | Eval loss | Macro F1 | Micro F1 | Macro precision | Macro recall | Runtime (min) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | RoBERTa-Base | 0.1942 | 0.4374 | 0.5569 | 0.6245 | 0.3641 | 59.7 |
| 2 | BERT-Base-Uncased | 0.1966 | 0.4221 | 0.5500 | 0.6438 | 0.3511 | 61.1 |
| — | DeBERTa-v3-Base | NaN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 41.1 |

## Training observations

- RoBERTa achieved the strongest valid validation result with macro F1
  **0.4374** and micro F1 **0.5569**.
- BERT achieved macro F1 **0.4221** and micro F1 **0.5500**.
- DeBERTa reported `NaN` loss and `NaN` gradient norms during training, so its
  zero-valued metrics are failure placeholders rather than model performance.
- The `UNEXPECTED` and `MISSING` checkpoint messages are expected when loading
  pretrained language-model weights into a new multilabel classification head.
- Tokenizer PAD/BOS/EOS alignment messages are informational and did not stop
  the successful BERT or RoBERTa runs.

## Recommendation

Use **RoBERTa-Base** as the current text-model baseline. DeBERTa should be
retrained separately after addressing its numerical instability, for example
with a lower learning rate, gradient clipping, or a compatible mixed-precision
configuration.
