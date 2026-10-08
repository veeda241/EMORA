# Video Models Training & Comparison Report

**Target Modality:** Video / Facial Emotion Recognition (FER)
**Classes:** 6 (`angry`, `disgust`, `fearful`, `happy`, `sad`, `surprised`)
**Embedding Output:** 512-dim multimodal projection vector

## Comparison Grid

| Model Key | Architecture | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Training Time |
|---|---|---|---|---|---|---|---|
| `mae-dfer` | MAE-DFER | `0.3894` | `0.2419` | `0.4131` | `0.2458` | `0.2537` | 112.2s |
| `est` | EST | `0.3649` | `0.2384` | `0.3974` | `0.2475` | `0.2527` | 94.4s |
| `hybrid-cnn-vit` | HYBRID-CNN-VIT | `0.4153` | `0.2435` | `0.4282` | `0.2443` | `0.2471` | 96.5s |


All models successfully trained and evaluated.
