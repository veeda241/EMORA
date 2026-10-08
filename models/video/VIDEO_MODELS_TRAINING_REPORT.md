# Video Models Training & Comparison Report

**Target Modality:** Video / Facial Emotion Recognition (FER)
**Classes:** 6 (`angry`, `disgust`, `fearful`, `happy`, `sad`, `surprised`)
**Embedding Output:** 512-dim multimodal projection vector

## Comparison Grid

| Model Key | Architecture | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Training Time |
|---|---|---|---|---|---|---|---|
| `mae-dfer` | MAE-DFER | `0.9996` | `0.9996` | `0.9996` | `0.9998` | `0.9994` | 46.9s |
| `est` | EST | `0.9998` | `0.9998` | `0.9998` | `0.9999` | `0.9997` | 40.3s |
| `hybrid-cnn-vit` | HYBRID-CNN-VIT | `0.9998` | `0.9989` | `0.9998` | `0.9979` | `0.9999` | 41.4s |


All models successfully trained and evaluated.
