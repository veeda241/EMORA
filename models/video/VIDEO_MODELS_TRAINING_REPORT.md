# Video Models Training & Comparison Report

**Target Modality:** Video / Facial Emotion Recognition (FER)
**Classes:** 6 (`angry`, `disgust`, `fearful`, `happy`, `sad`, `surprised`)
**Embedding Output:** 512-dim multimodal projection vector

## Comparison Grid

| Model Key | Architecture | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Training Time |
|---|---|---|---|---|---|---|---|
| `mae-dfer` | MAE-DFER | `0.9996` | `0.9985` | `0.9996` | `0.9994` | `0.9976` | 51.7s |
| `est` | EST | `0.9994` | `0.9994` | `0.9994` | `0.9997` | `0.9991` | 40.6s |
| `hybrid-cnn-vit` | HYBRID-CNN-VIT | `0.9996` | `0.9996` | `0.9996` | `0.9998` | `0.9994` | 39.4s |


All models successfully trained and evaluated.
