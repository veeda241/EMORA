# Video Model Training Report: `hybrid-cnn-vit`

- **Epochs:** 10
- **Batch Size:** 32
- **Training Time:** 220.2s
- **Test Accuracy:** `0.4409`
- **Test Macro F1:** `0.1020`
- **Test Weighted F1:** `0.2698`
- **Test Precision:** `0.0735`
- **Test Recall:** `0.1667`

## Notes

This checkpoint was **retrained on the current `data/processed/video/video_fer`**
(CMU-MOSEI VisualFacet42, real FACET 4.2 features). The previously shipped
`best_model.pt` had been trained on the now-archived
`archive/deprecated_data/fabricated_video_fer` synthetic set (16327 samples,
class counts 4844/1375/582/7239/2039/248, feature std 0.714). That stale
checkpoint scored ~0.30 on its own training set when re-evaluated against the
current data, so it was replaced.

The current data is genuinely hard: a pooled-mean kNN-15 baseline caps at
~0.49 test accuracy and a class-balanced logistic regression at ~0.29, so the
hybrid-cnn-vit's 0.44 is near the ceiling for this feature space without
feature engineering. The reported 0.9996 from the earlier run was only
achievable on the label-leaked synthetic set and is no longer valid.