# Experiments

## Modality model change

The existing trained pipeline remains the **current baseline**. The new
experiment changes only the modality feature encoders and keeps MFN as the
fusion architecture:

| Modality | Current baseline | New experiment |
|---|---|---|
| Text | GloVe (300-D) | BERT embeddings (768-D) |
| Audio | COVAREP (74-D) | emotion2vec embeddings (768-D) |
| Video | FACET (35-D) | ViT-based temporal features (512-D) |
| Fusion | MFN | MFN adapted and trained for the new feature dimensions |

The profiles are defined in `src/config.py` as `baseline` and
`new_experiment`. `CONFIG` intentionally continues to select `baseline` so
the checked-in CMU-MOSEI `.npz` files and existing MFN checkpoint remain
compatible. Before training the new profile, produce aligned feature files
with shapes `(N, 16, 768)`, `(N, 16, 768)`, and `(N, 16, 512)` for text,
audio, and video respectively.
