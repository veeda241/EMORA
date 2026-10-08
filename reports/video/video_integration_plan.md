# Video Branch Integration Plan — Multimodal Emotion Recognition System

## 1. Goal

The eventual system fuses three modalities into a single emotion-state inference pipeline:

```
                 ┌── DeBERTa ──→ Text embedding
                 │
User interaction ├── Wav2Vec2 ─→ Audio embedding
                 │
                 └── ViT ──────→ Video embedding
                              ↓
                       Multimodal Fusion
                              ↓
                       Emotion State
                              ↓
                     HAPA State Inference
                              ↓
                  MI / BCT Intervention
```

This document describes how the **video branch** should be designed so it can
plug into that fusion architecture once video data becomes available.

## 2. Current status

- **Text branch**: trained. Best model = DeBERTa-v3-base, 87.32% eval accuracy.
- **Audio branch**: design underway; training blocked pending arrival of audio files.
- **Video branch**: **cannot be trained yet** — no video files exist in `data/raw/`.

See `video_dataset_inspection_report.md` for the full dataset inspection.

## 3. What the video branch must eventually expose

For clean multimodal fusion, the video model should provide a uniform interface:

```python
{
    "emotion": "...",            # predicted emotion label string
    "probabilities": {...},      # per-class probabilities
    "embedding": [...]           # fixed-size video embedding for fusion
}
```

The embedding is the key output for the fusion stage — it must be a single
fixed-size vector per video/utterance, regardless of how many frames were used.

## 4. Proposed model interface

A `VideoEmotionModel` wrapper should eventually expose:

```python
class VideoEmotionModel:
    def __init__(self, model_path, config_path):
        ...

    def predict_from_video_path(self, video_path) -> dict:
        """Full pipeline: read video → sample frames → face crop → ViT → classify."""

    def predict_from_frames(self, frames: List[np.ndarray]) -> dict:
        """Accept pre-extracted frames."""

    def embed_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Return only the fused video embedding, for multimodal fusion."""
```

This mirrors the expected text and audio branch interfaces.

## 5. Intended architecture (once data exists)

```
Video file
  ↓
Frame sampling (fixed number, uniform temporal spacing)
  ↓
Face detection + crop (per frame)
  ↓
Resize to 224×224 (ViT input size)
  ↓
Normalize (ViT expected mean/std)
  ↓
google/vit-base-patch16-224 (frame-level embeddings)
  ↓
Temporal aggregation (mean pooling / attention pooling)
  ↓
Linear classification head → emotion probabilities
  ↓
Embedding vector for fusion
```

Design constraints:

- Fixed frames per sample (e.g., 8, 12, or 16) — chosen based on GPU memory.
- Resolution 224×224 for ViT-base.
- Temporal aggregation must be lightweight (mean pooling first; optional attention).
- No large 3D CNN or VideoMAE initially.
- Batch size may need to be 1–2 per GPU step, with gradient accumulation.

## 6. Data requirements

To train the ViT video pipeline, the project needs:

1. **Actual video files** — `.mp4` or similar — with one video/clip per labeled sample.
2. **Label metadata** — a CSV or similar mapping each video to an emotion label.
3. The label set must be defined explicitly, with a **contiguous integer mapping 0..N-1**.
4. The same label mapping must be used across train/val/test.
5. Train/val/test must not leak samples from the same person/session across splits.

The existing datasets (`MELD`, `CMU-MOSEI`, `EmoryNLP`) currently supply only
metadata and precomputed features — not raw video.

## 7. How video labels should align with text and audio branches

For multimodal fusion to make sense, all three branches should use a **shared
emotion label space**. The label mapping must be agreed on before training any
modality, because the fusion model will combine embeddings that all predict the
same set of emotion classes.

Possible shared label strategy:

- Choose one canonical label set and map all modalities to it.
- For example, if MELD-style labels are chosen:
  ```
  0 -> anger
  1 -> disgust
  2 -> fear
  3 -> joy
  4 -> neutral
  5 -> sadness
  6 -> surprise
  ```
- If the text branch used a different set (e.g., 6 classes from `emotion_split`),
  the video and audio branches must either match it or a mapping must be defined.

This needs a deliberate decision before training video/audio so that later fusion
is consistent.

## 8. Face processing design

If videos exist:

- Detect face in each sampled frame.
- Crop face region; discard frame if no face found (but track failure rate).
- Do **not** silently drop too many samples — report total failures.
- If the dataset already provides reliable face crops or visual features, prefer
  those over running new face detection to save compute.
- Do not save duplicate full-size frames unless needed for debugging.

For MOSEI, note that FACET 4.2 features already exist — but these are not raw
face images and cannot directly feed ViT. Using them would be a separate
feature-based pipeline, not the ViT video branch.

## 9. Memory planning for 6 GB VRAM

A rough planning outline:

- ViT-base patch16 224 has ~86M parameters.
- Each sample = N frames × 224×224×3.
- For N=8 frames, one sample is 8×224×224×3 ≈ 1.2 MB of image data before
  model activations; training memory depends heavily on batch size, sequence
  length through ViT, gradient storage, and optimizer state.
- Likely starting point: batch_size=1–2, gradient accumulation to simulate a
  larger effective batch, mixed precision to reduce memory.

Exact batch size must be determined empirically on the target GPU by running a
memory probe with the real frame count and model.

## 10. Training script outline (to be created when data exists)

File target: `src/models/train_video_vit.py` (or `models/train_video_vit.py`).

Organization to mirror existing pipelines:

1. Dataset inspection / label validation.
2. Build label mapping, save `label_mapping.json`.
3. Video frame sampling + face preprocessing dataset class.
4. ViT + temporal pooling + classifier model.
5. Training loop with:
   - fixed random seed
   - mixed precision
   - gradient accumulation if needed
   - per-epoch validation
   - best-model checkpointing
6. Metrics:
   - accuracy, precision, recall, macro F1, weighted F1
   - per-class metrics
   - confusion matrix
   - train/val loss, learning rate, training time
7. Outputs under `models/video/`:
   - `runs_vit/`
   - `video_training_report.md`
   - `video_training_log.txt`
   - `confusion_matrix.png`
   - `classification_report.csv`
   - `label_mapping.json`
   - preprocessing config for inference

## 11. Training safety checks (to repeat before GPU work)

- num_labels derived from dataset, not hard-coded
- label range check: 0 <= label < num_labels for all splits
- corrupt/empty video check
- zero-frame and too-few-frame sample check
- invalid pixel value check
- face-detection failure rate report
- train/val/test leakage check where IDs available
- class distribution review
- model output dimension == num_labels
- CUDA assert diagnosis plan (do not blindly rerun)

## 12. What can be done right now

Since no video exists yet:

1. ✅ Dataset inspection completed.
2. ✅ Inspection report written.
3. ✅ This integration plan drafted so the video branch is designed in advance.
4. ⏸ Actual model training — blocked until video files are added.
5. ⏸ Label-space decision — should be made once all three modalities are scoped,
   to keep text/audio/video consistent for fusion.

## 13. Next steps when video data arrives

1. Place video files in a known location under the project.
2. Re-run dataset inspection to confirm format, frame rate, resolution, label file.
3. Define the shared emotion label mapping across modalities.
4. Build the video dataset class with frame sampling + face preprocessing.
5. Run a memory probe to choose batch size / gradient accumulation.
6. Train ViT video model.
7. Save artifacts under `models/video/`.
8. Verify the model can produce embeddings for fusion.
9. Integrate with text and audio branches into multimodal fusion once all three
   branches exist.
