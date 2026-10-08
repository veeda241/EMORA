# Video Emotion Dataset Inspection Report
Generated before any video training.

## 1. Raw directory listing (`data/raw/`)

```
cmu_mosei/
  CMU_MOSEI_COVAREP.csd          (HDF5)
  CMU_MOSEI_Labels.csd           (HDF5)
  CMU_MOSEI_OpenFace2.csd        (HDF5 — TRUNCATED/CORRUPT)
  CMU_MOSEI_TimestampedPhones.csd
  CMU_MOSEI_TimestampedWordVectors.csd
  CMU_MOSEI_TimestampedWords.csd
  CMU_MOSEI_VisualFacet42.csd   (HDF5 — FACET 4.2 visual features)
emorynlp/
  emorynlp_dev_final.csv
  emorynlp_test_final.csv
  emorynlp_train_final.csv
emotion_split/
  test-00000-of-00001.parquet
  train-00000-of-00001.parquet
  validation-00000-of-00001.parquet
emotion_unsplit/
  train-00000-of-00001.parquet
meld/
  datasets.yaml
  dev_sent_emo.csv
  test_sent_emo.csv
  train_sent_emo.csv
meld_dya/
  dev_sent_emo_dya.csv
  test_sent_emo_dya.csv
  train_sent_emo_dya.csv
pickles/
  download-features.txt
```

## 2. Video file search results

Searched recursively for video extensions:
- `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`

**Result: 0 video files found in `data/raw/`.**

## 3. Per-dataset video feasibility

### MELD / MELD_DYA
- CSV metadata with string emotion labels.
- No video files.
- References TV episodes but no video present.

### CMU-MOSEI
- Contains HDF5 feature/label files.
- `CMU_MOSEI_VisualFacet42.csd` stores **FACET 4.2 facial visual features** per frame:
  - 35-dimensional features per frame (AU, head pose, gender, etc.)
  - Each entry has `features` (frames × 35) and `intervals` (frame start/end times)
  - This is **already-extracted visual feature data**, not raw video
  - No raw `.mp4`/`.avi` files
- `CMU_MOSEI_OpenFace2.csd` is **truncated/corrupt** (cannot be opened)
- `CMU_MOSEI_COVAREP.csd` contains acoustic features, not video
- **Conclusion**: MOSEI has facial features but **no raw video**. A ViT-based video pipeline cannot use this directly because ViT needs image frames, not precomputed features. Using FACET features would mean training a model directly on those 35-dim features (not ViT video), which is a different pipeline.

### EmoryNLP
- CSV metadata only, no video files.

### emotion_split / emotion_unsplit
- Parquet files used for text training.
- No video.

### pickles
- Contains `download-features.txt` hint file.
- No video.

## 4. Emotion label availability (for reference)

The emotion labels present in existing CSVs are for **text** classification:

### MELD (string labels, 7 classes)
anger, disgust, fear, joy, neutral, sadness, surprise

### EmoryNLP (string labels, 7 classes)
Joyful, Mad, Neutral, Peaceful, Powerful, Sad, Scared

These labels are **per-utterance**, not per-video, and are associated with transcript text — not with video frames or faces.

## 5. Missing / corrupted items

| Item | Status | Notes |
|------|--------|-------|
| Any `.mp4`/`.avi`/`.mkv` video | **Missing** | None present |
| `CMU_MOSEI_OpenFace2.csd` | **Corrupt** | Truncated HDF5, cannot open |
| Raw video feeds for any dataset | **Missing** | Only metadata/features present |

## 6. Face detection feasibility

Face detection requires video frames or images.
- No video → no frames → no face crops.
- The FACET 4.2 features are already extracted facial features, not raw images.
- ViT-base expects 224×224 RGB images, not precomputed 35-d features.

## 7. Summary and conclusion

**No video data exists in `data/raw/`.** Therefore:

- A video/facial emotion model using ViT cannot be trained from the current project files.
- The closest video-adjacent data is the FACET 4.2 precomputed visual features in MOSEI, but that is not raw video and cannot feed a ViT video pipeline.
- To proceed with a video branch, actual video files (or frame images) must be added to the project.

## 8. Recommendation

Do **not** proceed to video training now. Options:

1. **Add the actual MELD/CMU-MOSEI/IEMOCAP video files** to `data/raw/` (or a subdirectory), then re-run this inspection.
2. **Use the FACET 4.2 features directly** — but that would be a feature-based model, not a ViT video model, and would require a different architecture.
3. **Defer the video branch** until video data is available; continue developing text (done) and audio (needs audio files too).
