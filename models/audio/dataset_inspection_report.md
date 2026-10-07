# Audio Emotion Dataset Inspection Report
## Generated: before Wav2Vec2 training

### 1. Raw directory listing (`data/raw/`)

```
cmu_mosei/
  CMU_MOSEI_COVAREP.csd          (HDF5)
  CMU_MOSEI_Labels.csd           (HDF5)
  CMU_MOSEI_OpenFace2.csd        (HDF5)
  CMU_MOSEI_TimestampedPhones.csd
  CMU_MOSEI_TimestampedWordVectors.csd
  CMU_MOSEI_TimestampedWords.csd
  CMU_MOSEI_VisualFacet42.csd   (HDF5)
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

No `.wav`, `.flac`, `.mp3`, `.ogg`, `.aiff` files were found in any subdirectory.

---

### 2. Audio file search results

Searched recursively for audio extensions:
- `*.wav`, `*.flac`, `*.mp3`, `*.ogg`, `*.aiff`, `*.aif`, `*.m4a`, `*.aac`, `*.wma`

**Result: 0 audio files found in `data/raw/`.**

---

### 3. Per-dataset audio feasibility

#### MELD (`meld/`)
- Format: CSV metadata (train/dev/test_sent_emo.csv) + YAML dialogue index.
- Emotion labels present as **strings**: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.
- **No audio tracks** (`.wav`/`.flac`/`.mp3`) present.
- Each CSV row references `Season`/`Episode`/`StartTime`/`EndTime` from a TV series, but the audio itself is not in `data/raw/`.
- **Actionable only if the original MELD audio files are copied into the project.** As-is: not usable for Wav2Vec2.

#### CMU-MOSEI (`cmu_mosei/`)
- Format: HDF5 (`.csd`) files — COVAREP acoustic features, Labels, OpenFace visual, timestamped words/phones/word-vectors, visual facet.
- `CMU_MOSEI_Labels.csd` contains per-segment emotion/sentiment features, NOT raw waveforms. The dimension names are `["sentiment","happy","sad","anger","surprise","disgust","fear"]`.
- COVAREP file carries low-level acoustic features (pitch, voice quality, etc.).
- **No raw audio files**. These are precomputed features — not compatible with a Wav2Vec2 waveform pipeline.

#### EmoryNLP (`emorynlp/`)
- Format: CSV files (train/dev/test_final).
- **No audio files** present; directory contains only CSV metadata.
- Dataset content not yet inspected in detail (see limitation below).

#### `meld_dya/`
- Appears to be a variant/dynamic version of MELD CSVs.
- **No audio files** present.

#### `pickles/`
- Contains `download-features.txt` (a text hint file, not data).
- **No audio files**.

#### `emotion_split/` and `emotion_unsplit/`
- Parquet files — from the text-modality training pipeline.
- **No audio files**.

---

### 4. Number of emotion classes

From **MELD** (the most complete labeled dataset present):

| Split | Samples | Distinct classes |
|-------|---------|-----------------|
| train | 9,989   | 7 |
| dev   | 1,109   | 7 |
| test  | 2,610   | 7 |
| **total** | **13,708** | **7** |

Class names (exact strings):
```
anger
disgust
fear
joy
neutral
sadness
surprise
```

Distribution (overall):
```
neutral  : 6436  (46.95%)
joy      : 2308  (16.84%)
anger    : 1607  (11.72%)
surprise : 1636  (11.93%)
sadness  : 1002  (7.31%)
disgust  :  361  (2.63%)
fear     :  358  (2.61%)
```

**MELD is heavily imbalanced** — neutral dominates.

Class count = **7**. Classes are **not** integer-encoded; they are strings.

---

### 5. CMU-MOSEI emotion dimensions (if used)

The label dimension names in `CMU_MOSEI_Labels.csd`:

```
["sentiment","happy","sad","anger","surprise","disgust","fear"]
```

That is:
- sentiment (continuous, here treated as a dimension)
- happy
- sad
- anger
- surprise
- disgust
- fear

= **7 labels** if sentiment is included, or **6** pure emotion classes if sentiment excluded. These are **continuous-valued** features (regression-style), not discrete classes. Wav2Vec2 sequence classification normally targets discrete classes, so CMU-MOSEI labels may require thresholding/binning — not a clean fit without extra processing.

---

### 6. Samples in train/validation/test

Only **metadata counts** are available — no audio samples exist to count.

| Dataset   | Train  | Dev   | Test  | Audio available? |
|-----------|--------|-------|-------|-----------------|
| MELD      | 9989   | 1109  | 2610  | No |
| CMU-MOSEI | (HDF5 segment counts only) | No raw waveform |
| EmoryNLP  | (need CSV inspection) | No audio |
| meld_dya  | CSV only | No audio |

---

### 7. Missing / corrupted audio files

Since **no audio files exist**, there are no corrupt or missing audio files to check. The real gap is the **absence of audio entirely**.

---

### 8. Label continuity from 0

- MELD emotion labels are **strings**, not integers.
- Not contiguous from 0 in raw form.
- They **must be mapped** to contiguous integer IDs before training.
- Proposed mapping (canonical order):
  ```
  0 -> anger
  1 -> disgust
  2 -> fear
  3 -> joy
  4 -> neutral
  5 -> sadness
  6 -> surprise
  ```
  Number of classes = 7. Must verify this mapping is consistent across all splits.

---

### 9. Class distribution

See section 4 — MELD distribution shown above. Strong neutral bias.

---

### 10. IMPORTANT — no assumption of 6 or 7 classes

- MELD has **7 classes**.
- CMU-MOSEI label space depends on whether sentiment is treated as a class — could be 6 or 7.
- The final class count must be computed directly from the dataset used, not assumed.

---

## Conclusion and required actions

**At present, `data/raw/` does NOT contain any audio waveforms usable for Wav2Vec2.**

Wav2Vec2 needs:
- actual `.wav`/`.flac` audio files (or streams), **AND**
- a label for each audio segment.

None of the datasets currently in `data/raw/` satisfy both. The project contains only metadata/label files plus precomputed features.

### What would need to happen first

1. **Acquire the actual MELD audio files** (or another labeled audio dataset) and place them so the training script can locate them by file path. Without that, no audio model training is possible.
2. Alternatively, if the intention is to use **COVAREP features** from CMU-MOSEI instead of raw waveforms, then the model should not be Wav2Vec2 — it should be trained on those precomputed features directly.

### Recommendation

Do **not** proceed to Wav2Vec2 training yet. First obtain audio files or decide to switch the audio pipeline to the precomputed COVAREP/MOSEI feature space.

If audio files are later added under `data/raw/`, re-run this inspection, then:
- build the label mapping,
- validate label ranges,
- build a Wav2Vec2 preprocessing + training script similar in structure to the text pipeline but adapted for audio.
