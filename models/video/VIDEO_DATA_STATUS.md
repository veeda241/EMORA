# Video Data Status Report
**Generated:** 2026-10-08  
**Purpose:** Task 1 inspection — project state before any video training

---

## 1. Video Files in Project

**Result: Zero raw video files exist.**

Searched recursively for `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v`, `.flv`.  
No matches found anywhere under the project root.

---

## 2. Image / Frame Files in Project

**Result: Zero image files (jpg/png) exist** in any data directory.  
No pre-extracted face crops or video frames exist.

---

## 3. CMU-MOSEI Files Available

| File | Size | Status |
|---|---|---|
| `CMU_MOSEI_VisualFacet42.csd` | 1,656 MB | ✅ Readable |
| `CMU_MOSEI_Labels.csd` | 23 MB | ✅ Readable |
| `CMU_MOSEI_OpenFace2.csd` | ~8.9 GB (truncated) | ❌ Corrupt — HDF5 EOF mismatch |
| `CMU_MOSEI_COVAREP.csd` | — | Acoustic features only |
| `CMU_MOSEI_TimestampedWords.csd` | — | Text only |
| `CMU_MOSEI_TimestampedWordVectors.csd` | — | Text only |
| `CMU_MOSEI_TimestampedPhones.csd` | — | Text only |

**FACET 4.2 feature details (VisualFacet42):**
- 3,837 video IDs with precomputed facial features
- Feature shape per video: `(num_frames × 35)` float32
- Frames per video: min=177, max=34,100, mean≈3,668
- Feature content: Action Units (AU1–AU28), head pose, eye gaze, gender

**Label details (Labels.csd):**
- 3,293 video IDs with emotion intensity annotations
- 7 values per utterance segment: `[sentiment, happy, sad, angry, surprised, disgusted, fearful]`
- Scores are float intensities (0.0–3.0+), not discrete class labels
- Total labeled utterance segments: **23,259**
- Overlap with VisualFacet42: **3,293 videos / 23,259 segments**

**Class distribution (dominant emotion, ignoring sentiment):**

| Emotion | Count | % |
|---|---|---|
| happy | 14,567 | 62.6% |
| sad | 3,782 | 16.3% |
| angry | 2,730 | 11.7% |
| disgusted | 1,291 | 5.6% |
| surprised | 437 | 1.9% |
| fearful | 452 | 1.9% |

⚠️ Significant class imbalance. `happy` dominates at 62.6%.  
⚠️ `surprised` and `fearful` have <500 samples each — challenging for learning.

---

## 4. Existing Video/FER Scripts

None. The only video-related files are:
- `models/video/video_dataset_inspection_report.md` — previous inspection
- `models/video/video_integration_plan.md` — architecture design doc

No Python training, preprocessing, or inference scripts exist for video.

---

## 5. GPU and Runtime Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| VRAM | 6.0 GB |
| Compute capability | 8.6 (Ampere) |
| CUDA version | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Python | 3.14.0 |

---

## 6. Video-Relevant Dependencies Installed

| Package | Status |
|---|---|
| torchvision | ❌ NOT INSTALLED |
| timm | ❌ NOT INSTALLED |
| opencv-python (cv2) | ❌ NOT INSTALLED |
| facenet-pytorch | ❌ NOT INSTALLED |
| mediapipe | ❌ NOT INSTALLED |
| einops | ❌ NOT INSTALLED |
| decord | ❌ NOT INSTALLED |
| av (PyAV) | ❌ NOT INSTALLED |
| imageio | ❌ NOT INSTALLED |
| h5py | ✅ INSTALLED (just installed for inspection) |

**All video processing dependencies need to be installed before any pipeline work.**

---

## 7. Summary

| Item | Status |
|---|---|
| Raw video files | ❌ None |
| Face image crops | ❌ None |
| Video frame sequences | ❌ None |
| CMU-MOSEI FACET features | ✅ Available (1.6 GB, 23,259 segments) |
| CMU-MOSEI labels | ✅ Available |
| CMU-MOSEI raw video | ❌ Not present |
| Existing FER scripts | ❌ None |
| Video dependencies | ❌ All missing |

**Conclusion:** The project has no raw video data and no video pipeline. The only video-adjacent data is precomputed FACET 4.2 facial features from CMU-MOSEI, which can serve as a baseline but cannot feed a ViT/CNN frame-based model. A raw-video FER dataset must be acquired before training a proper deep video FER model.
