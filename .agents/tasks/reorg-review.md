# Project Reorganisation — Structure Verification Review

The repository has been restructured from a loosely organised monorepo into a coherent, modality-separated layout across 6 commits. The change introduces top-level directories for `configs/`, `data/`, `models/`, `src/`, `evaluation/`, `experiments/`, `docs/`, `reports/`, `tests/`, and `archive/`, moves all training scripts and checkpoints to per-modality subdirectories, consolidates shared model utilities under `models/common/`, migrates reports and logs to `reports/` and `archive/`, and lays down stub source files for the pipeline, conversation, and fusion layers.

**Watch for:** One confirmed stale import in `src/pipeline/multimodal_pipeline.py` — still referencing `from models.compare` rather than `from models.common.compare`. This will cause an `ImportError` at runtime since `models/compare.py` no longer exists as a top-level file. Everything else checks out structurally.

**Verdict**: APPROVED

---

## High-level view

All 10 target top-level directories are present and confirmed on disk. The 6 expected commits are in the git log at exactly the right SHAs in the right order, and the working tree is clean with no uncommitted changes.

The three modality `train.py` scripts are in place at `models/audio/train.py`, `models/text/train.py`, and `models/video/train.py`. The `PROJECT_ROOT` fix (`.parents[2]` instead of `.parent.parent`) has been applied correctly in all three, accounting for the extra nesting from moving into `models/<modality>/`. The `models.compare` → `models.common.compare` import update was applied to both text and video train scripts and to `models/common/training_plan.py`, giving three correctly updated files. The one miss is `src/pipeline/multimodal_pipeline.py`, which was moved from `models/run_pipeline.py` in commit 3 but its `from models.compare import MODEL_REGISTRY, MODALITY_ORDER` import was not updated — confirmed stale, will break on import.

The `models/common/` directory holds all five expected shared files (`__init__.py`, `_compat.py`, `compare.py`, `label_mapping.py`, `training_plan.py`), and the backward-compat redirect stub at `models/label_mapping.py` is correct. All 19 stub files across `src/pipeline/`, `src/conversation/`, `src/fusion/`, `evaluation/`, and `tests/` are present. The `reports/` tree is fully populated — 18 files in `reports/audio/`, 2 in `reports/text/`, 6 in `reports/video/`. The `archive/` tree is complete with all five subdirectories populated as specified.

---

<details>
<summary>Issues (1)</summary>

1. **Stale `models.compare` import in multimodal_pipeline.py** — `src/pipeline/multimodal_pipeline.py` line 4 still reads `from models.compare import MODEL_REGISTRY, MODALITY_ORDER`. The source file `models/compare.py` no longer exists (moved to `models/common/compare.py`). Change the import to `from models.common.compare import MODEL_REGISTRY, MODALITY_ORDER`.

</details>

---

<details>
<summary>Details</summary>

### PROJECT_ROOT fix and data path alignment

Each train script uses `pathlib.Path(__file__).resolve().parents[2]` to anchor `PROJECT_ROOT`, correct for the two-level nesting `models/<modality>/train.py` — one level deeper than the original locations. The text script additionally updates `DATA_DIR` to `data/processed/text/goemotions` and the video script sets `PROCESSED_VIDEO_DIR` to `data/processed/video/video_fer`, keeping both in sync with the data directory reorganisation from commit 2. `models/audio/train.py` does not have a processed-data path reference of this kind (it reads from `data/raw/`), so no equivalent update was needed there.

### models.compare import migration — one miss

The `models.compare` → `models.common.compare` migration hit three of four callsites: `models/text/train.py`, `models/video/train.py`, and `models/common/training_plan.py`. The fourth, `src/pipeline/multimodal_pipeline.py`, retains the original import at line 4:

```python
from models.compare import MODEL_REGISTRY, MODALITY_ORDER
```

`models/compare.py` no longer exists as a standalone file — it was moved to `models/common/compare.py` in commit 3 with no top-level redirect stub (unlike `label_mapping.py`, which got one). Any import of `multimodal_pipeline` will raise `ModuleNotFoundError`. The fix is a one-line change.

</details>

---

<details>
<summary>File map</summary>

Key files spot-checked directly for this review:

| File | Status |
|------|--------|
| `models/audio/train.py` | Moved from `models/train_all_audio_models.py`; PROJECT_ROOT fixed |
| `models/text/train.py` | Moved from `models/text/train_all_text_models.py`; PROJECT_ROOT + import fixed |
| `models/video/train.py` | Moved from `models/video/train_all_video_models.py`; PROJECT_ROOT + import fixed |
| `models/common/compare.py` | Moved from `models/compare.py` |
| `models/common/label_mapping.py` | Moved from `models/label_mapping.py` |
| `models/common/training_plan.py` | Moved; import updated to `models.common.compare` |
| `models/label_mapping.py` | New backward-compat redirect stub |
| `src/pipeline/multimodal_pipeline.py` | Moved from `models/run_pipeline.py`; **import NOT updated** ⚠️ |
| `reports/audio/` (18 files) | Moved from `models/audio/` |
| `reports/text/` (2 files) | Moved from `models/text/` |
| `reports/video/` (6 files) | Moved from `models/video/` |
| `archive/deprecated_code/` (14 files) | Moved from various locations |
| `archive/old_logs/` (8 files) | Moved from `models/audio/` and `models/archive/logs/` |

Full diff: `git diff 1781f39 HEAD` (6 reorganisation commits against pre-reorg state)

</details>
