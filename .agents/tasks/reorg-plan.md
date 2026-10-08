# Implementation Plan — Project Reorganisation

> Grounded in exploration of the repo as of commit `1781f39` (HEAD → main).
> Every path is relative to the project root unless stated otherwise.
>
> **Key rules applied throughout:**
> - `git mv` for every file that is already tracked by git (preserves blame/history).
> - `Move-Item` (PowerShell) for untracked files and directories whose contents are not tracked.
> - Nothing is deleted. Items that do not fit elsewhere go to `archive/`.
> - After all moves, broken imports are fixed in a single dedicated step.
> - The six commit groups are listed at the end of this file.

---

## Inventory: what exists vs. what must be created

### Exists (git-tracked, must use `git mv`)
- `models/__init__.py` → moves to `models/common/__init__.py`
- `models/_compat.py` → moves to `models/common/_compat.py`
- `models/compare.py` → moves to `models/common/compare.py`
- `models/label_mapping.py` → moves to `models/common/label_mapping.py`
- `models/training_plan.py` → moves to `models/common/training_plan.py`
- `models/embedding_shapes.md` → moves to `docs/architecture.md`
- `models/run_pipeline.py` → moves to `src/pipeline/` as `multimodal_pipeline.py`
- `models/train_all_audio_models.py` → moves to `models/audio/train.py`
- `models/text/train_all_text_models.py` → moves to `models/text/train.py`
- `models/text/train_all_text_models_dry_run.py` → moves to `archive/deprecated_code/`
- `models/text/__init__.py` → keep in place (it stays at `models/text/__init__.py`)
- `models/text/TEXT_MODELS_TRAINING_REPORT.md` → moves to `reports/text/`
- `models/text/text_model_analysis.md` → moves to `reports/text/`
- `models/video/train_all_video_models.py` → moves to `models/video/train.py`
- `models/video/VIDEO_MODELS_TRAINING_REPORT.md` → moves to `reports/video/`
- `models/video/VIDEO_DATASET_HYPERPARAMETER_PLAN.md` → moves to `reports/video/`
- `models/video/VIDEO_DATA_STATUS.md` → moves to `reports/video/`
- `models/video/VIDEO_FER_RESEARCH_REPORT.md` → moves to `reports/video/`
- `models/video/video_dataset_inspection_report.md` → moves to `reports/video/`
- `models/video/video_integration_plan.md` → moves to `reports/video/`
- `models/video/configs/mae_dfer_dfew.yaml` — stays under `models/video/configs/` (no move needed)
- `models/video/runs_mae_dfer/**` → moves to `models/video/checkpoints/runs_mae_dfer/`
- `models/video/runs_est/**` → moves to `models/video/checkpoints/runs_est/`
- `models/video/runs_hybrid_cnn_vit/**` → moves to `models/video/checkpoints/runs_hybrid_cnn_vit/`
- `models/audio/runs_wav2vec2/**` → moves to `models/audio/checkpoints/runs_wav2vec2/`
- `models/audio/runs_wavlm/**` → moves to `models/audio/checkpoints/runs_wavlm/`
- `models/audio/runs_emotion2vec/**` → moves to `models/audio/checkpoints/runs_emotion2vec/`
- `models/audio/AUDIO_MODELS_TRAINING_REPORT.md` → moves to `reports/audio/`
- `models/audio/audio_training_report.md` → moves to `reports/audio/`
- `models/audio/audio_readiness_report.md` → moves to `reports/audio/`
- `models/audio/model_comparison_summary.md` → moves to `reports/audio/`
- `models/audio/dataset_inspection_report.md` → moves to `reports/audio/`
- `models/audio/crema_d_file_format_diagnostic.md` → moves to `reports/audio/`
- `models/audio/classification_report.csv` → moves to `reports/audio/`
- `models/audio/emotion2vec_classification_report.csv` → moves to `reports/audio/`
- `models/audio/wav2vec2_classification_report.csv` → moves to `reports/audio/`
- `models/audio/wavlm_classification_report.csv` → moves to `reports/audio/`
- `models/audio/crema_d_file_format_diagnostic.csv` → moves to `reports/audio/`
- `models/audio/emotion2vec_confusion_matrix.png` → moves to `reports/audio/`
- `models/audio/wav2vec2_confusion_matrix.png` → moves to `reports/audio/`
- `models/audio/wavlm_confusion_matrix.png` → moves to `reports/audio/`
- `models/audio/ravdess_confusion_matrix.png` → moves to `reports/audio/`
- `models/audio/audio_training_log.txt` → moves to `archive/old_logs/`
- `models/audio/emotion2vec_training_log.txt` → moves to `archive/old_logs/`
- `models/audio/wav2vec2_training_log.txt` → moves to `archive/old_logs/`
- `models/audio/wavlm_training_log.txt` → moves to `archive/old_logs/`
- `models/audio/audio_inspection_only.log` → moves to `archive/old_logs/`
- `models/audio/audio_module_load_test.log` → moves to `archive/old_logs/`
- `models/audio/label_mapping.json` — leave in place (it is audio-specific metadata)
- `models/audio/audio_training_report.md` — duplicate of `audio_training_report.md` above; include in same move
- `models/archive/README.md` → moves to `archive/old_models/README.md`
- `models/archive/checkpoints/runs_bert-base-uncased/**` → moves to `models/text/checkpoints/runs_bert-base-uncased/`
- `models/archive/checkpoints/runs_deberta-v3-base/**` → moves to `models/text/checkpoints/runs_deberta-v3-base/`
- `models/archive/checkpoints/runs_roberta-base/**` → moves to `models/text/checkpoints/runs_roberta-base/`
- `models/archive/logs/train_all_log.txt` → moves to `archive/old_logs/`
- `models/archive/logs/train_run_20261006_052734.txt` → moves to `archive/old_logs/`
- `models/archive/reports/comparison_table.md` → moves to `archive/old_reports/`
- `models/archive/reports/training_plan.md` → moves to `archive/old_reports/`
- `models/preprocessing/**` — stays at `models/preprocessing/` (contents unchanged)
- `models/video/results/**` — stays at `models/video/results/` (no target stated for these)
- `models/video/MAE-DFER` (git submodule or tracked dir) — stays under `models/video/MAE-DFER`
- `replace_emotions.py` → moves to `archive/deprecated_code/`
- `scripts/dataset_inspection_report.md` → moves to `docs/datasets.md`
- `scripts/download_ravdess.py` → moves to `archive/deprecated_code/`
- `scripts/inspect_meld.py` → moves to `archive/deprecated_code/`
- `scripts/prepare_goemotions.py` → moves to `archive/deprecated_code/`
- `scripts/prepare_meld.py` → moves to `archive/deprecated_code/`
- `scripts/prepare_ravdess.py` → moves to `archive/deprecated_code/`
- `scripts/video/evaluate.py` → moves to `archive/deprecated_code/`
- `scripts/video/inference.py` → moves to `archive/deprecated_code/`
- `scripts/video/inspect_dfew.py` → moves to `archive/deprecated_code/`
- `scripts/video/prepare_dfew.py` → moves to `archive/deprecated_code/`
- `scripts/video/prepare_video_data.py` → moves to `archive/deprecated_code/`
- `scripts/video/train_cmu_mosei_baseline.py` → moves to `archive/deprecated_code/`
- `scripts/video/train_mae_dfer.py` → moves to `archive/deprecated_code/`
- `data/processed/ravdess/` — **not moved**; but a `data/processed/audio/` symlink/move is required per target spec — see step 6 below
- `data/processed/cmu_mosei/` → moves to `data/processed/audio/cmu_mosei/`
- `data/processed/goemotions/` → moves to `data/processed/text/goemotions/`
- `data/processed/meld/` (processed) → moves to `data/processed/text/meld/`
- `data/processed/video_fer/` → moves to `data/processed/video/video_fer/`

### Must be created as stubs / empty folders
- `configs/audio.yaml`, `configs/text.yaml`, `configs/video.yaml`
- `src/pipeline/text_pipeline.py`, `audio_pipeline.py`, `video_pipeline.py`
  (`multimodal_pipeline.py` comes from `models/run_pipeline.py` via `git mv`)
- `src/conversation/agent.py`, `prompts.py`, `memory.py`, `response_generator.py`
- `src/fusion/feature_fusion.py`, `emotion_fusion.py`
- `src/utils/__init__.py`
- `models/audio/inference.py` (stub)
- `models/text/inference.py` (stub)
- `models/video/inference.py` (stub)
- `models/label_mapping.py` (redirect stub at `models/` root pointing to `models/common/label_mapping.py`)
- `evaluation/audio/`, `evaluation/text/`, `evaluation/video/`, `evaluation/multimodal/`
- `evaluation/comparison.py`, `evaluation/metrics.py`
- `experiments/configs/`, `experiments/logs/`, `experiments/results/`, `experiments/notebooks/`
- `docs/architecture.md` (populated from `models/embedding_shapes.md`), `docs/methodology.md`,
  `docs/training.md`, `docs/experiments.md`, `docs/research_gap.md`
- `docs/datasets.md` (populated from `scripts/dataset_inspection_report.md`)
- `docs/paper_notes/` with `.gitkeep`
- `data/metadata/` with `.gitkeep`
- `data/processed/audio/`, `data/processed/text/`, `data/processed/video/`
- `reports/audio/`, `reports/text/`, `reports/video/`, `reports/multimodal/`
- `archive/old_models/`, `archive/old_runs/`, `archive/old_logs/`, `archive/old_reports/`, `archive/deprecated_code/`
- `tests/test_audio.py`, `tests/test_text.py`, `tests/test_video.py`, `tests/test_pipeline.py`
- `requirements.txt`, `.env.example`

---

## Ordered Implementation Steps

---

- [ ] 1. **Create all new empty directories and placeholder files**

  Create every folder in the target layout that does not yet exist, plus `.gitkeep` markers for intentionally-empty leaf folders. Do this first so that subsequent `git mv` commands have valid destination paths.

  Directories to create:
  ```
  .agents/tasks/            (already created)
  configs/
  src/pipeline/
  src/conversation/
  src/fusion/
  src/utils/
  models/common/
  models/audio/checkpoints/
  models/text/checkpoints/
  models/video/checkpoints/
  evaluation/audio/
  evaluation/text/
  evaluation/video/
  evaluation/multimodal/
  experiments/configs/
  experiments/logs/
  experiments/results/
  experiments/notebooks/
  docs/paper_notes/
  data/processed/audio/
  data/processed/text/
  data/processed/video/
  data/metadata/
  reports/audio/
  reports/text/
  reports/video/
  reports/multimodal/
  archive/old_models/
  archive/old_runs/
  archive/old_logs/
  archive/old_reports/
  archive/deprecated_code/
  tests/
  ```

  Files to touch (`.gitkeep` only — no content):
  ```
  docs/paper_notes/.gitkeep
  data/metadata/.gitkeep
  experiments/configs/.gitkeep
  experiments/logs/.gitkeep
  experiments/results/.gitkeep
  experiments/notebooks/.gitkeep
  archive/old_runs/.gitkeep
  evaluation/audio/.gitkeep
  evaluation/text/.gitkeep
  evaluation/video/.gitkeep
  evaluation/multimodal/.gitkeep
  reports/multimodal/.gitkeep
  ```

  PowerShell to create all dirs:
  ```powershell
  $root = "c:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support"
  $dirs = @(
    "configs","src\pipeline","src\conversation","src\fusion","src\utils",
    "models\common","models\audio\checkpoints","models\text\checkpoints","models\video\checkpoints",
    "evaluation\audio","evaluation\text","evaluation\video","evaluation\multimodal",
    "experiments\configs","experiments\logs","experiments\results","experiments\notebooks",
    "docs\paper_notes","data\processed\audio","data\processed\text","data\processed\video",
    "data\metadata","reports\audio","reports\text","reports\video","reports\multimodal",
    "archive\old_models","archive\old_runs","archive\old_logs","archive\old_reports","archive\deprecated_code",
    "tests"
  )
  foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path "$root\$d" | Out-Null }
  $gitkeeps = @(
    "docs\paper_notes\.gitkeep","data\metadata\.gitkeep",
    "experiments\configs\.gitkeep","experiments\logs\.gitkeep",
    "experiments\results\.gitkeep","experiments\notebooks\.gitkeep",
    "archive\old_runs\.gitkeep","evaluation\audio\.gitkeep","evaluation\text\.gitkeep",
    "evaluation\video\.gitkeep","evaluation\multimodal\.gitkeep","reports\multimodal\.gitkeep"
  )
  foreach ($f in $gitkeeps) { New-Item -Force -Path "$root\$f" -ItemType File | Out-Null }
  ```

  Files: various `.gitkeep` files as listed above.

  Verify: `Get-ChildItem -Path $root -Directory -Recurse | Where-Object { $_.Name -in @("configs","models\common","src","evaluation","experiments","docs","reports","archive","tests","data\metadata") } | Select-Object FullName` — all target folders present.

---

- [ ] 2. **Move models/common files (git mv) — models root utilities**

  These five files move from the `models/` root into the new `models/common/` package. All are git-tracked; use `git mv`.

  ```bash
  git mv models/__init__.py      models/common/__init__.py
  git mv models/_compat.py       models/common/_compat.py
  git mv models/compare.py       models/common/compare.py
  git mv models/label_mapping.py models/common/label_mapping.py
  git mv models/training_plan.py models/common/training_plan.py
  ```

  Also move `models/embedding_shapes.md` to `docs/architecture.md`:
  ```bash
  git mv models/embedding_shapes.md docs/architecture.md
  ```

  And move `models/run_pipeline.py` to its new `src/` home:
  ```bash
  git mv models/run_pipeline.py src/pipeline/multimodal_pipeline.py
  ```

  After the `git mv` commands, create the `models/label_mapping.py` redirect stub at the models root (new untracked file — do NOT `git mv` it; just create it):
  ```python
  # models/label_mapping.py
  # Redirect stub — real definition lives in models/common/label_mapping.py
  from models.common.label_mapping import *  # noqa: F401,F403
  ```

  Files modified:
  - `models/__init__.py` → `models/common/__init__.py` (git mv)
  - `models/_compat.py` → `models/common/_compat.py` (git mv)
  - `models/compare.py` → `models/common/compare.py` (git mv)
  - `models/label_mapping.py` → `models/common/label_mapping.py` (git mv)
  - `models/training_plan.py` → `models/common/training_plan.py` (git mv)
  - `models/embedding_shapes.md` → `docs/architecture.md` (git mv)
  - `models/run_pipeline.py` → `src/pipeline/multimodal_pipeline.py` (git mv)
  - `models/label_mapping.py` (new stub created at models root)

  Verify: `git status` shows all seven renames staged and the new stub appears as an untracked file; `python -c "from models.common.compare import MODEL_REGISTRY; print('ok')"` from project root succeeds.

---

- [ ] 3. **Move train scripts (git mv) — audio, text, video**

  Rename each modality's master training script into its final location:

  ```bash
  git mv models/train_all_audio_models.py models/audio/train.py
  git mv models/text/train_all_text_models.py models/text/train.py
  git mv models/video/train_all_video_models.py models/video/train.py
  ```

  Files modified:
  - `models/train_all_audio_models.py` → `models/audio/train.py`
  - `models/text/train_all_text_models.py` → `models/text/train.py`
  - `models/video/train_all_video_models.py` → `models/video/train.py`

  Verify: `git status` shows exactly three renames staged. The files exist at new paths:
  ```
  models/audio/train.py
  models/text/train.py
  models/video/train.py
  ```

---

- [ ] 4. **Move audio checkpoint runs (git mv)**

  Move the three audio model run directories into `models/audio/checkpoints/`:

  ```bash
  git mv models/audio/runs_wav2vec2   models/audio/checkpoints/runs_wav2vec2
  git mv models/audio/runs_wavlm      models/audio/checkpoints/runs_wavlm
  git mv models/audio/runs_emotion2vec models/audio/checkpoints/runs_emotion2vec
  ```

  Files modified (all tracked files under those directories):
  - `models/audio/runs_wav2vec2/config.json` etc. → `models/audio/checkpoints/runs_wav2vec2/`
  - `models/audio/runs_wavlm/config.json` etc. → `models/audio/checkpoints/runs_wavlm/`
  - `models/audio/runs_emotion2vec/preprocessor_config.json` → `models/audio/checkpoints/runs_emotion2vec/`

  Verify: `git status` shows the checkpoint-file renames staged; `ls models/audio/checkpoints/` lists three run folders.

---

- [ ] 5. **Move text checkpoints from models/archive (git mv)**

  Move the three text-model checkpoint directories from `models/archive/checkpoints/` into `models/text/checkpoints/`:

  ```bash
  git mv models/archive/checkpoints/runs_bert-base-uncased    models/text/checkpoints/runs_bert-base-uncased
  git mv models/archive/checkpoints/runs_deberta-v3-base      models/text/checkpoints/runs_deberta-v3-base
  git mv models/archive/checkpoints/runs_roberta-base         models/text/checkpoints/runs_roberta-base
  ```

  Files modified: all tracked files within those three checkpoint dirs.

  Verify: `git status` shows renames staged; `ls models/text/checkpoints/` lists three run folders.

---

- [ ] 6. **Move video checkpoint runs (git mv)**

  Move the three video run dirs into `models/video/checkpoints/`:

  ```bash
  git mv models/video/runs_mae_dfer        models/video/checkpoints/runs_mae_dfer
  git mv models/video/runs_est             models/video/checkpoints/runs_est
  git mv models/video/runs_hybrid_cnn_vit  models/video/checkpoints/runs_hybrid_cnn_vit
  ```

  Files modified: all tracked files within the three video run dirs.

  Verify: `git status` shows renames staged; `ls models/video/checkpoints/` lists three run folders.

---

- [ ] 7. **Move audio reports and logs (git mv)**

  Move report/artefact files from `models/audio/` to `reports/audio/`, and log files to `archive/old_logs/`:

  ```bash
  # Reports → reports/audio/
  git mv models/audio/AUDIO_MODELS_TRAINING_REPORT.md    reports/audio/AUDIO_MODELS_TRAINING_REPORT.md
  git mv models/audio/audio_training_report.md           reports/audio/audio_training_report.md
  git mv models/audio/audio_readiness_report.md          reports/audio/audio_readiness_report.md
  git mv models/audio/model_comparison_summary.md        reports/audio/model_comparison_summary.md
  git mv models/audio/dataset_inspection_report.md       reports/audio/dataset_inspection_report.md
  git mv models/audio/crema_d_file_format_diagnostic.md  reports/audio/crema_d_file_format_diagnostic.md
  git mv models/audio/classification_report.csv          reports/audio/classification_report.csv
  git mv models/audio/emotion2vec_classification_report.csv  reports/audio/emotion2vec_classification_report.csv
  git mv models/audio/wav2vec2_classification_report.csv    reports/audio/wav2vec2_classification_report.csv
  git mv models/audio/wavlm_classification_report.csv       reports/audio/wavlm_classification_report.csv
  git mv models/audio/crema_d_file_format_diagnostic.csv    reports/audio/crema_d_file_format_diagnostic.csv
  git mv models/audio/emotion2vec_confusion_matrix.png   reports/audio/emotion2vec_confusion_matrix.png
  git mv models/audio/wav2vec2_confusion_matrix.png      reports/audio/wav2vec2_confusion_matrix.png
  git mv models/audio/wavlm_confusion_matrix.png         reports/audio/wavlm_confusion_matrix.png
  git mv models/audio/ravdess_confusion_matrix.png       reports/audio/ravdess_confusion_matrix.png

  # Logs → archive/old_logs/
  git mv models/audio/audio_training_log.txt      archive/old_logs/audio_training_log.txt
  git mv models/audio/emotion2vec_training_log.txt archive/old_logs/emotion2vec_training_log.txt
  git mv models/audio/wav2vec2_training_log.txt    archive/old_logs/wav2vec2_training_log.txt
  git mv models/audio/wavlm_training_log.txt       archive/old_logs/wavlm_training_log.txt
  git mv models/audio/audio_inspection_only.log    archive/old_logs/audio_inspection_only.log
  git mv models/audio/audio_module_load_test.log   archive/old_logs/audio_module_load_test.log
  ```

  Note: `emotion2vec_training_report.md`, `wav2vec2_training_report.md`, `wavlm_training_report.md` are also in `models/audio/` and go to `reports/audio/`:
  ```bash
  git mv models/audio/emotion2vec_training_report.md reports/audio/emotion2vec_training_report.md
  git mv models/audio/wav2vec2_training_report.md    reports/audio/wav2vec2_training_report.md
  git mv models/audio/wavlm_training_report.md       reports/audio/wavlm_training_report.md
  ```

  Files modified: ~24 `git mv` operations as listed above.

  Verify: `git status` shows all audio report/log renames staged; `ls reports/audio/` lists 18+ files; `ls archive/old_logs/` lists 6 log files.

---

- [ ] 8. **Move text reports (git mv)**

  ```bash
  git mv models/text/TEXT_MODELS_TRAINING_REPORT.md reports/text/TEXT_MODELS_TRAINING_REPORT.md
  git mv models/text/text_model_analysis.md          reports/text/text_model_analysis.md
  ```

  Files modified: 2 `git mv` operations.

  Verify: `ls reports/text/` lists both files; `git status` shows the renames staged.

---

- [ ] 9. **Move video reports (git mv)**

  ```bash
  git mv models/video/VIDEO_MODELS_TRAINING_REPORT.md        reports/video/VIDEO_MODELS_TRAINING_REPORT.md
  git mv models/video/VIDEO_DATASET_HYPERPARAMETER_PLAN.md   reports/video/VIDEO_DATASET_HYPERPARAMETER_PLAN.md
  git mv models/video/VIDEO_DATA_STATUS.md                   reports/video/VIDEO_DATA_STATUS.md
  git mv models/video/VIDEO_FER_RESEARCH_REPORT.md           reports/video/VIDEO_FER_RESEARCH_REPORT.md
  git mv models/video/video_dataset_inspection_report.md     reports/video/video_dataset_inspection_report.md
  git mv models/video/video_integration_plan.md              reports/video/video_integration_plan.md
  ```

  Files modified: 6 `git mv` operations.

  Verify: `ls reports/video/` lists 6 files.

---

- [ ] 10. **Move models/archive contents to archive/old_models and archive/old_logs (git mv)**

  Move the models/archive's tracked metadata and log files. The checkpoint subdirs were already moved in step 5.

  ```bash
  git mv models/archive/README.md                        archive/old_models/README.md
  git mv models/archive/logs/train_all_log.txt           archive/old_logs/train_all_log.txt
  git mv models/archive/logs/train_run_20261006_052734.txt archive/old_logs/train_run_20261006_052734.txt
  git mv models/archive/reports/comparison_table.md     archive/old_reports/comparison_table.md
  git mv models/archive/reports/training_plan.md        archive/old_reports/training_plan.md
  ```

  Files modified: 5 `git mv` operations.

  Verify: `ls archive/old_models/`; `ls archive/old_logs/`; `ls archive/old_reports/` all list the moved files.

---

- [ ] 11. **Move deprecated code and scripts to archive/deprecated_code (git mv)**

  ```bash
  git mv replace_emotions.py                             archive/deprecated_code/replace_emotions.py
  git mv models/text/train_all_text_models_dry_run.py    archive/deprecated_code/train_all_text_models_dry_run.py
  git mv scripts/download_ravdess.py                     archive/deprecated_code/download_ravdess.py
  git mv scripts/inspect_meld.py                         archive/deprecated_code/inspect_meld.py
  git mv scripts/prepare_goemotions.py                   archive/deprecated_code/prepare_goemotions.py
  git mv scripts/prepare_meld.py                         archive/deprecated_code/prepare_meld.py
  git mv scripts/prepare_ravdess.py                      archive/deprecated_code/prepare_ravdess.py
  git mv scripts/video/evaluate.py                       archive/deprecated_code/video_evaluate.py
  git mv scripts/video/inference.py                      archive/deprecated_code/video_inference.py
  git mv scripts/video/inspect_dfew.py                   archive/deprecated_code/inspect_dfew.py
  git mv scripts/video/prepare_dfew.py                   archive/deprecated_code/prepare_dfew.py
  git mv scripts/video/prepare_video_data.py             archive/deprecated_code/prepare_video_data.py
  git mv scripts/video/train_cmu_mosei_baseline.py       archive/deprecated_code/train_cmu_mosei_baseline.py
  git mv scripts/video/train_mae_dfer.py                 archive/deprecated_code/train_mae_dfer.py
  git mv scripts/dataset_inspection_report.md            docs/datasets.md
  ```

  Note: the last line moves the dataset inspection report to `docs/datasets.md` as specified.

  Files modified: 15 `git mv` operations.

  Verify: `ls archive/deprecated_code/` lists 14 Python files; `ls docs/` contains `datasets.md`.

---

- [ ] 12. **Move processed data subdirectories**

  `data/processed/ravdess/` and `data/processed/cmu_mosei/` contain mostly untracked `.parquet` / `.csv` / `.npz` files (the git ls-files audit shows only a handful are tracked). Use a two-step approach: `git mv` for any tracked files, then `Move-Item` for the untracked remainder.

  Determine which files under these dirs are git-tracked first:
  ```powershell
  cd $root
  git ls-files data/processed/ravdess data/processed/cmu_mosei data/processed/goemotions data/processed/meld data/processed/video_fer
  ```

  **For any git-tracked file in those dirs** — `git mv` them individually to the new paths.

  **For all untracked content** — use `Move-Item` after git-tracked files are staged:
  ```powershell
  $root = "c:\Users\ADMIN\Downloads\spiderboy projects\LLM-based-conversational-agents-for-behaviour-change-support"
  # Audio processed data
  Move-Item "$root\data\processed\ravdess"   "$root\data\processed\audio\ravdess"   -Force
  Move-Item "$root\data\processed\cmu_mosei" "$root\data\processed\audio\cmu_mosei" -Force
  # Text processed data
  Move-Item "$root\data\processed\goemotions" "$root\data\processed\text\goemotions" -Force
  Move-Item "$root\data\processed\meld"       "$root\data\processed\text\meld"       -Force
  # Video processed data
  Move-Item "$root\data\processed\video_fer"  "$root\data\processed\video\video_fer" -Force
  ```

  **Important:** `data/processed/audio/`, `data/processed/text/`, and `data/processed/video/` folders were created in step 1, so the destination exists. Do not move `data/processed/audio/` (which itself may have items) into itself — create it clean in step 1 before running these moves.

  Files/dirs modified:
  - `data/processed/ravdess/` → `data/processed/audio/ravdess/`
  - `data/processed/cmu_mosei/` → `data/processed/audio/cmu_mosei/`
  - `data/processed/goemotions/` → `data/processed/text/goemotions/`
  - `data/processed/meld/` → `data/processed/text/meld/`
  - `data/processed/video_fer/` → `data/processed/video/video_fer/`

  Verify: `ls data/processed/audio/`; `ls data/processed/text/`; `ls data/processed/video/` each list the expected subdirectories.

---

- [ ] 13. **Create configs/ YAML stubs**

  Create minimal YAML stubs in `configs/`. Each file names its modality's key hyperparameters as discovered from the train scripts (learning rate, batch size, epochs, data paths).

  `configs/audio.yaml`:
  ```yaml
  # Audio model training configuration
  modality: audio
  data:
    raw_meld: data/raw/meld
    raw_ravdess: data/raw/ravdess
    processed: data/processed/audio
  training:
    learning_rate: 2.0e-5
    batch_size: 8
    epochs: 10
    warmup_steps: 0
  output_dir: models/audio/checkpoints
  ```

  `configs/text.yaml`:
  ```yaml
  # Text model training configuration
  modality: text
  data:
    processed: data/processed/text/goemotions
  training:
    learning_rate: 2.0e-5
    per_device_train_batch: 8
    per_device_eval_batch: 8
    gradient_accumulation_steps: 2
    epochs: 3
    warmup_steps: 0
    weight_decay: 0.01
    max_length: 128
    early_stopping_patience: 2
  output_dir: models/text/checkpoints
  ```

  `configs/video.yaml`:
  ```yaml
  # Video model training configuration
  modality: video
  data:
    processed: data/processed/video/video_fer
  training:
    learning_rate: 1.0e-4
    batch_size: 32
    epochs: 10
  output_dir: models/video/checkpoints
  ```

  Files created: `configs/audio.yaml`, `configs/text.yaml`, `configs/video.yaml`

  Verify: All three files exist and are valid YAML: `python -c "import yaml; [yaml.safe_load(open(f)) for f in ['configs/audio.yaml','configs/text.yaml','configs/video.yaml']]; print('ok')"`.

---

- [ ] 14. **Create inference stubs for audio, text, video**

  Create a minimal inference.py stub in each modality directory. Each stub follows the same pattern: imports the checkpoint dir constant, defines a placeholder `predict()` function with the correct signature, and raises `NotImplementedError`.

  `models/audio/inference.py`:
  ```python
  """Audio emotion inference stub.
  TODO: implement using saved checkpoints from models/audio/checkpoints/.
  """
  from __future__ import annotations
  import pathlib

  CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent / "checkpoints"


  def predict(audio_path: str | pathlib.Path, model: str = "emotion2vec") -> dict:
      """Return emotion probabilities for a single audio file.

      Args:
          audio_path: Path to .wav/.flac audio file.
          model: One of 'wav2vec2', 'wavlm', 'emotion2vec'.

      Returns:
          Dict mapping emotion label → probability.
      """
      raise NotImplementedError("Audio inference not yet implemented.")
  ```

  `models/text/inference.py`:
  ```python
  """Text emotion inference stub.
  TODO: implement using saved checkpoints from models/text/checkpoints/.
  """
  from __future__ import annotations
  import pathlib

  CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent / "checkpoints"


  def predict(text: str, model: str = "deberta-v3-base") -> dict:
      """Return emotion probabilities for a text utterance.

      Args:
          text: Raw utterance string.
          model: One of 'bert-base-uncased', 'roberta-base', 'deberta-v3-base'.

      Returns:
          Dict mapping emotion label → probability.
      """
      raise NotImplementedError("Text inference not yet implemented.")
  ```

  `models/video/inference.py`:
  ```python
  """Video FER inference stub.
  TODO: implement using saved checkpoints from models/video/checkpoints/.
  """
  from __future__ import annotations
  import pathlib

  CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent / "checkpoints"


  def predict(video_path: str | pathlib.Path, model: str = "mae-dfer") -> dict:
      """Return emotion probabilities for a video clip.

      Args:
          video_path: Path to video file.
          model: One of 'mae-dfer', 'est', 'hybrid-cnn-vit'.

      Returns:
          Dict mapping emotion label → probability.
      """
      raise NotImplementedError("Video inference not yet implemented.")
  ```

  Files created: `models/audio/inference.py`, `models/text/inference.py`, `models/video/inference.py`

  Verify: `python -c "import models.audio.inference; import models.text.inference; import models.video.inference; print('ok')"` from project root — no ImportError (will raise NotImplementedError only if `predict()` is called).

---

- [ ] 15. **Create src/ module stubs**

  Create all src/ Python source files as minimal stubs. Each file gets a module docstring describing its role and placeholder function/class definitions (raising `NotImplementedError`). This establishes the public API surface without implementing logic.

  Files to create:

  **`src/__init__.py`** — empty package marker.

  **`src/pipeline/__init__.py`** — empty package marker.

  **`src/pipeline/text_pipeline.py`**:
  ```python
  """Text emotion recognition pipeline."""
  from __future__ import annotations


  def run(utterance: str, model: str = "deberta-v3-base") -> dict:
      """Process a text utterance and return emotion scores."""
      raise NotImplementedError
  ```

  **`src/pipeline/audio_pipeline.py`**:
  ```python
  """Audio emotion recognition pipeline."""
  from __future__ import annotations
  import pathlib


  def run(audio_path: str | pathlib.Path, model: str = "emotion2vec") -> dict:
      """Process an audio file and return emotion scores."""
      raise NotImplementedError
  ```

  **`src/pipeline/video_pipeline.py`**:
  ```python
  """Video facial emotion recognition pipeline."""
  from __future__ import annotations
  import pathlib


  def run(video_path: str | pathlib.Path, model: str = "mae-dfer") -> dict:
      """Process a video clip and return emotion scores."""
      raise NotImplementedError
  ```

  **`src/conversation/__init__.py`** — empty package marker.

  **`src/conversation/agent.py`**:
  ```python
  """Conversational agent for behaviour-change support."""
  from __future__ import annotations


  class BehaviourChangeAgent:
      """LLM-based conversational agent that fuses multimodal emotion cues."""

      def __init__(self, config: dict | None = None) -> None:
          self.config = config or {}

      def respond(self, user_input: str, emotion_context: dict | None = None) -> str:
          raise NotImplementedError
  ```

  **`src/conversation/prompts.py`**:
  ```python
  """Prompt templates for the behaviour-change conversational agent."""
  from __future__ import annotations

  SYSTEM_PROMPT = (
      "You are a supportive behaviour-change coach. "
      "Use the detected emotion context to adapt your response."
  )

  def build_prompt(user_input: str, emotion_context: dict | None = None) -> str:
      raise NotImplementedError
  ```

  **`src/conversation/memory.py`**:
  ```python
  """Session memory for the conversational agent."""
  from __future__ import annotations
  from typing import List


  class ConversationMemory:
      def __init__(self) -> None:
          self.history: List[dict] = []

      def add_turn(self, role: str, content: str) -> None:
          self.history.append({"role": role, "content": content})

      def get_history(self) -> List[dict]:
          return list(self.history)

      def clear(self) -> None:
          self.history.clear()
  ```

  **`src/conversation/response_generator.py`**:
  ```python
  """LLM response generation for the conversational agent."""
  from __future__ import annotations


  def generate(prompt: str, model: str = "gpt-4o-mini", **kwargs) -> str:
      raise NotImplementedError
  ```

  **`src/fusion/__init__.py`** — empty package marker.

  **`src/fusion/feature_fusion.py`**:
  ```python
  """Feature-level fusion of multimodal emotion embeddings."""
  from __future__ import annotations
  import torch


  def fuse(text_emb: torch.Tensor, audio_emb: torch.Tensor, video_emb: torch.Tensor) -> torch.Tensor:
      """Concatenate and project multimodal embeddings."""
      raise NotImplementedError
  ```

  **`src/fusion/emotion_fusion.py`**:
  ```python
  """Decision-level fusion: combine per-modality emotion probability dicts."""
  from __future__ import annotations


  def fuse(text_probs: dict, audio_probs: dict, video_probs: dict,
           weights: tuple[float, float, float] = (0.33, 0.33, 0.34)) -> dict:
      """Weighted average of per-modality emotion probabilities."""
      raise NotImplementedError
  ```

  **`src/utils/__init__.py`** — empty package marker.

  Files created: 14 files across `src/`.

  Verify: `python -c "from src.pipeline import text_pipeline, audio_pipeline, video_pipeline; from src.conversation import agent, prompts, memory; from src.fusion import feature_fusion; print('ok')"`.

---

- [ ] 16. **Create evaluation/ stubs**

  Create `evaluation/comparison.py` and `evaluation/metrics.py` as minimal stubs.

  **`evaluation/__init__.py`** — empty package marker.

  **`evaluation/comparison.py`**:
  ```python
  """Cross-modality and cross-model comparison utilities."""
  from __future__ import annotations


  def compare_models(results: dict) -> dict:
      """Aggregate evaluation results across models and modalities."""
      raise NotImplementedError
  ```

  **`evaluation/metrics.py`**:
  ```python
  """Shared evaluation metrics for all modalities."""
  from __future__ import annotations
  from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


  def compute_metrics(y_true, y_pred, average: str = "weighted") -> dict:
      return {
          "accuracy": accuracy_score(y_true, y_pred),
          "f1": f1_score(y_true, y_pred, average=average),
      }
  ```

  Files created: `evaluation/__init__.py`, `evaluation/comparison.py`, `evaluation/metrics.py`

  Verify: `python -c "from evaluation import comparison, metrics; print('ok')"`.

---

- [ ] 17. **Create test stubs**

  Create four test files under `tests/`. Each imports the corresponding module and asserts the stub raises `NotImplementedError` (proving the module is importable and the API exists).

  **`tests/__init__.py`** — empty package marker.

  **`tests/test_audio.py`**:
  ```python
  import pytest
  from models.audio.inference import predict


  def test_predict_raises():
      with pytest.raises(NotImplementedError):
          predict("dummy.wav")
  ```

  **`tests/test_text.py`**:
  ```python
  import pytest
  from models.text.inference import predict


  def test_predict_raises():
      with pytest.raises(NotImplementedError):
          predict("I feel stressed today.")
  ```

  **`tests/test_video.py`**:
  ```python
  import pytest
  from models.video.inference import predict


  def test_predict_raises():
      with pytest.raises(NotImplementedError):
          predict("dummy.mp4")
  ```

  **`tests/test_pipeline.py`**:
  ```python
  import pytest
  from src.pipeline import text_pipeline, audio_pipeline, video_pipeline


  def test_text_pipeline_raises():
      with pytest.raises(NotImplementedError):
          text_pipeline.run("hello")


  def test_audio_pipeline_raises():
      with pytest.raises(NotImplementedError):
          audio_pipeline.run("dummy.wav")


  def test_video_pipeline_raises():
      with pytest.raises(NotImplementedError):
          video_pipeline.run("dummy.mp4")
  ```

  Files created: `tests/__init__.py`, `tests/test_audio.py`, `tests/test_text.py`, `tests/test_video.py`, `tests/test_pipeline.py`

  Verify: `python -m pytest tests/ -v` — all 6 tests pass (each correctly catches `NotImplementedError`).

---

- [ ] 18. **Create docs/ stubs**

  Create placeholder Markdown files for the documentation structure. `docs/architecture.md` was already populated by step 2 (from `models/embedding_shapes.md`). `docs/datasets.md` was already populated by step 11 (from `scripts/dataset_inspection_report.md`). Create the remaining stubs as minimal headers.

  **`docs/methodology.md`**:
  ```markdown
  # Methodology

  > TODO: describe the multimodal emotion recognition approach,
  > model selection rationale, and fusion strategy.
  ```

  **`docs/training.md`**:
  ```markdown
  # Training Guide

  > TODO: document how to run models/audio/train.py, models/text/train.py,
  > and models/video/train.py, including required configs and data paths.
  ```

  **`docs/experiments.md`**:
  ```markdown
  # Experiments

  > TODO: log experimental runs, hyperparameter sweeps, and ablations.
  ```

  **`docs/research_gap.md`**:
  ```markdown
  # Research Gap

  > TODO: summarise the gap this project addresses relative to
  > prior multimodal emotion recognition work.
  ```

  Files created: `docs/methodology.md`, `docs/training.md`, `docs/experiments.md`, `docs/research_gap.md`

  Verify: All four files exist and are non-empty: `Get-ChildItem docs/ -Filter "*.md" | Measure-Object`.

---

- [ ] 19. **Create requirements.txt and .env.example**

  There is no `requirements.txt` at the project root (confirmed by audit). Create minimal versions capturing the core dependencies visible in the train scripts.

  **`requirements.txt`**:
  ```
  # Core ML
  torch>=2.0.0
  transformers>=4.40.0
  datasets>=2.14.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  matplotlib>=3.7.0

  # Audio
  soundfile>=0.12.0
  librosa>=0.10.0
  funasr>=0.8.0

  # Data
  pandas>=2.0.0
  pyarrow>=12.0.0
  pyyaml>=6.0

  # Dev
  pytest>=7.4.0
  python-dotenv>=1.0.0
  ```

  **`.env.example`**:
  ```
  # Copy this file to .env and fill in your values.
  # Never commit .env to version control.

  # HuggingFace token (required for gated model downloads)
  HF_TOKEN=your_huggingface_token_here

  # OpenAI key (required for src/conversation/response_generator.py)
  OPENAI_API_KEY=your_openai_key_here

  # Paths (override defaults if your data lives elsewhere)
  # DATA_ROOT=/path/to/data
  ```

  Files created: `requirements.txt`, `.env.example`

  Verify: `python -c "import yaml; print('yaml ok')"` and `cat requirements.txt` show content; confirm `.env.example` is listed in `.gitignore` would be wrong — `.env.example` should NOT be in `.gitignore` (it is the template, not the secret file). Verify `.env` (without `.example`) is already in `.gitignore`:
  ```powershell
  Select-String "\.env" .gitignore
  ```

---

- [ ] 20. **Fix broken imports in all moved Python files**

  After all moves, these files have import paths that no longer resolve. Edit each one to use the new `models.common.*` paths.

  ### `models/audio/train.py` (was `models/train_all_audio_models.py`)

  The `PROJECT_ROOT` calculation uses `.parent.parent` (i.e., two levels up from the old `models/` root location). After moving to `models/audio/train.py`, the file is now one level deeper, so `.parent.parent` needs to become `.parent.parent.parent`.

  Find:
  ```python
  PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
  ```
  Replace with:
  ```python
  PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
  ```

  This file has no direct `from models.compare` import (it uses data path constants), so no further import changes are needed in `train.py` for audio.

  ### `models/text/train.py` (was `models/text/train_all_text_models.py`)

  `PROJECT_ROOT` already uses `.parent.parent` and the file was already in `models/text/`, so after being renamed to `train.py` in the same dir, the path calculation is unchanged.

  The import:
  ```python
  from models.compare import MODEL_REGISTRY
  ```
  Must become:
  ```python
  from models.common.compare import MODEL_REGISTRY
  ```

  ### `models/video/train.py` (was `models/video/train_all_video_models.py`)

  `PROJECT_ROOT` uses `.parents[2]` (two levels above `models/video/`), which correctly resolves to the project root. After moving to `models/video/train.py` in the same directory, `.parents[2]` still resolves to the project root — **no change needed**.

  The fallback import block:
  ```python
  try:
      from compare import MODEL_REGISTRY
  except ImportError:
      from models.compare import MODEL_REGISTRY
  ```
  Replace with:
  ```python
  from models.common.compare import MODEL_REGISTRY
  ```
  (Remove the try/except; the new path is always correct from project root.)

  Also update:
  ```python
  VIDEO_OUT_DIR = PROJECT_ROOT / "models" / "video"
  ```
  remains correct. But:
  ```python
  PROCESSED_VIDEO_DIR = PROJECT_ROOT / "data" / "processed" / "video_fer"
  ```
  Must become:
  ```python
  PROCESSED_VIDEO_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "video_fer"
  ```
  (Reflects the step 12 move of `data/processed/video_fer/` to `data/processed/video/video_fer/`.)

  ### `models/common/training_plan.py` (was `models/training_plan.py`)

  ```python
  from models.compare import MODEL_REGISTRY, MODALITY_ORDER
  ```
  Replace with:
  ```python
  from models.common.compare import MODEL_REGISTRY, MODALITY_ORDER
  ```

  ### `src/pipeline/multimodal_pipeline.py` (was `models/run_pipeline.py`)

  ```python
  from models.compare import MODEL_REGISTRY, MODALITY_ORDER
  ```
  Replace with:
  ```python
  from models.common.compare import MODEL_REGISTRY, MODALITY_ORDER
  ```

  ### `models/text/train.py` — data path update

  After step 12, the GoEmotions data moved. Update:
  ```python
  DATA_DIR = PROJECT_ROOT / "data" / "processed" / "goemotions"
  ```
  to:
  ```python
  DATA_DIR = PROJECT_ROOT / "data" / "processed" / "text" / "goemotions"
  ```

  ### `models/audio/train.py` — data path update

  Check whether the audio train script references `data/processed/ravdess`. If it does, update to `data/processed/audio/ravdess`.

  Files modified:
  - `models/audio/train.py` — `PROJECT_ROOT` fix + data path update
  - `models/text/train.py` — import fix + data path update
  - `models/video/train.py` — import fix + data path update
  - `models/common/training_plan.py` — import fix
  - `src/pipeline/multimodal_pipeline.py` — import fix

  Verify: From the project root, run:
  ```powershell
  python -c "from models.common.compare import MODEL_REGISTRY; print('compare ok')"
  python -c "from models.common.training_plan import modality_stream; print('training_plan ok')"
  python -c "from src.pipeline.multimodal_pipeline import embedding_shapes; print('pipeline ok')"
  ```
  All three should print their ok messages without ImportError.

---

- [ ] 21. **Verify project root still importable as a package**

  The old `models/__init__.py` moved to `models/common/__init__.py`. Create a new minimal `models/__init__.py` at the models root so `import models` still works (other code or IDE tooling may rely on it).

  `models/__init__.py` (new, at root of models/):
  ```python
  # models/__init__.py
  # Package marker. Core utilities live in models.common.
  ```

  Similarly, `models/text/__init__.py` already exists and was not moved — no action needed.

  Verify: `python -c "import models; import models.common; import models.audio; import models.text; import models.video; print('all packages ok')"`.

---

- [ ] 22. **Stage and commit — 6 commit groups**

  All changes should be committed in the following ordered groups. Each commit leaves the repo in a working state (imports correct, no broken references at that point).

  ---

  **Commit 1 — `chore: scaffold new directory structure`**
  Stage: all new `.gitkeep` files, new empty `__init__.py` stubs, `configs/`, `docs/` stubs, `evaluation/` stubs, `src/` stubs (pipeline, conversation, fusion, utils), `tests/` stubs, `requirements.txt`, `.env.example`, `models/label_mapping.py` redirect stub, `models/__init__.py` new root marker.
  ```bash
  git add configs/ src/ evaluation/ experiments/ docs/methodology.md docs/training.md \
          docs/experiments.md docs/research_gap.md docs/paper_notes/.gitkeep \
          data/metadata/.gitkeep reports/multimodal/.gitkeep \
          archive/old_runs/.gitkeep models/audio/inference.py \
          models/text/inference.py models/video/inference.py \
          models/label_mapping.py models/__init__.py \
          requirements.txt .env.example tests/
  git commit -m "chore: scaffold new directory structure and stubs"
  ```

  ---

  **Commit 2 — `refactor: move models/common utilities`**
  Stage: the 7 `git mv` renames from step 2 (compare, label_mapping, training_plan, _compat, __init__, embedding_shapes→docs/architecture, run_pipeline→src/pipeline).
  ```bash
  git add -u
  git commit -m "refactor: move models root utilities to models/common and src/pipeline"
  ```

  ---

  **Commit 3 — `refactor: rename train scripts and move checkpoints`**
  Stage: the `git mv` renames from steps 3, 4, 5, and 6 (train.py renames; audio/text/video checkpoint moves).
  ```bash
  git add -u
  git commit -m "refactor: rename train scripts and move checkpoint directories"
  ```

  ---

  **Commit 4 — `chore: move reports and logs to reports/ and archive/`**
  Stage: all `git mv` renames from steps 7, 8, 9, and 10 (audio/text/video reports, archive logs).
  ```bash
  git add -u
  git commit -m "chore: move training reports and logs to reports/ and archive/"
  ```

  ---

  **Commit 5 — `chore: archive deprecated scripts and move docs`**
  Stage: all `git mv` renames from step 11 (scripts → archive/deprecated_code, dataset_inspection_report → docs/datasets.md).
  ```bash
  git add -u
  git commit -m "chore: archive deprecated scripts and move dataset doc to docs/"
  ```

  ---

  **Commit 6 — `fix: update import paths and data paths after reorganisation`**
  Stage: all edited files from step 20 and step 21 (import and PROJECT_ROOT fixes).
  ```bash
  git add models/audio/train.py models/text/train.py models/video/train.py \
          models/common/training_plan.py src/pipeline/multimodal_pipeline.py
  git commit -m "fix: update import paths and data paths after reorganisation"
  ```

  Verify after all 6 commits:
  ```powershell
  git log --oneline -8
  python -c "from models.common.compare import MODEL_REGISTRY; print(list(MODEL_REGISTRY.keys()))"
  python -m pytest tests/ -v
  ```
  Expected: 6 clean commits visible, MODEL_REGISTRY loads correctly, all 6 stub tests pass.

---

## Unresolved items and assumptions

1. **`data/processed/ravdess/` git tracking**: The git audit shows no tracked files directly in `data/processed/ravdess/` (they appear to be .parquet/.csv files excluded by `.gitignore`). If `git ls-files` in step 12 reveals any tracked files there, `git mv` them individually before running `Move-Item`. Assumption: they are untracked; `Move-Item` is sufficient.

2. **`models/audio/test_meld/`**: Contains one tracked WAV file (`Episode3_12/dial3_3_utt1.wav`). It is test/fixture data for the audio model. Leave it in place under `models/audio/test_meld/` — no target specifies where it should go, and moving it could break any path reference in `train.py`.

3. **`models/video/MAE-DFER`**: Recorded as a git-tracked path but appears to be a submodule reference (`.gitmodules` exists). Do not move it; it stays at `models/video/MAE-DFER`.

4. **`data/processed/audio/` naming collision**: The `data/processed/` directory already has a subfolder called `audio/` (listed in the `data/processed` inventory). Check its contents before running step 12 — it may already contain some files. If so, move its contents into the new `data/processed/audio/` rather than overwriting.

5. **`models/video/results/`**: Contains tracked result files for `cmu_mosei_baseline`. The target layout does not specify a destination for `models/video/results/`. Leave it in place; no move planned. Note it in commit 5's message if desired.

6. **`models/preprocessing/`**: Stays fully in place per the target spec. No moves needed.
