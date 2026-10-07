# Copilot instructions

## Repository shape

This is a research-oriented Python repository for a future multimodal behaviour/stress
change-support system. The project-owned implementation is under `models/` and
`scripts/`; `data/external/CMU-MultimodalSDK-main/` is a vendored CMU SDK. The active dataset
set is GoEmotions (text), MELD (audio), and CMU-MOSEI (video/features). Treat
downloaded datasets, model checkpoints,
logs, and generated reports as artifacts rather than source code.

The current pipeline has three layers:

- `models/comparison_grid.py` is the central registry for text, face, and voice
  encoders and dataset metadata. `models/training_plan.py` and
  `models/run_pipeline.py` generate the intended per-modality and embedding-shape
  plans from that registry; they do not implement the final fusion runtime.
- `models/train_all_text_models.py` is the existing single-label Hugging Face
  Trainer pipeline. The active GoEmotions export is multi-label and lives under
  `data/processed/goemotions/`, so this trainer must be adapted for multi-label
  loss/metrics before it can replace the deleted six-class emotion experiment.
- `models/train_all_audio_models.py` trains the currently implemented Wav2Vec2
  MELD branch. It reads MELD CSVs, resolves aligned audio files, freezes the
  feature encoder, evaluates on dev/test, and writes model, metric, confusion
  matrix, CSV, report, and log artifacts under `models/audio/`.
- `scripts/prepare_meld.py` normalizes the MELD annotation CSVs into
  `data/processed/meld/`, preserving train/validation/test splits, integer label
  IDs, and episode-relative timestamps in milliseconds. The processed annotations
  do not include audio; aligned MELD waveforms are still required for Wav2Vec2.
- `scripts/inspect_meld.py` is the preflight check for MELD split sizes, labels,
  utterance-ID overlap, and audio availability. `models/preprocessing/` contains
  dataset inspection utilities, including CMU-MOSEI fold/modality reports.
- The vendored SDK models data as HDF5 computational sequences (`.csd`) with
  timestamped intervals/features. Its standard-dataset helpers download and align
  modalities; keep SDK changes isolated from project-owned training code.

All project scripts assume the repository root is the current working directory
unless they explicitly derive `PROJECT_ROOT`. In particular, the audio trainer's
`data/raw/meld`, `data/raw/audio`, and `models/audio` paths are relative paths.

## Setup and commands

There is no root `pyproject.toml`, `requirements.txt`, Makefile, test runner, or
configured linter. The vendored SDK has its own dependencies and package metadata:

```powershell
cd data/external/CMU-MultimodalSDK-main
python -m pip install -r requirements.txt
python -m pip install -e .
cd ..
```

The project training scripts additionally import PyTorch, Transformers, Datasets,
PyArrow, scikit-learn, librosa, and Matplotlib; install compatible versions in the
active environment before running them. The checked-in `.venv` is ignored and
should not be treated as a distributable environment.

Useful commands from the repository root:

```powershell
# Inspect MELD before attempting audio training
python scripts/inspect_meld.py
python scripts/prepare_meld.py

# Print the registry-derived model comparison and training plan
python models/comparison_grid.py
python models/training_plan.py
python models/run_pipeline.py

# Fine-tune all registered text encoders (downloads Hugging Face models if needed)
python models/train_all_text_models.py

# Train the Wav2Vec2 audio branch; audio must be externally supplied and aligned
python models/train_all_audio_models.py --audio-root data/raw/audio

# Fast audio smoke run, still requires aligned audio for the selected samples
python models/train_all_audio_models.py --max-train-samples 8 --epochs 1
```

There is no project-wide automated test suite or lint command. The vendored SDK's
standalone LSTHM smoke test is the closest existing single-test check:

```powershell
cd data/external/CMU-MultimodalSDK-main
python mmsdk/mmmodelsdk/modules/LSTHM/test.py
cd ..
```

For a lightweight syntax check of a changed project script, run for example:

```powershell
python -m py_compile models/comparison_grid.py
```

## Data and artifact conventions

- Raw data belongs under `data/raw/` and is intentionally ignored by Git. The
  tracked Parquet/CSV files may be absent in a fresh clone because the repository
  snapshot contains local experiment artifacts; do not hard-code a new data
  location without updating the consuming script and its report.
- Keep the three active datasets separated by modality: GoEmotions source files
  are archived under `data/archive/goemotions_source/` and its active training
  outputs are under `data/processed/goemotions/`, MELD under
  `data/raw/meld/` and `data/processed/meld/`, and CMU-MOSEI feature sequences
  under `data/raw/cmu_mosei/`.
- MELD labels are derived by sorting the union of emotion names and assigning
  integer IDs. Audio filenames are resolved from `(Season, Episode, Dialogue_ID,
  Utterance_ID)` using the `Episode{season}_{episode}/dial{season}_{dialogue}_utt{utterance}.wav`
  convention. Preserve this alignment contract when changing the audio loader.
- Text training uses the `label` -> `labels` rename expected by Hugging Face
  `Trainer`, derives `num_labels` from Parquet rather than assuming binary
  classification, evaluates per epoch, and selects the lowest evaluation loss.
- Audio training uses deterministic seeds, mono/resampled audio at the processor's
  sample rate, dynamic batch padding, AdamW with gradient clipping, best-dev-loss
  checkpoint selection, and explicit NaN/Inf failures. Preserve these checks when
  modifying the loop.
- Generated files under `models/` are ignored except for the small source modules
  explicitly unignored in `.gitignore`. Do not commit checkpoints, downloaded
  datasets, logs, or generated reports unless the task explicitly requests an
  artifact.
- `MODEL_REGISTRY` is the source of truth for model names and modality order.
  Add or rename a model there before changing registry-driven reports/plans, and
  keep the common projected embedding size (`PROJECT_DIM = 512`) consistent across
  modality plans.
