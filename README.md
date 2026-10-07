# LLM-Based Multimodal Behaviour-Change Support

Research repository for comparing text, audio, and video/face encoders for
emotion and behaviour-change support. The project currently prepares
GoEmotions, MELD, RAVDESS, and CMU-MOSEI data and generates registry-driven
multimodal training plans.

## Current model comparison

The active registry is `models/comparison_grid.py`.

| Modality | Primary | Comparison | Baseline |
|---|---|---|---|
| Text | DeBERTa-v3-Base | RoBERTa-Base | BERT-Base-Uncased |
| Audio | emotion2vec | WavLM-Base | Wav2Vec2-Base |
| Video/face | MAE-DFER | EST | Hybrid CNN + ViT |

The text and audio registries include model metadata for future comparison.
The implemented text trainer supports the three registered text encoders. The
implemented audio trainer currently trains Wav2Vec2; emotion2vec and WavLM
still require model-specific loader/training integration. The three video
models are registered as planned architectures and do not yet have training
implementations or checkpoints in this repository.

All modality streams are planned to project to the shared `PROJECT_DIM = 512`
embedding size before multimodal fusion.

## Repository layout

```text
data/
  raw/                 # Downloaded datasets; ignored by Git
  processed/           # Lightweight CSV/JSON manifests; Parquet is ignored
models/
  comparison_grid.py   # Source of truth for model and dataset metadata
  training_plan.py     # Registry-derived training plan
  run_pipeline.py      # Registry-derived embedding-shape plan
  train_all_text_models.py
  train_all_audio_models.py
scripts/
  prepare_goemotions.py
  prepare_meld.py
  prepare_ravdess.py
  inspect_meld.py
```

Downloaded datasets, archives, checkpoints, logs, generated plots, and other
large artifacts are intentionally excluded from Git. See `.gitignore`.

## Environment setup

Use a Python virtual environment with compatible versions of:

- PyTorch with CUDA support
- Transformers
- Datasets
- PyArrow
- pandas and NumPy
- scikit-learn
- librosa and matplotlib

The current development environment was validated with CUDA-enabled PyTorch
and an NVIDIA RTX 3050. Verify the GPU before training:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

## Dataset preprocessing

### GoEmotions

The active processed dataset is multi-label and is expected under
`data\processed\goemotions`. Prepare it with:

```powershell
python scripts\prepare_goemotions.py
```

### MELD

MELD annotation CSVs should be placed under `data\raw\meld`:

```text
data\raw\meld\
  train_sent_emo.csv
  dev_sent_emo.csv
  test_sent_emo.csv
```

Normalize the annotations with:

```powershell
python scripts\prepare_meld.py
python scripts\inspect_meld.py
```

The MELD audio trainer expects separately supplied aligned WAV files using:

```text
data\raw\audio\
  Episode{season}_{episode}\
    dial{season}_{dialogue}_utt{utterance}.wav
```

### RAVDESS

RAVDESS speech data is extracted under `data\raw\ravdess` and normalized by:

```powershell
python scripts\prepare_ravdess.py
```

This creates manifests under `data\processed\ravdess`:

- 768 train clips
- 192 validation clips
- 192 test clips
- 1,152 validated WAV files
- Six labels: angry, disgust, fearful, happy, sad, surprised
- Actor-disjoint split: actors 1–16 train, 17–20 validation, 21–24 test

The current `train_all_audio_models.py` loader is MELD-specific and does not
consume the RAVDESS manifest directly. A RAVDESS-specific loader is required
before training the selected audio comparison models on RAVDESS.

## Text training

Run the full text comparison from the repository root:

```powershell
python models\train_all_text_models.py
```

The trainer uses multi-label classification for GoEmotions and writes
checkpoints and run artifacts under ignored `models\runs_*` directories.
The comparison report is:

```text
models\text_model_analysis.md
```

The current validated run selected RoBERTa as the best valid result. DeBERTa
requires additional numerical-stability work when retrained.

## Audio training

The current implemented audio branch is Wav2Vec2 on MELD:

```powershell
python models\train_all_audio_models.py `
  --audio-root data\raw\audio `
  --epochs 3 `
  --batch-size 2
```

Run a small smoke test after aligned MELD audio is available:

```powershell
python models\train_all_audio_models.py `
  --audio-root data\raw\audio `
  --max-train-samples 8 `
  --epochs 1
```

Audio outputs are written under ignored `models\audio`. The trainer requires a
CUDA GPU and validates every referenced audio file before loading the model.

## Registry-derived documents

Regenerate the comparison and planning documents after changing the registry:

```powershell
python models\comparison_grid.py | Set-Content models\comparison_table.md
python models\training_plan.py | Set-Content models\training_plan.md
python models\run_pipeline.py | Set-Content models\embedding_shapes.md
```

These documents describe the selected emotion2vec/WavLM/Wav2Vec2 audio
comparison and MAE-DFER/EST/Hybrid CNN + ViT video comparison.

## Git and data policy

Raw audio/video archives and model checkpoints are not suitable for ordinary
GitHub pushes. GitHub rejects individual files larger than 100 MB, and the
complete local dataset is many gigabytes. The repository therefore tracks
source code and lightweight metadata/manifests while ignoring raw datasets,
Parquet files, checkpoints, ZIP archives, logs, and generated media.

Store large datasets in an appropriate artifact store or dataset host and keep
the repository scripts/configuration reproducible.
