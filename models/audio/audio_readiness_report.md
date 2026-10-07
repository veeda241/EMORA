# MELD Audio Training Readiness

## Status

The Wav2Vec2 audio trainer is GPU-ready but cannot start until the aligned MELD
audio files are supplied.

## Verified environment

- PyTorch: CUDA-enabled `2.11.0+cu128`
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- CUDA available: yes
- Audio dependencies: `librosa` and `matplotlib` installed
- Trainer: `models/train_all_audio_models.py` compiles successfully

## Dataset

- Train: 9,989 utterances
- Validation/dev: 1,109 utterances
- Test: 2,610 utterances
- Total: 13,708 utterances
- Classes: 7
- Label mapping: `models/audio/label_mapping.json`

Classes and IDs:

| ID | Class |
|---:|---|
| 0 | anger |
| 1 | disgust |
| 2 | fear |
| 3 | joy |
| 4 | neutral |
| 5 | sadness |
| 6 | surprise |

## Current blocker

No audio files are present under `data/raw/audio`. Add the aligned MELD audio
directory using the expected layout:

```text
data/raw/audio/
  Episode{season}_{episode}/
    dial{season}_{dialogue}_utt{utterance}.wav
```

The trainer resolves each waveform from the CSV fields
`Season`, `Episode`, `Dialogue_ID`, and `Utterance_ID`.

## Data-quality warning

The current raw CSVs contain utterance-ID overlap across splits:

- Train/dev: 675
- Train/test: 1,740
- Dev/test: 670

Do not treat final test metrics as leakage-free until the split construction is
reviewed. The existing trainer preserves the supplied splits and does not
silently remove or rewrite rows.

## Launch command after audio is supplied

```powershell
python models\train_all_audio_models.py --audio-root data\raw\audio
```

For a short smoke run:

```powershell
python models\train_all_audio_models.py --audio-root data\raw\audio --max-train-samples 8 --epochs 1
```
