# CREMA-D File Format Diagnostic

## Scope

- Directory inspected: `CREMA-D-master/AudioWAV/`
- Files scanned: 7442
- For each file, the first 256 bytes were read in binary mode.
- Checked: file size, RIFF/WAVE markers, and text-like headers.
- No files were modified, deleted, downloaded, or used for training.

## Category counts

- Total files scanned: 7442
- Category counts sum to: 7442

| Category | Count |
|---|---|
| git_lfs_pointer | 7442 |
| riff_wave_candidate | 0 |
| empty | 0 |
| html_or_text_response | 0 |
| unknown_malformed | 0 |

## Findings

All 7,442 files are **Git LFS pointer files**, not real WAV audio.

Each file is a small text pointer of about 130 bytes, e.g.:

```
version https://git-lfs.github.com/spec/v1
oid sha256:<hash>
size <bytes>
```

This is why every audio backend failed:

- `soundfile`: `LibsndfileError`
- `torchaudio`: `LibsndfileError`
- `ffmpeg`/`ffprobe`: `Invalid data found when processing input`
- Python `wave`: `file does not start with RIFF id`

No file in this directory starts with `RIFF` or contains `WAVE` near the header.
No file starts with HTML or other recognizable non-pointer text either.

## Representative examples

### git_lfs_pointer

- Filename: `1001_DFA_ANG_XX.wav`
- File size: 130 bytes
- First 64 bytes (hex): `76657273696f6e2068747470733a2f2f6769742d6c66732e6769746875622e636f6d2f737065632f76310a6f6964207368613235363a63366639346232353932`
- First 64 bytes (text): `version https://git-lfs.github.com/spec/v1
oid sha256:c6f94b2592`
- Has RIFF header: False
- Has WAVE near header: False
- Starts with Git LFS pointer: False
- Starts with HTML-like text: False

## Why the files cannot be decoded

A `.wav` extension is not proof of a valid WAV file.
In this directory, every `.wav` file is a Git LFS pointer, which is only a reference to the real file stored in Git LFS storage.
Pointer files contain no audio samples and cannot be decoded by audio libraries.

## Recommended fix

1. If this repository was cloned without Git LFS support, the pointer files were checked out instead of the real audio.
2. The correct recovery is to fetch the real files through Git LFS, for example:
   - `git lfs install`
   - `git lfs pull`
   - or re-checkout/re-download the repository with LFS enabled.

If Git LFS is not available or not intended, the CREMA-D audio must be obtained from the legitimate dataset download source and placed in a decodeable WAV format.

Do **not**:

- rename `.wav` files to make them look valid,
- fabricate RIFF/WAVE headers,
- treat pointer files as audio,
- or start Wav2Vec2 training until real audio can be decoded successfully.

## Machine-readable summary

- Total files scanned: 7442
- Git LFS pointer files: 7442
- Valid RIFF/WAVE candidates: 0
- Empty files: 0
- HTML/text responses: 0
- Unknown/malformed files: 0
- Sum of categories: 7442
