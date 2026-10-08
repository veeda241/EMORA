"""Download RAVDESS speech audio (~209 MB) from Zenodo and extract it.

All 24 actors are packaged in a single ZIP (Audio_Speech_Actors_01-24.zip)
on Zenodo record 1188976.  This script downloads and extracts it into:

    data/raw/ravdess/
        Actor_01/  ...  Actor_24/

After extraction, re-run the preprocessing step to regenerate the processed
outputs with validated audio paths:

    python scripts/prepare_ravdess.py

Then start training:

    python models/train_all_audio_models.py --dataset ravdess

Citation (required by the CC BY-NC-SA 4.0 license):
    Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of
    Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
    https://doi.org/10.1371/journal.pone.0196391
    Zenodo: https://zenodo.org/record/1188976
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import sys
import urllib.request
import zipfile

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "ravdess"

SPEECH_ZIP_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
)
SPEECH_ZIP_MD5 = "bc696df654c87fed845eb13823edef8a"


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
        sys.stdout.flush()


def download_and_extract(output_dir: pathlib.Path, force: bool) -> None:
    # Quick check: if Actor_01..Actor_24 already exist and have WAVs, skip.
    existing_actors = sorted(output_dir.glob("Actor_*"))
    existing_wavs = sum(len(list(a.glob("*.wav"))) for a in existing_actors)
    if not force and existing_wavs >= 1440:
        print(
            f"RAVDESS speech audio already present: {existing_wavs} WAV files "
            f"across {len(existing_actors)} actor folders.\n"
            "Pass --force to re-download."
        )
        return

    print(f"Downloading {SPEECH_ZIP_URL}")
    print("(~209 MB — this may take a minute on a slow connection)\n")

    tmp_path = output_dir / "_ravdess_speech_tmp.zip"
    output_dir.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(SPEECH_ZIP_URL, tmp_path, reporthook=_progress_hook)
    print()  # newline after progress bar

    print("Verifying MD5 ...", end=" ", flush=True)
    data = tmp_path.read_bytes()
    digest = _md5(data)
    if digest != SPEECH_ZIP_MD5:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"MD5 mismatch: expected {SPEECH_ZIP_MD5}, got {digest}.\n"
            "The downloaded file may be corrupt. Try again."
        )
    print("OK")

    print("Extracting ...", end=" ", flush=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        wav_entries = [n for n in zf.namelist() if n.lower().endswith(".wav")]
        for name in wav_entries:
            # Preserve Actor_NN/filename.wav structure
            parts = pathlib.PurePosixPath(name).parts
            # Drop any leading directory component that is the zip root
            actor_parts = [p for p in parts if p.startswith("Actor_")]
            if not actor_parts:
                continue
            actor_idx = parts.index(actor_parts[0])
            rel = pathlib.Path(*parts[actor_idx:])
            target = output_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())

    tmp_path.unlink(missing_ok=True)
    total_extracted = sum(
        len(list((output_dir / f"Actor_{i:02d}").glob("*.wav")))
        for i in range(1, 25)
        if (output_dir / f"Actor_{i:02d}").exists()
    )
    print(f"done. {total_extracted} WAV files extracted to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Directory to extract actor folders into (default: data/raw/ravdess).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if audio files are already present.",
    )
    args = parser.parse_args()

    download_and_extract(args.output_dir, args.force)

    print("\nNext steps:")
    print("  1. Re-run preprocessing to validate audio files:")
    print("       python scripts/prepare_ravdess.py")
    print("  2. Start training:")
    print("       python models/train_all_audio_models.py --dataset ravdess")


if __name__ == "__main__":
    main()
