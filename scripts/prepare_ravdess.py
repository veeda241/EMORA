"""Normalize the extracted RAVDESS speech annotations for audio training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "ravdess"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ravdess"
REQUIRED_COLUMNS = {
    "file_path",
    "actor_id",
    "emotion",
    "emotion_id",
    "statement",
    "intensity",
    "repetition",
    "split",
}


def normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def prepare(args: argparse.Namespace) -> None:
    metadata_path = args.input_dir / "ravdess_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing RAVDESS metadata: {metadata_path}")

    frame = pd.read_csv(metadata_path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{metadata_path} is missing required columns: {sorted(missing)}"
        )

    frame["file_path"] = frame["file_path"].map(normalize_path)
    frame["emotion"] = frame["emotion"].str.strip().str.lower()
    frame["split"] = frame["split"].str.strip().str.lower()
    for column in ("actor_id", "emotion_id", "statement", "intensity", "repetition"):
        frame[column] = frame[column].astype(int)

    if frame.empty:
        raise ValueError("RAVDESS metadata contains no rows.")
    if frame["file_path"].duplicated().any():
        raise ValueError("RAVDESS metadata contains duplicate file paths.")

    missing_files = [
        path
        for path in frame["file_path"]
        if not (args.input_dir / Path(path)).is_file()
    ]
    if missing_files:
        if not args.skip_audio_check:
            sample = "\n".join(missing_files[:10])
            raise FileNotFoundError(
                f"{len(missing_files)} metadata files are missing under {args.input_dir}.\n"
                f"First missing paths:\n{sample}\n\n"
                f"Run with --skip-audio-check to generate processed outputs "
                f"without validating audio files on disk."
            )
        print(
            f"[warn] {len(missing_files)} audio files not found under {args.input_dir} "
            f"(--skip-audio-check is set; continuing without file validation)."
        )

    allowed_splits = {"train", "validation", "test"}
    unexpected_splits = sorted(set(frame["split"]) - allowed_splits)
    if unexpected_splits:
        raise ValueError(f"Unexpected RAVDESS split names: {unexpected_splits}")

    labels = sorted(frame["emotion"].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    metadata_ids = frame.groupby("emotion")["emotion_id"].nunique()
    if (metadata_ids > 1).any():
        raise ValueError("An emotion maps to multiple emotion_id values.")
    frame["label_id"] = frame["emotion"].map(label2id).astype(int)
    frame["actor_id"] = frame["actor_id"].astype(int)
    frame.insert(
        0,
        "sample_id",
        frame["file_path"].map(lambda path: Path(path).stem),
    )
    frame["absolute_path"] = frame["file_path"].map(
        lambda path: str((args.input_dir / Path(path)).resolve())
    )

    columns = [
        "sample_id",
        "split",
        "file_path",
        "absolute_path",
        "actor_id",
        "emotion",
        "emotion_id",
        "label_id",
        "statement",
        "intensity",
        "repetition",
    ]
    frame = frame[columns].sort_values(["split", "actor_id", "file_path"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "ravdess_combined.csv", index=False)
    frame.to_parquet(args.output_dir / "ravdess_combined.parquet", index=False)
    for split in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        split_frame.to_csv(args.output_dir / f"ravdess_{split}.csv", index=False)
        split_frame.to_parquet(
            args.output_dir / f"ravdess_{split}.parquet", index=False
        )

    summary = {
        "dataset": "RAVDESS",
        "source_metadata": str(metadata_path),
        "audio_root": str(args.input_dir),
        "total_rows": int(len(frame)),
        "validated_audio_files": int(len(frame) - len(missing_files)),
        "missing_audio_files": int(len(missing_files)),
        "labels": labels,
        "label2id": label2id,
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "class_counts": frame["emotion"].value_counts().sort_index().to_dict(),
        "actors_by_split": {
            split: sorted(frame.loc[frame["split"].eq(split), "actor_id"].unique().tolist())
            for split in ("train", "validation", "test")
        },
        "audio_format": "WAV",
        "relative_path_column": "file_path",
        "label_column": "label_id",
        "split_policy": "Actor-disjoint splits supplied by the RAVDESS metadata.",
    }
    for filename in ("labels.json", "preprocessing_summary.json"):
        (args.output_dir / filename).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(f"Wrote {len(frame):,} RAVDESS rows to {args.output_dir}")
    print(frame["split"].value_counts().sort_index().to_string())
    print(f"Labels: {labels}")
    print(f"Validated audio files: {len(frame):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        default=False,
        help="Skip validation that each audio file exists on disk. "
             "Useful when only the metadata CSV is present.",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
