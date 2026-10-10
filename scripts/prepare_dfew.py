"""Prepare manifests for the official DFEW single-label frame release.

The DFEW repository contains code, not the video frames. Download the
single-label release from https://dfew-dataset.github.io/ and arrange it as:

    data/raw/dfew/data/<five-digit-video-id>/*.jpg
    data/raw/dfew/label/single_trainset_<fold>.csv
    data/raw/dfew/label/single_testset_<fold>.csv

This script validates the layout and writes CSV manifests containing a fixed
number of sampled frame paths per clip. It does not claim that the existing
CMU-MOSEI/FACET checkpoint was trained on DFEW; a DFEW-compatible RGB model
must be trained from these manifests separately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABEL_NAMES = {
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "angry",
    5: "surprise",
    6: "disgust",
    7: "fear",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

def locate_single_label_root(root: Path) -> Path:
    """Find the extracted DFEW single-label directory under ``root``."""
    candidates = [
        root,
        root / "single_label",
        root / "data_affine" / "single_label",
    ]
    for candidate in candidates:
        if (
            (candidate / "data").is_dir()
            and (candidate / "label").is_dir()
        ):
            return candidate
    return root


def sample_frames(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) <= count:
        return paths + [paths[-1]] * (count - len(paths))
    indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[i] for i in indices]


def read_split(label_path: Path) -> list[tuple[str, int]]:
    with label_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Label file is empty: {label_path}")
    required = {"video_name", "label"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{label_path} is missing columns: {sorted(missing)}")
    result = []
    for row in rows:
        video_id = str(row["video_name"]).strip().zfill(5)
        label = int(row["label"])
        if label not in LABEL_NAMES:
            raise ValueError(f"Unsupported DFEW label {label} in {label_path}")
        result.append((video_id, label))
    return result


def build_manifest(data_root: Path, label_root: Path, split: str, fold: int, frames: int) -> list[dict]:
    label_path = label_root / f"single_{split}set_{fold}.csv"
    entries = []
    for video_id, label in read_split(label_path):
        clip_dir = data_root / video_id
        frame_paths = sorted(
            p for p in clip_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ) if clip_dir.is_dir() else []
        if not frame_paths:
            raise FileNotFoundError(f"No frames found for DFEW clip {video_id}: {clip_dir}")
        selected = sample_frames(frame_paths, frames)
        entries.append({
            "video_id": video_id,
            "label_id": label,
            "label": LABEL_NAMES[label],
            "num_source_frames": len(frame_paths),
            "frame_paths": "|".join(str(p.resolve()) for p in selected),
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/dfew"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dfew"))
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    dataset_root = locate_single_label_root(args.root)
    data_root = dataset_root / "data"
    label_root = dataset_root / "label"
    required_label = label_root / f"single_trainset_{args.fold}.csv"
    if not data_root.is_dir() or not label_root.is_dir() or not required_label.is_file():
        raise FileNotFoundError(
            "DFEW files are not installed at "
            f"{args.root.resolve()}.\n"
            "Download and extract the official single-label DFEW release so it "
            "contains:\n"
            f"  <root>/single_label/data/00001/*.jpg\n"
            f"  <root>/single_label/label/single_trainset_{args.fold}.csv\n"
            f"  <root>/single_label/label/single_testset_{args.fold}.csv\n"
            "The GitHub DFEW repository contains training code only; it does "
            "not contain the dataset videos or annotations."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = build_manifest(data_root, label_root, split, args.fold, args.frames)
        output = args.output / f"{split}_fold{args.fold}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"WROTE {output} ({len(rows)} clips)")

    labels_path = args.output / "labels.json"
    labels_path.write_text(
        '{\n  "labels": ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]\n}\n',
        encoding="utf-8",
    )
    print(f"WROTE {labels_path}")


if __name__ == "__main__":
    main()
