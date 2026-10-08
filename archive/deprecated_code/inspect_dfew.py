"""
inspect_dfew.py
===============
Sanity-check the DFEW dataset before training:
  - File counts per split and fold
  - Class distribution
  - Frame count per clip
  - Corrupt / empty clip detection
  - Label range validation

Usage
-----
    python scripts/video/inspect_dfew.py --dfew_root data/raw/video_fer/dfew --fold 1
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DFEW_PROCESSED = PROJECT_ROOT / "data" / "processed" / "video_fer" / "dfew"

LABEL_NAMES = {0: "angry", 1: "disgust", 2: "fearful",
               3: "happy",  4: "sad",     5: "surprised"}


def inspect_raw(dfew_root: pathlib.Path) -> None:
    video_dir = dfew_root / "Clip" / "jpg_256"
    if not video_dir.exists():
        print(f"[WARN] {video_dir} not found — raw clip inspection skipped")
        return

    clips = sorted(video_dir.iterdir())
    print(f"\n=== Raw clips under {video_dir} ===")
    print(f"Total clip folders: {len(clips)}")

    frame_counts = []
    empty = []
    for clip in clips:
        frames = list(clip.glob("*.jpg")) + list(clip.glob("*.png"))
        frame_counts.append(len(frames))
        if len(frames) == 0:
            empty.append(clip.name)

    if frame_counts:
        import statistics
        print(f"Frames per clip — min:{min(frame_counts)}  max:{max(frame_counts)}"
              f"  mean:{statistics.mean(frame_counts):.1f}  median:{statistics.median(frame_counts):.1f}")
    if empty:
        print(f"[WARN] {len(empty)} empty clip folders: {empty[:5]}")
    else:
        print("No empty clip folders found. ✓")


def inspect_annotations(fold: int) -> None:
    fold_dir = DFEW_PROCESSED / f"fold{fold}"
    for split in ("train", "val", "test"):
        csv = fold_dir / f"{split}.csv"
        if not csv.exists():
            print(f"[WARN] {csv} missing — run prepare_dfew.py first")
            continue

        df = pd.read_csv(csv, header=None, sep=" ", names=["path", "label"])
        print(f"\n── {split} split (fold {fold}) ──────────────")
        print(f"  Total samples: {len(df)}")

        # label range check
        bad = df[~df["label"].isin(LABEL_NAMES.keys())]
        if len(bad):
            print(f"  [ERROR] {len(bad)} samples with out-of-range labels!")
        else:
            print("  Label range [0-5]: OK ✓")

        # class distribution
        dist = df["label"].value_counts().sort_index()
        print("  Class distribution:")
        for lbl, cnt in dist.items():
            pct = 100 * cnt / len(df)
            bar = "█" * int(pct / 2)
            print(f"    {LABEL_NAMES[lbl]:10s} ({lbl}): {cnt:4d}  {pct:5.1f}%  {bar}")

        # check frame folders exist (sample 10)
        missing = 0
        for path in df["path"].sample(min(10, len(df))):
            if not pathlib.Path(path).exists():
                missing += 1
        if missing:
            print(f"  [WARN] {missing}/10 sampled frame paths do not exist."
                  f" Check --dfew_root path.")
        else:
            print("  Sampled path check: OK ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DFEW dataset")
    parser.add_argument("--dfew_root", default="data/raw/video_fer/dfew")
    parser.add_argument("--fold", default="1")
    args = parser.parse_args()

    dfew_root = pathlib.Path(args.dfew_root)
    inspect_raw(dfew_root)

    folds = list(range(1, 6)) if args.fold == "all" else [int(args.fold)]
    for fold in folds:
        inspect_annotations(fold)


if __name__ == "__main__":
    main()
