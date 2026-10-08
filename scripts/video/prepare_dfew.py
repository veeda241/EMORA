"""
prepare_dfew.py
===============
Builds train/val/test annotation CSV files for DFEW in the format
MAE-DFER expects:

    <path_to_frame_folder>  <label_0_indexed>

6-class mapping (drops neutral from DFEW's 7 classes):
    DFEW original  → our index
    1 happy        → 3
    2 sad          → 4
    3 neutral      → DROPPED
    4 angry        → 0
    5 surprised    → 5
    6 disgust      → 1
    7 fearful      → 2

Usage
-----
    python scripts/video/prepare_dfew.py --dfew_root data/raw/video_fer/dfew --fold 1
    python scripts/video/prepare_dfew.py --dfew_root data/raw/video_fer/dfew --fold all

Expected DFEW folder layout (after download):
    data/raw/video_fer/dfew/
        Clip/
            jpg_256/
                00001/   ← frame folder (001.jpg, 002.jpg …)
                00002/
                ...
        EmoLabel_DataSplit/
            train(single-labeled)/
                set_1.csv
                set_2.csv
                ...
            test(single-labeled)/
                set_1.csv
                ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DFEW_PROCESSED = PROJECT_ROOT / "data" / "processed" / "video_fer" / "dfew"

# ─── label mapping ────────────────────────────────────────────────────────────
# DFEW uses 1-indexed labels internally; we convert to 0-indexed 6-class.
# Neutral (label 3 in DFEW, 0-indexed 2) is excluded.
DFEW_TO_6CLASS: dict[int, int] = {
    1: 3,   # happy     → 3
    2: 4,   # sad       → 4
    # 3 neutral  → dropped
    4: 0,   # angry     → 0
    5: 5,   # surprised → 5
    6: 1,   # disgust   → 1
    7: 2,   # fearful   → 2
}

LABEL_NAMES = {0: "angry", 1: "disgust", 2: "fearful",
               3: "happy",  4: "sad",     5: "surprised"}


def process_fold(fold: int, dfew_root: pathlib.Path) -> None:
    split_dir = dfew_root / "EmoLabel_DataSplit"
    video_dir = dfew_root / "Clip" / "jpg_256"
    save_dir  = DFEW_PROCESSED / f"fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train(single-labeled)" / f"set_{fold}.csv"
    test_csv  = split_dir / "test(single-labeled)"  / f"set_{fold}.csv"

    if not train_csv.exists():
        print(f"[WARN] {train_csv} not found — skipping fold {fold}")
        return
    if not test_csv.exists():
        print(f"[WARN] {test_csv} not found — skipping fold {fold}")
        return

    def build_anno(csv_path: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        rows = []
        dropped = 0
        for _, row in df.iterrows():
            orig_label = int(row["label"])
            if orig_label not in DFEW_TO_6CLASS:
                dropped += 1
                continue
            new_label = DFEW_TO_6CLASS[orig_label]
            frame_dir = video_dir / f"{int(row['video_name']):05d}"
            rows.append((str(frame_dir), new_label))
        print(f"  {csv_path.name}: {len(rows)} kept, {dropped} neutral dropped")
        return pd.DataFrame(rows, columns=["path", "label"])

    train_df = build_anno(train_csv)
    test_df  = build_anno(test_csv)

    # val == test (standard DFEW protocol)
    train_df.to_csv(save_dir / "train.csv", header=False, index=False, sep=" ")
    test_df.to_csv( save_dir / "val.csv",   header=False, index=False, sep=" ")
    test_df.to_csv( save_dir / "test.csv",  header=False, index=False, sep=" ")

    print(f"  Saved to {save_dir}")

    # class distribution
    print(f"  Train distribution:")
    for label_id, count in train_df["label"].value_counts().sort_index().items():
        print(f"    {LABEL_NAMES[label_id]:10s} ({label_id}): {count}")


def save_label_mapping() -> None:
    mapping = {
        "label_to_name": LABEL_NAMES,
        "name_to_label": {v: k for k, v in LABEL_NAMES.items()},
        "dfew_original_to_6class": DFEW_TO_6CLASS,
        "note": "neutral (DFEW label 3) is excluded; 6 classes 0-5",
    }
    path = DFEW_PROCESSED / "label_mapping.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2))
    print(f"Label mapping saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DFEW annotation CSVs for MAE-DFER")
    parser.add_argument("--dfew_root", default="data/raw/video_fer/dfew",
                        help="Path to the DFEW dataset root")
    parser.add_argument("--fold", default="1",
                        help="Fold number 1-5, or 'all' for all folds")
    args = parser.parse_args()

    dfew_root = pathlib.Path(args.dfew_root)
    if not dfew_root.exists():
        print(f"[ERROR] DFEW root not found: {dfew_root}")
        print("Download DFEW from https://dfew-dataset.github.io/ and place it there.")
        sys.exit(1)

    save_label_mapping()

    folds = list(range(1, 6)) if args.fold == "all" else [int(args.fold)]
    for fold in folds:
        print(f"\n── Fold {fold} ──────────────────────────")
        process_fold(fold, dfew_root)

    print("\nDone. Run train_mae_dfer.py next.")


if __name__ == "__main__":
    main()
