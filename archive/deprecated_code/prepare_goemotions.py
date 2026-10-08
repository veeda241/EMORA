"""Combine and normalize the Kaggle GoEmotions CSV exports.

The Kaggle export stores one-hot columns for the 28 GoEmotions labels. This
script keeps the original split assignment, converts those columns into a
pipe-separated ``labels`` field, and writes CSV and Parquet files that are
easier for downstream training code to consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/archive/goemotions_source")
DEFAULT_OUTPUT = Path("data/processed/goemotions")
SOURCE_FILES = ("goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv")


def prepare_frame(path: Path, label_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = {"text", "id", "example_very_unclear", *label_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["example_very_unclear"] = frame["example_very_unclear"].astype(str).str.lower()
    unclear = frame["example_very_unclear"].isin({"true", "1", "yes"})
    frame = frame.loc[~unclear & frame["text"].ne("")].copy()

    label_values = frame[label_columns].fillna(0).astype(int)
    frame["labels"] = label_values.apply(
        lambda row: "|".join(label for label, value in row.items() if value == 1),
        axis=1,
    )
    frame = frame.loc[frame["labels"].ne("")].copy()
    keep = [
        "id",
        "text",
        "labels",
        "author",
        "subreddit",
        "link_id",
        "parent_id",
        "created_utc",
    ]
    keep = [column for column in keep if column in frame.columns]
    result = frame[keep].reset_index(drop=True)
    return result, {
        "source_file": str(path),
        "source_rows": int(pd.read_csv(path, usecols=["id"]).shape[0]),
        "output_rows": int(len(result)),
        "removed_unclear_or_empty": int(unclear.sum()),
    }


def write_outputs(frame: pd.DataFrame, output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "goemotions_combined.csv", index=False)
    frame.to_parquet(output_dir / "goemotions_combined.parquet", index=False)

    for split in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        split_frame.to_csv(output_dir / f"goemotions_{split}.csv", index=False)
        split_frame.to_parquet(output_dir / f"goemotions_{split}.parquet", index=False)

    label_names = sorted(
        {label for labels in frame["labels"] for label in labels.split("|")}
    )
    metadata = {
        "dataset": "GoEmotions",
        "label_count": len(label_names),
        "labels": label_names,
        "multi_label": True,
        "split_strategy": "deduplicate by comment ID, then deterministic 80/10/10 hash split",
        "annotation_rows_before_deduplication": int(summary["rows_before"]),
        "duplicate_annotation_rows": int(summary["duplicate_ids"]),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "rows_removed": int(summary["rows_before"] - len(frame)),
        "source_files": summary["sources"],
    }
    (output_dir / "labels.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(metadata | {"details": summary["details"]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    first = pd.read_csv(args.input_dir / SOURCE_FILES[0])
    excluded = {
        "text",
        "id",
        "author",
        "subreddit",
        "link_id",
        "parent_id",
        "created_utc",
        "rater_id",
        "example_very_unclear",
    }
    label_columns = [column for column in first.columns if column not in excluded]
    if len(label_columns) != 28:
        raise ValueError(f"Expected 28 GoEmotions labels, found {len(label_columns)}")

    frames = []
    details = []
    for filename in SOURCE_FILES:
        prepared, detail = prepare_frame(args.input_dir / filename, label_columns)
        frames.append(prepared)
        details.append(detail)

    annotations = pd.concat(frames, ignore_index=True)
    duplicate_ids = int(annotations["id"].duplicated().sum())
    aggregations = []
    for comment_id, group in annotations.groupby("id", sort=False):
        labels = sorted(
            {
                label
                for values in group["labels"]
                for label in values.split("|")
                if label
            }
        )
        record = group.iloc[0].copy()
        record["labels"] = "|".join(labels)
        aggregations.append(record)
    combined = pd.DataFrame(aggregations).reset_index(drop=True)
    buckets = combined["id"].map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)
        % 100
    )
    combined["split"] = buckets.map(
        lambda bucket: "train"
        if bucket < 80
        else "validation"
        if bucket < 90
        else "test"
    )
    combined["label_count"] = combined["labels"].str.count(r"\|") + 1
    combined = combined[
        ["split", "id", "text", "labels", "label_count"]
        + [column for column in ("author", "subreddit", "link_id", "parent_id", "created_utc")
           if column in combined.columns]
    ]
    summary = {
        "rows_before": int(len(annotations)),
        "duplicate_ids": duplicate_ids,
        "sources": [item["source_file"] for item in details],
        "details": details,
    }
    write_outputs(combined, args.output_dir, summary)
    print(f"Wrote {len(combined):,} rows to {args.output_dir}")
    print(f"Labels: {len(label_columns)} | duplicate IDs: {duplicate_ids}")
    print(combined["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
