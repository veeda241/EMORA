"""Normalize the MELD CSV annotations for text/audio training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/raw/meld")
DEFAULT_OUTPUT = Path("data/processed/meld")
SPLITS = {
    "train": "train_sent_emo.csv",
    "validation": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}
REQUIRED_COLUMNS = {
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
    "StartTime",
    "EndTime",
}


def time_to_milliseconds(value: str) -> int:
    hours, minutes, rest = str(value).replace(",", ".").split(":", 2)
    seconds = float(rest)
    return round((int(hours) * 3600 + int(minutes) * 60 + seconds) * 1000)


def prepare_split(path: Path, split: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame.rename(
        columns={
            "Utterance": "text",
            "Emotion": "label",
            "Sentiment": "sentiment",
            "Dialogue_ID": "dialogue_id",
            "Utterance_ID": "utterance_id",
            "Season": "season",
            "Episode": "episode",
            "StartTime": "start_time",
            "EndTime": "end_time",
        }
    )
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["speaker"] = frame["Speaker"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].fillna("").astype(str).str.strip().str.lower()
    frame["sentiment"] = frame["sentiment"].fillna("").astype(str).str.strip().str.lower()
    frame = frame.loc[frame["text"].ne("") & frame["label"].ne("")].copy()

    frame["dialogue_id"] = frame["dialogue_id"].astype(int)
    frame["utterance_id"] = frame["utterance_id"].astype(int)
    frame["season"] = frame["season"].astype(int)
    frame["episode"] = frame["episode"].astype(int)
    frame["start_ms"] = frame["start_time"].map(time_to_milliseconds)
    frame["end_ms"] = frame["end_time"].map(time_to_milliseconds)
    if (frame["end_ms"] < frame["start_ms"]).any():
        raise ValueError(f"{path} contains an utterance ending before it starts")

    frame.insert(0, "split", split)
    frame.insert(
        1,
        "utterance_key",
        frame.apply(
            lambda row: (
                f"{split}:s{row.season}_e{row.episode}_d{row.dialogue_id}_u{row.utterance_id}"
            ),
            axis=1,
        ),
    )
    columns = [
        "split",
        "utterance_key",
        "text",
        "speaker",
        "label",
        "sentiment",
        "dialogue_id",
        "utterance_id",
        "season",
        "episode",
        "start_time",
        "end_time",
        "start_ms",
        "end_ms",
    ]
    output = frame[columns].reset_index(drop=True)
    return output, {
        "source_file": str(path),
        "source_rows": int(len(pd.read_csv(path, usecols=["Utterance"]))),
        "output_rows": int(len(output)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    details: list[dict] = []
    for split, filename in SPLITS.items():
        frame, detail = prepare_split(args.input_dir / filename, split)
        frames.append(frame)
        details.append(detail)

    combined = pd.concat(frames, ignore_index=True)
    raw_episode_keys = combined["utterance_key"].str.split(":", n=1).str[1]
    duplicate_keys = int(raw_episode_keys.duplicated().sum())

    labels = sorted(combined["label"].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    combined["label_id"] = combined["label"].map(label2id).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "meld_combined.csv", index=False)
    combined.to_parquet(args.output_dir / "meld_combined.parquet", index=False)
    for split in SPLITS:
        split_frame = combined.loc[combined["split"].eq(split)].reset_index(drop=True)
        split_frame.to_csv(args.output_dir / f"meld_{split}.csv", index=False)
        split_frame.to_parquet(args.output_dir / f"meld_{split}.parquet", index=False)

    metadata = {
        "dataset": "MELD",
        "labels": labels,
        "label2id": label2id,
        "split_counts": combined["split"].value_counts().sort_index().to_dict(),
        "total_rows": int(len(combined)),
        "duplicate_utterance_keys": duplicate_keys,
        "duplicate_key_note": (
            "Some source split files reuse episode/dialogue/utterance IDs for "
            "different rows. The processed key includes the source split; verify "
            "audio alignment before waveform training."
        ),
        "time_units": "milliseconds from episode start",
        "audio_filename_pattern": "Episode{season}_{episode}/dial{season}_{dialogue_id}_utt{utterance_id}.wav",
        "sources": details,
    }
    (args.output_dir / "labels.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output_dir / "preprocessing_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(combined):,} MELD rows to {args.output_dir}")
    print(combined["split"].value_counts().sort_index().to_string())
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()
