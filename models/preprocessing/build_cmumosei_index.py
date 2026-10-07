#!/usr/bin/env python3
"""Build a common CMU-MOSEI index aligned across text/audio/video/labels.

Output:
- models/preprocessing/data/cmumosei_index.json

This script does not train anything and does not modify raw files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT = Path(__file__).resolve()
MODELS = SCRIPT.parent.parent
REPO_ROOT = MODELS.parent
ROOT = REPO_ROOT

DATA_RAW = ROOT / "data" / "raw" / "cmu_mosei"
OUT_DIR = MODELS / "preprocessing" / "data"
OUT_INDEX = OUT_DIR / "cmumosei_index.json"

SDK_FOLDS_PY = (
    ROOT
    / "data"
    / "external"
    / "CMU-MultimodalSDK-main"
    / "mmsdk"
    / "mmdatasdk"
    / "dataset"
    / "standard_datasets"
    / "CMU_MOSEI"
    / "cmu_mosei_std_folds.py"
)


def resolve_sdk_folds() -> Path:
    candidates = [
        SDK_FOLDS_PY,
        REPO_ROOT.parent / "CMU-MultimodalSDK-main"
        / "mmsdk"
        / "mmdatasdk"
        / "dataset"
        / "standard_datasets"
        / "CMU_MOSEI"
        / "cmu_mosei_std_folds.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find cmu_mosei_std_folds.py"
    )


def load_folds() -> tuple[set[str], set[str], set[str]]:
    ns: dict[str, object] = {}
    ns["__name__"] = "cmu_mosei_std_folds"
    with resolve_sdk_folds().open("r", encoding="utf-8") as fh:
        exec(fh.read(), ns)
    return (
        set(ns["standard_train_fold"]),
        set(ns["standard_valid_fold"]),
        set(ns["standard_test_fold"]),
    )


def load_data_group(path: Path, top: str):
    f = h5py.File(path, "r")
    g = f[top]
    data = g["data"]
    return f, data


def words_for_clip(wdata, clip: str) -> tuple[np.ndarray, np.ndarray]:
    wf = np.asarray(wdata[clip]["intervals"][:])
    words = np.asarray(wdata[clip]["features"][:], dtype=object)
    out: list[str] = []
    for i in range(words.shape[0]):
        val = words[i][0]
        if isinstance(val, (bytes, bytearray)):
            out.append(val.decode("utf-8", "replace"))
        else:
            out.append(str(val))
    return wf, np.array(out)


def frames_for_intervals(
    data,
    clip: str,
    lo: float,
    hi: float,
):
    mids = np.asarray(data[clip]["intervals"][:])
    mask = (mids[:, 1] > lo - 1e-6) & (mids[:, 0] < hi + 1e-6)
    return data[clip]["features"][mask]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_folds()
    all_clips = train | valid | test

    _, wdata = load_data_group(
        DATA_RAW / "CMU_MOSEI_TimestampedWords.csd", "words"
    )
    _, fdata = load_data_group(
        DATA_RAW / "CMU_MOSEI_VisualFacet42.csd", "FACET 4.2"
    )
    _, ldata = load_data_group(
        DATA_RAW / "CMU_MOSEI_Labels.csd", "All Labels"
    )
    _, cdata = load_data_group(
        DATA_RAW / "CMU_MOSEI_COVAREP.csd", "COVAREP"
    )

    EmotionNames = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
    # Labels column order: sentiment, happy, sad, anger, surprise, disgust, fear
    LABEL_COL = {
        "sentiment": 0,
        "happy": 1,
        "sad": 2,
        "anger": 3,
        "surprise": 4,
        "disgust": 5,
        "fear": 6,
    }

    # We don't load all COVAREP frames into RAM. We only need utterance-windowed frames.
    covarep_clips = set(cdata.keys())
    facet_clips = set(fdata.keys())
    words_clips = set(wdata.keys())
    label_clips = set(ldata.keys())

    index: dict[str, object] = {
        "source": {
            "repo_root": str(REPO_ROOT),
            "data_raw": str(DATA_RAW),
            "sdk_folds": str(resolve_sdk_folds()),
        },
        "label_order_in_file": ["sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear"],
        "model_output_order": EmotionNames,
        "emotion_column_map_from_file_to_model_output": {
            name: int(LABEL_COL[name]) for name in EmotionNames
        },
        "folds": {
            "standard_train_fold": sorted(train),
            "standard_valid_fold": sorted(valid),
            "standard_test_fold": sorted(test),
        },
        "modality_clips": {
            "Labels": sorted(label_clips),
            "Words": sorted(words_clips),
            "Facet42": sorted(facet_clips),
            "COVAREP": sorted(covarep_clips),
        },
        "utterances": [],
        "counts": {},
    }

    utterances: list[dict] = []
    counts: dict[str, dict[str, int]] = {
        split: {
            "utterances": 0,
            "text": 0,
            "facet": 0,
            "covarep": 0,
            "multi_emotion_gt_0_33": 0,
        }
        for split in ("train", "valid", "test")
    }

    for clip in sorted(label_clips):
        split = (
            "train" if clip in train else
            "valid" if clip in valid else
            "test" if clip in test else
            "ungrouped"
        )
        if split == "ungrouped":
            continue

        lfeats = np.asarray(ldata[clip]["features"][:])
        lints = np.asarray(ldata[clip]["intervals"][:])
        wf, words = words_for_clip(wdata, clip)
        ff = np.asarray(fdata[clip]["intervals"][:]) if clip in facet_clips else None
        # we will fetch covarep frames on demand per utterance

        for i in range(lfeats.shape[0]):
            lo, hi = float(lints[i, 0]), float(lints[i, 1])
            row = lfeats[i]
            evec = [
                float(row[LABEL_COL["anger"]]),
                float(row[LABEL_COL["disgust"]]),
                float(row[LABEL_COL["fear"]]),
                float(row[LABEL_COL["happy"]]),
                float(row[LABEL_COL["sad"]]),
                float(row[LABEL_COL["surprise"]]),
            ]

            best_idx = 0
            best_val = -1.0
            best_set = False
            for idx, v in enumerate(evec):
                if not best_set or v > best_val + 1e-9:
                    best_val = v
                    best_idx = idx
                    best_set = True

            # text: words whose timestamps overlap the utterance interval
            text_tokens: list[str] = []
            if clip in words_clips:
                text_tokens = [
                    w for w, (ws, we) in zip(words.tolist(), wf.tolist())
                    if we > lo - 1e-6 and ws < hi + 1e-6
                ]

            # facet frames
            facet_frames = np.zeros((0, 35), dtype=np.float32)
            if ff is not None and clip in facet_clips:
                facet_frames = frames_for_intervals(fdata, clip, lo, hi)

            # covarep frames
            covarep_frames = np.zeros((0, 74), dtype=np.float32)
            if clip in covarep_clips:
                covarep_frames = frames_for_intervals(cdata, clip, lo, hi)

            entry: dict[str, object] = {
                "clip_id": clip,
                "utterance_id": f"{clip}_utt_{i}",
                "utterance_index_in_clip": i,
                "interval": [lo, hi],
                "split": split,
                "text": text_tokens,
                "text_tokens": len(text_tokens),
                "facet_frames": int(facet_frames.shape[0]),
                "covarep_frames": int(covarep_frames.shape[0]),
                "emotion_scores": {
                    name: round(v, 6) for name, v in zip(EmotionNames, evec)
                },
                "emotion_label": EmotionNames[best_idx],
                "emotion_label_max_score": round(best_val, 6),
                "multi_emotion_gt_0_33": int(sum(v > 0.33 for v in evec)),
                "has_text": len(text_tokens) > 0,
                "has_facet_frames": facet_frames.shape[0] > 0,
                "has_covarep_frames": covarep_frames.shape[0] > 0,
            }
            utterances.append(entry)
            counts[split]["utterances"] += 1
            if entry["has_text"]:
                counts[split]["text"] += 1
            if entry["has_facet_frames"]:
                counts[split]["facet"] += 1
            if entry["has_covarep_frames"]:
                counts[split]["covarep"] += 1
            if entry["multi_emotion_gt_0_33"] > 1:
                counts[split]["multi_emotion_gt_0_33"] += 1

    index["utterances"] = utterances
    index["counts"] = counts

    OUT_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("WROTE", OUT_INDEX)
    print("utterances:", len(utterances))
    print("counts:", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
