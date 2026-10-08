#!/usr/bin/env python3
"""Build the CMU-MOSEI video (FACET 4.2) processed dataset for video FER training.

Input:
- models/preprocessing/data/cmumosei_index.json  (utterances, official std folds)
- data/raw/cmu_mosei/CMU_MOSEI_VisualFacet42.csd  (raw HDF5, per-clip FACET frames)

Output (data/processed/video/video_fer/):
- train_data.npz / validation_data.npz / test_data.npz
    features: (N, 16, 35) float32  — REAL FACET 4.2 frames only
    labels:   (N,) int64
    utterance_ids: (N,) unicode    — for cross-modality alignment
- metadata.json (class distribution, normalization, provenance)

This script:
- uses ONLY the FACET 4.2 visual features — no emotion scores, no text, no metadata
  columns ever enter the feature tensor (fixes the label-leakage in the previous
  fabricated dataset, see archive/deprecated_data/fabricated_video_fer/)
- pools each utterance's frames to a fixed 16-frame sequence
    (uniform subsample if >16, tile-repeat if <16)
- computes per-dim normalization using ONLY the training split
- does not train any model and does not modify raw files
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import h5py

SCRIPT = Path(__file__).resolve()
MODELS = SCRIPT.parent.parent
ROOT = MODELS.parent

INDEX_JSON = MODELS / "preprocessing" / "data" / "cmumosei_index.json"
FACET_CSD = ROOT / "data" / "raw" / "cmu_mosei" / "CMU_MOSEI_VisualFacet42.csd"
OUT_DIR = ROOT / "data" / "processed" / "video" / "video_fer"

SEQ_LEN = 16
FACET_DIM = 35

# index emotion_label string -> train.py LABEL_NAMES index
LABEL_NAMES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
EMOTION_TO_IDX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
}

SPLIT_KEY = {"train": "train", "valid": "validation", "test": "test"}


def load_index() -> dict:
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


def pool_frames(frames: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Pool (n, 35) frames to (seq_len, 35). Uniform subsample or tile-repeat."""
    n = frames.shape[0]
    if n == 0:
        raise ValueError("no frames")
    if n == seq_len:
        return frames
    if n > seq_len:
        idx = np.linspace(0, n - 1, seq_len)
        idx = np.rint(idx).astype(np.int64)
        idx = np.unique(idx) if False else idx  # keep duplicates-free rint indices
        # rint on linspace can produce duplicates only when n >> seq_len? No:
        # linspace(0, n-1, 16) with n>16 always yields strictly increasing values,
        # and rint of strictly increasing floats spaced >= 1 stays strictly increasing
        # when spacing >= 1. When 16 < n < 31, spacing < 1, duplicates possible —
        # fall back to first-fit unique fix by nudging forward.
        for i in range(1, len(idx)):
            if idx[i] <= idx[i - 1]:
                idx[i] = idx[i - 1] + 1
        idx = np.clip(idx, 0, n - 1)
        return frames[idx]
    # n < seq_len: tile-repeat
    reps = int(np.ceil(seq_len / n))
    tiled = np.tile(frames, (reps, 1))[:seq_len]
    return tiled


def sanitize(frames: np.ndarray) -> np.ndarray:
    out = frames.astype(np.float32, copy=True)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()

    by_split: dict[str, list[dict]] = defaultdict(list)
    for u in index["utterances"]:
        if u.get("has_facet_frames") and u["facet_frames"] > 0:
            split = SPLIT_KEY.get(u["split"])
            if split:
                by_split[split].append(u)

    print("utterances with facet frames:", {k: len(v) for k, v in by_split.items()})

    # group utterances by clip so each CSD clip is read exactly once
    clip_to_utts: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for split, utts in by_split.items():
        for u in utts:
            clip_to_utts[u["clip_id"]].append((u, split))

    with h5py.File(FACET_CSD, "r") as f:
        cdata = f["FACET 4.2"]["data"]
        csd_clips = set(cdata.keys())
        missing = set(clip_to_utts) - csd_clips
        if missing:
            print(f"WARNING: {len(missing)} index clips missing from CSD, skipping")

        feats_by_split: dict[str, list[np.ndarray]] = defaultdict(list)
        labels_by_split: dict[str, list[int]] = defaultdict(list)
        uids_by_split: dict[str, list[str]] = defaultdict(list)
        skipped_no_overlap = 0

        for i, (clip, utts) in enumerate(sorted(clip_to_utts.items())):
            if clip not in csd_clips:
                continue
            intervals = np.asarray(cdata[clip]["intervals"][:], dtype=np.float64)
            features = np.asarray(cdata[clip]["features"][:], dtype=np.float32)
            for u, split in sorted(utts, key=lambda x: x[0]["utterance_index_in_clip"]):
                lo, hi = u["interval"]
                mask = (intervals[:, 1] > lo - 1e-6) & (intervals[:, 0] < hi + 1e-6)
                frames = sanitize(features[mask])
                if frames.shape[0] == 0:
                    skipped_no_overlap += 1
                    continue
                seq = pool_frames(frames)
                feats_by_split[split].append(seq)
                labels_by_split[split].append(EMOTION_TO_IDX[u["emotion_label"]])
                uids_by_split[split].append(u["utterance_id"])
            if (i + 1) % 400 == 0:
                print(f"  processed {i + 1}/{len(clip_to_utts)} clips")

    print("skipped (no frame overlap):", skipped_no_overlap)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "validation", "test"]:
        feats = np.stack(feats_by_split[split], axis=0).astype(np.float32)
        labels = np.asarray(labels_by_split[split], dtype=np.int64)
        uids = np.asarray(uids_by_split[split], dtype="U32")
        arrays[split] = {"features": feats, "labels": labels, "utterance_ids": uids}
        counts = Counter(labels.tolist())
        print(f"{split}: {feats.shape}, labels {dict(sorted(counts.items()))}")

    # train-only normalization per feature dim (over all pooled train frames)
    train_flat = arrays["train"]["features"].reshape(-1, FACET_DIM)
    norm_mean = train_flat.mean(axis=0).astype(np.float32)
    norm_std = train_flat.std(axis=0).astype(np.float32)
    norm_std = np.where(norm_std <= 0.0, 1.0, norm_std)
    norm_std = np.nan_to_num(norm_std, nan=1.0, posinf=1.0, neginf=1.0)

    for split in ["train", "validation", "test"]:
        x = arrays[split]["features"]
        x = (x - norm_mean[None, None, :]) / norm_std[None, None, :]
        arrays[split]["features"] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    for split in ["train", "validation", "test"]:
        out = OUT_DIR / f"{split}_data.npz"
        np.savez(out, **arrays[split])
        print("WROTE", out)

    metadata = {
        "source": "CMU_MOSEI VisualFacet42 (real FACET 4.2 features)",
        "replaces": "archive/deprecated_data/fabricated_video_fer (label-leaked synthetic set)",
        "label_names": LABEL_NAMES,
        "sequence_length": SEQ_LEN,
        "feature_dim": FACET_DIM,
        "normalization": {
            "computed_from_split": "train",
            "mean": norm_mean.tolist(),
            "std": norm_std.tolist(),
        },
        "num_samples": {s: int(len(arrays[s]["labels"])) for s in arrays},
        "class_distribution": {
            s: {LABEL_NAMES[c]: int(n) for c, n in sorted(Counter(arrays[s]["labels"].tolist()).items())}
            for s in arrays
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("WROTE", OUT_DIR / "metadata.json")


if __name__ == "__main__":
    main()
