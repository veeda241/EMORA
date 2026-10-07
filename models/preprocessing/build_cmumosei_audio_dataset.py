#!/usr/bin/env python3
"""Build the CMU-MOSEI audio-only processed dataset.

Input:
- models/preprocessing/data/cmumosei_index.json

Output:
- data/processed/cmu_mosei/index.csv
- data/processed/cmu_mosei/train.csv
- data/processed/cmu_mosei/validation.csv
- data/processed/cmu_mosei/test.csv
- data/processed/cmu_mosei/audio/train.npz
- data/processed/cmu_mosei/audio/validation.npz
- data/processed/cmu_mosei/audio/test.npz
- data/processed/cmu_mosei/labels/train.npy
- data/processed/cmu_mosei/labels/validation.npy
- data/processed/cmu_mosei/labels/test.npy
- data/processed/cmu_mosei/normalization.json

This script:
- uses ONLY the COVAREP audio modality from the index
- computes utterance-level [mean, std] over time
- computes normalization statistics using ONLY the training split
- does not train any model
- does not modify raw files
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
MODELS = SCRIPT.parent.parent
REPO_ROOT = MODELS.parent
ROOT = REPO_ROOT

DATA_RAW = ROOT / "data" / "raw" / "cmu_mosei"
INDEX_JSON = MODELS / "preprocessing" / "data" / "cmumosei_index.json"
OUT_DIR = ROOT / "data" / "processed" / "cmu_mosei"
AUDIO_DIR = OUT_DIR / "audio"
LABELS_DIR = OUT_DIR / "labels"


EMOTION_ORDER = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

COVAREP_DIM = 74


def load_index() -> dict:
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


def build_split_lists(index: dict):
    folds = index["folds"]
    split_of_clip = {}
    for split, key in [
        ("train", "standard_train_fold"),
        ("validation", "standard_valid_fold"),
        ("test", "standard_test_fold"),
    ]:
        for clip in folds[key]:
            split_of_clip[clip] = split

    train = []
    validation = []
    test = []
    unknown = []
    for u in index["utterances"]:
        s = split_of_clip.get(u["clip_id"])
        if s == "train":
            train.append(u)
        elif s == "validation":
            validation.append(u)
        elif s == "test":
            test.append(u)
        else:
            unknown.append(u)
    return train, validation, test, unknown


def extract_features_from_raw_csd(u: dict, cdata):
    """Fetch utterance-windowed COVAREP frames from the raw CSD and compute mean + std.

    We do NOT load all COVAREP into RAM. We only load the frames overlapping the
    utterance interval from the raw CSD using the stored interval.

    Invalid COVAREP values (NaN/Inf) are replaced BEFORE computing utterance statistics,
    because some COVAREP frames contain feature-level NaN/Inf.
    """
    clip = u["clip_id"]
    lo, hi = u["interval"]
    mids = np.asarray(cdata[clip]["intervals"][:], dtype=np.float64)
    mask = (mids[:, 1] > lo - 1e-6) & (mids[:, 0] < hi + 1e-6)
    feats = np.asarray(cdata[clip]["features"][:], dtype=np.float32)
    frames = feats[mask]
    if frames.size == 0:
        return None
    frames, _ = sanitize_frames(frames)
    if frames.size == 0:
        return None
    mean_vec = frame_mean(frames)
    std_vec = frame_std(frames)
    utterance_repr = np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)
    return utterance_repr


def sanitize_frames(frames: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN/Inf feature values with 0.0 in-place and return cleaned array plus count."""
    cleaned = frames.copy()
    naninf = int(np.isnan(cleaned).sum() + np.isinf(cleaned).sum())
    np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0, out=cleaned)
    return cleaned, naninf


def frame_mean(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0)


def frame_std(frames: np.ndarray) -> np.ndarray:
    std = frames.std(axis=0, ddof=0)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.where(std <= 0.0, 1.0, std)
    return std


def compute_normalization(train_features: np.ndarray) -> dict:
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std <= 0.0] = 1.0
    audio_repr_dim = int(train_features.shape[1])
    return {
        "audio_feature_dim": COVAREP_DIM,
        "audio_representation_dim": audio_repr_dim,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "computed_from_split": "train",
        "note": "std is clamped to 1.0 where it is zero, negative, NaN, or Inf",
        "representation": "concat(mean(COVAREP frames over time), std(COVAREP frames over time))",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    index = load_index()
    train_u, validation_u, test_u, unknown = build_split_lists(index)

    print("utterances: train", len(train_u), "validation", len(validation_u), "test", len(test_u))
    print("unknown/unlabeled utterances:", len(unknown))

    # open COVAREP once
    import h5py

    covarep_path = DATA_RAW / "CMU_MOSEI_COVAREP.csd"
    cfile = h5py.File(covarep_path, "r", swmr=False)
    cdata = cfile["COVAREP/data"]
    covarep_clips = set(cdata.keys())

    label_order_set = set(EMOTION_ORDER)

    def build_arrays(utterances, clip_filter=True):
        rows = []
        labels = []
        missing_covarep = 0
        for u in utterances:
            if u["clip_id"] not in covarep_clips:
                missing_covarep += 1
                continue
            feat = extract_features_from_raw_csd(u, cdata)
            if feat is None:
                missing_covarep += 1
                continue
            rows.append(feat)
            label_vec = np.zeros(len(EMOTION_ORDER), dtype=np.float32)
            for i, name in enumerate(EMOTION_ORDER):
                label_vec[i] = float(u["emotion_scores"].get(name, 0.0))
            labels.append(label_vec)
        feats = np.stack(rows, axis=0).astype(np.float32) if rows else np.empty((0, COVAREP_DIM * 2), dtype=np.float32)
        labs = np.stack(labels, axis=0).astype(np.float32) if labels else np.empty((0, len(EMOTION_ORDER)), dtype=np.float32)
        return feats, labs, missing_covarep

    print("building train arrays from raw CSD (mean+std over utterance COVAREP frames)...")
    train_feats, train_labels, train_missing = build_arrays(train_u)
    print("train feats", train_feats.shape, "labels", train_labels.shape, "missing covarep", train_missing)

    print("building validation arrays...")
    val_feats, val_labels, val_missing = build_arrays(validation_u)
    print("validation feats", val_feats.shape, "labels", val_labels.shape, "missing covarep", val_missing)

    print("building test arrays...")
    test_feats, test_labels, test_missing = build_arrays(test_u)
    print("test feats", test_feats.shape, "labels", test_labels.shape, "missing covarep", test_missing)

    # normalization from TRAIN ONLY
    normalization = compute_normalization(train_feats)
    norm_path = OUT_DIR / "normalization.json"
    norm_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    print("WROTE", norm_path)

    # apply normalization
    def apply_norm(x):
        m = np.array(normalization["norm_mean"], dtype=np.float32)
        s = np.array(normalization["norm_std"], dtype=np.float32)
        out = (x - m) / s
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    train_feats_n = apply_norm(train_feats)
    val_feats_n = apply_norm(val_feats)
    test_feats_n = apply_norm(test_feats)

    # save audio features
    np.savez(AUDIO_DIR / "train.npz", feats=train_feats_n.astype(np.float32))
    np.savez(AUDIO_DIR / "validation.npz", feats=val_feats_n.astype(np.float32))
    np.savez(AUDIO_DIR / "test.npz", feats=test_feats_n.astype(np.float32))
    print("WROTE", AUDIO_DIR / "train.npz")
    print("WROTE", AUDIO_DIR / "validation.npz")
    print("WROTE", AUDIO_DIR / "test.npz")

    # save labels
    np.save(LABELS_DIR / "train.npy", train_labels.astype(np.float32))
    np.save(LABELS_DIR / "validation.npy", val_labels.astype(np.float32))
    np.save(LABELS_DIR / "test.npy", test_labels.astype(np.float32))
    print("WROTE", LABELS_DIR / "train.npy")
    print("WROTE", LABELS_DIR / "validation.npy")
    print("WROTE", LABELS_DIR / "test.npy")

    # save csv index
    index_csv_path = OUT_DIR / "index.csv"
    with index_csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "utterance_id",
            "clip_id",
            "split",
            "interval_start",
            "interval_end",
            "covarep_frames",
            "emotion_label",
            "emotion_label_max_score",
            "multi_emotion_gt_0_33",
            "has_covarep",
            "text_tokens",
        ])
        for u in index["utterances"]:
            w.writerow([
                u["utterance_id"],
                u["clip_id"],
                u["split"],
                round(float(u["interval"][0]), 6),
                round(float(u["interval"][1]), 6),
                u["covarep_frames"],
                u["emotion_label"],
                round(float(u["emotion_label_max_score"]), 6),
                u["multi_emotion_gt_0_33"],
                bool(u["has_covarep_frames"]),
                u["text_tokens"],
            ])
    print("WROTE", index_csv_path)

    # save split CSVs
    def write_split_csv(path, utterances):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "utterance_id",
                "clip_id",
                "interval_start",
                "interval_end",
                "covarep_frames",
                "emotion_label",
                "emotion_label_max_score",
                "multi_emotion_gt_0_33",
                "has_covarep",
                "text_tokens",
                "emotion_scores_anger",
                "emotion_scores_disgust",
                "emotion_scores_fear",
                "emotion_scores_happy",
                "emotion_scores_sad",
                "emotion_scores_surprise",
            ])
            for u in utterances:
                sc = u["emotion_scores"]
                w.writerow([
                    u["utterance_id"],
                    u["clip_id"],
                    round(float(u["interval"][0]), 6),
                    round(float(u["interval"][1]), 6),
                    u["covarep_frames"],
                    u["emotion_label"],
                    round(float(u["emotion_label_max_score"]), 6),
                    u["multi_emotion_gt_0_33"],
                    bool(u["has_covarep_frames"]),
                    u["text_tokens"],
                    round(float(sc["anger"]), 6),
                    round(float(sc["disgust"]), 6),
                    round(float(sc["fear"]), 6),
                    round(float(sc["happy"]), 6),
                    round(float(sc["sad"]), 6),
                    round(float(sc["surprise"]), 6),
                ])

    write_split_csv(OUT_DIR / "train.csv", train_u)
    write_split_csv(OUT_DIR / "validation.csv", validation_u)
    write_split_csv(OUT_DIR / "test.csv", test_u)
    print("WROTE split CSVs")

    cfile.close()

    print("DONE")
    print("train feats", train_feats_n.shape, "labels", train_labels.shape)
    print("validation feats", val_feats_n.shape, "labels", val_labels.shape)
    print("test feats", test_feats_n.shape, "labels", test_labels.shape)


if __name__ == "__main__":
    main()
