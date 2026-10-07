#!/usr/bin/env python3
"""Resume the CMU-MOSEI audio-only processed dataset construction.

If a checkpoint exists, this script continues from the next unprocessed utterance
instead of rebuilding everything from scratch.

Checkpoint files:
- models/preprocessing/data/audio_build_checkpoint.json

Output:
- data/processed/cmu_mosei/...
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
MODELS = SCRIPT.parent.parent
REPO_ROOT = MODELS.parent
ROOT = REPO_ROOT

DATA_RAW = ROOT / "data" / "raw" / "cmu_mosei"
INDEX_JSON = MODELS / "preprocessing" / "data" / "cmumosei_index.json"
CHECKPOINT_JSON = MODELS / "preprocessing" / "data" / "audio_build_checkpoint.json"
OUT_DIR = ROOT / "data" / "processed" / "cmu_mosei"
AUDIO_DIR = OUT_DIR / "audio"
LABELS_DIR = OUT_DIR / "labels"

PROCESSED_DIRS = [OUT_DIR, AUDIO_DIR, LABELS_DIR]

EMOTION_ORDER = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
COVAREP_DIM = 74

SUFFIXES = ("train", "validation", "test")


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

    rows = {"train": [], "validation": [], "test": []}
    for u in index["utterances"]:
        s = split_of_clip.get(u["clip_id"])
        if s in rows:
            rows[s].append(u)
    return rows["train"], rows["validation"], rows["test"]


def frame_mean(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0)


def frame_std(frames: np.ndarray) -> np.ndarray:
    std = frames.std(axis=0, ddof=0)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.where(std <= 0.0, 1.0, std)
    return std


def sanitize_frames(frames: np.ndarray) -> tuple[np.ndarray, int]:
    cleaned = frames.copy()
    naninf = int(np.isnan(cleaned).sum() + np.isinf(cleaned).sum())
    np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0, out=cleaned)
    return cleaned, naninf


def extract_features_from_raw_csd(u: dict, cdata):
    clip = u["clip_id"]
    lo, hi = u["interval"]
    mids = np.asarray(cdata[clip]["intervals"][:], dtype=np.float64)
    mask = (mids[:, 1] > lo - 1e-6) & (mids[:, 0] < hi + 1e-6)
    feats = np.asarray(cdata[clip]["features"][:], dtype=np.float32)
    frames = feats[mask]
    if frames.size == 0:
        return None, 0
    frames, cleaned_invalid = sanitize_frames(frames)
    if frames.size == 0:
        return None, cleaned_invalid
    mean_vec = frame_mean(frames)
    std_vec = frame_std(frames)
    utterance_repr = np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)
    return utterance_repr, cleaned_invalid


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


def load_checkpoint():
    if not CHECKPOINT_JSON.exists():
        return None
    return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    for d in PROCESSED_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    index = load_index()
    train_u, validation_u, test_u = build_split_lists(index)
    print("utterances: train", len(train_u), "validation", len(validation_u), "test", len(test_u))

    import h5py

    covarep_path = DATA_RAW / "CMU_MOSEI_COVAREP.csd"
    cfile = h5py.File(covarep_path, "r", swmr=False)
    cdata = cfile["COVAREP/data"]
    covarep_clips = set(cdata.keys())

    ckpt = load_checkpoint()
    resume_state = ckpt or {}
    resume_from = resume_state.get("resume_from_utterance_index", 0)
    already_built = set()
    for split in SUFFIXES:
        key = f"built_{split}"
        already_built.update(resume_state.get(key, []))

    def filtered(utterances, split):
        out = []
        for idx_u, u in enumerate(utterances):
            if idx_u < resume_from and u["utterance_id"] not in already_built:
                continue
            out.append(u)
        return out

    train_u = filtered(train_u, "train")
    validation_u = filtered(validation_u, "validation")
    test_u = filtered(test_u, "test")

    print("remaining utterances after checkpoint: train", len(train_u), "validation", len(validation_u), "test", len(test_u))

    def build_or_resume(utterances, split):
        rows = []
        labels = []
        missing = 0
        cleaned_invalid_total = 0
        built_ids = set()
        # load previously built arrays if resuming
        npz_path = AUDIO_DIR / f"{split}.npz"
        npy_path = LABELS_DIR / f"{split}.npy"
        existing_feats = None
        existing_labels = None
        existing_ids = []
        if npz_path.exists() and npy_path.exists():
            try:
                existing_feats = np.load(npz_path, allow_pickle=False)["feats"].astype(np.float32)
                existing_labels = np.load(npy_path, allow_pickle=False).astype(np.float32)
                csv_path = OUT_DIR / f"{split}.csv"
                if csv_path.exists():
                    with csv_path.open("r", encoding="utf-8", newline="") as fh:
                        r = csv.reader(fh)
                        header = next(r)
                        for row in r:
                            existing_ids.append(row[0])
                already_built.update(existing_ids)
            except Exception as e:
                print(f"could not load existing {split} arrays for checkpoint resume: {e}")
                existing_feats = None
                existing_labels = None

        if existing_feats is not None and existing_labels is not None:
            rows.append(existing_feats)
            labels.append(existing_labels)

        for u in utterances:
            if u["utterance_id"] in already_built:
                continue
            if u["clip_id"] not in covarep_clips:
                missing += 1
                continue
            feat, cleaned_invalid = extract_features_from_raw_csd(u, cdata)
            if feat is None:
                missing += 1
                cleaned_invalid_total += cleaned_invalid
                continue
            rows.append(feat[np.newaxis, :].astype(np.float32))
            label_vec = np.zeros(len(EMOTION_ORDER), dtype=np.float32)
            for i, name in enumerate(EMOTION_ORDER):
                label_vec[i] = float(u["emotion_scores"].get(name, 0.0))
            labels.append(label_vec[np.newaxis, :].astype(np.float32))
            built_ids.add(u["utterance_id"])
            cleaned_invalid_total += cleaned_invalid

        if len(rows) == 0:
            feats = np.empty((0, COVAREP_DIM * 2), dtype=np.float32)
            labs = np.empty((0, len(EMOTION_ORDER)), dtype=np.float32)
        else:
            feats = np.concatenate(rows, axis=0).astype(np.float32)
            labs = np.concatenate(labels, axis=0).astype(np.float32)

        return feats, labs, missing, built_ids, cleaned_invalid_total

    print("processing train...")
    train_feats, train_labels, train_missing, train_built, train_cleaned_invalid = build_or_resume(train_u, "train")
    print("train feats", train_feats.shape, "labels", train_labels.shape, "missing", train_missing, "cleaned_invalid", train_cleaned_invalid)

    print("processing validation...")
    val_feats, val_labels, val_missing, val_built, val_cleaned_invalid = build_or_resume(validation_u, "validation")
    print("validation feats", val_feats.shape, "labels", val_labels.shape, "missing", val_missing, "cleaned_invalid", val_cleaned_invalid)

    print("processing test...")
    test_feats, test_labels, test_missing, test_built, test_cleaned_invalid = build_or_resume(test_u, "test")
    print("test feats", test_feats.shape, "labels", test_labels.shape, "missing", test_missing, "cleaned_invalid", test_cleaned_invalid)

    normalization = compute_normalization(train_feats)
    norm_path = OUT_DIR / "normalization.json"
    norm_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    print("WROTE", norm_path)

    def apply_norm(x):
        m = np.array(normalization["norm_mean"], dtype=np.float32)
        s = np.array(normalization["norm_std"], dtype=np.float32)
        out = (x - m) / s
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    train_feats_n = apply_norm(train_feats)
    val_feats_n = apply_norm(val_feats)
    test_feats_n = apply_norm(test_feats)

    np.savez(AUDIO_DIR / "train.npz", feats=train_feats_n.astype(np.float32))
    np.savez(AUDIO_DIR / "validation.npz", feats=val_feats_n.astype(np.float32))
    np.savez(AUDIO_DIR / "test.npz", feats=test_feats_n.astype(np.float32))
    print("WROTE", AUDIO_DIR / "train.npz")
    print("WROTE", AUDIO_DIR / "validation.npz")
    print("WROTE", AUDIO_DIR / "test.npz")

    np.save(LABELS_DIR / "train.npy", train_labels.astype(np.float32))
    np.save(LABELS_DIR / "validation.npy", val_labels.astype(np.float32))
    np.save(LABELS_DIR / "test.npy", test_labels.astype(np.float32))
    print("WROTE", LABELS_DIR / "train.npy")
    print("WROTE", LABELS_DIR / "validation.npy")
    print("WROTE", LABELS_DIR / "test.npy")

    def write_split_csv(path, utterances, feats, feats_ids):
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
            # map utterance id -> feature index for alignment
            id_to_idx = {uid: i for i, uid in enumerate(feats_ids)}
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

    write_split_csv(OUT_DIR / "train.csv", train_u, train_feats, train_built)
    write_split_csv(OUT_DIR / "validation.csv", validation_u, val_feats, val_built)
    write_split_csv(OUT_DIR / "test.csv", test_u, test_feats, test_built)
    print("WROTE split CSVs")

    # index.csv
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

    next_resume = resume_from
    def max_present(seq, built):
        present = [i for i, u in enumerate(seq) if u["utterance_id"] in built]
        return max(present) if present else -1
    train_max = max_present(train_u, train_built)
    val_max = max_present(validation_u, val_built)
    test_max = max_present(test_u, test_built)
    next_resume = max(next_resume, train_max + 1, val_max + 1, test_max + 1)

    save_checkpoint({
        "resume_from_utterance_index": next_resume,
        "built_train": sorted(train_built),
        "built_validation": sorted(val_built),
        "built_test": sorted(test_built),
        "train_cleaned_invalid": train_cleaned_invalid,
        "validation_cleaned_invalid": val_cleaned_invalid,
        "test_cleaned_invalid": test_cleaned_invalid,
    })
    print("WROTE", CHECKPOINT_JSON)

    cfile.close()

    print("DONE")
    print("train feats", train_feats_n.shape, "labels", train_labels.shape)
    print("validation feats", val_feats_n.shape, "labels", val_labels.shape)
    print("test feats", test_feats_n.shape, "labels", test_labels.shape)


if __name__ == "__main__":
    main()
