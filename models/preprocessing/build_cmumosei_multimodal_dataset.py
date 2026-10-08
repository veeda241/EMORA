#!/usr/bin/env python3
"""Build the time-aligned CMU-MOSEI multimodal dataset for MFN training.

Aligns, per utterance, the three modality streams to a shared (16, dim)
sequence so the Memory Fusion Network can step through them in lockstep:

- text:  GloVe 300-d word vectors (CMU_MOSEI_TimestampedWordVectors.csd)
- audio: COVAREP 74-d acoustic frames (CMU_MOSEI_COVAREP.csd)
- video: FACET 4.2 35-d facial frames (CMU_MOSEI_VisualFacet42.csd)

Output (data/processed/multimodal/):
- {split}_data.npz with:
    text:   (N, 16, 300) float32
    audio:  (N, 16, 74)  float32
    video:  (N, 16, 35)  float32
    labels: (N,) int64
    utterance_ids: (N,) unicode
- metadata.json (class distribution, normalization stats from TRAIN only)

Only utterances present in ALL three modalities are kept. Splits follow the
official CMU-MOSEI standard folds via cmumosei_index.json. No emotion scores
ever enter the feature tensors.
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
RAW_DIR = ROOT / "data" / "raw" / "cmu_mosei"
OUT_DIR = ROOT / "data" / "processed" / "multimodal"

SEQ_LEN = 16
TEXT_DIM = 300
AUDIO_DIM = 74
VIDEO_DIM = 35

LABEL_NAMES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
EMOTION_TO_IDX = {"anger": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5}

SPLIT_KEY = {"train": "train", "valid": "validation", "test": "test"}


def resample(frames: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Uniformly resample (n, d) frames to (seq_len, d)."""
    n = frames.shape[0]
    if n == seq_len:
        return frames
    if n > seq_len:
        idx = np.linspace(0, n - 1, seq_len)
        idx = np.rint(idx).astype(np.int64)
        for i in range(1, len(idx)):
            if idx[i] <= idx[i - 1]:
                idx[i] = idx[i - 1] + 1
        idx = np.clip(idx, 0, n - 1)
        return frames[idx]
    reps = int(np.ceil(seq_len / n))
    return np.tile(frames, (reps, 1))[:seq_len]


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def frames_in_interval(intervals: np.ndarray, features: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (intervals[:, 1] > lo - 1e-6) & (intervals[:, 0] < hi + 1e-6)
    return features[mask]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    # keep utterances with all three modalities
    by_split: dict[str, list[dict]] = defaultdict(list)
    for u in index["utterances"]:
        if u.get("has_text") and u.get("has_covarep_frames") and u.get("has_facet_frames"):
            split = SPLIT_KEY.get(u["split"])
            if split:
                by_split[split].append(u)
    print("utterances with all 3 modalities:", {k: len(v) for k, v in by_split.items()})

    clip_to_utts: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for split, utts in by_split.items():
        for u in utts:
            clip_to_utts[u["clip_id"]].append((u, split))

    feats: dict[str, dict[str, list]] = {
        s: {"text": [], "audio": [], "video": [], "labels": [], "uids": []}
        for s in ["train", "validation", "test"]
    }
    skipped = Counter()

    with h5py.File(RAW_DIR / "CMU_MOSEI_TimestampedWordVectors.csd", "r") as ft, \
         h5py.File(RAW_DIR / "CMU_MOSEI_COVAREP.csd", "r") as fa, \
         h5py.File(RAW_DIR / "CMU_MOSEI_VisualFacet42.csd", "r") as fv:

        tdata, adata, vdata = ft["glove_vectors"]["data"], fa["COVAREP"]["data"], fv["FACET 4.2"]["data"]

        for i, (clip, utts) in enumerate(sorted(clip_to_utts.items())):
            if clip not in tdata or clip not in adata or clip not in vdata:
                skipped["missing_clip"] += len(utts)
                continue
            t_iv = np.asarray(tdata[clip]["intervals"][:], dtype=np.float64)
            t_ft = np.asarray(tdata[clip]["features"][:], dtype=np.float32)
            a_iv = np.asarray(adata[clip]["intervals"][:], dtype=np.float64)
            a_ft = np.asarray(adata[clip]["features"][:], dtype=np.float32)
            v_iv = np.asarray(vdata[clip]["intervals"][:], dtype=np.float64)
            v_ft = np.asarray(vdata[clip]["features"][:], dtype=np.float32)

            for u, split in sorted(utts, key=lambda x: x[0]["utterance_index_in_clip"]):
                lo, hi = u["interval"]
                t_frames = sanitize(frames_in_interval(t_iv, t_ft, lo, hi))
                a_frames = sanitize(frames_in_interval(a_iv, a_ft, lo, hi))
                v_frames = sanitize(frames_in_interval(v_iv, v_ft, lo, hi))
                if t_frames.size == 0 or a_frames.size == 0 or v_frames.size == 0:
                    skipped[f"{split}_empty"] += 1
                    continue
                rec = feats[split]
                rec["text"].append(resample(t_frames))
                rec["audio"].append(resample(a_frames))
                rec["video"].append(resample(v_frames))
                rec["labels"].append(EMOTION_TO_IDX[u["emotion_label"]])
                rec["uids"].append(u["utterance_id"])
            if (i + 1) % 400 == 0:
                print(f"  processed {i + 1}/{len(clip_to_utts)} clips")

    print("skipped:", dict(skipped))

    # train-only normalization per modality per dim
    norm = {}
    for mod, dim in [("text", TEXT_DIM), ("audio", AUDIO_DIM), ("video", VIDEO_DIM)]:
        flat = np.stack(feats["train"][mod], axis=0).reshape(-1, dim)
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        std = np.where(std <= 0.0, 1.0, std)
        norm[mod] = (mean, std)

    for split in ["train", "validation", "test"]:
        out = {"labels": np.asarray(feats[split]["labels"], dtype=np.int64),
               "utterance_ids": np.asarray(feats[split]["uids"], dtype="U32")}
        for mod in ["text", "audio", "video"]:
            x = np.stack(feats[split][mod], axis=0)
            mean, std = norm[mod]
            x = (x - mean[None, None, :]) / std[None, None, :]
            out[mod] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        path = OUT_DIR / f"{split}_data.npz"
        np.savez(path, **out)
        counts = Counter(out["labels"].tolist())
        print(f"WROTE {path} — N={len(out['labels'])}, classes {dict(sorted(counts.items()))}")

    metadata = {
        "source": "CMU-MOSEI aligned GloVe/COVAREP/FACET computational sequences",
        "label_names": LABEL_NAMES,
        "seq_len": SEQ_LEN,
        "dims": {"text": TEXT_DIM, "audio": AUDIO_DIM, "video": VIDEO_DIM},
        "normalization": {
            m: {"mean": norm[m][0].tolist(), "std": norm[m][1].tolist()}
            for m in norm
        },
        "num_samples": {s: len(feats[s]["labels"]) for s in ["train", "validation", "test"]},
        "class_distribution": {
            s: {LABEL_NAMES[c]: int(n) for c, n in sorted(Counter(feats[s]["labels"]).items())}
            for s in ["train", "validation", "test"]
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("WROTE", OUT_DIR / "metadata.json")


if __name__ == "__main__":
    main()
