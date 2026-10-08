"""Video modality pipeline: FACET feature extraction and emotion inference.

Wraps the three trained video FER models (mae-dfer / est / hybrid-cnn-vit).
Input is a (T, 35) FACET 4.2 frame sequence; it is pooled to the fixed
16-frame length the models expect, normalized with the dataset's train
statistics, and classified.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from src.config import CONFIG

LABEL_NAMES = CONFIG["label_names"]

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "video_fer"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "video"

SEQ_LEN = CONFIG["seq_len"]
FACET_DIM = 35


def _load_normalization() -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    norm = meta["normalization"]
    return np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["std"], dtype=np.float32)


def pool_frames(frames: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Uniformly resample (n, 35) frames to (seq_len, 35)."""
    frames = np.nan_to_num(frames.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = frames.shape[0]
    if n == 0:
        return np.zeros((seq_len, FACET_DIM), dtype=np.float32)
    if n == seq_len:
        return frames
    if n > seq_len:
        idx = np.rint(np.linspace(0, n - 1, seq_len)).astype(np.int64)
        for i in range(1, len(idx)):
            if idx[i] <= idx[i - 1]:
                idx[i] = idx[i - 1] + 1
        idx = np.clip(idx, 0, n - 1)
        return frames[idx]
    reps = int(np.ceil(seq_len / n))
    return np.tile(frames, (reps, 1))[:seq_len]


class VideoPipeline:
    """Feature-level video emotion pipeline for one trained model."""

    MODEL_KEYS = ("mae-dfer", "est", "hybrid-cnn-vit")

    def __init__(self, model_key: str = "hybrid-cnn-vit", device: str | None = None):
        if model_key not in self.MODEL_KEYS:
            raise ValueError(f"unknown model {model_key!r}; expected one of {self.MODEL_KEYS}")
        self.model_key = model_key
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        from models.video.train import build_model

        self.model = build_model(model_key, in_dim=FACET_DIM, seq_len=SEQ_LEN).to(self.device)
        run_dir = CHECKPOINT_DIR / f"runs_{model_key.replace('-', '_')}"
        ckpt = run_dir / "best_model.pt"
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        self.model.eval()

        self.norm_mean, self.norm_std = _load_normalization()

    @torch.no_grad()
    def predict(self, facet_frames: np.ndarray) -> dict:
        """Classify a (n, 35) FACET frame sequence.

        Returns dict with probs (6,), label, label_index, confidence.
        """
        seq = pool_frames(np.asarray(facet_frames, dtype=np.float32))
        seq = (seq - self.norm_mean[None, :]) / self.norm_std[None, :]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            "probs": probs,
            "label_index": idx,
            "label": LABEL_NAMES[idx],
            "confidence": float(probs[idx]),
            "model": self.model_key,
        }
