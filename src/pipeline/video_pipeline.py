"""Video modality pipeline: FACET feature extraction and emotion inference.

Supports two input modes:
1. **Live webcam** — `predict_webcam(frame_rgb)` processes one frame at a time,
   maintains a rolling buffer of the last SEQ_LEN frames, and returns a
   prediction once the buffer has enough frames.
2. **Precomputed segment** — `predict(facet_frames)` accepts a (T, 35) FACET
   4.2 frame sequence, pools it to SEQ_LEN, and classifies it.

The winning video FER model is hybrid-cnn-vit (models/video/VIDEO_MODELS_TRAINING_REPORT.md).
"""

from __future__ import annotations

import collections
import json
import pathlib
from typing import Optional

import numpy as np
import torch

from src.config import CONFIG

LABEL_NAMES = CONFIG["label_names"]

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "video_fer"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "video"

SEQ_LEN  = CONFIG["seq_len"]   # 16
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
    """Feature-level video emotion pipeline with live webcam support."""

    MODEL_KEYS = ("hybrid-cnn-vit",)

    def __init__(self, model_key: str = "hybrid-cnn-vit", device: str | None = None):
        if model_key not in self.MODEL_KEYS:
            raise ValueError(f"unknown model {model_key!r}; expected one of {self.MODEL_KEYS}")
        self.model_key = model_key
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        from models.video.train import build_model

        self.model = build_model(model_key, in_dim=FACET_DIM, seq_len=SEQ_LEN).to(self.device)
        run_dir = CHECKPOINT_DIR / f"runs_{model_key.replace('-', '_')}"
        ckpt = run_dir / "best_model.pt"
        raw = torch.load(ckpt, map_location=self.device)
        # Checkpoint may be a plain state_dict or wrapped in {"model_state_dict": ...}
        state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        self.model.load_state_dict(state)
        self.model.eval()

        self.norm_mean, self.norm_std = _load_normalization()

        # Rolling frame buffer for live webcam inference
        self._frame_buffer: collections.deque = collections.deque(maxlen=SEQ_LEN)

        # Lazy-loaded live feature extractor
        self._extractor = None

    # ── precomputed segment ───────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, facet_frames: np.ndarray) -> dict:
        """Classify a (n, 35) precomputed FACET frame sequence."""
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
            "source": "precomputed",
        }

    # ── live webcam ───────────────────────────────────────────────────────

    def _get_extractor(self):
        if self._extractor is None:
            from src.pipeline.facet_live import FacetLiveExtractor
            # Extractor already applies normalisation from training stats
            self._extractor = FacetLiveExtractor(apply_norm=True)
        return self._extractor

    @torch.no_grad()
    def predict_webcam(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """
        Process one webcam frame (H, W, 3) RGB.

        Adds the extracted features to a rolling buffer of length SEQ_LEN.
        Returns a prediction dict once the buffer contains at least 4 frames,
        or None if no face is detected in this frame.

        The returned dict has an extra key ``"buffer_frames"`` showing how
        many real frames are in the buffer (so the UI can display fill status).
        """
        extractor = self._get_extractor()
        feat = extractor.extract(frame_rgb)   # (35,) or None
        if feat is None:
            return None  # no face in frame

        self._frame_buffer.append(feat)
        n = len(self._frame_buffer)

        if n < 4:
            # Not enough frames yet — return partial info
            return {
                "probs": np.full(len(LABEL_NAMES), 1.0 / len(LABEL_NAMES), dtype=np.float32),
                "label_index": -1,
                "label": "…",
                "confidence": 0.0,
                "model": self.model_key,
                "source": "live",
                "buffer_frames": n,
                "buffer_ready": False,
            }

        # Build (SEQ_LEN, 35) sequence from buffer (pad if needed)
        frames = np.stack(list(self._frame_buffer), axis=0)  # (n, 35), n <= SEQ_LEN
        seq = pool_frames(frames, SEQ_LEN)
        # Features already normalised by FacetLiveExtractor — do NOT normalise again
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
            "source": "live",
            "buffer_frames": n,
            "buffer_ready": True,
        }

    def clear_buffer(self):
        """Reset the rolling frame buffer (call on conversation reset)."""
        self._frame_buffer.clear()
