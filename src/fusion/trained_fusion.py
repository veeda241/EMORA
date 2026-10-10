"""Learned decision-level fusion: a trained classifier over the three frozen
models' probability vectors.

Drop-in replacement for EmotionFusion.fuse() — same input (a dict of
{modality: (6,) probs}) and same output keys — but the combination weights are
LEARNED (logistic regression fit on synchronized MELD predictions produced by
scripts/build_fusion_predictions.py + scripts/train_fusion.py) instead of fixed.

The exact feature-column order used at training time is stored alongside the
model in feature_order.json and is honoured here so live inference matches
training. If any modality is missing at inference time, we do NOT fabricate
probabilities: we fall back to the fixed-weight EmotionFusion over whatever
modalities are available (documented behaviour).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CONFIG
from src.fusion.emotion_fusion import EmotionFusion

LABEL_NAMES = CONFIG["label_names"]
MODALITIES = ("text", "audio", "video")

DEFAULT_MODEL_DIR = Path(CONFIG["model_dir"]).parent / "fusion" / "logistic_regression"


class TrainedFusion:
    """Learned fusion over per-modality probability vectors."""

    def __init__(self, model_dir: str | Path | None = None):
        import joblib

        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        model_path = model_dir / "fusion_logreg.joblib"
        order_path = model_dir / "feature_order.json"
        if not model_path.exists() or not order_path.exists():
            raise FileNotFoundError(f"no trained fusion at {model_dir}")

        self.model_dir = model_dir
        self.clf = joblib.load(model_path)
        meta = json.loads(order_path.read_text(encoding="utf-8"))
        self.feature_cols: list[str] = meta["feature_cols"]
        self.classes: list[str] = meta["classes"]
        # column order used by predict_proba
        self._clf_classes = list(self.clf.classes_)
        self._fallback = EmotionFusion()

    @staticmethod
    def available(model_dir: str | Path | None = None) -> bool:
        d = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        return (d / "fusion_logreg.joblib").exists() and (d / "feature_order.json").exists()

    def _feature_vector(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        vec = np.zeros(len(self.feature_cols), dtype=np.float64)
        index = {name: i for i, name in enumerate(self.feature_cols)}
        for mod in MODALITIES:
            probs = np.asarray(predictions[mod], dtype=np.float64)
            for j, cls in enumerate(LABEL_NAMES):
                col = f"{mod}_p_{cls}"
                if col in index and j < len(probs):
                    vec[index[col]] = probs[j]
        return vec

    def fuse(self, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
        present = [m for m in MODALITIES
                   if m in predictions and np.asarray(predictions[m]).size == len(LABEL_NAMES)]

        # Learned fusion requires all three modalities it was trained on.
        if len(present) != len(MODALITIES):
            out = self._fallback.fuse(predictions)
            out["fusion_mode"] = f"fixed-weight fallback ({'+'.join(present) or 'none'})"
            return out

        x = self._feature_vector(predictions).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        # reorder to LABEL_NAMES if the classifier's class order differs
        if self._clf_classes != list(range(len(LABEL_NAMES))):
            mapped = np.zeros(len(LABEL_NAMES), dtype=np.float64)
            for pos, c in enumerate(self._clf_classes):
                mapped[int(c)] = proba[pos]
            proba = mapped
        idx = int(np.argmax(proba))

        # disagreement: mean total-variation distance of each modality vs fused
        disagreement = float(np.mean(
            [np.abs(np.asarray(predictions[m], dtype=np.float64) - proba).sum() / 2
             for m in MODALITIES]
        ))
        return {
            "probs": proba,
            "label_index": idx,
            "label": LABEL_NAMES[idx],
            "confidence": float(proba[idx]),
            "modalities_used": list(MODALITIES),
            "disagreement": disagreement,
            "fusion_mode": "learned (logistic regression)",
        }
