"""Emotion-level fusion: combines per-modality emotion predictions into a single estimate.

Decision-level alternative to the MFN feature fusion: each unimodal model
emits a probability distribution over the 6 emotion classes, and this module
merges them. Useful when only a subset of modalities is available at
inference time (e.g. text-only chat turns).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import CONFIG

LABEL_NAMES = CONFIG["label_names"]
MODALITIES = ("text", "audio", "video")


class EmotionFusion:
    """Weighted probability fusion across modality predictions."""

    def __init__(self, weights: dict[str, float] | None = None):
        # default: equal weights; any subset of modalities is allowed
        self.weights = weights or {m: 1.0 for m in MODALITIES}

    def fuse(self, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
        """Fuse per-modality probability vectors.

        Args:
            predictions: {modality: (num_classes,) probability vector}.
                Missing modalities are simply excluded.

        Returns:
            dict with fused probs, label_index, label, confidence,
            and the per-modality disagreement (mean JS-like L1 distance).
        """
        probs, active = [], []
        for mod in MODALITIES:
            if mod in predictions:
                p = np.asarray(predictions[mod], dtype=np.float64)
                if p.sum() > 0:
                    p = p / p.sum()
                    probs.append((p, p * self.weights.get(mod, 1.0)))
                    active.append(mod)
        if not probs:
            raise ValueError("no valid modality predictions to fuse")

        fused = np.sum([w for _, w in probs], axis=0)
        fused = fused / fused.sum()
        idx = int(np.argmax(fused))

        # disagreement: mean absolute difference between each modality and the fusion
        disagreement = (
            float(np.mean([np.abs(p - fused).sum() / 2 for p, _ in probs]))
            if len(probs) > 1
            else 0.0
        )

        return {
            "probs": fused,
            "label_index": idx,
            "label": LABEL_NAMES[idx],
            "confidence": float(fused[idx]),
            "modalities_used": active,
            "disagreement": disagreement,
        }

    def fuse_batch(self, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
        """Batch variant.

        Args:
            predictions: {modality: (N, num_classes) probability matrix}.

        Returns:
            dict with fused (N, C) probs, (N,) indices, (N,) confidences.
        """
        mats, active = [], []
        for mod in MODALITIES:
            if mod in predictions:
                p = np.asarray(predictions[mod], dtype=np.float64)
                if p.size:
                    mats.append(p / p.sum(axis=1, keepdims=True) * self.weights.get(mod, 1.0))
                    active.append(mod)
        if not mats:
            raise ValueError("no valid modality predictions to fuse")

        fused = np.sum(mats, axis=0)
        fused = fused / fused.sum(axis=1, keepdims=True)
        idx = fused.argmax(axis=1)
        return {
            "probs": fused,
            "label_index": idx,
            "label": [LABEL_NAMES[i] for i in idx],
            "confidence": fused.max(axis=1),
            "modalities_used": active,
        }
