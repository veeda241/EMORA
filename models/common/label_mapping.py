"""Canonical emotion label mappings shared by all modality pipelines.

The project's shared emotion space is the 6-class Ekman set used by the
CMU-MOSEI based trackers:

    angry, disgust, fearful, happy, sad, surprised

Every pipeline converts its native label space into this one so that the
decision-level fusion (src/fusion/emotion_fusion.py) can average probability
vectors directly.
"""

from __future__ import annotations

SHARED_EMOTION_CLASSES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
SHARED_LABEL_TO_IDX = {name: i for i, name in enumerate(SHARED_EMOTION_CLASSES)}

# CMU-MOSEI index vocabulary (cmumosei_index.json / audio+video npz labels)
MOSEI_TO_SHARED = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprised",
}

# Audio checkpoints (models/audio/runs_*) were trained with the shared
# vocabulary directly: {0: angry, 1: disgust, 2: fearful, 3: happy, 4: sad, 5: surprised}

# Text checkpoint (GoEmotions 28-class deberta-v3-base) collapsed to the
# 6-class space using the official GoEmotions "Ekman" taxonomy.
# Neutral mass is dropped and the remaining probabilities renormalized.
GOEMOTIONS_TO_SHARED = {
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "nervousness": "fearful",
    "joy": "happy",
    "amusement": "happy",
    "approval": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "love": "happy",
    "optimism": "happy",
    "relief": "happy",
    "pride": "happy",
    "admiration": "happy",
    "desire": "happy",
    "caring": "happy",
    "sadness": "sad",
    "disappointment": "sad",
    "embarrassment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "surprise": "surprised",
    "realization": "surprised",
    "confusion": "surprised",
    "curiosity": "surprised",
    # "neutral" and "desire" (if unmapped) fall through; "neutral" is dropped.
    "neutral": None,
}


def collapse_goemotions_probs(label_names: list[str], probs) -> list[float]:
    """Collapse a GoEmotions probability vector (any order) onto the 6-class space."""
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    out = np.zeros(len(SHARED_EMOTION_CLASSES), dtype=np.float64)
    for i, name in enumerate(label_names):
        target = GOEMOTIONS_TO_SHARED.get(name)
        if target is None:
            continue
        out[SHARED_LABEL_TO_IDX[target]] += probs[i]
    total = out.sum()
    if total > 0:
        out /= total
    else:
        # all probability mass fell on unmapped labels (e.g. pure neutral):
        # fall back to a uniform distribution over the shared space.
        out[:] = 1.0 / len(SHARED_EMOTION_CLASSES)
    return out.tolist()
