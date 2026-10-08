"""Tests for the text emotion recognition pipeline."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.label_mapping import (  # noqa: E402
    SHARED_EMOTION_CLASSES,
    collapse_goemotions_probs,
)
from src.pipeline.text_pipeline import TextPipeline  # noqa: E402
from models.text.inference import _checkpoint_for  # noqa: E402


def test_goemotions_collapse_sums_to_one():
    labels = ["admiration", "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    probs = [0.2, 0.1, 0.05, 0.05, 0.3, 0.1, 0.1, 0.1]
    collapsed = collapse_goemotions_probs(labels, probs)
    assert len(collapsed) == 6
    assert abs(sum(collapsed) - 1.0) < 1e-9


def test_goemotions_collapse_maps_anger():
    labels = ["anger", "neutral"]
    probs = [0.8, 0.2]
    collapsed = collapse_goemotions_probs(labels, probs)
    idx = SHARED_EMOTION_CLASSES.index("angry")
    assert collapsed[idx] == 1.0


def test_goemotions_collapse_all_neutral_is_uniform():
    collapsed = collapse_goemotions_probs(["neutral"], [1.0])
    assert np.allclose(collapsed, 1.0 / 6.0)


def test_text_pipeline_produces_valid_distribution():
    tp = TextPipeline()
    result = tp.predict("I feel wonderful today")
    assert result["modality"] == "text"
    assert result["label"] in SHARED_EMOTION_CLASSES
    probs = np.asarray(result["probs_array"])
    assert probs.shape == (6,)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs >= 0).all()


def test_text_pipeline_empty_input_is_valid():
    tp = TextPipeline()
    result = tp.predict("")
    assert abs(sum(result["probs_array"]) - 1.0) < 1e-6


def test_missing_text_checkpoint_is_explicit():
    try:
        _checkpoint_for("model-that-does-not-exist")
    except FileNotFoundError as exc:
        assert "Run" in str(exc)
    else:
        raise AssertionError("missing text weights must not silently fall back")
