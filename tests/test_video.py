"""Tests for the video facial emotion recognition pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.pipeline.video_pipeline import VideoPipeline, pool_frames  # noqa: E402

CKPT = ROOT / "models" / "video" / "runs_hybrid_cnn_vit" / "best_model.pt"
DATA = ROOT / "data" / "processed" / "video" / "video_fer"

pytestmark = pytest.mark.skipif(
    not CKPT.exists() or not (DATA / "metadata.json").exists(),
    reason="trained video checkpoints or dataset metadata not available",
)


def test_pool_frames_downsamples():
    frames = np.arange(64, dtype=np.float32).reshape(64, 1)
    pooled = pool_frames(frames, seq_len=16)
    assert pooled.shape == (16, 1)


def test_pool_frames_upsamples():
    frames = np.arange(4, dtype=np.float32).reshape(4, 1)
    pooled = pool_frames(frames, seq_len=16)
    assert pooled.shape == (16, 1)


def test_pool_frames_empty():
    pooled = pool_frames(np.zeros((0, 35), dtype=np.float32))
    assert pooled.shape == (CONFIG["seq_len"], 35)


def test_pool_frames_indices_monotonic():
    rng = np.random.default_rng(1)
    frames = rng.normal(size=(23, 35)).astype(np.float32)
    pooled = pool_frames(frames)
    assert pooled.shape[0] == CONFIG["seq_len"]


@pytest.fixture(scope="module")
def pipeline():
    return VideoPipeline("hybrid-cnn-vit")


def test_predict_valid_distribution(pipeline):
    rng = np.random.default_rng(2)
    frames = rng.normal(0.0, 1.0, size=(30, 35)).astype(np.float32)
    result = pipeline.predict(frames)
    probs = np.asarray(result["probs"])
    assert probs.shape == (6,)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert result["label"] in CONFIG["label_names"]
    assert result["label_index"] == int(probs.argmax())


def test_predict_handles_nan_frames(pipeline):
    frames = np.full((10, 35), np.nan, dtype=np.float32)
    result = pipeline.predict(frames)
    assert np.isfinite(np.asarray(result["probs"])).all()
