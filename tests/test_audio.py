"""Tests for the audio emotion recognition pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.audio_pipeline import AudioPipeline  # noqa: E402

CKPT = ROOT / "models" / "audio" / "runs_emotion2vec" / "model_state_dict.pt"

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="trained emotion2vec head not available"
)


@pytest.fixture(scope="module")
def pipeline():
    return AudioPipeline("emotion2vec")


def test_labels_match_shared_vocabulary(pipeline):
    assert sorted(pipeline.label_names) == sorted(
        ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
    )


def test_predict_valid_distribution(pipeline):
    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.1, 16000).astype(np.float32)
    result = pipeline.predict(wav)
    probs = np.asarray(result["probs_array"])
    assert probs.shape == (6,)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert result["label"] in pipeline.label_names


def test_predict_rejects_2d_waveform(pipeline):
    with pytest.raises(ValueError):
        pipeline.predict(np.zeros((2, 16000), dtype=np.float32))


def test_unknown_model_key_raises():
    with pytest.raises(ValueError):
        AudioPipeline("not-a-model")
