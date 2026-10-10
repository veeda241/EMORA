"""Integration tests for the multimodal emotion recognition pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, get_config  # noqa: E402
from src.fusion.emotion_fusion import EmotionFusion  # noqa: E402
from src.fusion.feature_fusion import MemoryFusionNetwork  # noqa: E402
from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline  # noqa: E402


def test_config_consistency():
    assert CONFIG["total_lstm_hidden"] == sum(
        CONFIG[k] for k in ("lstm_hidden_text", "lstm_hidden_audio", "lstm_hidden_video")
    )
    assert CONFIG["fused_dim"] == CONFIG["total_lstm_hidden"] + CONFIG["mem_dim"]
    assert len(CONFIG["label_names"]) == CONFIG["num_classes"]


def test_new_experiment_modality_dimensions():
    cfg = get_config("new_experiment")
    assert (cfg["raw_text_dim"], cfg["raw_audio_dim"], cfg["raw_video_dim"]) == (
        768,
        768,
        512,
    )
    assert cfg["fused_dim"] == cfg["total_lstm_hidden"] + cfg["mem_dim"]


def test_mfn_forward_shapes():
    b, t = 4, CONFIG["seq_len"]
    mfn = MemoryFusionNetwork(CONFIG)
    fused = mfn(
        torch.randn(b, t, CONFIG["proj_text_dim"]),
        torch.randn(b, t, CONFIG["proj_audio_dim"]),
        torch.randn(b, t, CONFIG["proj_video_dim"]),
    )
    assert fused.shape == (b, CONFIG["fused_dim"])


def test_mfn_gradients_flow():
    mfn = MemoryFusionNetwork(CONFIG)
    fused = mfn(
        torch.randn(2, CONFIG["seq_len"], CONFIG["proj_text_dim"]),
        torch.randn(2, CONFIG["seq_len"], CONFIG["proj_audio_dim"]),
        torch.randn(2, CONFIG["seq_len"], CONFIG["proj_video_dim"]),
    )
    fused.sum().backward()
    grads = [p.grad for p in mfn.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_full_pipeline_forward():
    pipeline = MultimodalEmotionPipeline(CONFIG)
    logits = pipeline(
        torch.randn(2, CONFIG["seq_len"], CONFIG["raw_text_dim"]),
        torch.randn(2, CONFIG["seq_len"], CONFIG["raw_audio_dim"]),
        torch.randn(2, CONFIG["seq_len"], CONFIG["raw_video_dim"]),
    )
    assert logits.shape == (2, CONFIG["num_classes"])


def test_emotion_fusion_weights_and_disagreement():
    fusion = EmotionFusion(weights={"text": 1.0, "audio": 0.0, "video": 0.0})
    p_text = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
    p_video = np.array([0.05, 0.05, 0.05, 0.05, 0.7, 0.1])
    out = fusion.fuse({"text": p_text, "video": p_video})
    # text-only weight => fused equals text
    assert np.allclose(out["probs"], p_text, atol=1e-8)
    assert out["disagreement"] > 0.0


def test_emotion_fusion_single_modality():
    fusion = EmotionFusion()
    p = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
    out = fusion.fuse({"audio": p})
    assert np.allclose(out["probs"], p, atol=1e-8)
    assert out["disagreement"] == 0.0
    assert out["modalities_used"] == ["audio"]
