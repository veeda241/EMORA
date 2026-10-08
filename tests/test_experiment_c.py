"""Integrity tests for the controlled Experiment C MFN path."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.pipeline.experiment_c_pipeline import (
    ExperimentCPipeline,
    HYBRID_CHECKPOINT,
    HYBRID_TEMPORAL_DIM,
)

pytestmark = pytest.mark.skipif(
    not HYBRID_CHECKPOINT.exists(),
    reason="canonical Hybrid CNN-ViT checkpoint is unavailable",
)

BATCH = 2
STEPS = CONFIG["seq_len"]


def build_hybrid():
    from models.video.train import HybridCNNViTModel

    return HybridCNNViTModel(
        in_dim=CONFIG["raw_video_dim"],
        seq_len=CONFIG["seq_len"],
        project_dim=512,
        num_classes=CONFIG["num_classes"],
    )


def sample_inputs():
    text = torch.randn(BATCH, STEPS, CONFIG["raw_text_dim"])
    audio = torch.randn(BATCH, STEPS, CONFIG["raw_audio_dim"])
    facet = torch.randn(BATCH, STEPS, CONFIG["raw_video_dim"])
    return text, audio, facet


def test_extract_temporal_returns_pre_pooling_sequence():
    model = build_hybrid()
    temporal = model.extract_temporal(torch.randn(BATCH, STEPS, CONFIG["raw_video_dim"]))
    assert temporal.shape == (BATCH, STEPS, HYBRID_TEMPORAL_DIM)


def test_hybrid_forward_contract_unchanged():
    model = build_hybrid()
    logits, embedding = model(torch.randn(BATCH, STEPS, CONFIG["raw_video_dim"]))
    assert logits.shape == (BATCH, CONFIG["num_classes"])
    assert embedding.shape == (BATCH, 512)


def test_canonical_checkpoint_loads_strict():
    model = build_hybrid()
    state = torch.load(HYBRID_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)


def test_video_projection_returns_temporal_128():
    model = ExperimentCPipeline()
    facet = torch.randn(BATCH, STEPS, CONFIG["raw_video_dim"])
    with torch.no_grad():
        temporal = model.hybrid_video.extract_temporal(facet)
    projected = model.video_encoder(temporal)
    assert projected.shape == (BATCH, STEPS, CONFIG["proj_video_dim"])
    assert torch.isfinite(projected).all()


def test_experiment_c_end_to_end_shapes_frozen_and_backpropagates():
    model = ExperimentCPipeline()
    model.train()
    assert not model.hybrid_video.training
    assert all(not parameter.requires_grad for parameter in model.hybrid_video.parameters())

    text, audio, facet = sample_inputs()
    logits, temporal = model(text, audio, facet, return_temporal=True)

    assert temporal.shape == (BATCH, STEPS, HYBRID_TEMPORAL_DIM)
    assert logits.shape == (BATCH, CONFIG["num_classes"])
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()
    assert model.video_encoder.proj[0].weight.grad is not None
    assert model.mfn.lstm_video.weight_ih.grad is not None
    assert model.classifier[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.hybrid_video.parameters())


def test_experiment_c_rejects_wrong_temporal_length():
    model = ExperimentCPipeline()
    text, audio, facet = sample_inputs()
    with pytest.raises(AssertionError):
        model(text, audio, facet[:, :8, :])


def test_experiment_c_splits_have_disjoint_utterance_ids():
    split_dir = ROOT / "data" / "processed" / "multimodal"
    seen = set()
    for split in ("train", "validation", "test"):
        with np.load(split_dir / f"{split}_data.npz") as data:
            ids = data["utterance_ids"].tolist()
            assert len(ids) == len(set(ids))
            assert not seen.intersection(ids)
            seen.update(ids)
