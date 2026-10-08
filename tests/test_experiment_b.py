"""Integrity tests for the controlled Experiment B MFN path."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.pipeline.experiment_b_pipeline import ExperimentBPipeline, HYBRID_EMBED_DIM

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "video" / "runs_hybrid_cnn_vit" / "best_model.pt").exists(),
    reason="canonical Hybrid CNN-ViT checkpoint is unavailable",
)


def test_experiment_b_uses_frozen_512d_embedding_and_backpropagates():
    model = ExperimentBPipeline()
    model.train()
    assert not model.hybrid_video.training
    assert all(not parameter.requires_grad for parameter in model.hybrid_video.parameters())

    text = torch.randn(2, CONFIG["seq_len"], CONFIG["raw_text_dim"])
    audio = torch.randn(2, CONFIG["seq_len"], CONFIG["raw_audio_dim"])
    facet = torch.randn(2, CONFIG["seq_len"], CONFIG["raw_video_dim"])
    logits, embedding = model(text, audio, facet, return_embedding=True)

    assert embedding.shape == (2, HYBRID_EMBED_DIM)
    assert logits.shape == (2, CONFIG["num_classes"])
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()
    assert model.video_encoder.proj[0].weight.grad is not None
    assert model.mfn.lstm_video.weight_ih.grad is not None
    assert model.classifier[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.hybrid_video.parameters())


def test_experiment_b_splits_have_disjoint_utterance_ids():
    split_dir = ROOT / "data" / "processed" / "multimodal"
    seen = set()
    for split in ("train", "validation", "test"):
        with np.load(split_dir / f"{split}_data.npz") as data:
            ids = data["utterance_ids"].tolist()
            assert len(ids) == len(set(ids))
            assert not seen.intersection(ids)
            seen.update(ids)
