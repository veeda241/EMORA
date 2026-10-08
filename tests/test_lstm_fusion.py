"""Integrity tests for the LSTM fusion baseline (controlled comparison vs MFN)."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.pipeline.lstm_fusion_pipeline import LSTMFusionPipeline

LSTM_CHECKPOINT = ROOT / "models" / "multimodal" / "runs_lstm" / "best_model.pt"
MFN_CHECKPOINT = ROOT / "models" / "multimodal" / "runs_mfn" / "best_model.pt"
ABLATION_CSV = ROOT / "reports" / "ablation_results.csv"

BATCH = 2
STEPS = CONFIG["seq_len"]
FUSED_STEP_DIM = (
    CONFIG["proj_text_dim"] + CONFIG["proj_audio_dim"] + CONFIG["proj_video_dim"]
)

# Pinned measured rows — the controlled-experiment baseline must never drift.
PINNED_ROWS = {
    "MFN": ("0.47177505902554195", "0.27173548383205287", "0.46846071641749654"),
    "GloVe + COVAREP + HybridCNNViT + MFN": (
        "0.4518136939257351",
        "0.23786240106302314",
        "0.4333125765433986",
    ),
    "GloVe + COVAREP + temporal HybridCNNViT + MFN": (
        "0.4599699506331831",
        "0.2398372623178865",
        "0.4495704300035891",
    ),
}


def sample_inputs():
    text = torch.randn(BATCH, STEPS, CONFIG["raw_text_dim"])
    audio = torch.randn(BATCH, STEPS, CONFIG["raw_audio_dim"])
    video = torch.randn(BATCH, STEPS, CONFIG["raw_video_dim"])
    return text, audio, video


def test_input_shapes_and_output_contract():
    model = LSTMFusionPipeline()
    assert model.lstm.input_size == FUSED_STEP_DIM
    assert model.lstm.hidden_size == 128
    assert model.lstm.num_layers == 1
    assert not model.lstm.bidirectional

    text, audio, video = sample_inputs()
    logits = model(text, audio, video)
    assert logits.shape == (BATCH, CONFIG["num_classes"])

    fused = torch.cat(
        [
            model.text_encoder(text),
            model.audio_encoder(audio),
            model.video_encoder(video),
        ],
        dim=-1,
    )
    assert fused.shape == (BATCH, STEPS, FUSED_STEP_DIM)


def test_forward_produces_finite_values():
    model = LSTMFusionPipeline()
    model.eval()
    with torch.no_grad():
        logits = model(*sample_inputs())
    assert torch.isfinite(logits).all()


def test_temporal_alignment_is_preserved():
    model = LSTMFusionPipeline()
    model.eval()
    text, audio, video = sample_inputs()
    with torch.no_grad():
        baseline = model(text, audio, video)

        # Reversing time must change the output: the LSTM consumes the sequence
        # temporally rather than pooling it away.
        reversed_logits = model(text.flip(dims=[1]), audio.flip(dims=[1]), video.flip(dims=[1]))
        assert not torch.allclose(baseline, reversed_logits)

    with pytest.raises(AssertionError):
        model(text[:, :8, :], audio[:, :8, :], video[:, :8, :])


def test_backpropagation_reaches_all_parameters():
    model = LSTMFusionPipeline()
    logits = model(*sample_inputs())
    logits.square().mean().backward()
    assert model.text_encoder.proj[0].weight.grad is not None
    assert model.audio_encoder.proj[0].weight.grad is not None
    assert model.video_encoder.proj[0].weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.classifier.weight.grad is not None


def test_splits_have_disjoint_utterance_ids():
    split_dir = ROOT / "data" / "processed" / "multimodal"
    seen = set()
    for split in ("train", "validation", "test"):
        with np.load(split_dir / f"{split}_data.npz") as data:
            ids = data["utterance_ids"].tolist()
            assert len(ids) == len(set(ids))
            assert not seen.intersection(ids)
            seen.update(ids)


def test_mfn_baseline_remains_unchanged():
    from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline

    model = MultimodalEmotionPipeline(CONFIG)
    state = torch.load(MFN_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)

    with open(ABLATION_CSV, newline="", encoding="utf-8") as handle:
        import csv

        rows = {row["model"]: row for row in csv.DictReader(handle)}
    for name, (acc, macro_f1, weighted_f1) in PINNED_ROWS.items():
        assert rows[name]["accuracy"] == acc, f"{name} accuracy drifted"
        assert rows[name]["macro_f1"] == macro_f1, f"{name} macro_f1 drifted"
        assert rows[name]["weighted_f1"] == weighted_f1, f"{name} weighted_f1 drifted"


@pytest.mark.skipif(not LSTM_CHECKPOINT.exists(), reason="LSTM checkpoint not trained yet")
def test_lstm_checkpoint_loads_strict_and_reproduces_logits():
    model = LSTMFusionPipeline()
    state = torch.load(LSTM_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)

    model.eval()
    reference = LSTMFusionPipeline()
    reference.load_state_dict(
        torch.load(LSTM_CHECKPOINT, map_location="cpu", weights_only=True)
    )
    reference.eval()

    torch.manual_seed(0)
    inputs = sample_inputs()
    with torch.no_grad():
        assert torch.equal(model(*inputs), reference(*inputs))


def test_training_reduces_loss_on_small_batch():
    torch.manual_seed(0)
    model = LSTMFusionPipeline()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = sample_inputs()
    targets = torch.tensor([0, 5])
    model.train()
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = criterion(model(*inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
