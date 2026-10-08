"""Experiment C pipeline: GloVe + COVAREP + temporal Hybrid CNN-ViT + MFN."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.video.train import HybridCNNViTModel
from src.config import CONFIG
from src.fusion.feature_fusion import MemoryFusionNetwork
from src.pipeline.multimodal_pipeline import ModalityEncoder

ROOT = Path(__file__).resolve().parents[2]
HYBRID_CHECKPOINT = ROOT / "models" / "video" / "runs_hybrid_cnn_vit" / "best_model.pt"
HYBRID_TEMPORAL_DIM = 256


class ExperimentCPipeline(nn.Module):
    """MFN with frozen temporal Hybrid CNN-ViT features; no broadcasting."""

    def __init__(self, cfg: dict | None = None, checkpoint: Path = HYBRID_CHECKPOINT):
        super().__init__()
        self.cfg = cfg or CONFIG
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Canonical Hybrid CNN-ViT checkpoint is missing: {checkpoint}"
            )

        self.hybrid_video = HybridCNNViTModel(
            in_dim=self.cfg["raw_video_dim"],
            seq_len=self.cfg["seq_len"],
            project_dim=512,
            num_classes=self.cfg["num_classes"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.hybrid_video.load_state_dict(
            state.get("model_state_dict", state), strict=True
        )
        for parameter in self.hybrid_video.parameters():
            parameter.requires_grad_(False)
        self.hybrid_video.eval()

        self.text_encoder = ModalityEncoder(
            self.cfg["raw_text_dim"], self.cfg["proj_text_dim"]
        )
        self.audio_encoder = ModalityEncoder(
            self.cfg["raw_audio_dim"], self.cfg["proj_audio_dim"]
        )
        self.video_encoder = ModalityEncoder(
            HYBRID_TEMPORAL_DIM, self.cfg["proj_video_dim"]
        )
        self.mfn = MemoryFusionNetwork(self.cfg)
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg["fused_dim"], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.cfg["num_classes"]),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The video encoder is a frozen feature extractor, including BatchNorm.
        self.hybrid_video.eval()
        return self

    def forward(
        self,
        text_raw: torch.Tensor,
        audio_raw: torch.Tensor,
        facet_seq: torch.Tensor,
        return_temporal: bool = False,
    ):
        batch_size, steps, _ = facet_seq.shape
        assert steps == self.cfg["seq_len"], (
            f"expected T={self.cfg['seq_len']} timesteps, got {steps}"
        )
        assert text_raw.shape == (batch_size, steps, self.cfg["raw_text_dim"])
        assert audio_raw.shape == (batch_size, steps, self.cfg["raw_audio_dim"])
        assert facet_seq.shape == (batch_size, steps, self.cfg["raw_video_dim"])

        text = self.text_encoder(text_raw)
        audio = self.audio_encoder(audio_raw)
        with torch.no_grad():
            video_temporal = self.hybrid_video.extract_temporal(facet_seq)
        assert video_temporal.shape == (
            batch_size, steps, HYBRID_TEMPORAL_DIM,
        ), f"unexpected temporal video shape: {tuple(video_temporal.shape)}"

        video = self.video_encoder(video_temporal.detach())
        assert video.shape == (batch_size, steps, self.cfg["proj_video_dim"])
        fused = self.mfn(text, audio, video)
        logits = self.classifier(fused)
        if return_temporal:
            return logits, video_temporal
        return logits
