"""Experiment B pipeline: GloVe + COVAREP + frozen Hybrid CNN-ViT + MFN."""

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
HYBRID_EMBED_DIM = 512


class ExperimentBPipeline(nn.Module):
    """MFN with unchanged text/audio paths and frozen Hybrid video embeddings."""

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
            project_dim=HYBRID_EMBED_DIM,
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
            HYBRID_EMBED_DIM, self.cfg["proj_video_dim"]
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
        return_embedding: bool = False,
    ):
        text = self.text_encoder(text_raw)
        audio = self.audio_encoder(audio_raw)
        with torch.no_grad():
            _, video_embedding = self.hybrid_video(facet_seq)
        # Hybrid CNN-ViT exposes one segment embedding (B, 512), while MFN
        # consumes aligned sequences. The segment representation is therefore
        # broadcast across the existing 16-step MFN clock; no samples are
        # reordered and no classification logits enter the fusion path.
        video_embedding_seq = video_embedding.unsqueeze(1).expand(
            -1, self.cfg["seq_len"], -1
        )
        video = self.video_encoder(video_embedding_seq.detach())
        fused = self.mfn(text, audio, video)
        logits = self.classifier(fused)
        if return_embedding:
            return logits, video_embedding
        return logits
