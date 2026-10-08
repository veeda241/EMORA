"""Master multimodal emotion pipeline: encoders -> MFN fusion -> prediction.

Architecture (per project design):
    TEXT  (300-d GloVe tokens)  -> Linear/LayerNorm -> 256-D
    AUDIO (74-d COVAREP frames) -> Linear/LayerNorm -> 128-D
    VIDEO (35-d FACET frames)   -> Linear/LayerNorm -> 128-D
        -> Memory Fusion Network (System of LSTMs + DMAN + Gated Memory)
        -> classifier -> 6-class emotion logits

All inputs are time-aligned to CONFIG["seq_len"] timesteps upstream (see
models/preprocessing/build_cmumosei_multimodal_dataset.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG
from src.fusion.feature_fusion import MemoryFusionNetwork


class ModalityEncoder(nn.Module):
    """Project one modality's raw per-timestep features to its target dim."""

    def __init__(self, raw_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, raw_dim) -> (B, T, proj_dim)
        return self.proj(x)


class MultimodalEmotionPipeline(nn.Module):
    """Full model: three encoders + MFN core + emotion classifier."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or CONFIG

        self.text_encoder = ModalityEncoder(self.cfg["raw_text_dim"], self.cfg["proj_text_dim"])
        self.audio_encoder = ModalityEncoder(self.cfg["raw_audio_dim"], self.cfg["proj_audio_dim"])
        self.video_encoder = ModalityEncoder(self.cfg["raw_video_dim"], self.cfg["proj_video_dim"])

        self.mfn = MemoryFusionNetwork(self.cfg)

        self.classifier = nn.Sequential(
            nn.Linear(self.cfg["fused_dim"], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.cfg["num_classes"]),
        )

    def forward(
        self,
        text_raw: torch.Tensor,
        audio_raw: torch.Tensor,
        video_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Args: (B, T, raw_dim) per modality. Returns (B, num_classes) logits."""
        t = self.text_encoder(text_raw)
        a = self.audio_encoder(audio_raw)
        v = self.video_encoder(video_raw)
        fused = self.mfn(t, a, v)
        return self.classifier(fused)

    @torch.no_grad()
    def predict_emotion(
        self,
        text_raw: torch.Tensor,
        audio_raw: torch.Tensor,
        video_raw: torch.Tensor,
    ) -> dict:
        """Inference helper returning logits, softmax probs, and label name."""
        logits = self.forward(text_raw, audio_raw, video_raw)
        probs = F.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        names = self.cfg["label_names"]
        return {
            "logits": logits,
            "probs": probs,
            "label_index": idx,
            "label": [names[i] for i in idx.tolist()],
            "confidence": conf,
        }
