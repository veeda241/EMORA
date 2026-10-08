"""LSTM fusion baseline: per-timestep modality projections -> concat -> LSTM -> classifier.

Controlled comparison against the MFN baseline (src/pipeline/multimodal_pipeline.py).
Identical modality encoders and projection dimensions to MFN; the only changed
variable is the fusion mechanism: a single uni-directional LSTM over the
concatenated per-timestep features instead of the Memory Fusion Network.

Tensor flow (batch B, T = seq_len timesteps):
    text  (B, 16, 300) -> ModalityEncoder -> (B, 16, 256) ─┐
    audio (B, 16,  74) -> ModalityEncoder -> (B, 16, 128) ─┼-> concat (B, 16, 512)
    video (B, 16,  35) -> ModalityEncoder -> (B, 16, 128) ─┘
    -> nn.LSTM(512 -> 128, 1 layer, batch_first=True)      -> (B, 16, 128)
    -> last hidden state                                   -> (B, 128)
    -> Dropout(0.3) -> Linear(128, 6)                      -> (B, 6) logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CONFIG
from src.pipeline.multimodal_pipeline import ModalityEncoder

LSTM_HIDDEN = 128


class LSTMFusionPipeline(nn.Module):
    """Full model: three MFN-style encoders + temporal LSTM fusion + classifier."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or CONFIG

        self.text_encoder = ModalityEncoder(self.cfg["raw_text_dim"], self.cfg["proj_text_dim"])
        self.audio_encoder = ModalityEncoder(self.cfg["raw_audio_dim"], self.cfg["proj_audio_dim"])
        self.video_encoder = ModalityEncoder(self.cfg["raw_video_dim"], self.cfg["proj_video_dim"])

        fused_step_dim = (
            self.cfg["proj_text_dim"] + self.cfg["proj_audio_dim"] + self.cfg["proj_video_dim"]
        )
        self.lstm = nn.LSTM(
            input_size=fused_step_dim,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(LSTM_HIDDEN, self.cfg["num_classes"])

    def forward(
        self,
        text_raw: torch.Tensor,
        audio_raw: torch.Tensor,
        video_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Args: (B, T, raw_dim) per modality. Returns (B, num_classes) logits."""
        b, steps, _ = text_raw.shape
        assert steps == self.cfg["seq_len"], (
            f"expected seq_len={self.cfg['seq_len']} timesteps, got {steps}"
        )
        assert audio_raw.shape[1] == steps and video_raw.shape[1] == steps, (
            "modality sequences must share the same number of timesteps"
        )

        t = self.text_encoder(text_raw)
        a = self.audio_encoder(audio_raw)
        v = self.video_encoder(video_raw)
        fused = torch.cat([t, a, v], dim=-1)

        lstm_out, (hidden, _) = self.lstm(fused)
        assert lstm_out.shape == (b, steps, LSTM_HIDDEN)

        return self.classifier(self.dropout(hidden[-1]))
