"""Feature-level fusion: the Memory Fusion Network (MFN) core.

Implements the three MFN components from Zadeh et al. 2018:

1. System of LSTMs — one LSTMCell per modality, learning view-specific
   dynamics in isolation.
2. Delta-Memory Attention Network (DMAN) — cross-view attention over the
   concatenated hidden states of all views, computed from the delta
   (previous vs. current) hidden states.
3. Multi-view Gated Memory — a central memory vector gamma updated at every
   timestep via update/propose gates, accumulating cross-view interactions.

All dimensions are derived from the central CONFIG so a change to any
projection size automatically reconfigures the dependent layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CONFIG


class MemoryFusionNetwork(nn.Module):
    """Core MFN engine: System of LSTMs + DMAN + Multi-view Gated Memory."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or CONFIG

        # 1. System of LSTMs (isolated view-specific dynamics)
        self.lstm_text = nn.LSTMCell(self.cfg["proj_text_dim"], self.cfg["lstm_hidden_text"])
        self.lstm_audio = nn.LSTMCell(self.cfg["proj_audio_dim"], self.cfg["lstm_hidden_audio"])
        self.lstm_video = nn.LSTMCell(self.cfg["proj_video_dim"], self.cfg["lstm_hidden_video"])

        # 2. Delta-Memory Attention Network (DMAN)
        dman_input_dim = self.cfg["total_lstm_hidden"] * 2
        self.dman_net = nn.Sequential(
            nn.Linear(dman_input_dim, self.cfg["attn_dim"]),
            nn.ReLU(),
            nn.Linear(self.cfg["attn_dim"], self.cfg["total_lstm_hidden"]),
            nn.Softmax(dim=-1),
        )

        # 3. Multi-view Gated Memory
        mem_input_dim = self.cfg["total_lstm_hidden"] + self.cfg["mem_dim"]
        self.w_u = nn.Linear(mem_input_dim, self.cfg["mem_dim"])  # update gate
        self.w_g = nn.Linear(mem_input_dim, self.cfg["mem_dim"])  # propose gate

    def forward(
        self,
        text_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        video_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse time-aligned modality sequences.

        Args:
            text_seq:  (B, T, proj_text_dim)
            audio_seq: (B, T, proj_audio_dim)
            video_seq: (B, T, proj_video_dim)

        Returns:
            (B, fused_dim) — last unimodal hidden states + central memory gamma.
        """
        batch_size = text_seq.size(0)
        time_steps = text_seq.size(1)
        device = text_seq.device
        cfg = self.cfg

        h_t = torch.zeros(batch_size, cfg["lstm_hidden_text"], device=device)
        c_t = torch.zeros(batch_size, cfg["lstm_hidden_text"], device=device)
        h_a = torch.zeros(batch_size, cfg["lstm_hidden_audio"], device=device)
        c_a = torch.zeros(batch_size, cfg["lstm_hidden_audio"], device=device)
        h_v = torch.zeros(batch_size, cfg["lstm_hidden_video"], device=device)
        c_v = torch.zeros(batch_size, cfg["lstm_hidden_video"], device=device)
        gamma = torch.zeros(batch_size, cfg["mem_dim"], device=device)

        for t in range(time_steps):
            prev_h_all = torch.cat([h_t, h_a, h_v], dim=-1)

            # Step A: advance System of LSTMs
            h_t, c_t = self.lstm_text(text_seq[:, t, :], (h_t, c_t))
            h_a, c_a = self.lstm_audio(audio_seq[:, t, :], (h_a, c_a))
            h_v, c_v = self.lstm_video(video_seq[:, t, :], (h_v, c_v))

            # Step B: DMAN attention over the delta of hidden states
            curr_h_all = torch.cat([h_t, h_a, h_v], dim=-1)
            dman_input = torch.cat([prev_h_all, curr_h_all], dim=-1)
            attn_weights = self.dman_net(dman_input)
            attended = attn_weights * curr_h_all

            # Step C: Multi-view Gated Memory update
            memory_input = torch.cat([attended, gamma], dim=-1)
            gate_u = torch.sigmoid(self.w_u(memory_input))
            gate_g = torch.tanh(self.w_g(memory_input))
            gamma = (1 - gate_u) * gamma + gate_u * gate_g

        final_unimodal = torch.cat([h_t, h_a, h_v], dim=-1)
        return torch.cat([final_unimodal, gamma], dim=-1)
