"""Central configuration for the multimodal emotion recognition pipeline.

Single source of truth for every dimension in the system. The DMAN and
Multi-view Gated Memory layers derive their sizes from `total_lstm_hidden`,
so changing a projection or hidden size here propagates everywhere.

Raw input dims match the CMU-MOSEI processed features:
- text:  300-dim GloVe word vectors per token (TimestampedWordVectors.csd)
- audio: 74-dim COVAREP acoustic frames (CMU_MOSEI_COVAREP.csd)
- video: 35-dim FACET 4.2 facial frames (CMU_MOSEI_VisualFacet42.csd)
"""

from __future__ import annotations

CONFIG = {
    # 1. Raw input dimensions (from CMU-MOSEI computational sequences)
    "raw_text_dim": 300,
    "raw_audio_dim": 74,
    "raw_video_dim": 35,

    # 2. Target projection dimensions (architecture diagram)
    "proj_text_dim": 256,
    "proj_audio_dim": 128,
    "proj_video_dim": 128,

    # 3. System of LSTMs hidden dimensions (internal MFN sizes)
    "lstm_hidden_text": 128,
    "lstm_hidden_audio": 64,
    "lstm_hidden_video": 64,

    # 4. Central Multi-view Gated Memory configuration
    "mem_dim": 128,
    "attn_dim": 64,

    # 5. Output configuration
    "num_classes": 6,
    "label_names": ["angry", "disgust", "fearful", "happy", "sad", "surprised"],

    # 6. Temporal alignment (shared sequence length across modalities)
    "seq_len": 16,

    # 7. Training defaults
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 30,
}

CONFIG["total_lstm_hidden"] = (
    CONFIG["lstm_hidden_text"]
    + CONFIG["lstm_hidden_audio"]
    + CONFIG["lstm_hidden_video"]
)

# Fused representation: last unimodal hidden states + central memory
CONFIG["fused_dim"] = CONFIG["total_lstm_hidden"] + CONFIG["mem_dim"]

# Paths (relative to repo root)
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG["data_dir"] = str(PROJECT_ROOT / "data" / "processed" / "multimodal")
CONFIG["model_dir"] = str(PROJECT_ROOT / "models" / "multimodal")
