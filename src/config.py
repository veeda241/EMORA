"""Central configuration for the multimodal emotion recognition pipeline.

Single source of truth for every dimension in the system. The DMAN and
Multi-view Gated Memory layers derive their sizes from `total_lstm_hidden`,
so changing a projection or hidden size here propagates everywhere.

The default profile uses the current CMU-MOSEI baseline:
- text:  300-dim GloVe word vectors per token (TimestampedWordVectors.csd)
- audio: 74-dim COVAREP acoustic frames (CMU_MOSEI_COVAREP.csd)
- video: 35-dim FACET 4.2 facial frames (CMU_MOSEI_VisualFacet42.csd)

``MODEL_CONFIGS["new_experiment"]`` describes the replacement feature space:
BERT embeddings, emotion2vec embeddings, and temporal ViT features. The
corresponding extracted feature files must be generated before training.
"""

from __future__ import annotations

_COMMON_CONFIG = {
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

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _with_derived_values(config: dict) -> dict:
    """Add dimensions derived by the MFN and resolve project-relative paths."""
    config = dict(config)
    config["total_lstm_hidden"] = (
        config["lstm_hidden_text"]
        + config["lstm_hidden_audio"]
        + config["lstm_hidden_video"]
    )
    config["fused_dim"] = config["total_lstm_hidden"] + config["mem_dim"]
    config["data_dir"] = str(PROJECT_ROOT / "data" / "processed" / "multimodal")
    config["model_dir"] = str(PROJECT_ROOT / "models" / "multimodal")
    return config


MODEL_CONFIGS = {
    "baseline": _with_derived_values(_COMMON_CONFIG),
    "new_experiment": _with_derived_values(
        {
            **_COMMON_CONFIG,
            "raw_text_dim": 768,   # BERT embedding
            "raw_audio_dim": 768,  # emotion2vec embedding
            "raw_video_dim": 512,  # temporal ViT feature
        }
    ),
}

# Existing datasets/checkpoints use the baseline profile.
CONFIG = MODEL_CONFIGS["baseline"]


def get_config(profile: str = "baseline") -> dict:
    """Return a copy of a named modality/fusion configuration profile."""
    try:
        return dict(MODEL_CONFIGS[profile])
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model profile {profile!r}; choose {valid}") from exc
