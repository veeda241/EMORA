"""Demo runner: shows what the Cross-Modal Attention Transformer pipeline looks
like end-to-end for the MELD-based stress/burnout tracker."""

from __future__ import annotations

import torch
from models.common.compare import MODEL_REGISTRY, MODALITY_ORDER

PROJECT_DIM = 512  # shared embedding dimension fed into the Cross-Modal Transformer


def sample_shape(modality: str, model_key: str) -> torch.Tensor:
    """Return a fake batch that each stream would emit (before the projection)."""
    cfg = MODEL_REGISTRY[modality][model_key]
    in_dim = {"text": 768, "face": 512, "voice": 768}[modality]
    return torch.randn(2, in_dim)  # batch of 2


def embedding_shapes() -> str:
    out = ["# Modality embedding shapes (what the Cross-Modal Transformer consumes)",
           "",
           "Each encoder is trained independently, then a Linear + LayerNorm +",
           "GELU projection maps it to a uniform `PROJECT_DIM` vector before the",
           "cross-attention fusion.",
           "",
           "| Modality | Encoder | Pre-projection dim | Projected dim | Transformer input dim |",
           "|---|---|---|---|---|"]
    for mod in MODALITY_ORDER:
        for key, cfg in MODEL_REGISTRY[mod].items():
            pre = {"text": 768, "face": 512, "voice": 768}[mod]
            out += [f"| {mod.capitalize()} | {cfg['name']} | {pre} | {PROJECT_DIM} | {PROJECT_DIM} |"]
    out += [
        "",
        "## End-to-end flow",
        "",
        "```text",
        "MELD utterances                        MELD frames                    MELD audio",
        "  |  text transcripts                  |  video frames (MediaPipe     |  .wav ->",
        "  v                                    v                            v",
        " BERT/RoBERTa/DeBERTa                 MAE-DFER / EST / CNN+ViT       emotion2vec / WavLM / Wav2Vec2",
        "  |  [CLS]/mean pool                 |  GAP pool                   |  time-pooled block",
        "  v                                    v                            v",
        " 768-dim vector                     512-dim vector                 768-dim vector",
        "  |  Linear(768 -> 512)              |  Linear(512 -> 512)          |  Linear(768 -> 512)",
        "  v                                    v                            v",
        " 512-dim projected                  512-dim projected                512-dim projected",
        "  |  concat -> layer-norm            |                              |",
        "  v                                    v                            v",
        "  [v_face | v_text | v_voice] ------> Multi-head Cross-Attention ----> Stress score",
        "```",
        "",
        "## Sample tensor shapes",
        "",
    ]
    for mod in MODALITY_ORDER:
        for key in MODEL_REGISTRY[mod]:
            out += [
                f"### {mod.capitalize()} - {MODEL_REGISTRY[mod][key]['name']}",
                f"```python",
                f"model = {mod}_model  # e.g. Wav2Vec2Model",
                f"x = sample_{mod}(model, batch)           # (2, {768 if mod != 'face' else 512})",
                f"proj = nn.Sequential(",
                f"    nn.Linear({768 if mod != 'face' else 512}, {PROJECT_DIM}),",
                f"    nn.LayerNorm({PROJECT_DIM}),",
                f"    nn.GELU(),",
                f"    nn.Dropout(0.1),",
                f"    nn.Linear({PROJECT_DIM}, {PROJECT_DIM}),",
                f")",
                f"v = proj(x)                             # (2, {PROJECT_DIM})",
                f"print(v.shape)                          # torch.Size([2, {PROJECT_DIM}])",
                f"```",
            ]
    return "\n".join(out)


if __name__ == "__main__":
    print(embedding_shapes())
