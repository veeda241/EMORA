"""Text model inference entry point.

The fine-tuned RoBERTa checkpoint is preferred.  A lexicon fallback is not
silently substituted here: callers receive a clear error when the checkpoint
has not been trained yet.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from src.pipeline.text_pipeline import TextPipeline

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _checkpoint_for(model: str) -> Path:
    checkpoint_dir = CHECKPOINT_DIR / f"runs_{model}"
    candidates = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
            return candidate
    raise FileNotFoundError(
        f"No trained weights found for {model!r} under {checkpoint_dir}. "
        "Run `python models/text/train.py --model roberta-base` first."
    )


def predict(text: str, model: str = "roberta-base") -> dict[str, Any]:
    """Return emotion probabilities for a text utterance.

    Args:
        text: Raw utterance string.
        model: One of 'bert-base-uncased', 'roberta-base', 'deberta-v3-base'.

    Returns:
        Dict mapping emotion label to probability.
    """
    checkpoint = _checkpoint_for(model)
    return TextPipeline(checkpoint=checkpoint).predict(text)
