"""Text model inference entry point."""
from __future__ import annotations
import pathlib

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent / "checkpoints"


def predict(text: str, model: str = "deberta-v3-base") -> dict:
    """Return emotion probabilities for a text utterance.

    Args:
        text: Raw utterance string.
        model: One of 'bert-base-uncased', 'roberta-base', 'deberta-v3-base'.

    Returns:
        Dict mapping emotion label to probability.
    """
    raise NotImplementedError("Text inference not yet implemented.")
