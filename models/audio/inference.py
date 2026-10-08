"""Audio model inference entry point."""
from __future__ import annotations
import pathlib

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent / "checkpoints"


def predict(audio_path: str | pathlib.Path, model: str = "emotion2vec") -> dict:
    """Return emotion probabilities for a single audio file.

    Args:
        audio_path: Path to .wav/.flac audio file.
        model: One of 'wav2vec2', 'wavlm', 'emotion2vec'.

    Returns:
        Dict mapping emotion label to probability.
    """
    raise NotImplementedError("Audio inference not yet implemented.")
