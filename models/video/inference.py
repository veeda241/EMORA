"""Video model inference entry point."""
from __future__ import annotations
import pathlib

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent / "checkpoints"


def predict(video_path: str | pathlib.Path, model: str = "mae-dfer") -> dict:
    """Return emotion probabilities for a video clip.

    Args:
        video_path: Path to video file.
        model: One of 'mae-dfer', 'est', 'hybrid-cnn-vit'.

    Returns:
        Dict mapping emotion label to probability.
    """
    raise NotImplementedError("Video inference not yet implemented.")
