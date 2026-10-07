#!/usr/bin/env python3
"""PyTorch Dataset classes for the CMU-MOSEI audio-only processed dataset.

This module reads from:
- data/processed/cmu_mosei/audio/*.npz
- data/processed/cmu_mosei/labels/*.npy
- data/processed/cmu_mosei/normalization.json

It does NOT train anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class CMUMOSEIAudioDataset(Dataset):
    """COVAREP audio dataset.

    Each sample returns:
    - features: (148,) float32 tensor of normalized [mean, std] over utterance COVAREP frames
    - label_scores: (6,) float32 tensor of continuous emotion scores in the order
      [anger, disgust, fear, happy, sad, surprise]
    - label_index: int64 scalar of the hard argmax emotion label in the same order
    - metadata: dict with utterance_id, clip_id, split
    """

    EMOTION_ORDER = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

    def __init__(
        self,
        root: Path | str,
        split: str,
        norm_path: Optional[Path | str] = None,
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        if norm_path is None:
            norm_path = self.root / "normalization.json"
        self.norm = json.loads(norm_path.read_text(encoding="utf-8"))
        audio_path = self.root / "audio" / f"{split}.npz"
        labels_path = self.root / "labels" / f"{split}.npy"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        audio_data = np.load(audio_path, allow_pickle=False)
        if "feats" not in audio_data:
            raise ValueError(f"expected 'feats' key in {audio_path}")
        self.feats = np.ascontiguousarray(audio_data["feats"], dtype=np.float32)
        self.labels = np.ascontiguousarray(np.load(labels_path, allow_pickle=False), dtype=np.float32)
        if self.feats.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"feature/label count mismatch for {split}: {self.feats.shape[0]} vs {self.labels.shape[0]}"
            )
        if self.feats.shape[1] != self.norm["audio_representation_dim"]:
            raise ValueError(
                f"feature dim mismatch for {split}: {self.feats.shape[1]} vs {self.norm['audio_representation_dim']}"
            )

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = torch.from_numpy(self.feats[index])
        y_scores = torch.from_numpy(self.labels[index])
        y_index = int(y_scores.argmax().item())
        sample = {
            "features": x,
            "label_scores": y_scores,
            "label_index": torch.tensor(y_index, dtype=torch.long),
            "metadata": {
                "split": self.split,
                "index": index,
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class CMUMOSEITextDataset(Dataset):
    """Placeholder text dataset.

    The processed text data is not built in this audio-only stage.
    This class exists so multimodal code can reference a uniform text interface later.
    """

    def __init__(self, root: Path | str, split: str):
        self.root = Path(root)
        self.split = split

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("CMUMOSEITextDataset is not built yet in this stage.")


class CMUMOSEIVideoDataset(Dataset):
    """Placeholder video dataset.

    The processed video data is not built in this audio-only stage.
    This class exists so multimodal code can reference a uniform video interface later.
    """

    def __init__(self, root: Path | str, split: str):
        self.root = Path(root)
        self.split = split

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("CMUMOSEIVideoDataset is not built yet in this stage.")


class CMUMOSEIMultimodalDataset(Dataset):
    """Multimodal dataset that currently only supports audio.

    In a later stage this class will combine text + audio + video.
    For now it mirrors the audio dataset so downstream code can keep the same interface.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        norm_path: Optional[Path | str] = None,
        transform=None,
    ):
        self.audio = CMUMOSEIAudioDataset(root=root, split=split, norm_path=norm_path, transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {
            "audio": self.audio[index]["features"],
            "label_scores": self.audio[index]["label_scores"],
            "label_index": self.audio[index]["label_index"],
            "metadata": {
                "split": self.audio[index]["metadata"]["split"],
                "index": index,
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def audio_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed" / "cmu_mosei"
