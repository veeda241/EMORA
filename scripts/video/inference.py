"""
inference.py
============
Run emotion prediction on a single video clip or image folder using either:
  - MAE-DFER (fine-tuned on DFEW)
  - CMU-MOSEI FACET baseline

Usage
-----
    # Predict from a video file using MAE-DFER
    python scripts/video/inference.py --model mae_dfer \
        --input path/to/clip.mp4 \
        --ckpt models/video/checkpoints/mae_dfer_dfew/best_fold1.pth

    # Predict from a folder of frame images (jpg/png)
    python scripts/video/inference.py --model mae_dfer \
        --input path/to/frames/ \
        --ckpt models/video/checkpoints/mae_dfer_dfew/best_fold1.pth

Output
------
    {
        "emotion": "happy",
        "label_id": 3,
        "probabilities": {"angry": 0.02, "disgust": 0.01, ...},
        "embedding_512": [...]   # for multimodal fusion
    }
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT  = pathlib.Path(__file__).resolve().parents[2]
MAE_DFER_ROOT = PROJECT_ROOT / "models" / "video" / "MAE-DFER"
sys.path.insert(0, str(MAE_DFER_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

LABEL_NAMES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
NUM_CLASSES = 6


def load_frames_from_folder(folder: pathlib.Path, num_frames: int = 16) -> torch.Tensor:
    """Load uniformly sampled frames from a folder of images."""
    from PIL import Image
    from torchvision import transforms

    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No jpg/png frames found in {folder}")

    # uniform sample
    indices = np.linspace(0, len(paths) - 1, num_frames, dtype=int)
    selected = [paths[i] for i in indices]

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames = [transform(Image.open(p).convert("RGB")) for p in selected]
    # shape: (T, C, H, W) → (1, C, T, H, W)
    video  = torch.stack(frames, dim=0)        # (T, C, H, W)
    video  = video.permute(1, 0, 2, 3)         # (C, T, H, W)
    return video.unsqueeze(0)                  # (1, C, T, H, W)


def load_frames_from_video(video_path: pathlib.Path, num_frames: int = 16) -> torch.Tensor:
    """Sample frames from a video file using decord."""
    from decord import VideoReader, cpu
    from PIL import Image
    from torchvision import transforms

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_raw = vr.get_batch(indices).asnumpy()   # (T, H, W, C) uint8

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames = [transform(Image.fromarray(f)) for f in frames_raw]
    video  = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
    return video.unsqueeze(0)


class MAEDFERWithProjection(nn.Module):
    """MAE-DFER + 512-dim projection head for multimodal fusion."""
    def __init__(self, base_model: nn.Module, hidden_dim: int = 768):
        super().__init__()
        self.backbone = base_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # backbone forward gives logits; we hook the pre-head features
        feats  = self.backbone.forward_features(x)   # (B, D)
        logits = self.backbone.head(feats)
        emb    = self.proj(feats)
        return logits, emb


def predict_mae_dfer(input_path: pathlib.Path, ckpt_path: pathlib.Path,
                     device: torch.device) -> dict:
    from train_mae_dfer import build_model

    import logging
    model_base = build_model(ckpt_path, NUM_CLASSES, 160, logging.getLogger())
    model = MAEDFERWithProjection(model_base)
    model.to(device).eval()

    if input_path.is_dir():
        video_tensor = load_frames_from_folder(input_path)
    else:
        video_tensor = load_frames_from_video(input_path)

    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        logits, emb = model(video_tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred  = int(probs.argmax())

    return {
        "emotion":       LABEL_NAMES[pred],
        "label_id":      pred,
        "probabilities": {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)},
        "embedding_512": emb[0].cpu().numpy().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Video FER inference")
    parser.add_argument("--model",  choices=["mae_dfer", "cmu_mosei_baseline"], required=True)
    parser.add_argument("--input",  required=True, help="Path to video file or frame folder")
    parser.add_argument("--ckpt",   default=None,  help="Path to model checkpoint")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = pathlib.Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    if args.model == "mae_dfer":
        if not args.ckpt:
            print("[ERROR] --ckpt required for mae_dfer")
            sys.exit(1)
        result = predict_mae_dfer(input_path, pathlib.Path(args.ckpt), device)
    else:
        print("[ERROR] cmu_mosei_baseline inference not implemented (feature-based, needs FACET extractor)")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
