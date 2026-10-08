"""Cross-modality and cross-model comparison utilities.

Evaluates the unimodal video models (models/video/runs_*), the MFN fusion
model (models/multimodal/runs_mfn), and simple baselines on the aligned
CMU-MOSEI test split, then renders a side-by-side table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, majority_baseline  # noqa: E402

SHARED_EMOTION_CLASSES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]


def load_multimodal_test() -> dict:
    data = np.load(ROOT / "data" / "processed" / "multimodal" / "test_data.npz")
    return {k: data[k] for k in ("text", "audio", "video", "labels", "utterance_ids")}


def evaluate_mfn(device: str = "cuda") -> dict:
    """Evaluate the trained MFN on the multimodal test split."""
    import torch

    from src.config import CONFIG
    from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline

    ckpt = ROOT / "models" / "multimodal" / "runs_mfn" / "best_model.pt"
    if not ckpt.exists():
        return {"name": "MFN (text+audio+video)", "error": "no checkpoint"}

    test = load_multimodal_test()
    pipeline = MultimodalEmotionPipeline(CONFIG)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    pipeline.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    pipeline.to(device).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(test["labels"]), 256):
            batch = [
                torch.tensor(test[m][i : i + 256], dtype=torch.float32, device=device)
                for m in ("text", "audio", "video")
            ]
            preds.extend(pipeline(*batch).argmax(dim=-1).cpu().tolist())
    return {"name": "MFN (text+audio+video)", **compute_metrics(test["labels"], preds, SHARED_EMOTION_CLASSES)}


def evaluate_video_models(device: str = "cuda") -> list[dict]:
    """Evaluate the winning video model on the video-only test split."""
    import torch

    from models.video.train import build_model

    results = []
    video_dir = ROOT / "data" / "processed" / "video" / "video_fer"
    if not (video_dir / "test_data.npz").exists():
        return results

    data = np.load(video_dir / "test_data.npz")
    x, y = data["features"], data["labels"]

    # mae-dfer and est are archived under archive/old_models/video/; the
    # hybrid-cnn-vit is the production video model.
    registry = {"hybrid-cnn-vit": "hybrid_cnn_vit"}
    for key, subdir in registry.items():
        ckpt = ROOT / "models" / "video" / f"runs_{subdir}" / "best_model.pt"
        if not ckpt.exists():
            continue
        model = build_model(key, in_dim=x.shape[-1], seq_len=x.shape[1]).to(device).eval()
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                batch = torch.tensor(x[i : i + 256], dtype=torch.float32, device=device)
                preds.extend(model(batch)[0].argmax(dim=-1).cpu().tolist())
        results.append({"name": f"Video {key}", **compute_metrics(y, preds, SHARED_EMOTION_CLASSES)})
    return results


def compare_all(device: str = "cuda") -> str:
    """Build the full comparison table: baselines, unimodal video, MFN."""
    test = load_multimodal_test()
    rows = [{"name": "Majority baseline", **majority_baseline(test["labels"], SHARED_EMOTION_CLASSES), "macro_f1": 0.0}]
    rows += evaluate_video_models(device)
    rows += [evaluate_mfn(device)]

    header = f"{'Model':<28} {'Acc':>7} {'Macro F1':>9} {'W-F1':>7} {'N':>6}"
    lines = [header, "-" * len(header)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['name']:<28} {r['error']}")
            continue
        lines.append(
            f"{r['name']:<28} {r['accuracy']:>7.4f} {r.get('macro_f1', 0.0):>9.4f} "
            f"{r.get('weighted_f1', 0.0):>7.4f} {r.get('n_samples', ''):>6}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(compare_all())
