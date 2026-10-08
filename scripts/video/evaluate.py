"""
evaluate.py
===========
Evaluate a trained model on DFEW or CMU-MOSEI test data and generate a report.

Usage
-----
    # Evaluate MAE-DFER checkpoint on DFEW fold 1
    python scripts/video/evaluate.py --model mae_dfer --fold 1 \
        --ckpt models/video/checkpoints/mae_dfer_dfew/best_fold1.pth

    # Evaluate CMU-MOSEI baseline
    python scripts/video/evaluate.py --model cmu_mosei_baseline
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT  = pathlib.Path(__file__).resolve().parents[2]
MAE_DFER_ROOT = PROJECT_ROOT / "models" / "video" / "MAE-DFER"
sys.path.insert(0, str(MAE_DFER_ROOT))

LABEL_NAMES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
NUM_CLASSES = 6


def evaluate_mae_dfer(args: argparse.Namespace) -> None:
    from train_mae_dfer import build_dfew_dataset, build_model, compute_metrics, save_confusion_matrix
    from torch.utils.data import DataLoader

    processed = PROJECT_ROOT / "data" / "processed" / "video_fer" / "dfew" / f"fold{args.fold}"
    test_csv  = processed / "test.csv"

    class _Args:
        num_frames = 16; sampling_rate = 4; input_size = 160
        short_side_size = 160; reprob = 0.0; smoothing = 0.0

    test_ds     = build_dfew_dataset(test_csv, "test", _Args())
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = pathlib.Path(args.ckpt)
    model = build_model(ckpt_path, NUM_CLASSES, 160, __import__("logging").getLogger())
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            videos, labels = batch[0].to(device), batch[1]
            logits = model(videos)
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    m = compute_metrics(preds, labels)

    out_dir = PROJECT_ROOT / "models" / "video" / "results" / f"dfew_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _print_report(m, out_dir, f"MAE-DFER DFEW fold{args.fold}")


def evaluate_cmu_mosei_baseline() -> None:
    from train_cmu_mosei_baseline import load_data, build_model, compute_metrics, save_confusion_matrix
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    SEED = 42
    import numpy as np, torch
    X, y = load_data(__import__("logging").getLogger())
    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    _, X_te, _, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model().to(device)
    state  = torch.load(PROJECT_ROOT / "models/video/results/cmu_mosei_baseline/best_model.pt",
                        map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=64)
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            labels.extend(y_b.numpy())

    m = compute_metrics(np.array(preds), np.array(labels))
    out_dir = PROJECT_ROOT / "models" / "video" / "results" / "cmu_mosei_baseline"
    _print_report(m, out_dir, "CMU-MOSEI FACET Baseline")


def _print_report(m: dict, out_dir: pathlib.Path, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy  : {m['accuracy']:.4f}")
    print(f"Macro F1  : {m['macro_f1']:.4f}  ← primary")
    print(f"Weighted F1: {m['weighted_f1']:.4f}")
    print(f"UAR       : {m['macro_recall']:.4f}")
    print("\nPer-class F1:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:10s}: {m['per_class_f1'][i]:.4f}")

    # save updated confusion matrix
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fig.savefig(out_dir / "eval_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {out_dir / 'eval_confusion_matrix.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mae_dfer", "cmu_mosei_baseline"], required=True)
    parser.add_argument("--fold",  default=1, type=int)
    parser.add_argument("--ckpt",  default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    if args.model == "mae_dfer":
        evaluate_mae_dfer(args)
    else:
        evaluate_cmu_mosei_baseline()


if __name__ == "__main__":
    main()
