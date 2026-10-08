"""Train the Memory Fusion Network multimodal emotion model.

Usage:
  python models/multimodal/train_mfn.py --epochs 30 --batch-size 64 --lr 1e-3

Data: data/processed/multimodal/{split}_data.npz (built by
      models/preprocessing/build_cmumosei_multimodal_dataset.py)
Model: src.pipeline.multimodal_pipeline.MultimodalEmotionPipeline
Output: models/multimodal/runs_mfn/{best_model.pt, training_log.txt,
      classification_report.csv, confusion_matrix.png, training_report.md}
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in [str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import CONFIG
from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline

DATA_DIR = pathlib.Path(CONFIG["data_dir"])
OUT_DIR = pathlib.Path(CONFIG["model_dir"]) / "runs_mfn"
LABEL_NAMES = CONFIG["label_names"]
NUM_CLASSES = CONFIG["num_classes"]


def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_mfn")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [logging.FileHandler(OUT_DIR / "training_log.txt", encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_split(split: str) -> TensorDataset:
    d = np.load(DATA_DIR / f"{split}_data.npz")
    return TensorDataset(
        torch.tensor(d["text"], dtype=torch.float32),
        torch.tensor(d["audio"], dtype=torch.float32),
        torch.tensor(d["video"], dtype=torch.float32),
        torch.tensor(d["labels"], dtype=torch.long),
    )


def make_loader(ds: TensorDataset, batch_size: int, weighted: bool = False, shuffle: bool = True) -> DataLoader:
    sampler = None
    if weighted:
        counts = np.bincount(ds.tensors[3].numpy(), minlength=NUM_CLASSES).astype(np.float64)
        weights = (1.0 / np.maximum(counts, 1.0))[ds.tensors[3].numpy()]
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    for t, a, v, y in loader:
        t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
        logits = model(t, a, v)
        total_loss += criterion(logits, y).item() * len(y)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        gts.append(y.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return (
        total_loss / len(gts),
        accuracy_score(gts, preds),
        f1_score(gts, preds, average="macro", zero_division=0),
        gts,
        preds,
    )


def plot_cm(gts: np.ndarray, preds: np.ndarray, path: pathlib.Path) -> None:
    cm = confusion_matrix(gts, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(NUM_CLASSES), yticks=np.arange(NUM_CLASSES),
           xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
           title="MFN Confusion Matrix (test)", ylabel="True", xlabel="Predicted")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    lr = args.lr

    logger = setup_logging()
    device = torch.device(args.device)
    logger.info("device=%s lr=%g epochs=%d batch=%d", device, lr, args.epochs, args.batch_size)

    train_ds = load_split("train")
    if args.max_train_samples:
        sub = torch.randperm(len(train_ds))[: args.max_train_samples].tolist()
        train_ds = TensorDataset(*[t[sub] for t in train_ds.tensors])
    val_ds = load_split("validation")
    test_ds = load_split("test")
    logger.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = make_loader(train_ds, args.batch_size, weighted=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    model = MultimodalEmotionPipeline(CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model params: %d", n_params)

    class_counts = np.bincount(train_ds.tensors[3].numpy(), minlength=NUM_CLASSES)
    weights = torch.tensor(len(train_ds) / (NUM_CLASSES * np.maximum(class_counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for t, a, v, y in train_loader:
            t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(t, a, v)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            n += len(y)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_acc)
        history.append({"epoch": epoch, "train_loss": total_loss / n, "train_acc": correct / n,
                        "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1})
        logger.info("Epoch %2d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f Macro F1: %.4f",
                    epoch, args.epochs, total_loss / n, correct / n, val_loss, val_acc, val_f1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            logger.info("  saved best model (val acc %.4f)", val_acc)

    # final test eval with best weights
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device))
    test_loss, test_acc, test_f1, gts, preds = evaluate(model, test_loader, device, criterion)
    logger.info("TEST — loss %.4f acc %.4f macro F1 %.4f", test_loss, test_acc, test_f1)

    report = classification_report(gts, preds, target_names=LABEL_NAMES, zero_division=0)
    (OUT_DIR / "classification_report.csv").write_text(report, encoding="utf-8")
    plot_cm(gts, preds, OUT_DIR / "confusion_matrix.png")

    majority = max(class_counts.tolist()) / sum(class_counts.tolist())
    md = {
        "model": "MemoryFusionNetwork (text 256-D + audio 128-D + video 128-D)",
        "params": int(n_params),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "majority_baseline": float(majority),
        "history": history,
    }
    (OUT_DIR / "training_report.md").write_text(
        "# MFN Multimodal Training Report\n\n```json\n" + json.dumps(md, indent=2) + "\n```\n\n"
        "## Classification report\n\n```\n" + report + "\n```\n",
        encoding="utf-8",
    )
    logger.info("WROTE %s", OUT_DIR)


if __name__ == "__main__":
    main()
