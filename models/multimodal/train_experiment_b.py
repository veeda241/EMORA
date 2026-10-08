"""Train Experiment B: GloVe + COVAREP + Hybrid CNN-ViT embedding -> MFN."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.pipeline.experiment_b_pipeline import (
    ExperimentBPipeline,
    HYBRID_CHECKPOINT,
    HYBRID_EMBED_DIM,
)

DATA_DIR = ROOT / "data" / "processed" / "multimodal"
OUT_DIR = ROOT / "models" / "multimodal" / "runs_experiment_b"
CHECKPOINT_NAME = "best_model_glove_covarep_hybrid_cnn_vit.pt"
LABEL_NAMES = CONFIG["label_names"]
NUM_CLASSES = CONFIG["num_classes"]


def load_split(split: str) -> TensorDataset:
    with np.load(DATA_DIR / f"{split}_data.npz") as data:
        return TensorDataset(
            torch.tensor(data["text"], dtype=torch.float32),
            torch.tensor(data["audio"], dtype=torch.float32),
            torch.tensor(data["video"], dtype=torch.float32),
            torch.tensor(data["labels"], dtype=torch.long),
        )


def loader(dataset: TensorDataset, batch_size: int, weighted: bool = False):
    sampler = None
    shuffle = not weighted
    if weighted:
        labels = dataset.tensors[3].numpy()
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        weights = (1.0 / np.maximum(counts, 1.0))[labels]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    model.eval()
    losses, targets, predictions = [], [], []
    for text, audio, video, labels in data_loader:
        logits = model(text.to(device), audio.to(device), video.to(device))
        losses.append(criterion(logits, labels.to(device)).item() * len(labels))
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())
        targets.extend(labels.numpy())
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    return (
        sum(losses) / len(targets),
        accuracy_score(targets, predictions),
        f1_score(targets, predictions, average="macro", zero_division=0),
        targets,
        predictions,
    )


def plot_confusion(targets, predictions, path: Path):
    matrix = confusion_matrix(targets, predictions, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Purples)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="Experiment B Confusion Matrix",
    )
    threshold = matrix.max() / 2 if matrix.max() else 1
    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            ax.text(
                col, row, str(matrix[row, col]), ha="center", va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("experiment_b")
    device = torch.device(args.device)
    train_ds = load_split("train")
    if args.max_train_samples:
        indices = torch.randperm(len(train_ds))[:args.max_train_samples]
        train_ds = TensorDataset(*(tensor[indices] for tensor in train_ds.tensors))
    val_ds, test_ds = load_split("validation"), load_split("test")
    train_loader = loader(train_ds, args.batch_size, weighted=True)
    val_loader = loader(val_ds, args.batch_size)
    test_loader = loader(test_ds, args.batch_size)

    model = ExperimentBPipeline().to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in model.hybrid_video.parameters()]
    class_counts = np.bincount(train_ds.tensors[3].numpy(), minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        len(train_ds) / (NUM_CLASSES * np.maximum(class_counts, 1)),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = -1.0
    history = []
    checkpoint = OUT_DIR / CHECKPOINT_NAME
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        for text, audio, video, labels in train_loader:
            optimizer.zero_grad()
            logits = model(text.to(device), audio.to(device), video.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=-1).cpu() == labels).sum().item()
            seen += len(labels)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_acc)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(record)
        logger.info("%s", record)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment": "B",
                    "video_checkpoint": str(HYBRID_CHECKPOINT),
                    "hybrid_embedding_dim": HYBRID_EMBED_DIM,
                    "history": history,
                },
                checkpoint,
            )

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_loss, test_acc, test_f1, targets, predictions = evaluate(
        model, test_loader, device, criterion
    )
    report = classification_report(
        targets, predictions, target_names=LABEL_NAMES, output_dict=False, zero_division=0
    )
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(targets, predictions, OUT_DIR / "confusion_matrix.png")
    summary = {
        "experiment": "B",
        "architecture": "GloVe + COVAREP + frozen HybridCNNViT emb(512) + trainable projection + MFN",
        "hybrid_embedding_dim": HYBRID_EMBED_DIM,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": str(device),
            "weighted_sampler": True,
            "class_weighted_loss": True,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 5.0,
            "selection_metric": "validation accuracy",
        },
        "reproduce_command": (
            "python models/multimodal/train_experiment_b.py "
            "--epochs 30 --batch-size 64 --lr 0.001 --device cuda"
        ),
        "checkpoint": str(checkpoint),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "history": history,
    }
    (OUT_DIR / "training_report.md").write_text(
        "# Experiment B: GloVe + COVAREP + Hybrid CNN-ViT -> MFN\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"
        "## Per-class report\n\n```\n" + report + "\n```\n",
        encoding="utf-8",
    )
    logger.info("Experiment B test accuracy=%.4f macro_f1=%.4f", test_acc, test_f1)


if __name__ == "__main__":
    main()
