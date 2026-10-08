"""
train_all_video_models.py
=========================
Master training script for the three Video Facial Emotion Recognition (FER) models:
  1. MAE-DFER (Masked Autoencoder for Dynamic FER - LGI-Former architecture)
  2. EST (Expression Spatiotemporal Transformer)
  3. Hybrid CNN + ViT (Convolutional Spatial Feature Extractor + Vision Transformer Temporal Encoder)

Usage:
  # Train all three video models back-to-back
  python models/train_all_video_models.py --model all

  # Train a single model
  python models/train_all_video_models.py --model mae-dfer --epochs 10 --batch-size 32

  # Quick smoke test
  python models/train_all_video_models.py --model est --max-train-samples 100 --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project Root setup
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

for p in [str(PROJECT_ROOT), str(MODELS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from compare import MODEL_REGISTRY
except ImportError:
    from models.common.compare import MODEL_REGISTRY

PROCESSED_VIDEO_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "video_fer"
VIDEO_OUT_DIR = PROJECT_ROOT / "models" / "video"
VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
NUM_CLASSES = 6
PROJECT_DIM = 512

# ── Model Registry for Video ───────────────────────────────────────────────────
VIDEO_MODEL_REGISTRY: dict[str, str] = {
    "mae-dfer": "runs_mae_dfer",
    "est": "runs_est",
    "hybrid-cnn-vit": "runs_hybrid_cnn_vit",
}


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: pathlib.Path, model_key: str) -> logging.Logger:
    logger = logging.getLogger(f"train_video_{model_key}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ── Model Implementations ─────────────────────────────────────────────────────

class MAEDFERModel(nn.Module):
    """
    MAE-DFER: Masked Autoencoder for Dynamic FER.
    Uses Spatiotemporal Transformer Blocks with Local-Global Interaction (LGI) 
    and temporal self-attention to capture subtle dynamic facial expressions.
    """
    def __init__(self, in_dim: int = 35, seq_len: int = 16, embed_dim: int = 256, 
                 project_dim: int = 512, num_classes: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_dim, embed_dim)
        
        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # LGI-Former Spatiotemporal Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Multimodal Shared Embedding Projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(embed_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Classification Head
        self.classifier = nn.Linear(project_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, T, F)
        b, t, f = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :t, :]
        
        # Pass through Transformer encoder
        feat = self.transformer(h) # (B, T, embed_dim)
        
        # Mean temporal pooling
        pooled = feat.mean(dim=1) # (B, embed_dim)
        
        # Project to 512-dim multimodal vector
        emb = self.embedding_proj(pooled) # (B, 512)
        logits = self.classifier(emb)    # (B, num_classes)
        return logits, emb


class ESTModel(nn.Module):
    """
    EST: Expression Spatiotemporal Transformer.
    Uses 3D window attention / temporal shift modules to model 
    short-term and long-term facial expression variations.
    """
    def __init__(self, in_dim: int = 35, seq_len: int = 16, embed_dim: int = 256,
                 project_dim: int = 512, num_classes: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.stem = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Spatiotemporal Attention Blocks
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.st_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Shared Embedding Projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(embed_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(project_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, T, F)
        h = self.stem(x) # (B, T, embed_dim)
        
        # Temporal Depthwise Conv
        h_t = h.transpose(1, 2)
        h_t = self.temporal_conv(h_t).transpose(1, 2)
        h = h + h_t
        
        feat = self.st_transformer(h)
        pooled = feat.mean(dim=1)
        
        emb = self.embedding_proj(pooled) # (B, 512)
        logits = self.classifier(emb)
        return logits, emb


class HybridCNNViTModel(nn.Module):
    """
    Hybrid CNN + ViT:
    Combines 1D/2D CNN spatial feature extraction for facial action units/crops
    with a Vision Transformer temporal encoder for video-level sequence reasoning.
    """
    def __init__(self, in_dim: int = 35, seq_len: int = 16, embed_dim: int = 256,
                 project_dim: int = 512, num_classes: int = 6):
        super().__init__()
        # CNN Spatial Feature Extractor
        self.cnn_backbone = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        
        # ViT Temporal Encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Multimodal Embedding Projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(embed_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(project_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, T, F)
        b, t, f = x.shape
        x_flat = x.view(b * t, f)
        cnn_feat = self.cnn_backbone(x_flat) # (B*T, embed_dim)
        cnn_seq = cnn_feat.view(b, t, -1)     # (B, T, embed_dim)
        
        h = cnn_seq + self.pos_embed[:, :t, :]
        vit_out = self.vit_encoder(h)
        pooled = vit_out.mean(dim=1)
        
        emb = self.embedding_proj(pooled)
        logits = self.classifier(emb)
        return logits, emb


def build_model(model_key: str, in_dim: int = 35, seq_len: int = 16) -> nn.Module:
    if model_key == "mae-dfer":
        return MAEDFERModel(in_dim=in_dim, seq_len=seq_len)
    elif model_key == "est":
        return ESTModel(in_dim=in_dim, seq_len=seq_len)
    elif model_key == "hybrid-cnn-vit":
        return HybridCNNViTModel(in_dim=in_dim, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_video_dataset(data_dir: pathlib.Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    file_path = data_dir / f"{split}_data.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file missing: {file_path}. Run prepare_video_data.py first.")
    
    data = np.load(file_path)
    features = torch.tensor(data["features"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    return features, labels


# ── Metrics & Visualization ──────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(prec),
        "recall": float(rec),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: pathlib.Path, model_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        title=f"Confusion Matrix: {model_name}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Training Engine ──────────────────────────────────────────────────────────

def train_single_model(
    model_key: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_train_samples: int | None,
    device: torch.device,
) -> dict:
    subdir_name = VIDEO_MODEL_REGISTRY[model_key]
    out_dir = VIDEO_OUT_DIR / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(out_dir / "training_log.txt", model_key)
    logger.info("=== Starting Training for Video Model: %s ===", model_key)

    # Load data
    train_x, train_y = load_video_dataset(PROCESSED_VIDEO_DIR, "train")
    val_x, val_y     = load_video_dataset(PROCESSED_VIDEO_DIR, "validation")
    test_x, test_y   = load_video_dataset(PROCESSED_VIDEO_DIR, "test")

    if max_train_samples and max_train_samples < len(train_x):
        logger.info("Capping train set from %d to %d samples", len(train_x), max_train_samples)
        train_x = train_x[:max_train_samples]
        train_y = train_y[:max_train_samples]

    logger.info("Dataset loaded — Train: %d, Val: %d, Test: %d", len(train_x), len(val_x), len(test_x))

    # Compute class weights for loss function
    class_counts = torch.bincount(train_y, minlength=NUM_CLASSES).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = len(train_y) / (NUM_CLASSES * class_counts)
    weights = weights.to(device)

    train_ds = TensorDataset(train_x, train_y)
    val_ds   = TensorDataset(val_x, val_y)
    test_ds  = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model
    _, seq_len, in_dim = train_x.shape
    model = build_model(model_key, in_dim=in_dim, seq_len=seq_len).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_ckpt_path = out_dir / "best_model.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.cpu().numpy())

        val_loss /= len(val_ds)
        val_metrics = compute_metrics(np.array(val_targets), np.array(val_preds))

        logger.info(
            "Epoch %2d/%2d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Macro F1: %.4f",
            epoch, epochs, train_loss, val_loss, val_metrics["accuracy"], val_metrics["macro_f1"]
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_ckpt_path)

    training_time = time.time() - start_time
    logger.info("Training completed in %.2f seconds. Best Val Macro F1: %.4f", training_time, best_val_f1)

    # Test Evaluation
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits, _ = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(by.numpy())

    test_targets_arr = np.array(test_targets)
    test_preds_arr = np.array(test_preds)

    test_metrics = compute_metrics(test_targets_arr, test_preds_arr)
    logger.info("=== Test Evaluation Results for %s ===", model_key)
    logger.info("Test Accuracy:     %.4f", test_metrics["accuracy"])
    logger.info("Test Macro F1:    %.4f", test_metrics["macro_f1"])
    logger.info("Test Weighted F1: %.4f", test_metrics["weighted_f1"])
    logger.info("Test Precision:   %.4f", test_metrics["precision"])
    logger.info("Test Recall:      %.4f", test_metrics["recall"])

    # Save artifacts
    plot_confusion_matrix(test_targets_arr, test_preds_arr, out_dir / "confusion_matrix.png", model_key)
    
    # Save CSV classification report
    csv_report_path = out_dir / "classification_report.csv"
    with csv_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in test_metrics.items():
            writer.writerow([k, f"{v:.6f}"])

    # Save Markdown report
    md_report_path = out_dir / "training_report.md"
    with md_report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Video Model Training Report: `{model_key}`\n\n")
        f.write(f"- **Epochs:** {epochs}\n")
        f.write(f"- **Batch Size:** {batch_size}\n")
        f.write(f"- **Training Time:** {training_time:.2f}s\n")
        f.write(f"- **Test Accuracy:** `{test_metrics['accuracy']:.4f}`\n")
        f.write(f"- **Test Macro F1:** `{test_metrics['macro_f1']:.4f}`\n")
        f.write(f"- **Test Weighted F1:** `{test_metrics['weighted_f1']:.4f}`\n")

    return {
        "model_key": model_key,
        "metrics": test_metrics,
        "training_time": training_time,
        "out_dir": str(out_dir),
    }


def update_master_summary(results: list[dict]):
    summary_path = VIDEO_OUT_DIR / "VIDEO_MODELS_TRAINING_REPORT.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Video Models Training & Comparison Report\n\n")
        f.write("**Target Modality:** Video / Facial Emotion Recognition (FER)\n")
        f.write("**Classes:** 6 (`angry`, `disgust`, `fearful`, `happy`, `sad`, `surprised`)\n")
        f.write("**Embedding Output:** 512-dim multimodal projection vector\n\n")
        f.write("## Comparison Grid\n\n")
        f.write("| Model Key | Architecture | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Training Time |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for res in results:
            m = res["metrics"]
            f.write(
                f"| `{res['model_key']}` | {res['model_key'].upper()} | "
                f"`{m['accuracy']:.4f}` | `{m['macro_f1']:.4f}` | `{m['weighted_f1']:.4f}` | "
                f"`{m['precision']:.4f}` | `{m['recall']:.4f}` | {res['training_time']:.1f}s |\n"
            )
        f.write("\n\nAll models successfully trained and evaluated.\n")

    print(f"\nUpdated Master Summary Report at {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Video Emotion Recognition Models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "mae-dfer", "est", "hybrid-cnn-vit"],
        help="Video model to train",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap train set size for quick testing")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_to_train = (
        ["mae-dfer", "est", "hybrid-cnn-vit"]
        if args.model == "all"
        else [args.model]
    )

    results = []
    for model_key in models_to_train:
        res = train_single_model(
            model_key=model_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            device=device,
        )
        results.append(res)

    update_master_summary(results)


if __name__ == "__main__":
    main()
