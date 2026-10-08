"""
train_mae_dfer.py
=================
Fine-tune MAE-DFER on DFEW (6 classes: angry, disgust, fearful, happy, sad, surprised).

This script wraps MAE-DFER's run_class_finetuning.py with:
  - Paths pre-configured for this project
  - 6-class head (drops neutral)
  - RTX 3050 6 GB safe defaults (batch 4, grad_accum 4, bf16, 160px input)
  - Class-weighted loss for imbalanced DFEW
  - Macro F1 as primary metric (plus accuracy/WAR/UAR)

Usage
-----
    # Fine-tune on fold 1 from the public pretrained VoxCeleb2 checkpoint
    python scripts/video/train_mae_dfer.py --fold 1 --pretrain_ckpt models/video/checkpoints/mae_dfer_pretrained/checkpoint-99.pth

    # Fine-tune from an already fine-tuned DFEW checkpoint (faster)
    python scripts/video/train_mae_dfer.py --fold 1 --pretrain_ckpt models/video/checkpoints/mae_dfer_dfew/fold1_checkpoint.pth

    # Smoke test: 2 epochs, 64 samples
    python scripts/video/train_mae_dfer.py --fold 1 --pretrain_ckpt <ckpt> --max_train_samples 64 --epochs 2

Pretrained checkpoint
---------------------
Download the VoxCeleb2 pretrained MAE-DFER weights from the official repo:
  https://github.com/sunlicai/MAE-DFER
Place the file at:
  models/video/checkpoints/mae_dfer_pretrained/checkpoint-99.pth

Alternatively, use a fold-specific fine-tuned checkpoint from the same repo.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT  = pathlib.Path(__file__).resolve().parents[2]
MAE_DFER_ROOT = PROJECT_ROOT / "models" / "video" / "MAE-DFER"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "video_fer" / "dfew"
RESULTS_DIR   = PROJECT_ROOT / "models" / "video" / "results"
CKPT_DIR      = PROJECT_ROOT / "models" / "video" / "checkpoints" / "mae_dfer_dfew"

# add MAE-DFER to path so its modules are importable
sys.path.insert(0, str(MAE_DFER_ROOT))

LABEL_NAMES   = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
NUM_CLASSES   = 6


# ─── logging ─────────────────────────────────────────────────────────────────
def setup_logger(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("train_mae_dfer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ─── dataset ─────────────────────────────────────────────────────────────────
def build_dfew_dataset(anno_csv: pathlib.Path, mode: str, args: argparse.Namespace):
    """Build a VideoClsDatasetFrame (frame-folder based) dataset."""
    from datasets import build_dataset as _build  # MAE-DFER's datasets.py

    class _Args:
        """Minimal args object that MAE-DFER datasets expect."""
        data_set        = "DFEW"
        data_path       = str(anno_csv.parent)
        anno_path       = str(anno_csv)
        num_frames      = args.num_frames
        sampling_rate   = args.sampling_rate
        input_size      = args.input_size
        short_side_size = args.input_size
        reprob          = args.reprob if mode == "train" else 0.0
        remode          = "pixel"
        recount         = 1
        aa              = "rand-m7-n4-mstd0.5-inc1"
        smoothing       = args.smoothing if mode == "train" else 0.0
        train_interpolation = "bicubic"
        num_sample      = 1
        sparse_sample   = False
        test_num_segment = 2
        test_num_crop    = 2

    dataset, _ = _build(mode, _Args())
    return dataset


# ─── model ───────────────────────────────────────────────────────────────────
def build_model(pretrain_ckpt: pathlib.Path | None, num_classes: int,
                input_size: int, logger: logging.Logger) -> nn.Module:
    import modeling_finetune  # noqa — registers models in timm
    from timm.models import create_model

    model = create_model(
        f"vit_base_dim512_no_depth_patch16_{input_size}",
        pretrained=False,
        num_classes=num_classes,
        all_frames=16,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        head_drop_rate=0.5,
        depth=16,
        use_mean_pooling=True,
        init_scale=0.001,
        attn_type="local_global",
        lg_region_size=(2, 5, 10),
        lg_first_attn_type="self",
        lg_third_attn_type="cross",
        lg_attn_param_sharing_first_third=False,
        lg_classify_token_type="region",
        lg_no_second=False,
        lg_no_third=False,
    )

    if pretrain_ckpt is not None and pretrain_ckpt.exists():
        logger.info("Loading pretrained weights from %s", pretrain_ckpt)
        ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("module", ckpt))

        # adapt head if num_classes differs (7 → 6)
        if "head.weight" in state:
            ckpt_classes = state["head.weight"].shape[0]
            if ckpt_classes != num_classes:
                logger.warning(
                    "Head size mismatch: ckpt=%d, target=%d. "
                    "Re-initialising classification head.", ckpt_classes, num_classes
                )
                del state["head.weight"]
                del state["head.bias"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Missing keys  (%d): %s", len(missing), missing[:5])
        logger.info("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])
    else:
        logger.warning("No pretrained checkpoint — training from random init (not recommended).")

    return model


# ─── class weights ───────────────────────────────────────────────────────────
def compute_class_weights(anno_csv: pathlib.Path) -> torch.Tensor:
    import pandas as pd
    df  = pd.read_csv(anno_csv, header=None, sep=" ", names=["path", "label"])
    counts = np.bincount(df["label"].values, minlength=NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1.0, counts)            # avoid div-by-zero
    weights = len(df) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sampler_weights(anno_csv: pathlib.Path) -> list[float]:
    import pandas as pd
    df  = pd.read_csv(anno_csv, header=None, sep=" ", names=["path", "label"])
    counts = np.bincount(df["label"].values, minlength=NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_w = 1.0 / counts
    return [float(class_w[l]) for l in df["label"].values]


# ─── metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy":         float(accuracy_score(labels, preds)),
        "macro_f1":         float(f1_score(labels, preds, average="macro",       zero_division=0)),
        "weighted_f1":      float(f1_score(labels, preds, average="weighted",    zero_division=0)),
        "macro_precision":  float(precision_score(labels, preds, average="macro",zero_division=0)),
        "macro_recall":     float(recall_score(labels, preds, average="macro",   zero_division=0)),  # = UAR
        "per_class_f1":     f1_score(labels, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
    }


def save_confusion_matrix(cm: list[list[int]], out_path: pathlib.Path, title: str) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── train / eval loops ──────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device,
              grad_accum: int, is_train: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.train() if is_train else model.eval()
    total_loss, steps = 0.0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            # MAE-DFER datasets return (videos, labels)
            videos, labels = batch[0].to(device), batch[1].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(videos)
                loss   = criterion(logits, labels) / grad_accum

            if is_train:
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * grad_accum
            steps      += 1
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / max(steps, 1), np.array(all_preds), np.array(all_labels)


# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MAE-DFER on DFEW (6 classes)")
    parser.add_argument("--fold",          default=1,    type=int)
    parser.add_argument("--pretrain_ckpt", default=None, type=str,
                        help="Path to MAE-DFER pretrained or fine-tuned checkpoint")
    parser.add_argument("--epochs",        default=50,   type=int)
    parser.add_argument("--batch_size",    default=4,    type=int)
    parser.add_argument("--grad_accum",    default=4,    type=int)
    parser.add_argument("--lr",            default=1e-4, type=float)
    parser.add_argument("--backbone_lr",   default=5e-6, type=float)
    parser.add_argument("--weight_decay",  default=0.05, type=float)
    parser.add_argument("--warmup_epochs", default=5,    type=int)
    parser.add_argument("--num_frames",    default=16,   type=int)
    parser.add_argument("--sampling_rate", default=4,    type=int)
    parser.add_argument("--input_size",    default=160,  type=int)
    parser.add_argument("--smoothing",     default=0.1,  type=float)
    parser.add_argument("--reprob",        default=0.25, type=float)
    parser.add_argument("--seed",          default=42,   type=int)
    parser.add_argument("--num_workers",   default=2,    type=int)
    parser.add_argument("--max_train_samples", default=None, type=int,
                        help="Cap training samples for smoke test")
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    args = parser.parse_args()

    # ── reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── output dirs ──
    out_dir = RESULTS_DIR / f"dfew_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / f"training_log_fold{args.fold}.txt")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024**3)
    logger.info("Args: %s", args)

    # ── data ──
    fold_dir  = PROCESSED_DIR / f"fold{args.fold}"
    train_csv = fold_dir / "train.csv"
    val_csv   = fold_dir / "val.csv"
    test_csv  = fold_dir / "test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            logger.error("Annotation file missing: %s — run prepare_dfew.py first", p)
            sys.exit(1)

    logger.info("Building datasets...")
    train_ds = build_dfew_dataset(train_csv, "train",      args)
    val_ds   = build_dfew_dataset(val_csv,   "validation", args)
    test_ds  = build_dfew_dataset(test_csv,  "test",       args)

    if args.max_train_samples:
        from torch.utils.data import Subset
        indices = list(range(min(args.max_train_samples, len(train_ds))))
        train_ds = Subset(train_ds, indices)
        logger.info("Smoke test: capped train to %d samples", len(train_ds))

    # weighted sampler for class balance
    sampler_weights = compute_sampler_weights(train_csv)
    if args.max_train_samples:
        sampler_weights = sampler_weights[:args.max_train_samples]
    sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    logger.info("Train: %d  Val: %d  Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # ── model ──
    pretrain_ckpt = pathlib.Path(args.pretrain_ckpt) if args.pretrain_ckpt else None
    model = build_model(pretrain_ckpt, NUM_CLASSES, args.input_size, logger)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Total params: %.1f M | Trainable: %.1f M", total_params, train_params)

    # ── loss & optimizer ──
    class_weights = compute_class_weights(train_csv).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.smoothing)

    # layer-wise LR: head gets args.lr, backbone gets args.backbone_lr
    head_params     = list(model.head.parameters())
    head_ids        = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── training loop ──
    best_macro_f1  = -1.0
    best_state     = None
    patience_count = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_preds, tr_labels = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.grad_accum, is_train=True
        )
        scheduler.step()

        val_loss, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            device, args.grad_accum, is_train=False
        )

        tr_m  = compute_metrics(tr_preds, tr_labels)
        val_m = compute_metrics(val_preds, val_labels)
        elapsed = time.time() - t0

        logger.info(
            "epoch=%d  train_loss=%.4f  val_loss=%.4f  "
            "val_acc=%.4f  val_macro_f1=%.4f  val_uar=%.4f  time=%.1fs",
            epoch, tr_loss, val_loss,
            val_m["accuracy"], val_m["macro_f1"], val_m["macro_recall"], elapsed
        )
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
            "train_metrics": tr_m, "val_metrics": val_m,
        })

        if val_m["macro_f1"] > best_macro_f1:
            best_macro_f1  = val_m["macro_f1"]
            best_state     = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            logger.info("  ↑ New best val macro_f1=%.4f — saving checkpoint", best_macro_f1)
            torch.save(best_state, CKPT_DIR / f"best_fold{args.fold}.pth")
        else:
            patience_count += 1
            if patience_count >= args.early_stopping_patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.early_stopping_patience)
                break

    # ── final test evaluation ──
    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, optimizer, scaler,
        device, args.grad_accum, is_train=False
    )
    test_m = compute_metrics(test_preds, test_labels)

    logger.info("\n=== TEST RESULTS ===")
    logger.info("Accuracy:    %.4f", test_m["accuracy"])
    logger.info("Macro F1:    %.4f  ← primary metric", test_m["macro_f1"])
    logger.info("Weighted F1: %.4f", test_m["weighted_f1"])
    logger.info("UAR:         %.4f", test_m["macro_recall"])
    logger.info("Per-class F1:")
    for i, name in enumerate(LABEL_NAMES):
        logger.info("  %-10s: %.4f", name, test_m["per_class_f1"][i])

    # ── save confusion matrix ──
    save_confusion_matrix(
        test_m["confusion_matrix"],
        out_dir / f"confusion_matrix_fold{args.fold}.png",
        f"MAE-DFER DFEW fold{args.fold} (test)"
    )

    # ── save full results JSON ──
    results = {
        "fold": args.fold,
        "best_val_macro_f1": best_macro_f1,
        "test_metrics": test_m,
        "history": history,
        "args": vars(args),
    }
    (out_dir / f"results_fold{args.fold}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # ── markdown report ──
    lines = [
        f"# MAE-DFER Training Report — DFEW Fold {args.fold}",
        "",
        "## Training configuration",
        f"- Fold: {args.fold}",
        f"- Epochs run: {len(history)} (max {args.epochs})",
        f"- Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})",
        f"- Learning rate (head): {args.lr}",
        f"- Learning rate (backbone): {args.backbone_lr}",
        f"- Input size: {args.input_size}×{args.input_size}",
        f"- Num frames: {args.num_frames}",
        f"- Early stopping patience: {args.early_stopping_patience}",
        "",
        "## Per-epoch metrics",
        "",
        "| Epoch | Train Loss | Val Loss | Val Acc | Val Macro F1 | Val UAR |",
        "|---|---|---|---|---|---|",
    ]
    for h in history:
        vm = h["val_metrics"]
        lines.append(
            f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} "
            f"| {vm['accuracy']:.4f} | {vm['macro_f1']:.4f} | {vm['macro_recall']:.4f} |"
        )
    lines += [
        "",
        "## Test results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Macro F1** | **{test_m['macro_f1']:.4f}** |",
        f"| Accuracy (WAR) | {test_m['accuracy']:.4f} |",
        f"| UAR | {test_m['macro_recall']:.4f} |",
        f"| Weighted F1 | {test_m['weighted_f1']:.4f} |",
        "",
        "### Per-class F1 (test)",
        "",
        "| Class | F1 |",
        "|---|---|",
    ]
    for i, name in enumerate(LABEL_NAMES):
        lines.append(f"| {name} | {test_m['per_class_f1'][i]:.4f} |")

    (out_dir / f"training_report_fold{args.fold}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    logger.info("Results saved to %s", out_dir)
    logger.info("DONE — best val Macro F1: %.4f | test Macro F1: %.4f",
                best_macro_f1, test_m["macro_f1"])


if __name__ == "__main__":
    main()
