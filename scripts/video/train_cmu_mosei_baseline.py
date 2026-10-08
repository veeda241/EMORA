"""
train_cmu_mosei_baseline.py
===========================
Train a temporal MLP on CMU-MOSEI FACET 4.2 features as the video baseline.

Architecture:
    FACET 35-dim features (frames × 35)
        ↓  temporal mean-pool  → (35,)
        ↓  MLP: Linear(35→256) → BN → GELU → Dropout(0.3)
               → Linear(256→128) → BN → GELU → Dropout(0.2)
               → Linear(128→6)
        ↓  6-class softmax

6-class mapping (drops sentiment col 0 from CMU-MOSEI):
    0 = angry       (CMU-MOSEI col 3)
    1 = disgust     (CMU-MOSEI col 5)
    2 = fearful     (CMU-MOSEI col 6)
    3 = happy       (CMU-MOSEI col 1)
    4 = sad         (CMU-MOSEI col 2)
    5 = surprised   (CMU-MOSEI col 4)

Dominant emotion = argmax of the 6 intensity scores for that segment.

Usage
-----
    python scripts/video/train_cmu_mosei_baseline.py
"""
from __future__ import annotations

import json
import logging
import pathlib
import random
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MOSEI_DIR    = PROJECT_ROOT / "data" / "raw" / "cmu_mosei"
RESULTS_DIR  = PROJECT_ROOT / "models" / "video" / "results" / "cmu_mosei_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
NUM_CLASSES  = 6

# CMU-MOSEI label column indices (col 0 = sentiment, skip it)
# features shape per segment: (1, 7) → [sentiment, happy, sad, angry, surprised, disgusted, fearful]
COL_MAP = {
    "happy":     1,
    "sad":       2,
    "angry":     3,
    "surprised": 4,
    "disgusted": 5,
    "fearful":   6,
}
# map CMU-MOSEI argmax index (0-5 over cols 1-6) → our class index
MOSEI_ARGMAX_TO_CLASS = {
    0: 3,  # happy     → 3
    1: 4,  # sad       → 4
    2: 0,  # angry     → 0
    3: 5,  # surprised → 5
    4: 1,  # disgust   → 1
    5: 2,  # fearful   → 2
}


def setup_logger(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("cmu_mosei_baseline")
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


def load_data(logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """Load FACET features + labels, mean-pool frames, assign dominant class."""
    vis_path = MOSEI_DIR / "CMU_MOSEI_VisualFacet42.csd"
    lbl_path = MOSEI_DIR / "CMU_MOSEI_Labels.csd"

    logger.info("Loading CMU-MOSEI data...")
    with h5py.File(vis_path, "r") as vf, h5py.File(lbl_path, "r") as lf:
        vis_grp = vf["FACET 4.2"]["data"]
        lbl_grp = lf["All Labels"]["data"]

        vis_ids = set(vis_grp.keys())
        lbl_ids = set(lbl_grp.keys())
        overlap = sorted(vis_ids & lbl_ids)
        logger.info("Videos with both visual + labels: %d", len(overlap))

        features, labels = [], []
        for vid in overlap:
            vis_feats = vis_grp[vid]["features"][()]   # (frames, 35)
            lbl_feats = lbl_grp[vid]["features"][()]   # (num_segs, 7)

            # mean-pool frames → (35,) per video
            feat_pooled = vis_feats.mean(axis=0)

            # one label row per segment — use video-level dominant emotion
            # (average across segments, then argmax)
            emotion_scores = lbl_feats[:, 1:].mean(axis=0)   # (6,) — skip sentiment
            dominant = int(np.argmax(emotion_scores))
            label = MOSEI_ARGMAX_TO_CLASS[dominant]

            features.append(feat_pooled)
            labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int64)
    logger.info("Dataset: %d samples, feature dim %d", len(X), X.shape[1])
    return X, y


def build_model() -> nn.Module:
    # LayerNorm instead of BatchNorm1d — works with any batch size including 1
    return nn.Sequential(
        nn.Linear(35, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, NUM_CLASSES),
    )


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_f1":        float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(labels, preds, average="macro",    zero_division=0)),
        "per_class_f1":    f1_score(labels, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
    }


def save_confusion_matrix(cm: list, out_path: pathlib.Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("CMU-MOSEI FACET Baseline — Confusion Matrix (test)")
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


def main() -> None:
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    logger = setup_logger(RESULTS_DIR / "training_log.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── data ──
    X, y = load_data(logger)

    # 70 / 15 / 15 split (stratified)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    logger.info("Train: %d  Val: %d  Test: %d", len(X_tr), len(X_val), len(X_te))

    # class distribution
    counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    logger.info("Train class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        logger.info("  %-10s: %d (%.1f%%)", name, counts[i], 100 * counts[i] / len(y_tr))

    # tensors
    def make_loader(Xd, yd, shuffle=False, weighted=False) -> DataLoader:
        ds = TensorDataset(torch.tensor(Xd), torch.tensor(yd))
        if weighted:
            c = np.bincount(yd, minlength=NUM_CLASSES).astype(float)
            c = np.where(c == 0, 1.0, c)
            w = torch.tensor([1.0 / c[l] for l in yd])
            sampler = WeightedRandomSampler(w, len(w), replacement=True)
            return DataLoader(ds, batch_size=64, sampler=sampler)
        return DataLoader(ds, batch_size=64, shuffle=shuffle)

    train_loader = make_loader(X_tr,  y_tr,  weighted=True)
    val_loader   = make_loader(X_val, y_val)
    test_loader  = make_loader(X_te,  y_te)

    # ── model ──
    model = build_model().to(device)
    counts_f = torch.tensor(counts.astype(float), dtype=torch.float32)
    class_w  = (len(y_tr) / (NUM_CLASSES * counts_f)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    EPOCHS   = 30
    PATIENCE = 5
    best_f1, best_state, patience_count = -1.0, None, 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_steps = 0.0, 0
        tr_preds, tr_labels = [], []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_loss  += loss.item(); tr_steps += 1
            tr_preds.extend(logits.argmax(-1).cpu().numpy())
            tr_labels.extend(y_b.cpu().numpy())
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item(); val_steps += 1
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_labels.extend(y_b.cpu().numpy())

        tm = compute_metrics(np.array(tr_preds),  np.array(tr_labels))
        vm = compute_metrics(np.array(val_preds), np.array(val_labels))
        elapsed = time.time() - t0

        logger.info("epoch=%d  tr_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_macro_f1=%.4f  %.1fs",
                    epoch, tr_loss/tr_steps, val_loss/val_steps, vm["accuracy"], vm["macro_f1"], elapsed)
        history.append({"epoch": epoch, "train_loss": tr_loss/tr_steps,
                         "val_loss": val_loss/val_steps, "val_metrics": vm})

        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── test ──
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b.to(device))
            test_preds.extend(logits.argmax(-1).cpu().numpy())
            test_labels.extend(y_b.numpy())

    test_m = compute_metrics(np.array(test_preds), np.array(test_labels))

    logger.info("\n=== TEST RESULTS ===")
    logger.info("Accuracy:    %.4f", test_m["accuracy"])
    logger.info("Macro F1:    %.4f  ← primary metric", test_m["macro_f1"])
    logger.info("Weighted F1: %.4f", test_m["weighted_f1"])
    logger.info("UAR:         %.4f", test_m["macro_recall"])
    for i, name in enumerate(LABEL_NAMES):
        logger.info("  %-10s: %.4f", name, test_m["per_class_f1"][i])

    # save
    save_confusion_matrix(test_m["confusion_matrix"], RESULTS_DIR / "confusion_matrix.png")
    torch.save(best_state, RESULTS_DIR / "best_model.pt")
    (RESULTS_DIR / "results.json").write_text(
        json.dumps({"test_metrics": test_m, "history": history}, indent=2))

    # markdown report
    lines = [
        "# CMU-MOSEI FACET Baseline — Training Report",
        "",
        "## Architecture",
        "- Input: 35-dim FACET 4.2 facial features (temporal mean-pool per utterance)",
        "- MLP: Linear(35→256) → BN → GELU → Dropout(0.3) → Linear(256→128) → BN → GELU → Dropout(0.2) → Linear(128→6)",
        "",
        "## Dataset",
        f"- Source: CMU-MOSEI VisualFacet42.csd",
        f"- Total samples: {len(X)}",
        f"- Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}",
        "",
        "## Test Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Macro F1** | **{test_m['macro_f1']:.4f}** |",
        f"| Accuracy | {test_m['accuracy']:.4f} |",
        f"| UAR | {test_m['macro_recall']:.4f} |",
        f"| Weighted F1 | {test_m['weighted_f1']:.4f} |",
        "",
        "### Per-class F1",
        "",
        "| Class | F1 |",
        "|---|---|",
    ]
    for i, name in enumerate(LABEL_NAMES):
        lines.append(f"| {name} | {test_m['per_class_f1'][i]:.4f} |")
    (RESULTS_DIR / "training_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved to %s", RESULTS_DIR)
    logger.info("DONE — test Macro F1: %.4f", test_m["macro_f1"])


if __name__ == "__main__":
    main()
