"""Run reproducible unimodal, fusion, and MFN ablation experiments.

The script uses the aligned CMU-MOSEI feature arrays already consumed by MFN.
Frozen-feature rows are mean pooled over time and evaluated with a balanced
multinomial logistic regression for unimodal and simple-concatenation
baselines.  The LSTM Fusion row is evaluated from its saved runs_lstm
checkpoint when available, with the legacy inline sequence baseline
(--lstm-epochs) as fallback.  MFN is evaluated from its saved checkpoint
when available.

This intentionally reports missing artifacts instead of substituting
fabricated metrics.  Run from the repository root:

    python evaluation/ablation.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, majority_baseline

MODALITIES = ("text", "audio", "video")
LABELS = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]


def load_split(split: str) -> dict[str, np.ndarray]:
    path = ROOT / "data" / "processed" / "multimodal" / f"{split}_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing aligned split: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in (*MODALITIES, "labels")}


def pooled(data: dict[str, np.ndarray], modalities: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([data[name].mean(axis=1) for name in modalities], axis=1)


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(train_x, train_y)
    return model.predict(test_x)


def train_lstm(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    epochs: int,
    device_name: str,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(device_name)
    train_x = np.concatenate([train[name] for name in MODALITIES], axis=2)
    test_x = np.concatenate([test[name] for name in MODALITIES], axis=2)
    class SequenceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(train_x.shape[2], 64, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Linear(64, len(LABELS))

        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            return self.head(self.dropout(hidden[-1]))

    classifier = SequenceClassifier().to(device)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train["labels"], dtype=torch.long),
        ),
        batch_size=128,
        shuffle=True,
    )
    counts = np.bincount(train["labels"], minlength=len(LABELS))
    weights = torch.tensor(
        len(train["labels"]) / (len(LABELS) * np.maximum(counts, 1)),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    classifier.train()
    for _ in range(max(1, epochs)):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(classifier(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            optimizer.step()
    classifier.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_x), 256):
            x_batch = torch.tensor(test_x[start:start + 256], dtype=torch.float32, device=device)
            predictions.extend(classifier(x_batch).argmax(dim=-1).cpu().numpy())
    return np.asarray(predictions)


def evaluate_rows(lstm_epochs: int = 0, device: str = "cpu") -> list[dict]:
    train, test = load_split("train"), load_split("test")
    rows = [{
        "model": "Majority",
        **majority_baseline(test["labels"], LABELS),
        "macro_f1": 0.0,
        "notes": "test-set class prior",
    }]

    combinations = {
        "Text": ("text",),
        "Audio": ("audio",),
        "Video": ("video",),
        "Text + Audio": ("text", "audio"),
        "Text + Video": ("text", "video"),
        "Audio + Video": ("audio", "video"),
        "Simple Concatenation": MODALITIES,
    }
    for name, modalities in combinations.items():
        preds = fit_logistic(pooled(train, modalities), train["labels"], pooled(test, modalities))
        rows.append({"model": name, **compute_metrics(test["labels"], preds, LABELS),
                     "notes": "balanced logistic regression on temporal mean features"})

    # A single row is kept for MFN so this table can be compared directly with
    # the simpler baselines.  The checkpoint's input schema is validated first.
    mfn_checkpoint = ROOT / "models" / "multimodal" / "runs_mfn" / "best_model.pt"
    if mfn_checkpoint.exists():
        import torch
        from src.config import CONFIG
        from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultimodalEmotionPipeline(CONFIG).to(device)
        state = torch.load(mfn_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(test["text"], dtype=torch.float32, device=device),
                torch.tensor(test["audio"], dtype=torch.float32, device=device),
                torch.tensor(test["video"], dtype=torch.float32, device=device),
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        rows.append({"model": "MFN", **compute_metrics(test["labels"], preds, LABELS),
                     "notes": "saved MFN checkpoint"})
    else:
        rows.append({"model": "MFN", "status": "missing", "notes": str(mfn_checkpoint)})

    experiment_b_checkpoint = (
        ROOT / "models" / "multimodal" / "runs_experiment_b"
        / "best_model_glove_covarep_hybrid_cnn_vit.pt"
    )
    if experiment_b_checkpoint.exists():
        import torch
        from src.pipeline.experiment_b_pipeline import ExperimentBPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ExperimentBPipeline().to(device)
        state = torch.load(experiment_b_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(test["text"], dtype=torch.float32, device=device),
                torch.tensor(test["audio"], dtype=torch.float32, device=device),
                torch.tensor(test["video"], dtype=torch.float32, device=device),
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        rows.append({
            "model": "GloVe + COVAREP + HybridCNNViT + MFN",
            **compute_metrics(test["labels"], preds, LABELS),
            "notes": "Experiment B; frozen canonical Hybrid CNN-ViT 512-D embedding",
        })
    else:
        rows.append({
            "model": "GloVe + COVAREP + HybridCNNViT + MFN",
            "status": "missing",
            "notes": str(experiment_b_checkpoint),
        })

    experiment_c_checkpoint = (
        ROOT / "models" / "multimodal" / "runs_experiment_c"
        / "best_model_glove_covarep_hybrid_cnn_vit_temporal.pt"
    )
    if experiment_c_checkpoint.exists():
        import torch
        from src.pipeline.experiment_c_pipeline import ExperimentCPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ExperimentCPipeline().to(device)
        state = torch.load(experiment_c_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(test["text"], dtype=torch.float32, device=device),
                torch.tensor(test["audio"], dtype=torch.float32, device=device),
                torch.tensor(test["video"], dtype=torch.float32, device=device),
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        rows.append({
            "model": "GloVe + COVAREP + temporal HybridCNNViT + MFN",
            **compute_metrics(test["labels"], preds, LABELS),
            "notes": "Experiment C; frozen Hybrid CNN-ViT temporal (16x256) features, no broadcast",
        })
    else:
        rows.append({
            "model": "GloVe + COVAREP + temporal HybridCNNViT + MFN",
            "status": "missing",
            "notes": str(experiment_c_checkpoint),
        })

    lstm_checkpoint = ROOT / "models" / "multimodal" / "runs_lstm" / "best_model.pt"
    if lstm_checkpoint.exists():
        import torch
        from src.pipeline.lstm_fusion_pipeline import LSTMFusionPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMFusionPipeline().to(device)
        state = torch.load(lstm_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(test["text"], dtype=torch.float32, device=device),
                torch.tensor(test["audio"], dtype=torch.float32, device=device),
                torch.tensor(test["video"], dtype=torch.float32, device=device),
            )
        preds = logits.argmax(dim=-1).cpu().numpy()
        rows.append({
            "model": "LSTM Fusion",
            **compute_metrics(test["labels"], preds, LABELS),
            "notes": "LSTM temporal fusion using GloVe + COVAREP + FACET with the same CMU-MOSEI split and evaluation protocol as MFN.",
        })
    elif lstm_epochs > 0:
        preds = train_lstm(train, test, lstm_epochs, device)
        rows.append({"model": "LSTM Fusion", **compute_metrics(test["labels"], preds, LABELS),
                     "notes": f"sequence baseline, {lstm_epochs} epochs"})
    else:
        rows.append({"model": "LSTM Fusion", "status": "not_run",
                     "notes": "pass --lstm-epochs to train this baseline"})
    return rows


def write_outputs(rows: list[dict]) -> None:
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    fields = ["model", "accuracy", "macro_f1", "weighted_f1", "n_samples", "status", "notes"]
    with (out_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    lines = [
        "# Modality and Fusion Ablation",
        "",
        "| Model | Accuracy | Macro F1 | Status | Notes |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row.get('accuracy', '')} | "
            f"{row.get('macro_f1', '')} | {row.get('status', 'measured')} | "
            f"{row.get('notes', '')} |"
        )
    (out_dir / "ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lstm-epochs", type=int, default=0,
                        help="Train the LSTM baseline for this many epochs.")
    args = parser.parse_args()
    rows = evaluate_rows(args.lstm_epochs, args.device)
    write_outputs(rows)
    print((ROOT / "reports" / "ablation_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
