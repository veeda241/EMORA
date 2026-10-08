"""Evaluation metrics: accuracy, F1, confusion matrix helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, label_names: list[str] | None = None) -> dict:
    """Standard classification metrics over integer labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_samples": int(len(y_true)),
    }


def majority_baseline(y_true, label_names: list[str] | None = None) -> dict:
    """Accuracy of always predicting the most frequent class."""
    y_true = np.asarray(y_true)
    counts = np.bincount(y_true)
    top = int(np.argmax(counts))
    return {
        "majority_class": top,
        "majority_class_name": label_names[top] if label_names else str(top),
        "accuracy": float(counts[top] / len(y_true)),
    }


def compute_report(y_true, y_pred, label_names: list[str]) -> str:
    """Full sklearn classification report as a string."""
    return classification_report(y_true, y_pred, target_names=label_names, zero_division=0)


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
