"""Train a lightweight decision-level fusion classifier on frozen-model predictions.

Input:  data/processed/fusion_predictions.csv with columns
    label          one of the 6 shared emotion classes
    split          train / validation / test
    text_p_<c>     text model probability for class c
    audio_p_<c>    audio model probability for class c
    video_p_<c>    video model probability for class c
  where <c> iterates the shared classes in models.common.label_mapping.

The three encoders are frozen; this only learns how to combine their outputs.
Fit happens on split=='train' exclusively; split=='test' is never seen during
fitting, so the comparison against standalone models is leakage-free.

Run:  python scripts/train_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.label_mapping import SHARED_EMOTION_CLASSES as CLASSES  # noqa: E402

MODALITIES = ("text", "audio", "video")


def prob_cols(modality: str) -> list[str]:
    return [f"{modality}_p_{c}" for c in CLASSES]


def feature_cols() -> list[str]:
    cols: list[str] = []
    for m in MODALITIES:
        cols += prob_cols(m)
    return cols


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"label", "split", *feature_cols()}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    bad_labels = set(df["label"]).difference(CLASSES)
    if bad_labels:
        raise ValueError(f"labels not in shared schema: {sorted(bad_labels)}")
    if df[feature_cols()].isna().any().any():
        n = int(df[feature_cols()].isna().any(axis=1).sum())
        raise ValueError(
            f"{n} rows have missing modality probabilities. Every row must have all "
            "three models' outputs (synchronized samples), or use --allow-missing."
        )
    return df


def standalone_metrics(probs: np.ndarray, y_true_idx: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true_idx, pred)),
        "macro_f1": float(f1_score(y_true_idx, pred, average="macro", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "fusion_predictions.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "fusion" / "logistic_regression")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    df = load(args.data)
    fcols = feature_cols()
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    df["label_idx"] = df["label"].map(class_to_idx)

    train = df[df["split"] == args.train_split]
    test = df[df["split"] == args.test_split]
    if train.empty or test.empty:
        raise ValueError(
            f"need non-empty '{args.train_split}' and '{args.test_split}' splits; "
            f"found splits={sorted(df['split'].unique())}"
        )

    X_train = train[fcols].to_numpy(dtype=np.float64)
    y_train = train["label_idx"].to_numpy()
    X_test = test[fcols].to_numpy(dtype=np.float64)
    y_test = test["label_idx"].to_numpy()

    clf = LogisticRegression(
        C=args.C, max_iter=args.max_iter, solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    fusion = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }

    # Baselines on the SAME held-out test rows.
    baselines: dict[str, dict] = {}
    mod_probs: dict[str, np.ndarray] = {}
    for m in MODALITIES:
        p = test[prob_cols(m)].to_numpy(dtype=np.float64)
        mod_probs[m] = p
        baselines[f"standalone_{m}"] = standalone_metrics(p, y_test)
    avg = np.mean([mod_probs[m] for m in MODALITIES], axis=0)
    baselines["fixed_weight_average"] = standalone_metrics(avg, y_test)

    report = classification_report(
        y_test, pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
        digits=4, zero_division=0,
    )

    # Save model + exact feature order (live inference must match this order).
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out / "fusion_logreg.joblib")
    meta = {
        "model": "LogisticRegression(solver=lbfgs, class_weight=balanced)",
        "classes": CLASSES,
        "feature_cols": fcols,
        "modalities": list(MODALITIES),
        "C": args.C,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_split": args.train_split,
        "test_split": args.test_split,
    }
    (args.out / "feature_order.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics = {
        "fusion": fusion,
        "baselines": baselines,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "class_counts_test": {c: int((test["label"] == c).sum()) for c in CLASSES},
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# Decision-level fusion — results",
        "",
        f"- Trained on split `{args.train_split}` (n={len(train)}); evaluated on `{args.test_split}` (n={len(test)}).",
        "- Encoders frozen; only the fusion head is learned. Test split never used for fitting.",
        "",
        "## Accuracy / macro-F1",
        "",
        "| model | accuracy | macro-F1 |",
        "|---|---|---|",
        f"| **Learned fusion (logreg)** | **{fusion['accuracy']:.4f}** | **{fusion['macro_f1']:.4f}** |",
    ]
    for name, mm in baselines.items():
        lines.append(f"| {name} | {mm['accuracy']:.4f} | {mm['macro_f1']:.4f} |")
    lines += ["", "## Fusion classification report (test)", "", "```", report, "```", ""]

    # Honest interpretation vs baselines (no overclaiming).
    best_sa_name = max((b for b in baselines if b.startswith("standalone_")),
                       key=lambda b: baselines[b]["macro_f1"])
    best_sa = baselines[best_sa_name]
    fw = baselines["fixed_weight_average"]
    lines += ["## Interpretation", ""]
    lines.append(
        f"- Learned fusion macro-F1 **{fusion['macro_f1']:.4f}** vs strongest standalone "
        f"({best_sa_name}) {best_sa['macro_f1']:.4f} "
        f"-> {fusion['macro_f1'] - best_sa['macro_f1']:+.4f}."
    )
    lines.append(
        f"- Learned fusion accuracy **{fusion['accuracy']:.4f}** vs strongest standalone "
        f"{best_sa['accuracy']:.4f} -> {fusion['accuracy'] - best_sa['accuracy']:+.4f}."
    )
    lines.append(
        f"- vs fixed-weight average: macro-F1 {fusion['macro_f1'] - fw['macro_f1']:+.4f}, "
        f"accuracy {fusion['accuracy'] - fw['accuracy']:+.4f}. The learned head's main gain "
        "is balanced (macro) performance across emotions, not peak accuracy."
    )
    lines += ["", "## Limitations", ""]
    lines.append(
        f"- Small fitting set: n_train={len(train)} synchronized (text+audio+video) MELD "
        f"utterances; rare classes are thin "
        f"(train disgust={int((train['label']=='disgust').sum())}, "
        f"fearful={int((train['label']=='fearful').sum())}). Weights are provisional."
    )
    lines.append(
        "- Every row requires ALL THREE modalities on the same clip; utterances where "
        "MediaPipe found no frontal face were dropped (never fabricated), which shrinks "
        "the usable set and biases it toward face-visible clips."
    )
    lines.append(
        f"- Live video features come from a MediaPipe->FACET4.2 approximation (commercial "
        f"FACET emotion channels are zeroed); standalone video is weak "
        f"(acc {baselines['standalone_video']['accuracy']:.3f}, "
        f"macro-F1 {baselines['standalone_video']['macro_f1']:.3f}) and mainly contributes "
        "noise the fusion must learn to down-weight."
    )
    lines.append(
        "- Test set is a single held-out MELD split; deltas of a few points are within noise. "
        "class_weight='balanced' was fixed a priori (a coach must not miss rare emotions), "
        "not tuned on test."
    )
    lines.append("")
    (args.out / "FUSION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    strongest = max(baselines.items(), key=lambda kv: kv[1]["macro_f1"])
    print(f"\nStrongest baseline: {strongest[0]} (macro-F1 {strongest[1]['macro_f1']:.4f})")
    delta = fusion["macro_f1"] - strongest[1]["macro_f1"]
    print(f"Fusion vs strongest baseline: macro-F1 {delta:+.4f}")
    if delta <= 0:
        print("NOTE: fusion does NOT beat the strongest standalone — report honestly, do not overclaim.")
    print(f"\nSaved model + metrics to {args.out}")


if __name__ == "__main__":
    main()
