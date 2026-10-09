"""Fine-tune the registered text encoders on multilabel GoEmotions."""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.compare import MODEL_REGISTRY

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "text" / "goemotions"
OUTPUT_DIR = PROJECT_ROOT / "models" / "text" / "checkpoints"
OUTPUT_MD = PROJECT_ROOT / "models" / "text_model_analysis.md"
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH = 8
PER_DEVICE_EVAL_BATCH = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
# transformers 5 removed warmup_ratio; use explicit warmup steps (0 = linear decay from the
# first training step), which keeps the old "no warmup" behaviour identical.
WARMUP_STEPS = 0
MAX_LENGTH = 128
LOGGING_STEPS = 50
EARLY_STOPPING_PATIENCE = 2
THRESHOLD = 0.5


def load_emotion_data(data_dir: Path) -> DatasetDict:
    files = {
        split: str(data_dir / f"goemotions_{split}.parquet")
        for split in ("train", "validation", "test")
    }
    return load_dataset("parquet", data_files=files)


def load_label_names(data_dir: Path) -> list[str]:
    metadata = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    labels = metadata.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(
            "GoEmotions labels.json does not contain a non-empty labels list."
        )
    return [str(label) for label in labels]


def build_label_index(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(labels)}


def tokenize_and_encode(
    batch: dict, tokenizer, label_index: dict[str, int]
) -> dict:
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    targets = np.zeros(
        (len(batch["text"]), len(label_index)), dtype=np.float32
    )
    for row_index, values in enumerate(batch["labels"]):
        for label in str(values).split("|"):
            if label:
                targets[row_index, label_index[label]] = 1.0
    encoded["labels"] = targets.tolist()
    return encoded


def prepare_datasets(data: DatasetDict, tokenizer, label_names: list[str]) -> DatasetDict:
    label_index = build_label_index(label_names)
    tokenized = data.map(
        lambda batch: tokenize_and_encode(batch, tokenizer, label_index),
        batched=True,
        remove_columns=data["train"].column_names,
    )
    return tokenized


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= THRESHOLD).astype(int)
    labels = np.asarray(labels).astype(int)
    return {
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def train_one_model(
    model_key: str,
    model_name: str,
    model_cfg: dict,
    data: DatasetDict,
    label_names: list[str],
    dry_run: bool = False,
    output_dir: Path | None = None,
    max_train_samples: int | None = None,
    num_epochs: int = NUM_EPOCHS,
) -> dict:
    print(f"\n{'=' * 60}\nTraining: {model_key} ({model_cfg['name']})\n{'=' * 60}")
    start = time.time()
    # DeBERTa-v3 ships a SentencePiece tokenizer.  Transformers 5 may try to
    # convert it through TikToken when fast-tokenizer loading is enabled,
    # which fails unless an unrelated optional dependency is installed.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=model_key != "deberta-v3-base",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(label_names)),
        label2id={label: index for index, label in enumerate(label_names)},
    )
    dataset = prepare_datasets(data, tokenizer, label_names)
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    args = TrainingArguments(
        output_dir=str(output_dir or OUTPUT_DIR / f"runs_{model_key}"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="linear",
        report_to="none",
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        # Keep training on CUDA, but avoid GradScaler FP16 unscale failures
        # observed with this Windows/PyTorch/Transformers combination.
        fp16=False,
        use_cpu=False,
        # Each checkpoint is ~1.5 GB (model + AdamW optimizer state); one
        # checkpoint (the best, via load_best_model_at_end) is all the app
        # needs.
        save_total_limit=1,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    if dry_run:
        result = trainer.evaluate()
        print("dry_run | evaluated validation set in one epoch")
        elapsed = time.time() - start
        return {
            "model_key": model_key,
            "name": model_cfg["name"],
            "hf_id": model_name,
            "epochs": 1,
            "best_train_loss": None,
            "best_eval_loss": float(result.get("eval_loss", float("inf"))),
            "best_eval_macro_f1": float(result.get("eval_macro_f1", 0.0)),
            "best_eval_micro_f1": float(result.get("eval_micro_f1", 0.0)),
            "elapsed_minutes": elapsed / 60.0,
            "failed": False,
        }

    train_result = trainer.train()
    eval_results = trainer.evaluate(dataset["validation"], metric_key_prefix="validation")
    test_results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    elapsed = time.time() - start
    result = {
        "model_key": model_key,
        "name": model_cfg["name"],
        "hf_id": model_name,
        "epochs": NUM_EPOCHS,
        "best_train_loss": float(train_result.training_loss),
        "best_eval_loss": float(eval_results.get("validation_loss", eval_results.get("eval_loss", float("inf")))),
        "best_eval_macro_f1": float(eval_results.get("validation_macro_f1", eval_results.get("eval_macro_f1", 0.0))),
        "best_eval_micro_f1": float(eval_results.get("validation_micro_f1", eval_results.get("eval_micro_f1", 0.0))),
        "test_macro_f1": float(test_results.get("test_macro_f1", 0.0)),
        "test_micro_f1": float(test_results.get("test_micro_f1", 0.0)),
        "elapsed_minutes": elapsed / 60.0,
        "failed": False,
    }
    print(
        f"completed in {elapsed / 60:.1f} min | "
        f"eval macro F1: {result['best_eval_macro_f1']:.4f}"
    )
    return result


def generate_report(results: list[dict], path: Path, label_names: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "# GoEmotions Text Model Training Analysis",
                "",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Labels: {len(label_names)} | Multilabel threshold: {THRESHOLD}",
                "",
                "## Hyperparameters",
                "",
                "| Parameter | Value |",
                "|---|---|",
                f"| Epochs | {NUM_EPOCHS} |",
                f"| Train/eval batch | {PER_DEVICE_TRAIN_BATCH} / {PER_DEVICE_EVAL_BATCH} |",
                f"| Gradient accumulation | {GRADIENT_ACCUMULATION_STEPS} |",
                f"| Effective batch | {PER_DEVICE_TRAIN_BATCH * GRADIENT_ACCUMULATION_STEPS} |",
                f"| Learning rate | {LEARNING_RATE} |",
                f"| Weight decay | {WEIGHT_DECAY} |",
                f"| Warmup steps | {WARMUP_STEPS} |",
                f"| Max sequence length | {MAX_LENGTH} |",
                "| Scheduler | Linear |",
                "| Best metric | Macro F1 |",
                "",
                "## Comparison",
                "",
                "| Model | Validation loss | Validation macro F1 | Test macro F1 | Test micro F1 | Time (min) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            + [
                f"| {r['model_key']} ({r['name']}) | {r['best_eval_loss']:.4f} | "
                f"{r['best_eval_macro_f1']:.4f} | {r.get('test_macro_f1', 0.0):.4f} | "
                f"{r.get('test_micro_f1', 0.0):.4f} | "
                f"{r['elapsed_minutes']:.1f} |"
                for r in results
            ]
        ),
        encoding="utf-8",
    )


def main(
    dry_run: bool = False,
    model_keys: list[str] | None = None,
    max_train_samples: int | None = None,
    num_epochs: int = NUM_EPOCHS,
) -> None:
    label_names = load_label_names(DATA_DIR)
    data = load_emotion_data(DATA_DIR)
    print(f"GoEmotions data: {len(data['train']):,} train / {len(data['validation']):,} validation")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Install a CUDA-enabled PyTorch build "
            "before starting text-model training."
        )
    print(
        f"Labels: {len(label_names)} | GPU: {torch.cuda.get_device_name(0)} | "
        f"CUDA: {torch.version.cuda}"
    )
    results = []
    selected = model_keys or list(MODEL_REGISTRY["text"])
    unknown = sorted(set(selected) - set(MODEL_REGISTRY["text"]))
    if unknown:
        raise ValueError(f"Unknown text model(s): {', '.join(unknown)}")
    for model_key in selected:
        config = MODEL_REGISTRY["text"][model_key]
        try:
            results.append(
                train_one_model(
                    model_key,
                    config["hugging_face"],
                    config,
                    data,
                    label_names,
                    dry_run=dry_run,
                    max_train_samples=max_train_samples,
                    num_epochs=num_epochs,
                )
            )
        except Exception as exc:
            print(f"FAILED for {model_key}: {exc}")
            results.append(
                {
                    "model_key": model_key,
                    "name": config["name"],
                    "best_eval_loss": float("inf"),
                    "best_eval_macro_f1": 0.0,
                    "best_eval_micro_f1": 0.0,
                    "elapsed_minutes": 0.0,
                    "failed": True,
                }
            )
    results.sort(key=lambda item: item["best_eval_macro_f1"], reverse=True)
    generate_report(results, OUTPUT_MD, label_names)
    print("Training report written to", OUTPUT_MD)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models",
                        help="Model key to train; repeat to train multiple models.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load the model and evaluate without training.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    main(args.dry_run, args.models, args.max_train_samples, args.epochs)
