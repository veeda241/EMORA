import argparse
import csv
import json
import logging
import math
import pathlib
import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

from matplotlib import pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_MELD = PROJECT_ROOT / "data" / "raw" / "meld"
RAW_RAVDESS = PROJECT_ROOT / "data" / "raw" / "ravdess"
MODEL_OUT = PROJECT_ROOT / "models" / "audio" / "runs_wav2vec2"
AUDIO_DEFAULT = PROJECT_ROOT / "data" / "raw" / "audio"


def setup_logging(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("train_audio_wav2vec2")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_meld_csv(csv_path: pathlib.Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ravdess_metadata(csv_path: pathlib.Path) -> list[dict]:
    """Load RAVDESS metadata CSV prepared under data/raw/ravdess/."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_label_mapping(rows: list[dict]) -> tuple[dict[str, int], dict[int, str], int]:
    labels = sorted({r["Emotion"] for r in rows if r.get("Emotion")})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label, len(labels)


def build_label_mapping_ravdess() -> tuple[dict[str, int], dict[int, str], int]:
    """Canonical 6-class RAVDESS label mapping requested for this project."""
    label_names = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label, len(label_names)


def audio_present(audio_root: pathlib.Path) -> bool:
    if not audio_root.exists():
        return False
    return any(
        p.suffix.lower() in MELD_AUDIO_EXTS for p in audio_root.rglob("*") if p.is_file()
    )


def default_resolve_audio_path(row: dict, audio_root: pathlib.Path) -> pathlib.Path:
    season = str(row.get("Season", "")).strip()
    episode = str(row.get("Episode", "")).strip()
    dialogue_id = str(row.get("Dialogue_ID", "")).strip()
    utterance_id = str(row.get("Utterance_ID", "")).strip()
    candidate = audio_root / f"Episode{season}_{episode}" / f"dial{season}_{dialogue_id}_utt{utterance_id}.wav"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot locate audio for utterance "
        f"Dialogue_ID={dialogue_id!r} Utterance_ID={utterance_id!r} "
        f"Season={season!r} Episode={episode!r} under {audio_root!r}. "
        f"Tried: {candidate!r}"
    )


def load_and_resample(
    row: dict,
    processor: Wav2Vec2Processor,
    audio_root: pathlib.Path,
    resolver,
    logger: logging.Logger,
) -> np.ndarray:
    path = resolver(row, audio_root)
    try:
        import librosa

        waveform, orig_sr = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path!r}: {e}") from e

    if waveform.size == 0:
        raise ValueError(f"Empty waveform after loading {path!r}")
    if np.any(~np.isfinite(waveform)):
        raise ValueError(f"Non-finite waveform values in {path!r}")

    target_sr = processor.feature_extractor.sampling_rate
    if orig_sr != target_sr:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=orig_sr, target_sr=target_sr
        )

    return waveform


class MelDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict],
        processor: Wav2Vec2Processor,
        label2id: dict[str, int],
        audio_root: pathlib.Path,
        resolver,
        max_train_samples: int | None = None,
    ):
        self.rows = rows[:max_train_samples] if max_train_samples else rows
        self.processor = processor
        self.label2id = label2id
        self.audio_root = audio_root
        self.resolver = resolver
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        row = self.rows[idx]
        waveform = load_and_resample(
            row,
            self.processor,
            self.audio_root,
            self.resolver,
            logging.getLogger("train_audio_wav2vec2"),
        )
        emotion = row.get("emotion") or row.get("Emotion")
        if emotion is None:
            raise KeyError(f"Dataset row is missing an emotion label: {row!r}")
        label = int(self.label2id[emotion])
        inputs = self.processor(
            waveform,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="np",
            padding=False,
        )
        out = {
            "input_values": inputs.input_values[0],
            "labels": label,
            "row": row,
        }
        self._cache[idx] = out
        return out


def collate_fn(batch: list[dict]) -> dict:
    input_values = [b["input_values"] for b in batch]
    labels = [b["labels"] for b in batch]

    max_len = max(iv.shape[0] for iv in input_values)
    padded = np.zeros((len(input_values), max_len), dtype=np.float32)
    for i, iv in enumerate(input_values):
        padded[i, : iv.shape[0]] = iv

    return {
        "input_values": torch.from_numpy(padded),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def compute_metrics(
    preds: np.ndarray, labels: np.ndarray, label_names: list[str]
) -> dict:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(
        labels, preds, labels=list(range(len(label_names)))
    )

    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
        "preds": preds.tolist(),
        "labels": labels.tolist(),
        "num_classes": int(len(label_names)),
    }


def _nan_check(name: str, value: float, logger: logging.Logger) -> None:
    if math.isnan(value) or math.isinf(value):
        logger.error("Detected NaN/Inf metric: %s = %r", name, value)
        raise RuntimeError(f"Training produced non-finite metric {name}={value}.")


def _resolve_audio_path_rows(rows: list[dict], audio_root: pathlib.Path) -> tuple[dict[str, int], list[str]]:
    present_by_split: dict[str, int] = {}
    missing_examples: list[str] = []
    for r in rows:
        file_path = r.get("file_path")
        if file_path:
            candidate = pathlib.Path(file_path)
            if candidate.is_absolute():
                path = candidate
            else:
                path = audio_root / candidate
        else:
            season = str(r.get("Season", "")).strip()
            episode = str(r.get("Episode", "")).strip()
            dialogue_id = str(r.get("Dialogue_ID", "")).strip()
            utterance_id = str(r.get("Utterance_ID", "")).strip()
            path = audio_root / f"Episode{season}_{episode}" / f"dial{season}_{dialogue_id}_utt{utterance_id}.wav"
        split_key = r.get("split", "unknown")
        if split_key not in present_by_split:
            present_by_split[split_key] = 0
        if path.exists():
            present_by_split[split_key] += 1
        else:
            missing_examples.append(
                f"{split_key}: missing {str(path)}"
            )
            if len(missing_examples) >= 10:
                break
    return present_by_split, missing_examples


def validate_dataset(
    train_rows: list[dict],
    dev_rows: list[dict],
    test_rows: list[dict],
    label2id: dict[str, int],
    audio_root: pathlib.Path,
    logger: logging.Logger,
) -> dict:
    report: dict = {}

    report["counts"] = {
        split: dict(Counter(r["Emotion"] for r in rows))
        for split, rows in [
            ("train", train_rows),
            ("dev", dev_rows),
            ("test", test_rows),
        ]
    }
    report["num_classes"] = len(label2id)
    report["classes"] = list(label2id.keys())

    problems: list[str] = []
    for split_name, rows in [
        ("train", train_rows),
        ("dev", dev_rows),
        ("test", test_rows),
    ]:
        for r in rows:
            label = r.get("Emotion")
            if label not in label2id:
                problems.append(f"unknown label {label!r} in {split_name}")
            else:
                lid = label2id[label]
                if not (0 <= lid < len(label2id)):
                    problems.append(
                        f"out-of-range label {label!r} -> {lid} in {split_name}"
                    )

    if problems:
        raise ValueError("Dataset label problems:\n" + "\n".join(problems))

    has_audio = audio_present(audio_root)
    report["audio_present"] = has_audio
    if not has_audio:
        raise RuntimeError(
            "No MELD audio files were found under the configured audio root.\n"
            "This is a dataset-readiness failure, not a CUDA or model error.\n"
            f"Configured audio root: {audio_root!r}\n"
            "\nBefore running this script, either:\n"
            "1. Place the aligned MELD audio directory at the configured path, or\n"
            "2. Point this script at the correct directory with --audio-root.\n"
            "\nExpected MELD audio layout:\n"
            "  Episode{season}_{episode}/\n"
            "    dial{season}_{dialogue}_utt{utterance}.wav\n"
            "Example candidate resolved from CSV fields:\n"
            "  Episode1_1/dial1_0_utt0.wav\n"
            "\nDataset inspection found complete MELD CSVs, but training cannot "
            "proceed until aligned audio is available.\n"
        )

    present_by_split, missing_examples = _resolve_audio_path_rows(
        train_rows + dev_rows + test_rows, audio_root
    )
    report["audio_files_present_by_split"] = present_by_split
    report["audio_files_present_total"] = sum(present_by_split.values())
    if missing_examples:
        raise FileNotFoundError(
            "Audio validation failed. Some dataset rows have no matching audio file.\n"
            "The first missing files are:\n"
            + "\n".join(missing_examples)
            + "\nEnsure every dataset row has its aligned WAV file.\n"
        )

    logger.info("Dataset validation passed.")
    return report


def validate_dataset_ravdess(
    train_rows: list[dict],
    dev_rows: list[dict],
    test_rows: list[dict],
    label2id: dict[str, int],
    audio_root: pathlib.Path,
    logger: logging.Logger,
) -> dict:
    report: dict = {}

    report["counts"] = {
        split: dict(Counter(r["emotion"] for r in rows))
        for split, rows in [
            ("train", train_rows),
            ("dev", dev_rows),
            ("test", test_rows),
        ]
    }
    report["num_classes"] = len(label2id)
    report["classes"] = list(label2id.keys())

    problems: list[str] = []
    for split_name, rows in [
        ("train", train_rows),
        ("dev", dev_rows),
        ("test", test_rows),
    ]:
        for r in rows:
            label = r.get("emotion")
            if label not in label2id:
                problems.append(f"unknown label {label!r} in {split_name}")
            else:
                lid = label2id[label]
                if not (0 <= lid < len(label2id)):
                    problems.append(
                        f"out-of-range label {label!r} -> {lid} in {split_name}"
                    )

    if problems:
        raise ValueError("RAVDESS label problems:\n" + "\n".join(problems))

    has_audio = audio_present(audio_root)
    report["audio_present"] = has_audio
    if not has_audio:
        raise RuntimeError(
            "No RAVDESS audio files were found under the configured audio root.\n"
            "This is a dataset-readiness failure, not a CUDA or model error.\n"
            f"Configured audio root: {audio_root!r}\n"
            "\nExpected RAVDESS layout for this project:\n"
            "  data/raw/ravdess/\n"
            "    Actor_{actor_id}/\n"
            "      <RAVDESS_filename>.wav\n"
            "Example candidate filename:\n"
            "  Actor_01/03-01-05-01-01-01-01.wav\n"
        )

    present_by_split, missing_examples = _resolve_audio_path_rows(
        train_rows + dev_rows + test_rows, audio_root
    )
    report["audio_files_present_by_split"] = present_by_split
    report["audio_files_present_total"] = sum(present_by_split.values())
    if missing_examples:
        raise FileNotFoundError(
            "RAVDESS audio validation failed. Some metadata rows have no matching audio file.\n"
            "The first missing files are:\n"
            + "\n".join(missing_examples)
            + "\nEnsure every metadata row points to an extracted WAV under data/raw/ravdess/.\n"
        )

    logger.info("RAVDESS dataset validation passed.")
    return report


def resolve_ravdess_audio_path(row: dict, audio_root: pathlib.Path) -> pathlib.Path:
    """Resolve a RAVDESS waveform from the prepared metadata CSV row.

    The CSV may contain relative paths like 'data/raw/ravdess/Actor_01/...'.
    Those are interpreted as already rooted under audio_root, so we do not
    attach audio_root twice.
    """
    file_path = row.get("file_path")
    if not file_path:
        raise KeyError("RAVDESS metadata row is missing file_path")
    candidate = pathlib.Path(file_path)
    if not candidate.is_absolute():
        path = audio_root / candidate
        if path.exists():
            return path
    try:
        rel = candidate.relative_to(audio_root)
    except ValueError:
        rel = None
    if rel is not None and rel != pathlib.Path("."):
        path = audio_root / rel
        if path.exists():
            return path
    if audio_root in candidate.parents:
        path = candidate
        if path.exists():
            return path
    path = audio_root / candidate
    if not path.exists():
        raise FileNotFoundError(
            "Cannot locate RAVDESS audio file for metadata row. "
            f"Resolved path: {str(path)}. "
            f"audio_root: {str(audio_root)}."
        )
    return path


def train(
    args: argparse.Namespace,
    logger: logging.Logger,
    audio_root: pathlib.Path,
) -> dict:
    torch.set_float32_matmul_precision(
        "high" if torch.cuda.is_available() else "medium"
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Install a CUDA-enabled PyTorch build "
            "before starting MELD audio training."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s (%s)", device, torch.cuda.get_device_name(0))
    logger.info("CUDA version: %s", torch.version.cuda)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_name = str(args.dataset).lower()
    if dataset_name == "ravdess":
        ravdess_csv = RAW_RAVDESS / "ravdess_metadata.csv"
        if not ravdess_csv.exists():
            raise FileNotFoundError(
                "RAVDESS metadata CSV not found at expected path: "
                f"{ravdess_csv}\n"
                "Expected: data/raw/ravdess/ravdess_metadata.csv"
            )
        all_rows = load_ravdess_metadata(ravdess_csv)
        train_rows = [r for r in all_rows if r["split"] == "train"]
        dev_rows = [r for r in all_rows if r["split"] == "validation"]
        test_rows = [r for r in all_rows if r["split"] == "test"]
        logger.info(
            "RAVDESS rows from metadata CSV: train=%d dev=%d test=%d",
            len(train_rows),
            len(dev_rows),
            len(test_rows),
        )
        label2id, id2label, num_labels = build_label_mapping_ravdess()
        label_names = [id2label[i] for i in range(num_labels)]
        logger.info("num_labels=%d label_names=%s", num_labels, label_names)
        report = validate_dataset_ravdess(
            train_rows, dev_rows, test_rows, label2id, audio_root, logger
        )
    else:
        train_rows = load_meld_csv(RAW_MELD / "train_sent_emo.csv")
        dev_rows = load_meld_csv(RAW_MELD / "dev_sent_emo.csv")
        test_rows = load_meld_csv(RAW_MELD / "test_sent_emo.csv")

        logger.info(
            "MELD rows: train=%d dev=%d test=%d",
            len(train_rows),
            len(dev_rows),
            len(test_rows),
        )

        label2id, id2label, num_labels = build_label_mapping(
            train_rows + dev_rows + test_rows
        )
        label_names = [id2label[i] for i in range(num_labels)]
        logger.info("num_labels=%d label_names=%s", num_labels, label_names)

        report = validate_dataset(
            train_rows, dev_rows, test_rows, label2id, audio_root, logger
        )

    logger.info("Loading facebook/wav2vec2-base")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=num_labels,
        label2id={str(k): v for k, v in label2id.items()},
        id2label={str(k): v for k, v in id2label.items()},
        attention_dropout=0.1,
        hidden_dropout=0.1,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )

    if args.dataset == "ravdess":
        resolver = resolve_ravdess_audio_path
    else:
        resolver = default_resolve_audio_path

    train_ds = MelDDataset(
        train_rows, processor, label2id, audio_root, resolver, args.max_train_samples
    )
    dev_ds = MelDDataset(
        dev_rows, processor, label2id, audio_root, resolver
    )
    test_ds = MelDDataset(
        test_rows, processor, label2id, audio_root, resolver
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_epochs = int(args.epochs)
    global_step = 0
    best_dev_loss = float("inf")
    best_state = None
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        lr_now = float(optimizer.param_groups[0]["lr"])
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(**batch)
                loss = out.loss
            if loss is None or not torch.isfinite(loss):
                logger.error(
                    "Non-finite training loss at epoch=%d step=%d", epoch, batch_idx
                )
                raise RuntimeError("Training loss became non-finite.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item())
            epoch_steps += 1
            global_step += 1

            if batch_idx % args.log_interval == 0:
                logger.info(
                    "epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    float(loss.item()),
                    lr_now,
                )

        train_loss = epoch_loss / max(epoch_steps, 1)
        _nan_check("train_loss", train_loss, logger)

        model.eval()
        dev_loss = 0.0
        dev_steps = 0
        dev_preds: list[int] = []
        dev_labels: list[int] = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(**batch)
                loss = out.loss
                logits = out.logits
                if not torch.isfinite(loss):
                    logger.error("Non-finite dev loss")
                    raise RuntimeError("Validation loss became non-finite.")
                dev_loss += float(loss.item())
                dev_steps += 1
                dev_preds.extend(logits.argmax(dim=-1).detach().tolist())
                dev_labels.extend(batch["labels"].detach().tolist())

        dev_loss_avg = dev_loss / max(dev_steps, 1)
        _nan_check("dev_loss", dev_loss_avg, logger)

        metrics = compute_metrics(
            np.array(dev_preds), np.array(dev_labels), label_names
        )
        logger.info(
            "epoch=%d train_loss=%.4f dev_loss=%.4f accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
            epoch,
            train_loss,
            dev_loss_avg,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss_avg,
                "learning_rate": lr_now,
                "metrics": metrics,
            }
        )

        if dev_loss_avg < best_dev_loss:
            best_dev_loss = dev_loss_avg
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    train_seconds = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    def evaluate(loader):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(dim=-1).detach().tolist())
                labels.extend(batch["labels"].detach().tolist())
        return compute_metrics(
            np.array(preds), np.array(labels), label_names
        )

    dev_metrics = evaluate(dev_loader)
    test_metrics = evaluate(test_loader)

    model.eval()

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    processor.save_pretrained(MODEL_OUT)
    logger.info("Saved model to %s", MODEL_OUT)

    cm = np.array(test_metrics["confusion_matrix"])

    if args.dataset == "ravdess":
        cm_title = "Wav2Vec2 confusion matrix (RAVDESS test)"
        cm_path = PROJECT_ROOT / "models" / "audio" / "ravdess_confusion_matrix.png"
    else:
        cm_title = "Wav2Vec2 confusion matrix (MELD test)"
        cm_path = PROJECT_ROOT / "models" / "audio" / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(cm_title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", cm_path)

    csv_path = PROJECT_ROOT / "models" / "audio" / "classification_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_csv = []
    for i, name in enumerate(label_names):
        rows_csv.append(
            {
                "class": name,
                "precision": test_metrics["per_class_precision"][i],
                "recall": test_metrics["per_class_recall"][i],
                "f1": test_metrics["per_class_f1"][i],
                "support": test_metrics["support"][i],
            }
        )
    rows_csv.append(
        {
            "class": "macro_avg",
            "precision": test_metrics["macro_precision"],
            "recall": test_metrics["macro_recall"],
            "f1": test_metrics["macro_f1"],
            "support": "",
        }
    )
    rows_csv.append(
        {
            "class": "weighted_avg",
            "precision": test_metrics["weighted_precision"],
            "recall": test_metrics["weighted_recall"],
            "f1": test_metrics["weighted_f1"],
            "support": "",
        }
    )
    rows_csv.append(
        {
            "class": "accuracy",
            "precision": "",
            "recall": "",
            "f1": test_metrics["accuracy"],
            "support": "",
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["class", "precision", "recall", "f1", "support"],
        )
        w.writeheader()
        w.writerows(rows_csv)
    logger.info("Saved classification report to %s", csv_path)

    if args.dataset == "ravdess":
        dataset_header = "# Audio Model Training Report - Wav2Vec2 (RAVDESS)"
        dataset_source_lines = [
            "## Dataset",
            "- Source: `data/raw/ravdess/`",
            "- Metadata: `data/raw/ravdess/ravdess_metadata.csv`",
            "- Emotion set: angry, disgust, fearful, happy, sad, surprised",
            "- Excluded: neutral, calm",
            f"- Train samples: {len(train_rows)}",
            f"- Validation samples: {len(dev_rows)}",
            f"- Test samples: {len(test_rows)}",
            f"- Total samples: {len(train_rows) + len(dev_rows) + len(test_rows)}",
            "",
            "## Speaker-independent split",
            "- Split by Actor ID, not by individual utterances.",
            f"- Train actors: {', '.join(sorted({r['actor_id'] for r in train_rows}))}",
            f"- Validation actors: {', '.join(sorted({r['actor_id'] for r in dev_rows}))}",
            f"- Test actors: {', '.join(sorted({r['actor_id'] for r in test_rows}))}",
            "",
            "## RAVDESS label mapping",
            f"- label2id: {json.dumps(label2id)}",
            f"- id2label: {json.dumps({int(k): v for k, v in id2label.items()})}",
        ]
    else:
        dataset_header = "# Audio Model Training Report - Wav2Vec2 (MELD)"
        dataset_source_lines = [
            "## Dataset",
            "- Source: `data/raw/meld/`",
            "- Files used:",
            "  - `train_sent_emo.csv`",
            "  - `dev_sent_emo.csv`",
            "  - `test_sent_emo.csv`",
            f"- Train samples: {len(train_rows)}",
            f"- Validation samples: {len(dev_rows)}",
            f"- Test samples: {len(test_rows)}",
            f"- Total samples: {len(train_rows) + len(dev_rows) + len(test_rows)}",
        ]

    report_lines = [
        dataset_header,
        "",
    ] + dataset_source_lines + [
        "",
        "## Classes",
        f"- Number of classes: {num_labels}",
        "- Class names: " + ", ".join(label_names),
        "- Label mapping saved to: `models/audio/label_mapping.json`",
        "",
        "## Audio source",
        f"- Configured audio root: {audio_root!r}",
        "- Audio found: yes",
        "- Preprocessing: load -> mono -> resample to 16 kHz -> Wav2Vec2 processor -> dynamic padding",
        "",
        "## Wav2Vec2 configuration",
        "- Pretrained checkpoint: `facebook/wav2vec2-base`",
        f"- num_labels: {num_labels}",
        f"- label2id: {json.dumps(label2id)}",
        f"- id2label: {json.dumps({int(k): v for k, v in id2label.items()})}",
        "- Feature encoder: frozen during training",
        "- Gradient checkpointing: enabled",
        "",
        "## Training configuration",
        f"- Epochs: {num_epochs}",
        f"- Learning rate: {args.learning_rate}",
        f"- Batch size: {args.batch_size}",
        f"- Optimizer: AdamW (weight_decay=0.01)",
        "- Gradient clipping: norm 1.0",
        f"- Mixed precision: {'yes' if use_amp else 'no'}",
        f"- Random seed: {seed}",
        f"- Device: {device}",
        f"- Train batch steps per epoch: {len(train_loader)}",
        "",
        "## Training metrics per epoch",
        "",
        "| Epoch | Train loss | Dev loss | LR | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in history:
        m = h["metrics"]
        report_lines.append(
            "| {epoch} | {train_loss:.4f} | {dev_loss:.4f} | "
            "{learning_rate:.2e} | {accuracy:.4f} | {macro_f1:.4f} | "
            "{weighted_f1:.4f} |".format(
                epoch=h["epoch"],
                train_loss=h["train_loss"],
                dev_loss=h["dev_loss"],
                learning_rate=h["learning_rate"],
                accuracy=m["accuracy"],
                macro_f1=m["macro_f1"],
                weighted_f1=m["weighted_f1"],
            )
        )

    report_lines += [
        "",
        "## Final results (test)",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Macro F1: {test_metrics['macro_f1']:.4f}",
        f"- Weighted F1: {test_metrics['weighted_f1']:.4f}",
        f"- Macro precision: {test_metrics['macro_precision']:.4f}",
        f"- Macro recall: {test_metrics['macro_recall']:.4f}",
        "",
        "### Per-class results (test)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for i, name in enumerate(label_names):
        report_lines.append(
            "| {name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |".format(
                name=name,
                precision=test_metrics["per_class_precision"][i],
                recall=test_metrics["per_class_recall"][i],
                f1=test_metrics["per_class_f1"][i],
                support=test_metrics["support"][i],
            )
        )

    report_lines += [
        "",
        "### Confusion matrix (test)",
        "",
        "![confusion_matrix](confusion_matrix.png)",
        "",
        "### Classification report CSV",
        "",
        "Saved to `models/audio/classification_report.csv`.",
        "",
        "## Training time",
        "",
        f"- Total wall-clock training time: {train_seconds:.2f} seconds",
        "",
        "## GPU memory considerations",
        "",
        "- GPU memory usage depends heavily on waveform length and batch size.",
        "- Wav2Vec2 feature encoder was frozen and gradient checkpointing was enabled to "
        "reduce memory pressure.",
        "- For ~6 GB VRAM GPUs, reduce `batch_size`, shorten long utterances during "
        "preprocessing, or increase gradient accumulation if needed.",
        "- No claim is made here about which audio model is best; only Wav2Vec2 was "
        "trained in this stage.",
        "",
        "## Output artifacts",
        "",
        "- Model: `models/audio/runs_wav2vec2/`",
        "- Processor: `models/audio/runs_wav2vec2/`",
        "- Report: `models/audio/audio_training_report.md`",
        "- Log: `models/audio/audio_training_log.txt`",
        "- Confusion matrix: `models/audio/confusion_matrix.png` or `models/audio/ravdess_confusion_matrix.png`",
        "- Classification report: `models/audio/classification_report.csv`",
        "- Label mapping: `models/audio/label_mapping.json`",
        "",
        "## Ready for multimodal use?",
        "",
        "This model can serve as the **audio branch** of a future multimodal "
        "emotion-recognition system once:",
        "",
        "- For MELD: the MELD audio directory is available and aligned to the MELD CSVs, and",
        "- For RAVDESS: the prepared `data/raw/ravdess/ravdess_metadata.csv` and extracted WAV files are in place, and",
        "- Train/dev/test splits are made disjoint at the speaker level before "
        "any serious evaluation is trusted.",
        "",
        "At that point, this Wav2Vec2 branch can emit a fixed audio representation "
        "that can be fused with the text branch (already trained to 87.32% accuracy "
        "with DeBERTa-v3-base) and a future face branch.",
    ]

    report_path = PROJECT_ROOT / "models" / "audio" / "audio_training_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    return {
        "history": history,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "train_seconds": train_seconds,
        "model_path": str(MODEL_OUT),
    }


MELD_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma")


def meld_audio_preflight(audio_root: pathlib.Path, logger: logging.Logger) -> None:
    """Fail fast if the configured MELD audio root is missing or empty.

    This runs before model download / GPU training so the run does not waste
    time initializing the trainer when aligned MELD waveforms are absent.
    It does not change dataset or training behavior.
    """
    logger.info("Running MELD audio pre-flight check for: %r", audio_root)

    if not audio_root.exists():
        raise FileNotFoundError(
            "Configured MELD audio root does not exist.\n"
            f"Checked path: {audio_root}\n"
            "\nFix options:\n"
            "- Place the aligned MELD audio directory at this path, or\n"
            "- Point this script at the correct directory with --audio-root.\n"
            "\nExpected MELD audio layout:\n"
            "  Episode{season}_{episode}/\n"
            "    dial{season}_{dialogue}_utt{utterance}.wav\n"
            "Example candidate resolved from CSV fields:\n"
            "  Episode1_1/dial1_0_utt0.wav\n"
        )

    audio_files = [
        p for p in audio_root.rglob("*") if p.is_file() and p.suffix.lower() in MELD_AUDIO_EXTS
    ]

    if not audio_files:
        raise FileNotFoundError(
            "No MELD audio files were found under the configured audio root.\n"
            f"Checked path: {audio_root}\n"
            "\nFix options:\n"
            "- Place the aligned MELD audio directory at this path, or\n"
            "- Point this script at the correct directory with --audio-root.\n"
            "\nExpected MELD audio layout:\n"
            "  Episode{season}_{episode}/\n"
            "    dial{season}_{dialogue}_utt{utterance}.wav\n"
            "Example candidate resolved from CSV fields:\n"
            "  Episode1_1/dial1_0_utt0.wav\n"
        )

    logger.info(
        "MELD audio pre-flight OK: %d audio file(s) found under %r",
        len(audio_files),
        audio_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log-interval", default=20, type=int)
    parser.add_argument(
        "--dataset",
        default="meld",
        choices=["meld", "ravdess"],
        help="Which emotion dataset to train on. 'meld' uses the existing MELD CSVs; 'ravdess' uses data/raw/ravdess/ravdess_metadata.csv.",
    )
    parser.add_argument(
        "--audio-root",
        default=str(AUDIO_DEFAULT),
        help="Root folder containing the audio files for the selected dataset.",
    )
    parser.add_argument(
        "--max-train-samples",
        default=None,
        type=int,
        help="Optional cap on training samples for a quick smoke test.",
    )
    args = parser.parse_args()

    audio_root = pathlib.Path(args.audio_root)

    log_path = PROJECT_ROOT / "models" / "audio" / "audio_training_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path)

    logger.info("audio root: %r", audio_root)
    logger.info("args: %s", args)

    meld_audio_preflight(audio_root, logger)

    logger.info("Selected dataset: %s", args.dataset)
    logger.info("Audio root: %r", audio_root)

    try:
        result = train(args, logger, audio_root)
    except Exception:
        logger.exception("Training failed")
        raise

    logger.info("Done. model_path=%s", result["model_path"])


if __name__ == "__main__":
    main()
