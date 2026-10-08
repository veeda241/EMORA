"""Train audio emotion-recognition models on RAVDESS or MELD.

Supported models (--model flag):
  wav2vec2   facebook/wav2vec2-base          — baseline
  wavlm      microsoft/wavlm-base            — comparison
  emotion2vec emotion2vec_base_finetuned     — primary (best for speech emotion)

Usage examples:
  # Train all three models back-to-back on RAVDESS
  python models/train_all_audio_models.py --model all

  # Train a single model
  python models/train_all_audio_models.py --model wavlm --epochs 10 --batch-size 8

  # Quick smoke test
  python models/train_all_audio_models.py --model wav2vec2 --max-train-samples 32 --epochs 1
"""
from __future__ import annotations

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
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_MELD     = PROJECT_ROOT / "data" / "raw" / "meld"
RAW_RAVDESS  = PROJECT_ROOT / "data" / "raw" / "ravdess"
AUDIO_OUT    = PROJECT_ROOT / "models" / "audio"
AUDIO_DEFAULT_MELD    = PROJECT_ROOT / "data" / "raw" / "audio"
AUDIO_DEFAULT_RAVDESS = PROJECT_ROOT / "data" / "raw" / "ravdess"

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry:  display_name -> (hf_checkpoint, output_subdir)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "wav2vec2":   ("facebook/wav2vec2-base",         "runs_wav2vec2"),
    "wavlm":      ("microsoft/wavlm-base",           "runs_wavlm"),
    "emotion2vec": ("iic/emotion2vec_base_finetuned", "runs_emotion2vec"),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: pathlib.Path, model_key: str) -> logging.Logger:
    name = f"train_audio_{model_key}"
    logger = logging.getLogger(name)
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_meld_csv(csv_path: pathlib.Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ravdess_metadata(csv_path: pathlib.Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_label_mapping_meld(rows: list[dict]) -> tuple[dict[str, int], dict[int, str], int]:
    labels = sorted({r["Emotion"] for r in rows if r.get("Emotion")})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label  = {i: l for l, i in label2id.items()}
    return label2id, id2label, len(labels)


def build_label_mapping_ravdess() -> tuple[dict[str, int], dict[int, str], int]:
    """Canonical 6-class RAVDESS mapping (neutral/calm excluded)."""
    names = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
    label2id = {n: i for i, n in enumerate(names)}
    id2label  = {i: n for n, i in label2id.items()}
    return label2id, id2label, len(names)

# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def audio_present(audio_root: pathlib.Path) -> bool:
    if not audio_root.exists():
        return False
    return any(
        p.suffix.lower() in AUDIO_EXTS
        for p in audio_root.rglob("*")
        if p.is_file()
    )


def resolve_meld_audio_path(row: dict, audio_root: pathlib.Path) -> pathlib.Path:
    season      = str(row.get("Season",      "")).strip()
    episode     = str(row.get("Episode",     "")).strip()
    dialogue_id = str(row.get("Dialogue_ID", "")).strip()
    utterance_id= str(row.get("Utterance_ID","")).strip()
    candidate = (
        audio_root
        / f"Episode{season}_{episode}"
        / f"dial{season}_{dialogue_id}_utt{utterance_id}.wav"
    )
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot locate MELD audio for Dialogue_ID={dialogue_id!r} "
        f"Utterance_ID={utterance_id!r} Season={season!r} Episode={episode!r}. "
        f"Tried: {candidate!r}"
    )


def resolve_ravdess_audio_path(row: dict, audio_root: pathlib.Path) -> pathlib.Path:
    file_path = row.get("file_path")
    if not file_path:
        raise KeyError(f"RAVDESS row is missing file_path: {row!r}")
    candidate = pathlib.Path(file_path)
    # Try relative to audio_root first
    path = audio_root / candidate if not candidate.is_absolute() else candidate
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Cannot locate RAVDESS audio. Resolved path: {path}. "
        f"audio_root: {audio_root}."
    )


def load_waveform(path: pathlib.Path, target_sr: int) -> np.ndarray:
    """Load an audio file as a mono float32 waveform resampled to target_sr."""
    import librosa
    waveform, orig_sr = librosa.load(str(path), sr=None, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Empty waveform: {path}")
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Non-finite samples in: {path}")
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(torch.utils.data.Dataset):
    """Generic audio dataset compatible with any HF processor."""

    def __init__(
        self,
        rows: list[dict],
        processor,                      # any HF feature-extractor / processor
        label2id: dict[str, int],
        audio_root: pathlib.Path,
        resolver,                       # (row, audio_root) -> Path
        target_sr: int,
        emotion_key: str = "emotion",   # "emotion" for RAVDESS, "Emotion" for MELD
        max_samples: int | None = None,
    ):
        self.rows       = rows[:max_samples] if max_samples else rows
        self.processor  = processor
        self.label2id   = label2id
        self.audio_root = audio_root
        self.resolver   = resolver
        self.target_sr  = target_sr
        self.emotion_key= emotion_key
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        row = self.rows[idx]
        path = self.resolver(row, self.audio_root)
        waveform = load_waveform(path, self.target_sr)

        emotion = row.get(self.emotion_key) or row.get("Emotion") or row.get("emotion")
        if emotion is None:
            raise KeyError(f"Row missing emotion label: {row!r}")
        label = int(self.label2id[emotion])

        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="np",
            padding=False,
        )
        out = {"input_values": inputs.input_values[0], "labels": label}
        self._cache[idx] = out
        return out


def collate_fn(batch: list[dict]) -> dict:
    input_values = [b["input_values"] for b in batch]
    labels       = [b["labels"]       for b in batch]
    max_len = max(iv.shape[0] for iv in input_values)
    padded  = np.zeros((len(input_values), max_len), dtype=np.float32)
    for i, iv in enumerate(input_values):
        padded[i, : iv.shape[0]] = iv
    return {
        "input_values": torch.from_numpy(padded),
        "labels":       torch.tensor(labels, dtype=torch.long),
    }

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray, labels: np.ndarray, label_names: list[str]
) -> dict:
    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(label_names))), zero_division=0
    )
    macro_p,    macro_r,    macro_f1,    _ = precision_recall_fscore_support(labels, preds, average="macro",    zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    return {
        "accuracy":           acc,
        "macro_precision":    float(macro_p),
        "macro_recall":       float(macro_r),
        "macro_f1":           float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall":    float(weighted_r),
        "weighted_f1":        float(weighted_f1),
        "per_class_precision":prec.tolist(),
        "per_class_recall":   rec.tolist(),
        "per_class_f1":       f1.tolist(),
        "support":            support.tolist(),
        "confusion_matrix":   cm.tolist(),
        "label_names":        label_names,
        "num_classes":        len(label_names),
    }


def _nan_check(name: str, value: float, logger: logging.Logger) -> None:
    if math.isnan(value) or math.isinf(value):
        logger.error("NaN/Inf metric: %s = %r", name, value)
        raise RuntimeError(f"Non-finite metric {name}={value}.")

# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def _check_audio_files(
    rows: list[dict],
    audio_root: pathlib.Path,
    resolver,
) -> list[str]:
    """Return up to 10 missing-file messages."""
    missing = []
    for r in rows:
        try:
            resolver(r, audio_root)
        except (FileNotFoundError, KeyError) as exc:
            missing.append(str(exc))
            if len(missing) >= 10:
                break
    return missing


def validate_data(
    train_rows: list[dict],
    dev_rows:   list[dict],
    test_rows:  list[dict],
    label2id:   dict[str, int],
    audio_root: pathlib.Path,
    resolver,
    emotion_key: str,
    logger: logging.Logger,
    skip_audio_check: bool = False,
) -> None:
    """Validate labels and (optionally) audio file presence."""
    problems: list[str] = []
    for split_name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
        for r in rows:
            label = r.get(emotion_key)
            if label not in label2id:
                problems.append(f"unknown label {label!r} in {split_name}")
    if problems:
        raise ValueError("Label problems:\n" + "\n".join(problems[:20]))

    if not audio_present(audio_root):
        if skip_audio_check:
            logger.warning("No audio files found under %r — skip_audio_check is set, continuing.", audio_root)
            return
        raise RuntimeError(
            f"No audio files found under {audio_root}.\n"
            "Place audio files there or pass --skip-audio-check."
        )

    missing = _check_audio_files(train_rows + dev_rows + test_rows, audio_root, resolver)
    if missing:
        raise FileNotFoundError(
            "Missing audio files:\n" + "\n".join(missing)
        )
    logger.info("Dataset validation passed.")

# ---------------------------------------------------------------------------
# Model factory — builds processor + model for each backend
# ---------------------------------------------------------------------------

def _load_wav2vec2(checkpoint: str, num_labels: int, label2id: dict, id2label: dict):
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        label2id={str(k): v for k, v in label2id.items()},
        id2label={str(k): v for k, v in id2label.items()},
        attention_dropout=0.1,
        hidden_dropout=0.1,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    target_sr = processor.feature_extractor.sampling_rate
    return processor, model, target_sr


def _load_wavlm(checkpoint: str, num_labels: int, label2id: dict, id2label: dict):
    """WavLM uses the same API as Wav2Vec2 for sequence classification."""
    from transformers import (
        WavLMForSequenceClassification,
        Wav2Vec2FeatureExtractor,
    )
    # WavLM ships a Wav2Vec2FeatureExtractor-compatible extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
    model = WavLMForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        label2id={str(k): v for k, v in label2id.items()},
        id2label={str(k): v for k, v in id2label.items()},
        attention_dropout=0.1,
        hidden_dropout=0.1,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    target_sr = feature_extractor.sampling_rate
    return feature_extractor, model, target_sr


def _load_emotion2vec(checkpoint: str, num_labels: int, label2id: dict, id2label: dict):
    """emotion2vec via funasr AutoModel, with a classification head on top.

    Uses funasr.AutoModel (the recommended interface) to extract utterance-level
    embeddings from iic/emotion2vec_base_finetuned, then trains a linear
    classification head on top for our target label set.
    """
    try:
        from funasr import AutoModel
    except ImportError:
        raise ImportError(
            "emotion2vec requires the funasr package.\n"
            "Install it with:  pip install modelscope funasr addict\n"
            "Then re-run with --model emotion2vec."
        )

    # Load backbone once — funasr downloads + caches the model on first call
    backbone = AutoModel(model=checkpoint)

    # emotion2vec exposes embeddings; we wrap it with a classification head.
    class Emotion2VecClassifier(torch.nn.Module):
        def __init__(self, backbone_model, hidden_size: int, n_labels: int):
            super().__init__()
            self.backbone = backbone_model
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 256),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, n_labels),
            )
            self.loss_fn = torch.nn.CrossEntropyLoss()

        def forward(self, input_values: torch.Tensor, labels: torch.Tensor | None = None):
            # Run backbone on CPU numpy, collect utterance-level embeddings
            batch_embs = []
            for wav in input_values:
                wav_np = wav.cpu().numpy().astype("float32")
                result = self.backbone.generate(
                    wav_np,
                    output_dir=None,
                    granularity="utterance",
                    extract_embedding=True,
                )
                # result is a list of dicts; grab the embedding from first entry
                # use explicit None-check to avoid ambiguous truth-value on arrays
                r0   = result[0]
                emb  = r0.get("feats") if r0.get("feats") is not None else r0.get("embedding")
                emb_t = torch.tensor(emb, dtype=torch.float32)
                if emb_t.dim() == 2:
                    emb_t = emb_t.mean(0)   # mean-pool over time frames
                batch_embs.append(emb_t)
            embeddings = torch.stack(batch_embs).to(input_values.device)
            logits = self.classifier(embeddings)
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            # Return an object with .loss and .logits (same interface as HF models)
            class _Out:
                pass
            out = _Out()
            out.loss   = loss
            out.logits = logits
            return out

    # Probe embedding dimension with a short dummy waveform so we never hard-code the wrong size
    import numpy as _np
    _dummy = _np.zeros(16000, dtype="float32")
    _probe = backbone.generate(_dummy, output_dir=None, granularity="utterance", extract_embedding=True)
    _r0    = _probe[0]
    _emb   = _r0.get("feats") if _r0.get("feats") is not None else _r0.get("embedding")
    _emb_arr = _np.array(_emb)
    hidden_size = int(_emb_arr.shape[-1])   # last dim is embedding size (768 for base)

    model = Emotion2VecClassifier(backbone, hidden_size=hidden_size, n_labels=num_labels)

    # Use a plain Wav2Vec2FeatureExtractor as the processor (16 kHz mono)
    from transformers import Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    target_sr = 16000
    return feature_extractor, model, target_sr


# Map model key -> loader function
_MODEL_LOADERS = {
    "wav2vec2":    _load_wav2vec2,
    "wavlm":       _load_wavlm,
    "emotion2vec": _load_emotion2vec,
}

# ---------------------------------------------------------------------------
# Core training loop (shared across all models)
# ---------------------------------------------------------------------------

def run_training(
    args:        argparse.Namespace,
    logger:      logging.Logger,
    audio_root:  pathlib.Path,
    model_key:   str,
    train_rows:  list[dict],
    dev_rows:    list[dict],
    test_rows:   list[dict],
    label2id:    dict[str, int],
    id2label:    dict[int, str],
    num_labels:  int,
    label_names: list[str],
    emotion_key: str,
    resolver,
    output_dir:  pathlib.Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Device: %s (%s)", device, torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA unavailable — running on CPU (slow).")

    checkpoint, _ = MODEL_REGISTRY[model_key]
    logger.info("Loading checkpoint: %s", checkpoint)
    processor, model, target_sr = _MODEL_LOADERS[model_key](
        checkpoint, num_labels, label2id, id2label
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=0.01
    )

    # Build DataLoaders
    mk_ds = lambda rows, max_s=None: AudioDataset(
        rows, processor, label2id, audio_root, resolver,
        target_sr, emotion_key=emotion_key, max_samples=max_s,
    )
    train_ds = mk_ds(train_rows, args.max_train_samples)
    dev_ds   = mk_ds(dev_rows)
    test_ds  = mk_ds(test_rows)

    mk_dl = lambda ds, shuffle: torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    train_loader = mk_dl(train_ds, shuffle=True)
    dev_loader   = mk_dl(dev_ds,   shuffle=False)
    test_loader  = mk_dl(test_ds,  shuffle=False)

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dev_loss = float("inf")
    best_state    = None
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        # --- train ---
        model.train()
        epoch_loss, epoch_steps = 0.0, 0
        lr_now = float(optimizer.param_groups[0]["lr"])
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(**batch)
                loss = out.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss  += float(loss.item())
            epoch_steps += 1
            if step % args.log_interval == 0:
                logger.info(
                    "epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                    epoch, step, len(train_loader), float(loss.item()), lr_now,
                )

        train_loss = epoch_loss / max(epoch_steps, 1)
        _nan_check("train_loss", train_loss, logger)

        # --- validate ---
        model.eval()
        dev_loss, dev_steps = 0.0, 0
        dev_preds: list[int] = []
        dev_labels: list[int] = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(**batch)
                if not torch.isfinite(out.loss):
                    raise RuntimeError("Non-finite dev loss.")
                dev_loss  += float(out.loss.item())
                dev_steps += 1
                dev_preds.extend(out.logits.argmax(-1).cpu().tolist())
                dev_labels.extend(batch["labels"].cpu().tolist())

        dev_loss_avg = dev_loss / max(dev_steps, 1)
        _nan_check("dev_loss", dev_loss_avg, logger)
        m = compute_metrics(np.array(dev_preds), np.array(dev_labels), label_names)
        logger.info(
            "epoch=%d train_loss=%.4f dev_loss=%.4f acc=%.4f macro_f1=%.4f",
            epoch, train_loss, dev_loss_avg, m["accuracy"], m["macro_f1"],
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "dev_loss": dev_loss_avg, "learning_rate": lr_now, "metrics": m,
        })
        if dev_loss_avg < best_dev_loss:
            best_dev_loss = dev_loss_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- final evaluation ---
    def evaluate(loader):
        model.eval()
        preds, lbls = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                lbls.extend(batch["labels"].cpu().tolist())
        return compute_metrics(np.array(preds), np.array(lbls), label_names)

    dev_metrics  = evaluate(dev_loader)
    test_metrics = evaluate(test_loader)

    # --- save model ---
    output_dir.mkdir(parents=True, exist_ok=True)
    # HF models have save_pretrained; emotion2vec wrapper does not
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(output_dir)
    logger.info("Saved model to %s", output_dir)

    # --- confusion matrix ---
    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{model_key} confusion matrix (test)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(label_names))); ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    cm_path = AUDIO_OUT / f"{model_key}_confusion_matrix.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", cm_path)

    # --- classification CSV ---
    csv_path = AUDIO_OUT / f"{model_key}_classification_report.csv"
    rows_csv = [
        {"class": name,
         "precision": test_metrics["per_class_precision"][i],
         "recall":    test_metrics["per_class_recall"][i],
         "f1":        test_metrics["per_class_f1"][i],
         "support":   test_metrics["support"][i]}
        for i, name in enumerate(label_names)
    ]
    rows_csv += [
        {"class": "macro_avg",    "precision": test_metrics["macro_precision"],    "recall": test_metrics["macro_recall"],    "f1": test_metrics["macro_f1"],    "support": ""},
        {"class": "weighted_avg", "precision": test_metrics["weighted_precision"], "recall": test_metrics["weighted_recall"], "f1": test_metrics["weighted_f1"], "support": ""},
        {"class": "accuracy",     "precision": "",                                  "recall": "",                              "f1": test_metrics["accuracy"],    "support": ""},
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class","precision","recall","f1","support"])
        w.writeheader(); w.writerows(rows_csv)
    logger.info("Saved classification report to %s", csv_path)

    # --- markdown report ---
    report_lines = [
        f"# Audio Training Report — {model_key} ({args.dataset.upper()})",
        "",
        "## Model",
        f"- Key: `{model_key}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "## Dataset",
        f"- Name: {args.dataset.upper()}",
        f"- Train: {len(train_rows)}, Dev: {len(dev_rows)}, Test: {len(test_rows)}",
        f"- Emotion classes ({num_labels}): {', '.join(label_names)}",
        "",
        "## Training config",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Learning rate: {args.learning_rate}",
        f"- Optimizer: AdamW (weight_decay=0.01)",
        f"- Grad clip norm: 1.0",
        f"- Mixed precision: {'yes' if use_amp else 'no'}",
        f"- Seed: {args.seed}",
        f"- Device: {device}",
        "",
        "## Per-epoch metrics",
        "",
        "| Epoch | Train Loss | Dev Loss | Accuracy | Macro F1 |",
        "|---|---|---|---|---|",
    ]
    for h in history:
        m2 = h["metrics"]
        report_lines.append(
            f"| {h['epoch']} | {h['train_loss']:.4f} | {h['dev_loss']:.4f} "
            f"| {m2['accuracy']:.4f} | {m2['macro_f1']:.4f} |"
        )
    report_lines += [
        "",
        "## Test results",
        f"- Accuracy:   {test_metrics['accuracy']:.4f}",
        f"- Macro F1:   {test_metrics['macro_f1']:.4f}",
        f"- Weighted F1:{test_metrics['weighted_f1']:.4f}",
        "",
        "### Per-class (test)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for i, name in enumerate(label_names):
        report_lines.append(
            f"| {name} | {test_metrics['per_class_precision'][i]:.4f} "
            f"| {test_metrics['per_class_recall'][i]:.4f} "
            f"| {test_metrics['per_class_f1'][i]:.4f} "
            f"| {test_metrics['support'][i]} |"
        )
    report_lines += [
        "",
        f"## Training time: {train_seconds:.1f} s",
        "",
        f"## Output",
        f"- Weights: `{output_dir}`",
        f"- Confusion matrix: `{cm_path}`",
        f"- Classification report: `{csv_path}`",
    ]

    report_path = AUDIO_OUT / f"{model_key}_training_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    return {
        "model_key":     model_key,
        "history":       history,
        "dev_metrics":   dev_metrics,
        "test_metrics":  test_metrics,
        "train_seconds": train_seconds,
        "output_dir":    str(output_dir),
    }

# ---------------------------------------------------------------------------
# MELD preflight
# ---------------------------------------------------------------------------

def meld_preflight(audio_root: pathlib.Path, logger: logging.Logger) -> None:
    if not audio_root.exists():
        raise FileNotFoundError(
            f"MELD audio root does not exist: {audio_root}\n"
            "Place aligned MELD audio there or pass --audio-root."
        )
    files = [p for p in audio_root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise FileNotFoundError(
            f"No audio files found under {audio_root}.\n"
            "Place aligned MELD WAV files there first."
        )
    logger.info("MELD preflight OK: %d audio file(s) found.", len(files))

# ---------------------------------------------------------------------------
# Top-level train() — sets up data then dispatches run_training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, logger: logging.Logger, audio_root: pathlib.Path, model_key: str) -> dict:
    torch.set_float32_matmul_precision("high" if torch.cuda.is_available() else "medium")

    seed = int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    dataset_name = str(args.dataset).lower()

    if dataset_name == "ravdess":
        ravdess_csv = RAW_RAVDESS / "ravdess_metadata.csv"
        if not ravdess_csv.exists():
            raise FileNotFoundError(f"RAVDESS metadata not found: {ravdess_csv}")
        all_rows   = load_ravdess_metadata(ravdess_csv)
        train_rows = [r for r in all_rows if r["split"] == "train"]
        dev_rows   = [r for r in all_rows if r["split"] == "validation"]
        test_rows  = [r for r in all_rows if r["split"] == "test"]
        label2id, id2label, num_labels = build_label_mapping_ravdess()
        emotion_key = "emotion"
        resolver    = resolve_ravdess_audio_path
    else:
        train_rows = load_meld_csv(RAW_MELD / "train_sent_emo.csv")
        dev_rows   = load_meld_csv(RAW_MELD / "dev_sent_emo.csv")
        test_rows  = load_meld_csv(RAW_MELD / "test_sent_emo.csv")
        label2id, id2label, num_labels = build_label_mapping_meld(train_rows + dev_rows + test_rows)
        emotion_key = "Emotion"
        resolver    = resolve_meld_audio_path

    label_names = [id2label[i] for i in range(num_labels)]
    logger.info(
        "%s rows — train=%d dev=%d test=%d | labels=%s",
        dataset_name.upper(), len(train_rows), len(dev_rows), len(test_rows), label_names,
    )

    validate_data(
        train_rows, dev_rows, test_rows,
        label2id, audio_root, resolver, emotion_key, logger,
        skip_audio_check=args.skip_audio_check,
    )

    _, out_subdir = MODEL_REGISTRY[model_key]
    output_dir = AUDIO_OUT / out_subdir

    return run_training(
        args, logger, audio_root, model_key,
        train_rows, dev_rows, test_rows,
        label2id, id2label, num_labels, label_names,
        emotion_key, resolver, output_dir,
    )

# ---------------------------------------------------------------------------
# Summary table (written after --model all)
# ---------------------------------------------------------------------------

def write_summary_table(results: list[dict]) -> None:
    lines = [
        "# Audio Model Comparison Summary",
        "",
        "| Model | Test Accuracy | Macro F1 | Weighted F1 | Train time (s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        tm = r["test_metrics"]
        lines.append(
            f"| {r['model_key']} "
            f"| {tm['accuracy']:.4f} "
            f"| {tm['macro_f1']:.4f} "
            f"| {tm['weighted_f1']:.4f} "
            f"| {r['train_seconds']:.1f} |"
        )
    path = AUDIO_OUT / "model_comparison_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparison summary saved to {path}")
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    model_choices = list(MODEL_REGISTRY.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default="wav2vec2", choices=model_choices,
        help=(
            "Which model to train. "
            "'wav2vec2' = facebook/wav2vec2-base (baseline), "
            "'wavlm' = microsoft/wavlm-base (comparison), "
            "'emotion2vec' = emotion2vec_base_finetuned (primary, needs modelscope+funasr), "
            "'all' = train all three sequentially."
        ),
    )
    parser.add_argument(
        "--dataset", default="ravdess", choices=["meld", "ravdess"],
        help="Dataset to train on.",
    )
    parser.add_argument("--epochs",           default=10,  type=int)
    parser.add_argument("--batch-size",       default=8,   type=int)
    parser.add_argument("--learning-rate",    default=2e-5, type=float)
    parser.add_argument("--seed",             default=42,  type=int)
    parser.add_argument("--log-interval",     default=20,  type=int)
    parser.add_argument(
        "--audio-root", default=None,
        help="Override audio root. Defaults to data/raw/ravdess or data/raw/audio.",
    )
    parser.add_argument(
        "--max-train-samples", default=None, type=int,
        help="Cap training rows for a quick smoke test.",
    )
    parser.add_argument(
        "--skip-audio-check", action="store_true", default=False,
        help="Skip file-existence validation at startup.",
    )
    args = parser.parse_args()

    # Resolve audio root default
    if args.audio_root is None:
        args.audio_root = str(
            AUDIO_DEFAULT_RAVDESS if args.dataset == "ravdess" else AUDIO_DEFAULT_MELD
        )
    audio_root = pathlib.Path(args.audio_root)

    # Which models to run
    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)

    # MELD preflight only when needed
    if args.dataset == "meld":
        tmp_log = setup_logging(AUDIO_OUT / "audio_training_log.txt", "preflight")
        meld_preflight(audio_root, tmp_log)

    all_results: list[dict] = []
    for model_key in models_to_run:
        log_path = AUDIO_OUT / f"{model_key}_training_log.txt"
        logger   = setup_logging(log_path, model_key)
        logger.info("=" * 60)
        logger.info("Training model: %s  dataset: %s", model_key, args.dataset)
        logger.info("audio_root: %s", audio_root)
        logger.info("args: %s", args)
        logger.info("=" * 60)
        try:
            result = train(args, logger, audio_root, model_key)
            all_results.append(result)
            logger.info(
                "DONE %s — test_accuracy=%.4f macro_f1=%.4f",
                model_key,
                result["test_metrics"]["accuracy"],
                result["test_metrics"]["macro_f1"],
            )
        except ImportError as exc:
            logger.error("Skipping %s — missing dependency: %s", model_key, exc)
        except Exception:
            logger.exception("Training failed for model: %s", model_key)
            if args.model != "all":
                raise   # re-raise if user asked for a single model

    if len(all_results) > 1:
        write_summary_table(all_results)
    elif all_results:
        r  = all_results[0]
        tm = r["test_metrics"]
        print(
            f"\n{r['model_key']} — "
            f"accuracy={tm['accuracy']:.4f}  macro_f1={tm['macro_f1']:.4f}  "
            f"weighted_f1={tm['weighted_f1']:.4f}  "
            f"time={r['train_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()
