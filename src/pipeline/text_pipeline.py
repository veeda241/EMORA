"""Text modality pipeline: GoEmotions RoBERTa inference collapsed to 6 classes.

Primary path loads the fine-tuned checkpoint from
models/text/checkpoints/runs_roberta-base/checkpoint-*/ — the GoEmotions
28-class probabilities are collapsed onto the shared 6-class space using the
official GoEmotions->Ekman taxonomy (models/common/label_mapping.py).

If the checkpoint directory has no model weights (only config/tokenizer), the
pipeline falls back to a transparent lexicon scorer so the conversational agent
still runs end-to-end. The fallback is clearly flagged in its output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.label_mapping import (  # noqa: E402
    SHARED_EMOTION_CLASSES,
    collapse_goemotions_probs,
)

TEXTCHECKPOINT_DIR = ROOT / "models" / "text" / "checkpoints" / "runs_roberta-base"

# Lexicon fallback (used only when no fine-tuned weights are available).
LEXICON = {
    "angry": ["angry", "furious", "mad", "annoyed", "irritated", "hate", "outraged", "pissed", "frustrated", "unfair"],
    "disgust": ["disgusted", "gross", "revolting", "sickening", "nasty", "appalling", "yuck"],
    "fearful": ["scared", "afraid", "terrified", "anxious", "worried", "nervous", "panic", "frightened", "overwhelmed", "stressed"],
    "happy": ["happy", "great", "good", "excited", "glad", "love", "awesome", "wonderful", "joy", "delighted", "proud", "grateful", "hopeful", "better", "fine", "okay", "calm"],
    "sad": ["sad", "down", "depressed", "unhappy", "miserable", "lonely", "hurt", "cry", "tired", "exhausted", "disappointed", "hopeless", "burnt", "burned", "drained"],
    "surprised": ["surprised", "shocked", "wow", "unexpected", "unbelievable", "amazed", "confused", "what", "really"],
}


class TextPipeline:
    def __init__(self, checkpoint: str | None = None):
        self.backend = None
        self.model_dir = None

        candidates = [checkpoint] if checkpoint else sorted(
            TEXTCHECKPOINT_DIR.glob("checkpoint-*"), key=lambda p: p.name
        )
        for cand in candidates:
            if cand is None:
                continue
            cand = Path(cand)
            if not (cand / "config.json").exists():
                continue
            if any((cand / n).exists() for n in ("model.safetensors", "pytorch_model.bin")):
                self._load_transformers(cand)
                return

        # No weights anywhere — use the lexicon fallback.
        self.backend = "lexicon"

    def _load_transformers(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.backend = "transformers"
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.goemotions_labels = [id2label[i] for i in range(len(id2label))]
        self.model_name = f"goemotions-{model_dir.parent.name.removeprefix('runs_')}"

    def _predict_transformers(self, text: str) -> dict:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # GoEmotions is multi-label
        shared = np.asarray(collapse_goemotions_probs(self.goemotions_labels, probs))
        return self._package(shared, self.model_name)

    def _predict_lexicon(self, text: str) -> dict:
        tokens = set(t.strip(".,!?;:'\"()").lower() for t in text.split())
        scores = np.zeros(len(SHARED_EMOTION_CLASSES), dtype=np.float64)
        for i, emo in enumerate(SHARED_EMOTION_CLASSES):
            scores[i] = sum(1.0 for w in LEXICON[emo] if w in tokens)
        total = scores.sum()
        probs = scores / total if total > 0 else np.full(len(SHARED_EMOTION_CLASSES), 1.0 / len(SHARED_EMOTION_CLASSES))
        return self._package(probs, "lexicon-fallback")

    def _package(self, probs: np.ndarray, model_name: str) -> dict:
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(SHARED_EMOTION_CLASSES), 1.0 / len(SHARED_EMOTION_CLASSES))
        idx = int(np.argmax(probs))
        return {
            "probs": {SHARED_EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(probs))},
            "probs_array": probs.tolist(),
            "label": SHARED_EMOTION_CLASSES[idx],
            "confidence": float(probs[idx]),
            "modality": "text",
            "model": model_name,
        }

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return self._package(np.ones(len(SHARED_EMOTION_CLASSES)), self.backend_name())
        if self.backend == "transformers":
            return self._predict_transformers(text)
        return self._predict_lexicon(text)

    def backend_name(self) -> str:
        return self.model_name if self.backend == "transformers" else "lexicon-fallback"
