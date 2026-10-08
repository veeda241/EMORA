"""Audio modality pipeline: Wav2Vec2/WavLM speech-emotion inference.

Loads a fine-tuned checkpoint from models/audio/runs_{model_key}/ (weights +
processor + config, already in the shared 6-class vocabulary), runs waveform
through it, and returns probabilities over:

    angry, disgust, fearful, happy, sad, surprised
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIO_RUNS = ROOT / "models" / "audio"

SHARED_EMOTION_CLASSES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]

SUPPORTED_MODELS = {"wav2vec2": "runs_wav2vec2", "wavlm": "runs_wavlm"}


class AudioPipeline:
    def __init__(self, model_key: str = "wav2vec2", device: str | None = None):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown audio model key: {model_key}. Supported: {list(SUPPORTED_MODELS)}")
        self.model_key = model_key
        self.model_dir = AUDIO_RUNS / SUPPORTED_MODELS[model_key]
        if not (self.model_dir / "model.safetensors").exists():
            raise FileNotFoundError(
                f"No weights at {self.model_dir}. Fine-tune with models/audio/train.py first."
            )

        import torch
        from transformers import AutoModelForAudioClassification, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_dir).to(device)
        self.model.eval()

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label_names = [id2label[i] for i in range(len(id2label))]
        if sorted(self.label_names) != sorted(SHARED_EMOTION_CLASSES):
            raise ValueError(
                f"Checkpoint labels {self.label_names} do not match the shared vocabulary "
                f"{SHARED_EMOTION_CLASSES}"
            )

    def predict(self, waveform: np.ndarray, sampling_rate: int = 16000) -> dict:
        """Classify a 1-D float waveform. Returns probs, label, confidence."""
        import torch

        wav = np.asarray(waveform, dtype=np.float32).squeeze()
        if wav.ndim != 1:
            raise ValueError("waveform must be 1-D")
        inputs = self.processor(wav, sampling_rate=sampling_rate, return_tensors="pt")
        pixel = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_values", "attention_mask")}
        with torch.no_grad():
            logits = self.model(**pixel).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "probs": {self.label_names[i]: float(probs[i]) for i in range(len(probs))},
            "probs_array": probs.tolist(),
            "label": self.label_names[idx],
            "confidence": float(probs[idx]),
            "modality": "audio",
            "model": self.model_key,
        }
