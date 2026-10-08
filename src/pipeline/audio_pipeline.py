"""Audio modality pipeline: speech-emotion inference with the winning model.

Primary (and only) supported model is `emotion2vec` — the best audio model in
models/audio/model_comparison_summary.md (92.7% test acc on RAVDESS vs 75.5%
wavlm / 72.4% wav2vec2, both archived under archive/old_models/audio/).

The backbone is funasr's AutoModel over `iic/emotion2vec_base_finetuned`
(cached after first download), and the trained classification head lives at
models/audio/runs_emotion2vec/model_state_dict.pt (produced by
models/audio/train.py). Output is a probability vector over:

    angry, disgust, fearful, happy, sad, surprised
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIO_RUNS = ROOT / "models" / "audio"

SHARED_EMOTION_CLASSES = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]

SUPPORTED_MODELS = {"emotion2vec": "runs_emotion2vec"}
BACKBONE_HUB_ID = "iic/emotion2vec_base_finetuned"


class AudioPipeline:
    def __init__(self, model_key: str = "emotion2vec", device: str | None = None):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown audio model key: {model_key}. Supported: {list(SUPPORTED_MODELS)}")
        self.model_key = model_key
        self.model_dir = AUDIO_RUNS / SUPPORTED_MODELS[model_key]
        head_path = self.model_dir / "model_state_dict.pt"
        if not head_path.exists():
            raise FileNotFoundError(
                f"No trained head at {head_path}. Fine-tune with models/audio/train.py --model {model_key} first."
            )

        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "emotion2vec requires the funasr package.\n"
                "Install it with:  pip install modelscope funasr addict"
            )

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.label_names = SHARED_EMOTION_CLASSES

        # Backbone runs on CPU numpy, exactly as in models/audio/train.py
        self.backbone = AutoModel(model=BACKBONE_HUB_ID)

        probe = self.backbone.generate(
            np.zeros(16000, dtype="float32"),
            output_dir=None,
            granularity="utterance",
            extract_embedding=True,
        )
        r0 = probe[0]
        emb = r0.get("feats") if r0.get("feats") is not None else r0.get("embedding")
        hidden_size = int(np.array(emb).shape[-1])

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, len(SHARED_EMOTION_CLASSES)),
        )
        state = torch.load(head_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k.removeprefix("classifier."): v for k, v in state.items()}
        self.head.load_state_dict(state)
        self.head.to(self.device).eval()

    def _embed(self, waveform: np.ndarray) -> np.ndarray:
        result = self.backbone.generate(
            waveform.astype("float32"),
            output_dir=None,
            granularity="utterance",
            extract_embedding=True,
        )
        r0 = result[0]
        emb = r0.get("feats") if r0.get("feats") is not None else r0.get("embedding")
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 2:
            emb = emb.mean(0)
        return emb

    def predict(self, waveform: np.ndarray, sampling_rate: int = 16000) -> dict:
        """Classify a 1-D float waveform. Returns probs, label, confidence."""
        import torch

        wav = np.asarray(waveform, dtype=np.float32).squeeze()
        if wav.ndim != 1:
            raise ValueError("waveform must be 1-D")
        if sampling_rate != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)

        emb = self._embed(wav)
        with torch.no_grad():
            logits = self.head(
                torch.tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
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
