"""Conversational agent: LLM-based behaviour change coach.

Orchestrates the full loop:
    user input (text / audio / video features)
      -> per-modality emotion estimate (or MFN feature fusion)
      -> ConversationMemory update
      -> composite EmotionalState (current + recent history + behavioural trend)
      -> ResponseStrategy selection
      -> ResponseGenerator reply following that strategy
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from src.config import CONFIG
from src.conversation.emotional_state import assess_emotional_state, select_strategy
from src.conversation.memory import ConversationMemory
from src.conversation.response_generator import ResponseGenerator
from src.fusion.emotion_fusion import EmotionFusion
from src.fusion.trained_fusion import TrainedFusion
from src.pipeline.multimodal_pipeline import MultimodalEmotionPipeline

MODEL_PATH = f"{CONFIG['model_dir']}/runs_mfn/best_model.pt"


class BehaviourChangeAgent:
    def __init__(
        self,
        goal: str = "general wellbeing",
        user_id: str = "default",
        llm_config: dict[str, Any] | None = None,
        fusion_weights: dict[str, float] | None = None,
        device: str | None = None,
        memory_path: str | None = None,
    ):
        self.memory = ConversationMemory(user_id=user_id, goal=goal, storage_path=memory_path)
        # Prefer learned decision-level fusion when a trained model exists;
        # otherwise fall back to fixed-weight averaging. Both expose .fuse().
        if TrainedFusion.available():
            self.fusion = TrainedFusion()
        else:
            self.fusion = EmotionFusion(fusion_weights)
        self.generator = ResponseGenerator(llm_config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.mfn: MultimodalEmotionPipeline | None = None
        try:
            model = MultimodalEmotionPipeline(CONFIG)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.mfn = model
        except (FileNotFoundError, OSError):
            # MFN not trained yet - agent still works via decision-level fusion
            self.mfn = None

    @property
    def uses_mfn(self) -> bool:
        return self.mfn is not None

    # ── emotion estimation ────────────────────────────────────────────────

    @torch.no_grad()
    def _mfn_predict(self, text_seq: np.ndarray, audio_seq: np.ndarray, video_seq: np.ndarray) -> dict:
        t = torch.tensor(text_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor(audio_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = torch.tensor(video_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.mfn.predict_emotion(t, a, v)

    def estimate_emotion(
        self,
        modality_predictions: dict[str, np.ndarray] | None = None,
        aligned_sequences: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict | None:
        """Estimate emotion from either aligned sequences (MFN) or
        per-modality probability vectors (decision fusion)."""
        if aligned_sequences is not None and self.mfn is not None:
            res = self._mfn_predict(*aligned_sequences)
            return {
                "label": res["label"][0],
                "label_index": int(res["label_index"][0].item()),
                "confidence": float(res["confidence"][0].item()),
                "probs": res["probs"][0].cpu().numpy(),
                "modalities_used": ["text", "audio", "video"],
            }
        if modality_predictions:
            return self.fusion.fuse(modality_predictions)
        return None

    # ── main loop ─────────────────────────────────────────────────────────

    def respond(
        self,
        user_text: str,
        modality_predictions: dict[str, np.ndarray] | None = None,
        aligned_sequences: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict:
        """Process one user turn and return the agent's reply.

        The returned dict exposes every pipeline block so the chain is
        inspectable end to end: raw emotion estimate, composite emotional
        state, selected response strategy, and the memory summary.
        """
        emotion = self.estimate_emotion(modality_predictions, aligned_sequences)
        self.memory.add_user_turn(
            user_text,
            emotion=emotion["label"] if emotion else None,
            confidence=emotion["confidence"] if emotion else None,
        )
        state = assess_emotional_state(
            self.memory,
            emotion["label"] if emotion else None,
            emotion["confidence"] if emotion else None,
        )
        strategy = select_strategy(state)
        reply = self.generator.generate(self.memory, state, strategy)
        self.memory.add_agent_turn(reply)
        return {
            "reply": reply,
            "emotion": emotion,
            "emotional_state": asdict(state),
            "strategy": asdict(strategy),
            "memory": self.memory.summary(),
        }

    def status(self) -> dict:
        return {
            **self.memory.summary(),
            "mfn_loaded": self.uses_mfn,
            "fusion_mode": "trained" if isinstance(self.fusion, TrainedFusion) else "fixed-weight",
            "llm_mode": self.generator.uses_llm,
        }


def demo() -> None:
    """Text-only demo loop with random-text emotion placeholders."""
    agent = BehaviourChangeAgent(goal="reduce daily stress")
    print(f"[agent ready] mfn={agent.uses_mfn} llm={agent.generator.uses_llm}")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text or user_text.lower() in {"quit", "exit"}:
            break
        out = agent.respond(user_text)
        emo = out["emotion"]
        tag = f" ({emo['label']}, {emo['confidence']:.2f})" if emo else ""
        strat = out["strategy"]["name"]
        trend = out["emotional_state"]["trend"]
        print(f"Agent{tag} [strategy={strat}, trend={trend}]: {out['reply']}\n")


if __name__ == "__main__":
    demo()
