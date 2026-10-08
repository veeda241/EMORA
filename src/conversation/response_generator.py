"""Response generator: produces empathetic replies conditioned on emotion state.

Two modes:
- LLM mode: any OpenAI-compatible chat endpoint (base_url + api_key + model).
  If unavailable, silently falls back to template mode.
- Template mode (default): deterministic motivational-interviewing style
  responses composed from prompts.py - no external service required.
"""

from __future__ import annotations

from typing import Any

from src.conversation.prompts import (
    COACHING_QUESTIONS,
    EMOTION_ACKNOWLEDGEMENTS,
    EMOTIONAL_SHIFT_REACTION,
    SYSTEM_PROMPT,
    build_user_prompt,
)

VALID_EMOTIONS = {"angry", "disgust", "fearful", "happy", "sad", "surprised"}
POSITIVE = {"happy"}


class ResponseGenerator:
    def __init__(self, llm_config: dict[str, Any] | None = None):
        self.llm_config = llm_config or {}
        self._client = None
        if self.llm_config.get("api_key"):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.llm_config.get("base_url", "https://api.openai.com/v1"),
                    api_key=self.llm_config["api_key"],
                )
            except ImportError:
                self._client = None

    @property
    def uses_llm(self) -> bool:
        return self._client is not None

    def generate(self, memory, detected_emotion: str | None, confidence: float | None = None) -> str:
        if self._client is not None:
            reply = self._generate_llm(memory, detected_emotion, confidence)
            if reply:
                return reply
        return self._generate_template(memory, detected_emotion)

    def _generate_llm(self, memory, detected_emotion, confidence) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(goal=memory.goal or "general wellbeing")},
                    {"role": "user", "content": build_user_prompt(memory, detected_emotion, confidence)},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    def _generate_template(self, memory, detected_emotion: str | None) -> str:
        emotion = detected_emotion if detected_emotion in VALID_EMOTIONS else None
        parts: list[str] = []

        if emotion:
            parts.append(EMOTION_ACKNOWLEDGEMENTS[emotion])
        elif memory.turn_count == 0:
            parts.append("Welcome - I'm glad you're here. What would you like to work on together?")
        else:
            parts.append("Thanks for sharing that.")

        # note trajectory change once there is history to compare against
        traj = memory.emotion_trajectory
        if len(traj) >= 3 and traj[-1].emotion != traj[0].emotion:
            improved = (traj[-1].emotion in POSITIVE) and (traj[0].emotion not in POSITIVE)
            declined = (traj[-1].emotion not in POSITIVE) and (traj[0].emotion in POSITIVE)
            if improved:
                parts.append(EMOTIONAL_SHIFT_REACTION["improved"])
            elif declined:
                parts.append(EMOTIONAL_SHIFT_REACTION["declined"])

        if emotion:
            parts.append(COACHING_QUESTIONS[emotion])
        elif memory.goal:
            parts.append(f"How does this connect with your goal of {memory.goal}?")

        return " ".join(parts)
