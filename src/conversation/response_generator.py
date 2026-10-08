"""Response generator: produces empathetic replies conditioned on emotion state.

Three modes:
- LLM mode: any OpenAI-compatible chat endpoint (base_url + api_key + model).
- Local LLM mode: llm_config {"backend": "local"} -> a small instruction-tuned
  Hugging Face model (default Qwen/Qwen2.5-1.5B-Instruct) loaded once and
  shared across agent instances, so it fits a 6 GB GPU and survives agent
  resets. The counselor-style prompting (SYSTEM_PROMPT + build_user_prompt)
  already conditions generation on emotion state, trend and strategy.
- Template mode (fallback): deterministic motivational-interviewing style
  responses composed from prompts.py - no external service required.
"""

from __future__ import annotations

from typing import Any

from src.conversation.prompts import (
    COACHING_QUESTIONS,
    EMOTION_ACKNOWLEDGEMENTS,
    EMOTIONAL_SHIFT_REACTION,
    SYSTEM_PROMPT,
    TREND_ACKNOWLEDGEMENTS,
    build_user_prompt,
)

VALID_EMOTIONS = {"angry", "disgust", "fearful", "happy", "sad", "surprised"}

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_LOCAL_LLM: dict[str, tuple[Any, Any, str]] = {}


def _load_local_llm(model_id: str) -> tuple[Any, Any, str]:
    """Load (tokenizer, model, device) once per model_id, shared process-wide."""
    if model_id not in _LOCAL_LLM:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        model.eval()
        _LOCAL_LLM[model_id] = (tokenizer, model, device)
    return _LOCAL_LLM[model_id]


class ResponseGenerator:
    def __init__(self, llm_config: dict[str, Any] | None = None):
        self.llm_config = llm_config or {}
        self._client = None
        self._local_model_id: str | None = None
        if self.llm_config.get("api_key"):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.llm_config.get("base_url", "https://api.openai.com/v1"),
                    api_key=self.llm_config["api_key"],
                )
            except ImportError:
                self._client = None
        elif self.llm_config.get("backend") == "local":
            try:
                self._local_model_id = self.llm_config.get("model_id", DEFAULT_LOCAL_MODEL)
                _load_local_llm(self._local_model_id)
            except Exception:
                # model unavailable (not downloaded / no network) -> template mode
                self._local_model_id = None

    @property
    def uses_llm(self) -> bool:
        return self._client is not None or self._local_model_id is not None

    def generate(self, memory, emotional_state, strategy) -> str:
        """Produce a reply following the selected response strategy."""
        if self._client is not None:
            reply = self._generate_llm(memory, emotional_state, strategy)
            if reply:
                return reply
        if self._local_model_id is not None:
            reply = self._generate_local(memory, emotional_state, strategy)
            if reply:
                return reply
        return self._generate_template(memory, emotional_state)

    def _messages(self, memory, emotional_state, strategy) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT.format(goal=memory.goal or "general wellbeing")},
            {"role": "user", "content": build_user_prompt(memory, emotional_state, strategy)},
        ]

    def _generate_llm(self, memory, emotional_state, strategy) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=self._messages(memory, emotional_state, strategy),
                max_tokens=200,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    def _generate_local(self, memory, emotional_state, strategy) -> str | None:
        try:
            import torch
            tokenizer, model, device = _load_local_llm(self._local_model_id)
            enc = tokenizer.apply_chat_template(
                self._messages(memory, emotional_state, strategy),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(device)
            with torch.no_grad():
                output = model.generate(
                    **enc,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(output[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            return reply.strip() or None
        except Exception:
            return None

    def _generate_template(self, memory, emotional_state) -> str:
        emotion = emotional_state.current if emotional_state.current in VALID_EMOTIONS else None
        parts: list[str] = []

        if emotion:
            parts.append(EMOTION_ACKNOWLEDGEMENTS[emotion])
        elif memory.turn_count == 0:
            parts.append("Welcome - I'm glad you're here. What would you like to work on together?")
        else:
            parts.append("Thanks for sharing that.")

        if emotional_state.trend in TREND_ACKNOWLEDGEMENTS:
            parts.append(TREND_ACKNOWLEDGEMENTS[emotional_state.trend])
        elif emotional_state.direction != "stable":
            parts.append(EMOTIONAL_SHIFT_REACTION[emotional_state.direction])

        if emotion:
            parts.append(COACHING_QUESTIONS[emotion])
        elif memory.goal:
            parts.append(f"How does this connect with your goal of {memory.goal}?")

        return " ".join(parts)
