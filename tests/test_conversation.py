"""Tests for the conversation chain: emotional state -> strategy -> response.

Covers the composite emotional state (current + recent history + trend),
the non-clinical strategy mapping, and the end-to-end agent wiring.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conversation.agent import BehaviourChangeAgent  # noqa: E402
from src.conversation.emotional_state import (  # noqa: E402
    NO_DIAGNOSIS_RULE,
    EmotionalState,
    assess_emotional_state,
    select_strategy,
)
from src.conversation.memory import ConversationMemory  # noqa: E402
from src.conversation.response_generator import ResponseGenerator  # noqa: E402

CLINICAL_TERMS = ("depression", "anxiety", "disorder", "diagnos", "mental illness", "clinical")


def memory_with_emotions(emotions: list[str]) -> ConversationMemory:
    memory = ConversationMemory(goal="reduce stress")
    for emotion in emotions:
        memory.add_user_turn(f"feeling {emotion}", emotion=emotion, confidence=0.8)
    return memory


def test_composite_state_persistent_negative():
    memory = memory_with_emotions(["sad", "sad", "fearful"])
    state = assess_emotional_state(memory, "fearful", 0.8)
    assert state.current == "fearful"
    assert state.recent == ["sad", "sad", "fearful"]
    assert state.trend == "persistent_negative"
    assert state.negative_streak == 3
    lowered = state.description.lower()
    assert not any(term in lowered for term in CLINICAL_TERMS)


def test_supportive_steady_strategy():
    memory = memory_with_emotions(["sad", "sad", "fearful"])
    state = assess_emotional_state(memory, "fearful", 0.8)
    strategy = select_strategy(state)
    assert strategy.name == "supportive_steady"
    assert strategy.tone
    assert NO_DIAGNOSIS_RULE in strategy.guidelines
    assert all(isinstance(g, str) and g for g in strategy.guidelines)


def test_reinforce_progress_strategy():
    memory = memory_with_emotions(["sad", "sad", "happy"])
    state = assess_emotional_state(memory, "happy", 0.9)
    assert state.direction == "improved"
    assert state.trend == "positive"
    assert select_strategy(state).name == "reinforce_progress"

    improved_with_negative_current = EmotionalState(
        current="sad", confidence=0.5, recent=["happy", "sad"],
        trend="neutral_or_mixed", direction="improved",
        negative_streak=1, description="mixed or neutral emotional signals",
    )
    assert select_strategy(improved_with_negative_current).name == "reinforce_progress"


def test_template_reply_non_clinical():
    memory = memory_with_emotions(["sad", "sad", "fearful"])
    state = assess_emotional_state(memory, "fearful", 0.8)
    strategy = select_strategy(state)
    reply = ResponseGenerator().generate(memory, state, strategy)
    assert reply
    lowered = reply.lower()
    assert not any(term in lowered for term in CLINICAL_TERMS)
    assert "?" in reply


def test_agent_respond_end_to_end():
    agent = BehaviourChangeAgent(goal="reduce stress", device="cpu")
    probs = {
        "text": np.array([0.05, 0.05, 0.8, 0.05, 0.03, 0.02]),
        "audio": np.array([0.05, 0.05, 0.1, 0.05, 0.7, 0.05]),
    }
    out = agent.respond("I feel nervous about everything", modality_predictions=probs)
    assert out["reply"]
    assert out["emotion"]["label"] == "fearful"
    assert 0.0 < out["emotion"]["confidence"] <= 1.0
    assert {"reply", "emotion", "emotional_state", "strategy", "memory"} <= set(out)


def test_agent_chain_wiring():
    agent = BehaviourChangeAgent(goal="reduce stress", device="cpu")
    probs = {
        "text": np.array([0.05, 0.05, 0.8, 0.05, 0.03, 0.02]),
        "audio": np.array([0.05, 0.05, 0.1, 0.05, 0.7, 0.05]),
    }
    out = agent.respond("I feel nervous about everything", modality_predictions=probs)
    state = out["emotional_state"]
    assert state["current"] == "fearful"
    assert state["recent"][-1] == "fearful"
    assert agent.memory.turn_count == 2  # one user turn + one agent turn
    assert out["memory"]["turn_count"] == 2
    recomputed = select_strategy(assess_emotional_state(agent.memory, "fearful"))
    assert out["strategy"]["name"] == recomputed.name


def test_agent_no_emotion_fallback():
    agent = BehaviourChangeAgent(goal="exercise more", device="cpu")
    out = agent.respond("just checking in")
    assert out["emotion"] is None
    assert out["emotional_state"]["trend"] == "unknown"
    assert out["strategy"]["name"] == "open_exploration"
    assert out["reply"]
