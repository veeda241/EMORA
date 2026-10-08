"""Prompt templates for the behaviour change conversational agent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an empathetic behaviour-change coaching agent.
You support the user in working toward their goal: {goal}.

Guidelines:
- Acknowledge the user's emotional state before offering guidance.
- Use motivational-interviewing style: open questions, reflective listening,
  avoid lecturing.
- Keep replies to 2-4 sentences unless the user asks for detail.
- Never diagnose or provide medical advice. Detected emotions are model
  labels, not conditions: describe them in everyday words ("sounds heavy",
  "a lot to carry"), never as disorders, symptoms, or diagnoses.
- Suggest professional help only for crisis-level distress.
- Track progress gently across turns; celebrate small wins.
"""

EMOTION_ACKNOWLEDGEMENTS = {
    "angry": "I can hear that frustration coming through, and it makes sense given what you're dealing with.",
    "disgust": "That sounds genuinely off-putting - thanks for being honest about it.",
    "fearful": "That sounds unsettling. It takes courage to put words to it.",
    "happy": "It's great to hear that lift in your voice - let's hold onto what's working.",
    "sad": "That sounds heavy. I'm here to work through it with you, at your pace.",
    "surprised": "That caught you off guard - let's unpack what happened.",
}

COACHING_QUESTIONS = {
    "angry": "What part of this feels most out of your control right now?",
    "disgust": "What would need to change for this to feel more acceptable to you?",
    "fearful": "What's the smallest first step that would feel safe enough to try?",
    "happy": "What specifically contributed to this good moment, so we can repeat it?",
    "sad": "What usually helps, even a little, when you feel this way?",
    "surprised": "Now that the surprise has landed, what matters most about it?",
}

EMOTIONAL_SHIFT_REACTION = {
    "improved": "I've noticed things sound lighter than earlier in our conversation - what's changed?",
    "declined": "I've noticed things feel heavier than when we started. Want to talk about what shifted?",
    "stable": "Thanks for staying with this - steady is progress too.",
}

TREND_ACKNOWLEDGEMENTS = {
    "persistent_negative": "It sounds like this has been weighing on you for a while now, not just today.",
}


def build_user_prompt(
    memory,
    emotional_state,
    strategy,
) -> str:
    """Compose the user-turn prompt from memory + emotional state + strategy."""
    parts = [f"Conversation goal: {memory.goal or 'general wellbeing'}"]
    if memory.turn_count:
        parts.append("Recent conversation:\n" + memory.recent_transcript())

    recent = " -> ".join(emotional_state.recent) if emotional_state.recent else "none yet"
    current = emotional_state.current or "none detected"
    state_line = (
        f"Emotional state: {emotional_state.description}; "
        f"current emotion: {current}"
    )
    if emotional_state.confidence is not None:
        state_line += f" (confidence {emotional_state.confidence:.2f})"
    state_line += f"; recent trajectory: {recent}"
    parts.append(state_line)

    parts.append(
        f"Response strategy: {strategy.name} ({strategy.tone}). "
        + " ".join(f"- {g}." for g in strategy.guidelines)
    )
    parts.append("Write the agent's next reply following that strategy.")
    return "\n\n".join(parts)
