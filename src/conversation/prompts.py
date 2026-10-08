"""Prompt templates for the behaviour change conversational agent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an empathetic behaviour-change coaching agent.
You support the user in working toward their goal: {goal}.

Guidelines:
- Acknowledge the user's emotional state before offering guidance.
- Use motivational-interviewing style: open questions, reflective listening,
  avoid lecturing.
- Keep replies to 2-4 sentences unless the user asks for detail.
- Never diagnose or provide medical advice; suggest professional help for
  crisis-level distress.
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


def build_user_prompt(memory, detected_emotion: str | None, confidence: float | None) -> str:
    """Compose the user-turn prompt from memory + current emotion estimate."""
    parts = [f"Conversation goal: {memory.goal or 'general wellbeing'}"]
    if memory.turn_count:
        parts.append("Recent conversation:\n" + memory.recent_transcript())
    if detected_emotion:
        parts.append(
            f"Current detected user emotion (multimodal model): {detected_emotion} "
            f"(confidence {confidence:.2f})" if confidence is not None
            else f"Current detected user emotion (multimodal model): {detected_emotion}"
        )
        if memory.dominant_emotion and memory.dominant_emotion != detected_emotion:
            parts.append(f"Dominant emotion so far in this conversation: {memory.dominant_emotion}")
    parts.append("Write the agent's next reply.")
    return "\n\n".join(parts)
