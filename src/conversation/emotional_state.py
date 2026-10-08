"""Emotional state and response strategy: the blocks between emotion detection
and the response generator.

The detected 6-class label is never used alone and never mapped to a clinical
condition.  The emotional state combines:

    current emotion + recent emotion history + conversation context
        -> EmotionalState -> ResponseStrategy

Example:  current="sad", recent=["sad", "sad", "fearful"]
          -> trend="persistent_negative"
          -> strategy="supportive_steady" (supportive + reflective +
             open-ended question), NOT a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.conversation.memory import ConversationMemory

NEGATIVE_EMOTIONS = {"angry", "disgust", "fearful", "sad"}
POSITIVE_EMOTIONS = {"happy"}
RECENT_WINDOW = 3

TREND_DESCRIPTIONS = {
    "persistent_negative": "sustained negative feelings across recent turns",
    "mildly_negative": "negative feelings in this turn",
    "positive": "positive mood in this turn",
    "neutral_or_mixed": "mixed or neutral emotional signals",
    "unknown": "no clear emotional signal yet",
}

NO_DIAGNOSIS_RULE = (
    "describe feelings in everyday, non-clinical words - "
    "never diagnose a condition from detected emotion labels"
)


@dataclass
class EmotionalState:
    """Composite view of how the user is doing right now."""

    current: str | None          # this turn's detected emotion label (or None)
    confidence: float | None
    recent: list[str]            # last RECENT_WINDOW detected emotions, oldest first
    trend: str                   # persistent_negative | mildly_negative | positive | ...
    direction: str               # improved | declined | stable (within the window)
    negative_streak: int         # consecutive trailing negative-emotion turns
    description: str             # short non-clinical summary for prompts and logs


@dataclass
class ResponseStrategy:
    """How the response generator should respond to the emotional state."""

    name: str
    tone: str
    guidelines: list[str] = field(default_factory=list)


def assess_emotional_state(
    memory: ConversationMemory,
    current: str | None,
    confidence: float | None = None,
) -> EmotionalState:
    """Combine the current prediction with conversation history and context.

    Call after the current turn has been added to memory, so the trajectory
    includes this turn.
    """
    trajectory = memory.emotion_trajectory
    recent = [t.emotion for t in trajectory[-RECENT_WINDOW:]]
    trend = memory.behavioral_state()["state"]

    negative_streak = 0
    for turn in reversed(trajectory):
        if turn.emotion in NEGATIVE_EMOTIONS:
            negative_streak += 1
        else:
            break

    direction = "stable"
    if len(trajectory) >= 3:
        first, last = trajectory[0].emotion, trajectory[-1].emotion
        if last in POSITIVE_EMOTIONS and first not in POSITIVE_EMOTIONS:
            direction = "improved"
        elif last not in POSITIVE_EMOTIONS and first in POSITIVE_EMOTIONS:
            direction = "declined"

    return EmotionalState(
        current=current,
        confidence=confidence,
        recent=recent,
        trend=trend,
        direction=direction,
        negative_streak=negative_streak,
        description=TREND_DESCRIPTIONS[trend],
    )


def select_strategy(state: EmotionalState) -> ResponseStrategy:
    """Map the emotional state to a supportive, non-clinical response strategy."""
    if state.trend == "persistent_negative":
        return ResponseStrategy(
            name="supportive_steady",
            tone="warm, patient, unhurried",
            guidelines=[
                "acknowledge that the difficulty has lasted across turns",
                "normalize the feeling without labeling the person",
                "reflect back what the user said in their own words",
                "ask one open-ended question about what usually helps, even a little",
                NO_DIAGNOSIS_RULE,
            ],
        )
    if state.current in POSITIVE_EMOTIONS or state.direction == "improved":
        return ResponseStrategy(
            name="reinforce_progress",
            tone="encouraging, specific",
            guidelines=[
                "name the positive shift you noticed",
                "celebrate it as a small win without exaggerating",
                "ask what contributed, so it can be repeated",
                NO_DIAGNOSIS_RULE,
            ],
        )
    if state.current in NEGATIVE_EMOTIONS or state.current == "surprised":
        return ResponseStrategy(
            name="acknowledge_explore",
            tone="attentive, curious",
            guidelines=[
                "acknowledge this turn's feeling before anything else",
                "reflect it back briefly in the user's own words",
                "ask one open-ended question to explore what is behind it",
                NO_DIAGNOSIS_RULE,
            ],
        )
    return ResponseStrategy(
        name="open_exploration",
        tone="gentle, inviting",
        guidelines=[
            "invite the user to share more at their own pace",
            "connect the conversation back to their goal",
            "ask one open-ended question",
            NO_DIAGNOSIS_RULE,
        ],
    )
