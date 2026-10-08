"""Conversation memory: stores and retrieves dialogue history and emotion trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    speaker: str  # "user" | "agent"
    text: str
    emotion: str | None = None  # detected emotion label for user turns
    confidence: float | None = None
    turn_index: int = 0


@dataclass
class ConversationMemory:
    """Rolling memory of a behaviour-change coaching conversation."""

    user_id: str = "default"
    goal: str = ""  # e.g. "reduce stress", "exercise more"
    max_turns: int = 50
    turns: list[Turn] = field(default_factory=list)

    def add_user_turn(self, text: str, emotion: str | None = None, confidence: float | None = None) -> Turn:
        turn = Turn(speaker="user", text=text, emotion=emotion, confidence=confidence,
                    turn_index=len(self.turns))
        self._append(turn)
        return turn

    def add_agent_turn(self, text: str) -> Turn:
        turn = Turn(speaker="agent", text=text, turn_index=len(self.turns))
        self._append(turn)
        return turn

    def _append(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    @property
    def emotion_trajectory(self) -> list[Turn]:
        """User turns that carry a detected emotion, in order."""
        return [t for t in self.turns if t.speaker == "user" and t.emotion is not None]

    @property
    def dominant_emotion(self) -> str | None:
        """Most frequent detected emotion across the conversation."""
        traj = self.emotion_trajectory
        if not traj:
            return None
        counts: dict[str, int] = {}
        for t in traj:
            counts[t.emotion] = counts.get(t.emotion, 0) + 1
        return max(counts, key=counts.get)

    @property
    def last_emotion(self) -> str | None:
        traj = self.emotion_trajectory
        return traj[-1].emotion if traj else None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def recent_transcript(self, n_turns: int = 8) -> str:
        """Formatted transcript of the last n turns, for prompting."""
        lines = []
        for t in self.turns[-n_turns:]:
            tag = f" [{t.emotion}]" if t.emotion else ""
            lines.append(f"{t.speaker.capitalize()}{tag}: {t.text}")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "goal": self.goal,
            "turn_count": self.turn_count,
            "last_emotion": self.last_emotion,
            "dominant_emotion": self.dominant_emotion,
            "emotion_counts": {
                e: sum(1 for t in self.emotion_trajectory if t.emotion == e)
                for e in {t.emotion for t in self.emotion_trajectory}
            },
        }
