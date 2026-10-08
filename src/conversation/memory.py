"""Conversation memory: stores and retrieves dialogue history and emotion trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
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
    storage_path: str | None = None
    turns: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.storage_path:
            self._init_storage()
            self._load()

    def _connect(self) -> sqlite3.Connection:
        if not self.storage_path:
            raise RuntimeError("persistent storage is not configured")
        connection = sqlite3.connect(self.storage_path)
        connection.execute(
            """CREATE TABLE IF NOT EXISTS conversation_turns (
                user_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                emotion TEXT,
                confidence REAL,
                PRIMARY KEY (user_id, turn_index)
            )"""
        )
        connection.commit()
        return connection

    def _init_storage(self) -> None:
        with self._connect():
            pass

    def _load(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT turn_index, speaker, text, emotion, confidence "
                "FROM conversation_turns WHERE user_id = ? ORDER BY turn_index DESC LIMIT ?",
                (self.user_id, self.max_turns),
            ).fetchall()
        self.turns = [
            Turn(
                speaker=row[1],
                text=row[2],
                emotion=row[3],
                confidence=row[4],
                turn_index=row[0],
            )
            for row in reversed(rows)
        ]

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
        if self.storage_path:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO conversation_turns "
                    "(user_id, turn_index, speaker, text, emotion, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (self.user_id, turn.turn_index, turn.speaker, turn.text,
                     turn.emotion, turn.confidence),
                )

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

    def behavioral_state(self, window: int = 5) -> dict[str, Any]:
        """Summarize the recent emotional trajectory for application memory."""
        trajectory = self.emotion_trajectory[-window:]
        negative = {"angry", "disgust", "fearful", "sad"}
        if not trajectory:
            state = "unknown"
        elif len(trajectory) >= 3 and all(turn.emotion in negative for turn in trajectory[-3:]):
            state = "persistent_negative"
        elif len(trajectory) >= 2 and trajectory[-1].emotion in negative:
            state = "mildly_negative"
        elif trajectory[-1].emotion == "happy":
            state = "positive"
        else:
            state = "neutral_or_mixed"
        return {
            "state": state,
            "window": len(trajectory),
            "emotions": [turn.emotion for turn in trajectory],
        }

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
            "behavioral_state": self.behavioral_state(),
            "emotion_counts": {
                e: sum(1 for t in self.emotion_trajectory if t.emotion == e)
                for e in {t.emotion for t in self.emotion_trajectory}
            },
        }
