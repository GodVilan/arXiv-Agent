"""
memory.py – Two-tier memory for the arXiv agent.

ConversationMemory
    Rolling window of the last N turns (user/assistant pairs).
    Injected into the ReAct prompt so the agent understands follow-up questions.

ResearchMemory
    Persistent key-value store for findings the user explicitly saves.
    Stored on disk as JSON so notes survive session restarts.
    The agent can read these notes as additional context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from rag import config

log = logging.getLogger(__name__)


# ── Conversation memory ────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConversationMemory:
    """
    Rolling window of conversation turns.
    Older turns are dropped when the window is full so context stays focused.
    """

    def __init__(self, max_turns: int = config.MEMORY_WINDOW) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add_user(self, content: str) -> None:
        self._turns.append(Turn(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._turns.append(Turn(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        # Keep pairs: drop the oldest complete turn-pair when over limit
        while len(self._turns) > self.max_turns * 2:
            self._turns.pop(0)

    def format_for_prompt(self) -> str:
        """Return conversation history as a readable string for the agent prompt."""
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns[:-1]:   # exclude current user message (already in prompt)
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)


# ── Research memory ────────────────────────────────────────────────────────────

@dataclass
class ResearchNote:
    key: str                # short label, e.g. "BGE vs MiniLM"
    content: str            # the actual finding
    source: str = ""        # paper title or URL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ResearchMemory:
    """
    Persistent store for research findings.
    Users can save notes manually; the agent reads them as additional context.
    Backed by a JSON file so notes survive restarts.
    """

    def __init__(self, path: Path = config.RESEARCH_NOTES_PATH) -> None:
        self._path  = Path(path)
        self._notes: dict[str, ResearchNote] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save_note(self, key: str, content: str, source: str = "") -> ResearchNote:
        note = ResearchNote(key=key, content=content, source=source)
        self._notes[key] = note
        self._persist()
        log.info("Research note saved: '%s'", key)
        return note

    def get_note(self, key: str) -> ResearchNote | None:
        return self._notes.get(key)

    def delete_note(self, key: str) -> bool:
        if key in self._notes:
            del self._notes[key]
            self._persist()
            return True
        return False

    def all_notes(self) -> list[ResearchNote]:
        return list(self._notes.values())

    def format_for_prompt(self, max_notes: int = 5) -> str:
        """
        Return the most recent research notes as a readable block
        for injection into the agent prompt.
        """
        notes = sorted(
            self._notes.values(),
            key=lambda n: n.created_at,
            reverse=True,
        )[:max_notes]

        if not notes:
            return ""

        lines = ["**Saved research notes (for context):**"]
        for note in notes:
            src = f" [Source: {note.source}]" if note.source else ""
            lines.append(f"• [{note.key}]{src}: {note.content}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                self._notes = {k: ResearchNote(**v) for k, v in raw.items()}
                log.info("Loaded %d research notes from %s", len(self._notes), self._path)
            except Exception as exc:
                log.warning("Could not load research notes: %s", exc)
                self._notes = {}

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({k: asdict(v) for k, v in self._notes.items()}, f, indent=2)
        except Exception as exc:
            log.error("Could not save research notes: %s", exc)
