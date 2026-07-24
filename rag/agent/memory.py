"""
memory.py – Conversation memory + persistent research notes.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from rag import config

log = logging.getLogger(__name__)


@dataclass
class Turn:
    role:      str
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConversationMemory:
    def __init__(self, max_turns: int = config.MEMORY_WINDOW) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add_user(self, content: str) -> None:
        self._turns.append(Turn("user", content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._turns.append(Turn("assistant", content))
        self._trim()

    def _trim(self) -> None:
        while len(self._turns) > self.max_turns * 2:
            self._turns.pop(0)

    def format_for_prompt(self) -> str:
        if not self._turns:
            return ""
        lines = []
        for t in self._turns[:-1]:
            role = "User" if t.role == "user" else "Assistant"
            lines.append(f"{role}: {t.content[:300]}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)


@dataclass
class ResearchNote:
    key:        str
    content:    str
    source:     str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ResearchMemory:
    def __init__(self, path: Path = config.RESEARCH_NOTES_PATH) -> None:
        self._path  = Path(path)
        self._notes: dict[str, ResearchNote] = {}
        self._load()

    def save_note(self, key: str, content: str, source: str = "") -> ResearchNote:
        note = ResearchNote(key=key, content=content, source=source)
        self._notes[key] = note
        self._persist()
        return note

    def delete_note(self, key: str) -> bool:
        if key in self._notes:
            del self._notes[key]
            self._persist()
            return True
        return False

    def all_notes(self) -> list[ResearchNote]:
        return sorted(self._notes.values(), key=lambda n: n.created_at, reverse=True)

    def format_for_prompt(self, max_notes: int = 4) -> str:
        notes = self.all_notes()[:max_notes]
        if not notes:
            return ""
        lines = ["**Saved research notes:**"]
        for n in notes:
            src = f" [{n.source}]" if n.source else ""
            lines.append(f"• [{n.key}]{src}: {n.content[:200]}")
        return "\n".join(lines)

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                self._notes = {k: ResearchNote(**v) for k, v in raw.items()}
            except Exception as exc:
                log.warning("Could not load notes: %s", exc)

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({k: asdict(v) for k, v in self._notes.items()}, f, indent=2)
        except Exception as exc:
            log.error("Could not save notes: %s", exc)
