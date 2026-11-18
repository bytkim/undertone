"""Where transcripts are kept: a folder with one JSON Lines file per conversation, one call a line.

Plain files on purpose: a conversation can be read, copied or handed to a tool without the engine running.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from .transcript import Transcript

#: a conversation's id is also its file's name, so it is kept to characters that cannot leave the folder
CONVERSATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True)
class Summary:
    """A conversation at a glance, for a history list."""

    id: str
    turns: int
    created: str
    updated: str
    #: the first thing the user said, cut to a line
    title: str


class TranscriptStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, conversation: str) -> Path:
        if not CONVERSATION_ID.fullmatch(conversation):
            raise ValueError(f"Not a conversation id: {conversation!r}")
        return self.root / f"{conversation}.jsonl"

    def append(self, transcript: Transcript) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._path(transcript.conversation).open("a", encoding="utf-8") as f:
            f.write(transcript.line() + "\n")

    def conversation(self, conversation: str) -> list[Transcript]:
        """Every call of a conversation, in turn order (empty when there is none)."""
        path = self._path(conversation)
        if not path.exists():
            return []
        calls = [Transcript.parse(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return sorted(calls, key=lambda t: t.turn)

    def conversations(self) -> list[Summary]:
        """Every conversation, the most recently continued first."""
        out = []
        for path in self.root.glob("*.jsonl") if self.root.exists() else []:
            calls = self.conversation(path.stem)
            if not calls:
                continue
            first = next((m.content for m in calls[0].request.messages if m.role == "user"), "")
            title = " ".join(first.split())
            out.append(Summary(id=path.stem, turns=len(calls), created=calls[0].created, updated=calls[-1].created, title=title[:80] + ("…" if len(title) > 80 else "")))
        return sorted(out, key=lambda s: s.updated, reverse=True)
