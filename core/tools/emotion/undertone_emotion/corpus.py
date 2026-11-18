"""The story corpus as transcripts, so it can be captured like anything else.

The repository's stories (stories/corpus) are plain rows of text. The capture tool reads transcripts. This module
writes each story as a transcript: one fixed user message asking for a story, and the story as the reply. A neutral
dialogue becomes a transcript too: everything up to its last assistant turn is what was sent, and that turn is the
reply.

Nothing is generated and no model is asked anything: the text is the corpus's own.
"""
import json
from pathlib import Path
from typing import Any, Iterator

#: the message every story is the reply to. It is what the stories were captured after when the method was
#: developed; changing it changes what the model has read before the story, and so the vectors.
STORY_PROMPT = "Write a one-paragraph story."
TRANSCRIPT_SCHEMA = "undertone.transcript.v1"


class CorpusError(ValueError):
    """A corpus folder that is not there or is not laid out as stories/corpus is."""


def stories(corpus: str | Path) -> Iterator[dict[str, Any]]:
    """Every story row of the corpus, author by author."""
    yield from _rows(Path(corpus) / "stories")


def neutral(corpus: str | Path) -> Iterator[dict[str, Any]]:
    """Every neutral dialogue row of the corpus."""
    yield from _rows(Path(corpus) / "neutral_dialogues")


def _rows(folder: Path) -> Iterator[dict[str, Any]]:
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        raise CorpusError(f"No corpus files in {folder}.")
    for file in files:
        for line in file.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def labels(corpus: str | Path) -> dict[str, dict[str, str]]:
    """Each story's emotion, situation and author, by its id."""
    return {r["story_id"]: {"emotion": r["emotion"], "topic": r["topic_id"], "author": r["author"]} for r in stories(corpus)}


def story_transcript(row: dict[str, Any], turn: int = 1) -> dict[str, Any]:
    return {"schema": TRANSCRIPT_SCHEMA, "id": row["story_id"], "conversation": f"{row['author']}.{row['emotion'].replace(' ', '_')}", "turn": turn,
            "request": {"messages": [{"role": "user", "content": STORY_PROMPT}]}, "response": {"content": row["text"]}}


def neutral_transcript(row: dict[str, Any], turn: int = 1) -> dict[str, Any]:
    turns = [{"role": t["role"], "content": t["text"]} for t in row["turns"]]
    return {"schema": TRANSCRIPT_SCHEMA, "id": row["story_id"], "conversation": f"{row['author']}.neutral", "turn": turn,
            "request": {"messages": turns[:-1]}, "response": {"content": turns[-1]["content"]}}


def write_transcripts(corpus: str | Path, out: str | Path) -> dict[str, int]:
    """Write the corpus as transcripts: `out/stories/<author>.jsonl` and `out/neutral/<author>.jsonl`."""
    out = Path(out)
    counts = {}
    for kind, rows, make in (("stories", stories(corpus), story_transcript), ("neutral", neutral(corpus), neutral_transcript)):
        by_author: dict[str, list[str]] = {}
        seen: dict[str, int] = {}
        for row in rows:
            t = make(row)
            seen[t["conversation"]] = t["turn"] = seen.get(t["conversation"], 0) + 1
            by_author.setdefault(row["author"], []).append(json.dumps(t, ensure_ascii=False))
        (out / kind).mkdir(parents=True, exist_ok=True)
        for author, lines in by_author.items():
            (out / kind / f"{author}.jsonl").write_text("\n".join(lines) + "\n")
        counts[kind] = sum(len(v) for v in by_author.values())
    return counts
