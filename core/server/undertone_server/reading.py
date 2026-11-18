"""One reply's reading: its emotions and its lens words, made from the transcript in one pass.

The three tools are run in memory, in their own order, on one transcript:

    capture   the pinned model reads the transcript again; each block's output at the reply's tokens
    emotion   which emotions stand out, over the reply and token by token        (undertone-emotion)
    lens      the words the model leans toward at each depth                     (undertone-jlens)

and what the page needs of the two readings is put into one record (`undertone.reading.v1`). Nothing new is
measured here: every number is the tools' own, made by the same functions their command lines call. The capture
itself is not kept; only the reading is.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from undertone_capture import Nothing, capture
from undertone_emotion import read as read_emotions
from undertone_models import DEPTHS

SCHEMA = "undertone.reading.v1"


class Unreadable(ValueError):
    """A transcript there is nothing to read in, or that belongs to another model."""


@dataclass
class Tools:
    """Everything a reading is made with, loaded once. A tool that is not there (`reader` or `vectors` is None)
    leaves its part of the reading empty."""

    model: Any            # the loaded model (undertone_models: adapter.load)
    reader: Any = None    # undertone_jlens.Reader: the model's last step, a lens and the vocabulary's word types
    baseline: Any = None  # undertone_jlens.Baseline to rank the lens's words against, or None for probability
    vectors: Any = None   # undertone_emotion.VectorSet
    top: int = 8          # how many emotions and how many words a depth a reading keeps

    @property
    def contract(self) -> Any:
        return self.model.contract

    @property
    def blocks(self) -> list[int]:
        """The blocks the tools read: the lens's depths and the vectors' band. Only these are captured."""
        wanted: set[int] = set()
        if self.reader is not None:
            wanted |= {b for layers in self.reader.depths.values() for b in layers}
        if self.vectors is not None:
            wanted |= set(self.vectors.blocks)
        return sorted(wanted)

    @property
    def settings(self) -> dict[str, Any]:
        """What a reading was made with. Readings made with other settings are other readings."""
        c = self.contract
        return {
            "model": {"name": c.name, "repo": c.repo, "revision": c.revision},
            "lens": self.reader.lens.source if self.reader is not None else None,
            "baseline": self.baseline.summary["sha256"] if self.baseline is not None else None,
            "vectors": self.vectors.record["sha256"] if self.vectors is not None else None,
            "top": self.top,
        }


def read(tools: Tools, transcript: Mapping[str, Any]) -> dict[str, Any]:
    """The reading of one transcript. Raises `Unreadable` when the reply has nothing to read or the transcript
    names another model's contract. A transcript outside the setting the model was checked in is read all the same,
    and the reading says so (`flags`, `qualified`)."""
    named = transcript.get("model")
    if named and named != tools.contract.name:
        raise Unreadable(f"this transcript was made for another model's contract ({named})")
    if not tools.blocks:
        raise Unreadable("no tool is loaded to read with")
    try:
        record, blocks = capture(tools.model, transcript, blocks=tools.blocks)
    except Nothing as nothing:
        raise Unreadable(str(nothing)) from None
    emotions = read_emotions(tools.vectors, record, blocks, top=tools.top) if tools.vectors is not None else None
    lens = tools.reader.read(record, blocks, top=tools.top, baseline=tools.baseline) if tools.reader is not None else None

    reply = transcript["response"].get("content") or ""
    texts = pieces(tools.model.tokenizer, record, reply)
    words = lambda at: {d: [w["word"] for w in at[d]] for d in DEPTHS}
    tokens = []
    for k, text in enumerate(texts):
        piece: dict[str, Any] = {"text": text}
        marks = emotions["tokens"][k]["emotions"] if emotions else []
        if marks:   # a piece where nothing stands out carries only its text
            piece["emotions"] = [{"name": m["name"], "family": m["family"], "strength": m["strength"], "share": m["share"]} for m in marks]
            if lens:
                piece["lens"] = words(lens["tokens"][k]["lens"])
        tokens.append(piece)
    return {
        "schema": SCHEMA,
        "transcript": record["transcript"],
        "conversation": record["conversation"],
        "turn": record["turn"],
        "model": tools.contract.name,
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flags": list(record["flags"]),
        "qualified": record["qualified"],
        "fits": "".join(texts) == reply,
        "made_with": {
            **tools.settings,
            "ids": record["ids"]["source"],
            "lens_ranked_by": lens["ranked_by"] if lens else None,
            "marks": emotions["marks"] if emotions else None,
        },
        "overall": {
            "emotions": [{"name": e["name"], "family": e["family"], "level": e["level"], "share": e["share"]} for e in emotions["overall"]] if emotions else [],
            "lens": words(lens["overall"]) if lens else {d: [] for d in DEPTHS},
        },
        "tokens": tokens,
    }


def pieces(tokenizer: Any, record: Mapping[str, Any], reply: str) -> list[str]:
    """Each captured token's text, so that joined they are the reply to the character wherever that can be done.

    A capture usually has this already. Two things can stand in the way. A character split over two tokens decodes
    to nothing readable on either alone: the tokens are then decoded together, and the character goes to the token
    that completes it. And a model's template may trim a message's edges: the white space it dropped is put back on
    the first and last piece, since the page lays these pieces over the reply's own text.
    """
    texts = [t["text"] for t in record["tokens"]]
    if not texts:
        return texts
    if "".join(texts) not in (reply, reply.strip()):
        ids = [t["id"] for t in record["tokens"]]
        again, done = [], ""
        for k in range(len(ids)):
            whole = tokenizer.decode(ids[: k + 1], skip_special_tokens=False)
            if whole.endswith("�") or not whole.startswith(done):   # mid-character: wait for the rest
                again.append("")
                continue
            again.append(whole[len(done):])
            done = whole
        if "".join(again) in (reply, reply.strip()):
            texts = again
    joined = "".join(texts)
    if joined != reply and joined and joined == reply.strip():
        start = reply.index(joined)
        texts = [reply[:start] + texts[0], *texts[1:]]
        texts[-1] += reply[start + len(joined):]
    return texts
