"""Where readings are asked for: the engine's transcripts on one side, the readings kept on the other.

    transcripts/   what the engine keeps: one JSON Lines file per conversation, one call a line   (read, never written)
    readings/      what this keeps: <transcript id>.json, one reading each, and readings.json for the folder

A reading is made the first time it is asked for and kept; after that the file is answered. The model does one
thing at a time, so readings are made one after another however many are asked for at once.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from .reading import SCHEMA, Tools, Unreadable, read

FOLDER_SCHEMA = "undertone.readings.v1"
TRANSCRIPT_SCHEMA = "undertone.transcript.v1"
#: a transcript's id is also its reading's file name, so it is kept to characters that cannot leave the folder
TRANSCRIPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class NoTranscript(LookupError):
    """No transcript with that id in the transcripts folder."""


class DeskError(ValueError):
    """A readings folder that holds readings made another way."""


class Transcripts:
    """The transcripts in a folder, found by id. The folder is the engine's to write: a file is read again whenever
    it has changed, and a line that is not a whole transcript (one still being written, say) is passed over."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._seen: dict[Path, tuple[int, int]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def _look(self) -> None:
        for file in sorted(self.root.glob("*.jsonl")) if self.root.is_dir() else []:
            try:
                stat = file.stat()
                mark = (stat.st_mtime_ns, stat.st_size)
                if self._seen.get(file) == mark:
                    continue
                lines = file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(t, dict) and t.get("schema") == TRANSCRIPT_SCHEMA and isinstance(t.get("id"), str) and isinstance(t.get("response"), dict):
                    self._by_id[t["id"]] = t
            self._seen[file] = mark

    def find(self, transcript: str) -> dict[str, Any] | None:
        self._look()
        return self._by_id.get(transcript)

    def all(self) -> list[dict[str, Any]]:
        """Every transcript, a conversation's turns in order."""
        self._look()
        return sorted(self._by_id.values(), key=lambda t: (t.get("conversation") or "", t.get("turn") or 0))


class Desk:
    """Readings by transcript id: kept ones answered from their files, new ones made with the tools."""

    def __init__(self, tools: Tools, transcripts: str | Path, out: str | Path, log: Callable[[str], None] | None = None):
        self.tools, self.transcripts, self.out = tools, Transcripts(transcripts), Path(out)
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self._one_at_a_time = Lock()
        record = self.out / "readings.json"
        if record.is_file() and json.loads(record.read_text()).get("settings") != tools.settings:
            raise DeskError(f"{self.out} already holds readings made with another model, lens, baseline or vector set. Use another folder.")
        self.out.mkdir(parents=True, exist_ok=True)
        _write(record, {"schema": FOLDER_SCHEMA, "settings": tools.settings})

    def kept(self, transcript: str) -> dict[str, Any] | None:
        """The reading already made of a transcript, or None."""
        if not TRANSCRIPT_ID.fullmatch(transcript):
            return None
        path = self.out / f"{transcript}.json"
        if not path.is_file():
            return None
        try:
            reading = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return reading if isinstance(reading, dict) and reading.get("schema") == SCHEMA else None

    def count(self) -> int:
        return sum(1 for p in self.out.glob("*.json") if p.name != "readings.json")

    def reading(self, transcript: str) -> dict[str, Any]:
        """The reading of a transcript, made now if it has not been. Raises `NoTranscript` or `Unreadable`."""
        if not TRANSCRIPT_ID.fullmatch(transcript):
            raise NoTranscript(transcript)
        reading = self.kept(transcript)
        if reading is not None:
            return reading
        with self._one_at_a_time:
            reading = self.kept(transcript)   # made while this was waiting its turn
            if reading is not None:
                return reading
            found = self.transcripts.find(transcript)
            if found is None:
                raise NoTranscript(transcript)
            started = time.monotonic()
            try:
                reading = read(self.tools, found)
            except Unreadable:
                raise
            except Exception:
                _free()   # one reply too long for the memory does not take the next one down with it
                raise
            _write(self.out / f"{transcript}.json", reading)
            first = ", ".join(f"{e['name']} {e['level']}" for e in reading["overall"]["emotions"][:3]) or "no emotions"
            marked = sum(1 for t in reading["tokens"] if t.get("emotions"))
            self.log(f"{transcript}  turn {reading['turn']}  {len(reading['tokens'])} tokens, {marked} marked  {first}  "
                     f"{', '.join(reading['flags']) or 'no flags'}  {time.monotonic() - started:.1f}s")
            return reading

    def read_all(self) -> dict[str, Any]:
        """Make the reading of every transcript in the folder that has none yet, and say what happened to each."""
        made, there, left = [], [], []
        for t in self.transcripts.all():
            tid = t["id"]
            if self.kept(tid) is not None:
                there.append(tid)
                continue
            try:
                self.reading(tid)
                made.append(tid)
            except (Unreadable, NoTranscript) as error:
                left.append({"transcript": tid, "reason": str(error)})
            except Exception as error:
                left.append({"transcript": tid, "reason": f"{type(error).__name__}: {error}"[:500]})
        return {"made": made, "already_there": there, "left_out": left}


def _write(path: Path, value: dict[str, Any]) -> None:
    """Written beside the file and moved into place, so a reading is never answered half-written."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _free() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
