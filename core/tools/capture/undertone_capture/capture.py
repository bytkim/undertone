"""Capture: what every block of the model puts out while it reads a transcript.

A capture is made after the fact. The model that wrote the reply is not asked again; the transcript's text is read
once more by the pinned model (core/tools/models), and the output of each block is kept at the tokens that matter:
the reply's, by default. That is the raw material the other tools read (a lens, emotion vectors).

One transcript becomes two files in the capture folder:

    <transcript id>.safetensors   the numbers: "blocks" [blocks, tokens, width], or "mean" [blocks, width]
    <transcript id>.json          what they are of: the tokens, their text, the settings, the flags, the file's hash

and the folder has one `capture.json` for the run. Nothing is refused for being outside the setting the model was
qualified in: the item records the flags, so whatever is read from it can say so.
"""
import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from undertone_models import ContractError

SCHEMA = "undertone.capture.v1"
RUN_SCHEMA = "undertone.capture.run.v1"
TRANSCRIPT_SCHEMA = "undertone.transcript.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class CaptureError(ValueError):
    """A capture that cannot be made or read."""


class Nothing(CaptureError):
    """A transcript with nothing to capture (an empty reply, or a reply shorter than what is skipped)."""


def transcripts(paths: Iterable[str | Path], conversation: str | None = None) -> Iterator[dict[str, Any]]:
    """Every transcript in these files and folders (`*.jsonl`, one transcript a line), a conversation's turns in
    order. `conversation` keeps one conversation only."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob("*.jsonl"))
        elif p.is_file():
            files.append(p)
        else:
            raise CaptureError(f"No transcripts at {p}.")
    for file in files:
        lines = []
        for n, line in enumerate(file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError as error:
                raise CaptureError(f"{file}, line {n} is not JSON: {error}") from None
            if not isinstance(t, dict) or t.get("schema") != TRANSCRIPT_SCHEMA:
                raise CaptureError(f'{file}, line {n} is not a transcript ("{TRANSCRIPT_SCHEMA}").')
            if conversation is None or t.get("conversation") == conversation:
                lines.append(t)
        yield from sorted(lines, key=lambda t: (t.get("conversation") or "", t.get("turn") or 0))


def capture(model: Any, transcript: Mapping[str, Any], *, select: str = "reply", skip: int = 0, mean: bool = False, blocks: Sequence[int] | None = None) -> tuple[dict[str, Any], Any]:
    """Read one transcript with a loaded model and answer (record, tensor).

    select  "reply": the reply's own tokens (the default). "messages": every message's own tokens, the reply's
            included. Template tokens are never selected.
    skip    leave out the first `skip` tokens of the reply (with "reply" only): the start of a text says little
            about the rest of it.
    mean    keep the average over the selected tokens, in float32, instead of every token: [blocks, width].
    blocks  which blocks to keep, counted from 0 (default: all of them).

    The tensor is [blocks, tokens, width] in the weights' own precision, or the float32 mean.
    """
    import torch

    if select not in ("reply", "messages"):
        raise CaptureError('select is "reply" or "messages".')
    if skip < 0 or (skip and select != "reply"):
        raise CaptureError('skip leaves out the start of the reply, so it goes with select "reply" only.')
    contract = model.contract
    keep = list(range(contract.blocks)) if blocks is None else sorted(set(int(b) for b in blocks))
    if not keep or keep[0] < 0 or keep[-1] >= contract.blocks:
        raise CaptureError(f"{contract.title} has blocks 0 to {contract.blocks - 1}.")

    r = model.render(transcript)
    if select == "reply":
        if r.reply is None:
            raise Nothing("the reply has no tokens")
        spans, positions = [r.reply], list(r.reply.positions)[skip:]
        if not positions:
            raise Nothing(f"the reply has {len(r.reply.positions)} tokens and {skip} are skipped")
    else:
        spans = list(r.spans)
        positions = [p for s in spans for p in s.positions]
        if not positions:
            raise Nothing("no message has tokens")

    out = model.run(r.ids, positions)[keep]
    tensor = out.float().mean(1) if mean else out
    if not bool(torch.isfinite(tensor.float()).all()):
        raise CaptureError("the model answered numbers that are not finite")

    if r.offsets is not None and r.text is not None:
        pieces = [r.text[a:b] for a, b in (r.offsets[p] for p in positions)]
    else:   # the endpoint's ids are used as they are: each token's text is what it decodes to on its own
        pieces = [model.tokenizer.decode([r.ids[p]], skip_special_tokens=False) for p in positions]
    reply = transcript["response"].get("content") or ""
    joins = None
    if select == "reply" and not skip:
        joins = "".join(pieces) == reply   # the tokens' text is the reply to the character
    record = {
        "schema": SCHEMA,
        "transcript": transcript.get("id"),
        "conversation": transcript.get("conversation"),
        "turn": transcript.get("turn"),
        "model": {"name": contract.name, "repo": contract.repo, "revision": contract.revision},
        "ids": {"source": r.source, "count": len(r.ids), "input_ids": list(r.ids)},
        "flags": list(r.flags),
        "qualified": r.qualified,
        "select": select,
        "skip": skip,
        "mean": mean,
        "blocks": keep,
        "spans": [{"turn": s.turn, "role": s.role, "start": s.start, "stop": s.stop} for s in spans],
        "tokens": [{"position": p, "id": int(r.ids[p]), "text": piece} for p, piece in zip(positions, pieces)],
        "joins": joins,
        "tensor": {"name": "mean" if mean else "blocks", "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")},
    }
    return record, tensor


def settings(model: Any, *, select: str, skip: int, mean: bool, blocks: Sequence[int] | None) -> dict[str, Any]:
    """What must be the same for two captures to sit in one folder."""
    c = model.contract
    return {"model": {"name": c.name, "repo": c.repo, "revision": c.revision}, "select": select, "skip": skip, "mean": mean,
            "blocks": list(range(c.blocks)) if blocks is None else sorted(set(int(b) for b in blocks))}


def run(model: Any, items: Iterable[Mapping[str, Any]], out: str | Path, *, select: str = "reply", skip: int = 0, mean: bool = False,
        blocks: Sequence[int] | None = None, log: Any = None) -> dict[str, Any]:
    """Capture every transcript into the folder `out` and answer the run's record (also written as `capture.json`).

    A folder holds captures made one way, with one model. Running again with the same settings carries on: a
    transcript already captured there is left alone. Other settings, or another model, are refused, so two kinds of
    capture are never mixed in one folder. A transcript that has nothing to capture is skipped and one that fails is
    recorded with its error; neither stops the rest.
    """
    from safetensors.torch import save_file

    out = Path(out)
    wanted = settings(model, select=select, skip=skip, mean=mean, blocks=blocks)
    before = _read_json(out / "capture.json") if (out / "capture.json").is_file() else None
    if before is not None and before.get("settings") != wanted:
        raise CaptureError(f"{out} already holds captures made with other settings or another model. Use another folder.")
    out.mkdir(parents=True, exist_ok=True)
    started = _now()
    done, skipped, failed, kept = [], [], [], []
    for t in items:
        tid = t.get("id")
        if not isinstance(tid, str) or not _ID.fullmatch(tid):
            failed.append({"transcript": tid, "error": "the transcript has no usable id"})
            continue
        named = t.get("model")
        if named and named != model.contract.name:
            skipped.append({"transcript": tid, "reason": f"made for another model's contract ({named})"})
            continue
        if (out / f"{tid}.json").is_file():
            kept.append(tid)
            continue
        try:
            record, tensor = capture(model, t, select=select, skip=skip, mean=mean, blocks=blocks)
        except Nothing as nothing:
            skipped.append({"transcript": tid, "reason": str(nothing)})
            continue
        except ContractError:
            raise   # the tokenizer or the model is not the pinned one: nothing captured here could be trusted
        except Exception as error:   # one transcript (too long for the memory, say) does not stop the rest
            failed.append({"transcript": tid, "error": f"{type(error).__name__}: {error}"[:500]})
            _free()
            continue
        file = out / f"{tid}.safetensors"
        save_file({record["tensor"]["name"]: tensor.contiguous().cpu()}, str(file))
        record["tensor"].update(file=file.name, sha256=_sha(file), bytes=file.stat().st_size)
        _write_json(out / f"{tid}.json", record)   # written last: an item is there only once its numbers are
        done.append(tid)
        if log:
            log(f"{tid}  turn {record['turn']}  {record['tensor']['shape']}  {', '.join(record['flags']) or 'no flags'}")
    result = {
        "schema": RUN_SCHEMA,
        "settings": wanted,
        "runtime": runtime(model),
        "started": started,
        "finished": _now(),
        "captured": done,
        "already_there": kept,
        "skipped": skipped,
        "failed": failed,
    }
    record_of_folder = {**result, "captured": sorted(set((before or {}).get("captured", [])) | set(done) | set(kept))}
    _write_json(out / "capture.json", record_of_folder)
    return result


def items(out: str | Path) -> list[dict[str, Any]]:
    """Every capture in a folder: its record."""
    out = Path(out)
    return [_read_json(p) for p in sorted(out.glob("*.json")) if p.name != "capture.json"]


def read(out: str | Path, transcript: str) -> tuple[dict[str, Any], Any]:
    """One capture: (record, tensor). The file is checked against its recorded hash first."""
    from safetensors.torch import load_file

    out = Path(out)
    path = out / f"{transcript}.json"
    if not path.is_file():
        raise CaptureError(f"No capture of {transcript} in {out}.")
    record = _read_json(path)
    file = out / record["tensor"]["file"]
    if not file.is_file() or _sha(file) != record["tensor"]["sha256"]:
        raise CaptureError(f"{file.name} is missing or is not the file that was captured (its hash differs).")
    return record, load_file(str(file))[record["tensor"]["name"]]


def verify(out: str | Path) -> dict[str, Any]:
    """Read every capture in a folder back: hash, shape, precision, and that every number is finite."""
    import torch

    problems, count = [], 0
    for record in items(out):
        count += 1
        tid = record.get("transcript")
        try:
            _, tensor = read(out, tid)
        except CaptureError as error:
            problems.append({"transcript": tid, "problem": str(error)})
            continue
        want = record["tensor"]
        tokens = len(record["tokens"])
        expected = [len(record["blocks"]), want["shape"][-1]] if record["mean"] else [len(record["blocks"]), tokens, want["shape"][-1]]
        if list(tensor.shape) != want["shape"] or want["shape"] != expected:
            problems.append({"transcript": tid, "problem": f"shape {list(tensor.shape)} is not {expected}"})
        elif str(tensor.dtype).removeprefix("torch.") != want["dtype"]:
            problems.append({"transcript": tid, "problem": f"precision {tensor.dtype} is not {want['dtype']}"})
        elif not bool(torch.isfinite(tensor.float()).all()):
            problems.append({"transcript": tid, "problem": "not every number is finite"})
    return {"captures": count, "passed": not problems, "problems": problems}


def runtime(model: Any) -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "transformers", "safetensors", "accelerate", "huggingface-hub", "undertone-models", "undertone-capture"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() and str(model.device).startswith("cuda") else None
    return {"python": platform.python_version(), "packages": packages, "device": str(model.device), "gpu": gpu}


def _free() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
