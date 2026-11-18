"""Read a capture through the lens: the words the model leans toward at each depth.

For one captured token and one depth:

  1. each block of that depth is moved to the last block by its map (the last block needs none);
  2. the model's own last step turns it into a probability for every piece of the vocabulary;
  3. the pieces that spell the same word are added up (words.py);
  4. the depth's blocks are averaged.

That gives a mass for every word type at every token and depth. The whole reply is the average over its tokens.

The words can be ranked two ways. By probability, the list is what the model would most likely say, which is mostly
common words. Against a baseline (the same masses averaged over many ordinary texts), a word ranks by how much more
it is leaned toward here than usual, weighted by how much it is leaned toward at all: mass x log(mass / usual mass).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from undertone_capture import CaptureError, items as capture_items, read as read_capture
from undertone_models import DEPTHS

from .lens import Lens, _sha
from .words import Words

SCHEMA = "undertone.lens.v1"
RUN_SCHEMA = "undertone.lens.run.v1"
BASELINE_SCHEMA = "undertone.lens.baseline.v1"
#: keeps the ratio finite for a word with no mass in the baseline
FLOOR = 1e-7
#: how many tokens are turned into vocabulary-wide probabilities at once
CHUNK = 16


class ReadingError(ValueError):
    """A capture that cannot be read through the lens, or a folder or baseline that does not match."""


class Reader:
    """A model's last step, a lens and the vocabulary's word types: everything needed to read captures."""

    def __init__(self, head: Any, lens: Lens, words: Words | None = None):
        self.head, self.lens, self.contract = head, lens, head.contract
        self.words = words or Words.of(head.tokenizer, head.vocab_size, device=str(head.device))
        self.depths = {d: list(self.contract.depths[d]) for d in DEPTHS}

    def masses(self, record: Mapping[str, Any], blocks: Any) -> Iterator[Any]:
        """The word masses of a capture, a few tokens at a time: tensors [depths, tokens in this chunk, word types],
        float32, in the capture's token order."""
        import torch

        if record.get("mean"):
            raise ReadingError("this capture holds an average over tokens, not the tokens: capture again without --mean")
        row = {int(b): k for k, b in enumerate(record["blocks"])}
        missing = sorted({b for layers in self.depths.values() for b in layers} - set(row))
        if missing:
            raise ReadingError(f"this capture leaves out blocks the depths need ({missing[0]} to {missing[-1]})")
        if record["model"]["name"] != self.contract.name or record["model"]["revision"] != self.contract.revision:
            raise ReadingError(f"this capture was made with {record['model']['name']} at another revision")
        n = blocks.shape[1]
        with torch.inference_mode():
            for lo in range(0, n, CHUNK):
                out = torch.zeros(len(DEPTHS), min(CHUNK, n - lo), len(self.words), dtype=torch.float32, device=self.head.device)
                for d, depth in enumerate(DEPTHS):
                    for block in self.depths[depth]:
                        moved = self.lens.transport(block, blocks[row[block], lo : lo + CHUNK].to(self.head.device))
                        out[d] += self.words.merge(self.head.head(moved).softmax(-1))
                    out[d] /= len(self.depths[depth])
                yield out

    def read(self, record: Mapping[str, Any], blocks: Any, top: int = 8, baseline: "Baseline | None" = None) -> dict[str, Any]:
        """One capture as words: for each token and for the whole reply, the `top` words at each depth."""
        import torch

        if baseline is not None:
            baseline.fits(self)
        tokens = [dict(position=t["position"], id=t["id"], text=t["text"], lens={}) for t in record["tokens"]]
        total = torch.zeros(len(DEPTHS), len(self.words), dtype=torch.float64, device=self.head.device)
        at = 0
        for chunk in self.masses(record, blocks):
            total += chunk.sum(1)
            for k in range(chunk.shape[1]):
                tokens[at + k]["lens"] = {depth: self._top(chunk[d, k], top, baseline, d) for d, depth in enumerate(DEPTHS)}
            at += chunk.shape[1]
        mean = (total / max(at, 1)).float()
        return {
            "schema": SCHEMA,
            "transcript": record.get("transcript"),
            "conversation": record.get("conversation"),
            "turn": record.get("turn"),
            "model": record["model"],
            "lens": self.lens.source,
            "capture": {"sha256": record["tensor"].get("sha256"), "select": record.get("select"), "skip": record.get("skip")},
            "flags": list(record.get("flags", [])),
            "qualified": record.get("qualified"),
            "ranked_by": "lift over a baseline" if baseline is not None else "probability",
            "baseline": baseline.summary if baseline is not None else None,
            "top": top,
            "depths": self.depths,
            "word_types": len(self.words),
            "overall": {depth: self._top(mean[d], top, baseline, d) for d, depth in enumerate(DEPTHS)},
            "tokens": tokens,
        }

    def _top(self, mass: Any, top: int, baseline: "Baseline | None", d: int) -> list[dict[str, Any]]:
        """The `top` words of one mass vector [word types], most leaned-toward first."""
        if baseline is None:
            best = mass.topk(min(top, mass.numel()))
            return [{"word": self.words.types[j], "p": _round(p)} for p, j in zip(best.values.tolist(), best.indices.tolist()) if p > 0]
        usual = baseline.mass[d]
        ratio = (mass + FLOOR) / (usual + FLOOR)
        score = mass * ratio.log()
        best = score.topk(min(top, score.numel()))
        return [{"word": self.words.types[j], "p": _round(float(mass[j])), "lift": _round(float(ratio[j]))} for s, j in zip(best.values.tolist(), best.indices.tolist()) if s > 0]


class Baseline:
    """The usual word mass at each depth: an average over many tokens of ordinary text, to rank words against."""

    def __init__(self, mass: Any, summary: dict[str, Any]):
        self.mass, self.summary = mass, summary

    def fits(self, reader: Reader) -> None:
        s = self.summary
        if s["model"]["revision"] != reader.contract.revision or s["lens"]["sha256"] != reader.lens.source["sha256"] or s["word_types_sha256"] != reader.words.sha256:
            raise ReadingError("The baseline was made with another model, lens or vocabulary.")


def build_baseline(reader: Reader, captures: str | Path, out: str | Path, note: str | None = None, log: Any = None) -> dict[str, Any]:
    """Average the word masses over every token of every capture in a folder, and keep that in `out` as
    `baseline.safetensors` and `baseline.json`. The captures should be ordinary text: what "usual" is to mean.
    `note` says what they were, for whoever reads the baseline's record later."""
    import torch
    from safetensors.torch import save_file

    out = Path(out)
    if (out / "baseline.json").exists():
        raise ReadingError(f"{out} already holds a baseline.")
    total = torch.zeros(len(DEPTHS), len(reader.words), dtype=torch.float64, device=reader.head.device)
    tokens, used, left = 0, [], []
    for record in capture_items(captures):
        try:
            _, blocks = read_capture(captures, record["transcript"])
            for chunk in reader.masses(record, blocks):
                total += chunk.sum(1)
                tokens += chunk.shape[1]
        except (ReadingError, CaptureError) as error:
            left.append({"transcript": record.get("transcript"), "reason": str(error)})
            continue
        used.append(record["transcript"])
        if log:
            log(f"{record['transcript']}  {len(record['tokens'])} tokens")
    if not tokens:
        raise ReadingError(f"No capture in {captures} could be read.")
    out.mkdir(parents=True, exist_ok=True)
    save_file({"mass": (total / tokens).float().cpu()}, str(out / "baseline.safetensors"))
    summary = {
        "schema": BASELINE_SCHEMA,
        "model": {"name": reader.contract.name, "repo": reader.contract.repo, "revision": reader.contract.revision},
        "lens": reader.lens.source,
        "depths": reader.depths,
        "word_types": len(reader.words),
        "word_types_sha256": reader.words.sha256,
        "captures": len(used),
        "tokens": tokens,
        "note": note,
        "left_out": left,
        "made": _now(),
        "sha256": _sha(out / "baseline.safetensors"),
    }
    _write(out / "baseline.json", summary)
    return summary


def load_baseline(folder: str | Path, device: str = "cpu") -> Baseline:
    from safetensors.torch import load_file

    folder = Path(folder)
    if not (folder / "baseline.json").is_file():
        raise ReadingError(f"No baseline in {folder}.")
    summary = json.loads((folder / "baseline.json").read_text())
    file = folder / "baseline.safetensors"
    if summary.get("schema") != BASELINE_SCHEMA or not file.is_file() or _sha(file) != summary["sha256"]:
        raise ReadingError(f"{folder} does not hold the baseline its record describes.")
    return Baseline(load_file(str(file))["mass"].to(device), summary)


#: where the baselines that come with the tool are kept: beside the package in a checkout, inside it in a wheel
_SHIPPED = (Path(__file__).resolve().parent.parent / "baselines", Path(__file__).resolve().parent / "baselines")


def shipped_baseline(reader: Reader, roots: Any = None) -> Baseline | None:
    """The baseline that comes with the tool for this reader's model and lens, or None when there is none or it
    does not fit (another model revision, another lens file, or a vocabulary that decodes differently)."""
    name = f"{reader.contract.name}-{reader.lens.name}"
    for root in roots or _SHIPPED:
        if (Path(root) / name / "baseline.json").is_file():
            try:
                baseline = load_baseline(Path(root) / name, device=str(reader.head.device))
                baseline.fits(reader)
            except ReadingError:
                return None
            return baseline
    return None


def run(reader: Reader, captures: str | Path, out: str | Path, top: int = 8, baseline: Baseline | None = None, log: Any = None) -> dict[str, Any]:
    """Read every capture in a folder and keep each as `<transcript id>.json` in `out`, with one `lens.json` for
    the run. A folder holds readings made one way; running again with the same settings carries on, and other
    settings are refused. A capture that cannot be read is recorded with the reason, and the rest carry on."""
    out = Path(out)
    settings = {
        "model": {"name": reader.contract.name, "repo": reader.contract.repo, "revision": reader.contract.revision},
        "lens": reader.lens.source,
        "top": top,
        "baseline": baseline.summary["sha256"] if baseline is not None else None,
    }
    before = json.loads((out / "lens.json").read_text()) if (out / "lens.json").is_file() else None
    if before is not None and before.get("settings") != settings:
        raise ReadingError(f"{out} already holds readings made with other settings. Use another folder.")
    out.mkdir(parents=True, exist_ok=True)
    started, done, kept, left = _now(), [], [], []
    for record in capture_items(captures):
        tid = record["transcript"]
        if (out / f"{tid}.json").is_file():
            kept.append(tid)
            continue
        try:
            _, blocks = read_capture(captures, tid)   # the file is checked against its hash first
            reading = reader.read(record, blocks, top=top, baseline=baseline)
        except (ReadingError, CaptureError) as error:
            left.append({"transcript": tid, "reason": str(error)})
            continue
        _write(out / f"{tid}.json", reading)
        done.append(tid)
        if log:
            words = ", ".join(w["word"] for w in reading["overall"]["middle"][:4])
            log(f"{tid}  turn {reading['turn']}  {len(reading['tokens'])} tokens  middle: {words}")
    result = {"schema": RUN_SCHEMA, "settings": settings, "started": started, "finished": _now(), "read": done, "already_there": kept, "left_out": left}
    _write(out / "lens.json", {**result, "read": sorted(set((before or {}).get("read", [])) | set(done) | set(kept))})
    return result


def readings(out: str | Path) -> list[dict[str, Any]]:
    """Every reading in a folder."""
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob("*.json")) if p.name != "lens.json"]


def _round(x: float) -> float:
    return float(f"{x:.4g}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
