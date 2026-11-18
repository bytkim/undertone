"""A model's contract: the facts the undertone tools need about one model, read from its `contract.json`.

The file is the boundary. Anything that only needs facts (the engine, a page, a script in another language) reads the
JSON; nothing here needs torch. A contract is found by its folder's name (`qwen3.5-4b`) or by a path.
"""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "undertone.model.v1"
#: the four depths every model's blocks are grouped into, from its early layers to its output
DEPTHS = ("early", "middle", "late", "output")

_HERE = Path(__file__).resolve().parent
_SHA = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """A contract that cannot be found or read, or a model that does not match its contract."""


@dataclass(frozen=True)
class Contract:
    """One model's contract. The fields the tools use most are spelled out; `data` is the whole file."""

    name: str
    title: str
    #: the folder the contract lives in (its adapter is beside it)
    folder: Path
    data: dict[str, Any]

    @property
    def repo(self) -> str:
        return self.data["source"]["repo"]

    @property
    def revision(self) -> str:
        return self.data["source"]["revision"]

    @property
    def files(self) -> dict[str, dict[str, Any]]:
        """Every file of the pinned model: its SHA-256 and size."""
        return self.data["source"]["files"]

    @property
    def served_as(self) -> list[str]:
        """The names a server is known to serve these exact weights under."""
        return self.data["served_as"]

    @property
    def blocks(self) -> int:
        return self.data["shape"]["blocks"]

    @property
    def width(self) -> int:
        return self.data["shape"]["width"]

    @property
    def depths(self) -> dict[str, list[int]]:
        """Which blocks (counted from 0) make up each of the four depths."""
        return self.data["depths"]

    @property
    def thinking(self) -> bool:
        return self.data["text"]["thinking"]

    @property
    def max_tokens(self) -> int:
        return self.data["text"]["max_tokens"]

    @property
    def lenses(self) -> dict[str, dict[str, Any]]:
        """The lens files that exist for this model, each pinned to a revision and a hash."""
        return self.data.get("artifacts", {}).get("lenses", {})

    @property
    def checks(self) -> dict[str, float]:
        """The tolerances the model was qualified to."""
        return self.data["qualified"]["checks"]


def roots() -> list[Path]:
    """Where model folders are looked for: UNDERTONE_MODELS if set, then beside this package (a checkout), then
    inside it (an installed wheel)."""
    extra = [Path(p) for p in os.environ.get("UNDERTONE_MODELS", "").split(os.pathsep) if p]
    return [*extra, _HERE.parent, _HERE / "models"]


def names() -> list[str]:
    """Every model that has a contract here."""
    found = {p.parent.name for root in roots() if root.is_dir() for p in root.glob("*/contract.json")}
    return sorted(found)


def contract(ref: str | os.PathLike[str]) -> Contract:
    """A contract by name (`qwen3.5-4b`), or by the path of its folder or of its `contract.json`."""
    path = Path(ref)
    if path.is_dir():
        path = path / "contract.json"
    if not path.is_file():
        hits = [root / str(ref) / "contract.json" for root in roots()]
        path = next((h for h in hits if h.is_file()), path)
    if not path.is_file():
        known = ", ".join(names()) or "none"
        raise ContractError(f"No model contract {str(ref)!r}. Known here: {known}.")
    return read(path)


def read(path: Path) -> Contract:
    """Read and check one `contract.json`. A contract that is not whole is refused here, so no tool has to wonder."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ContractError(f"{path} is not JSON: {error}") from error
    problems = _problems(data)
    if problems:
        raise ContractError(f"{path} is not a usable model contract: " + "; ".join(problems))
    return Contract(name=data["name"], title=data.get("title", data["name"]), folder=path.resolve().parent, data=data)


def _problems(d: Any) -> list[str]:
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        return [f'its schema is not "{SCHEMA}"']
    out: list[str] = []
    for key in ("name", "source", "served_as", "shape", "text", "depths", "qualified"):
        if key not in d:
            out.append(f'"{key}" is missing')
    if out:
        return out
    source, shape = d["source"], d["shape"]
    if not source.get("repo") or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", ""))):
        out.append("the source needs a repo and a full 40-character revision")
    files = source.get("files") or {}
    if not files or any(not _SHA.fullmatch(str(f.get("sha256", ""))) for f in files.values()):
        out.append("every source file needs a SHA-256")
    blocks, width = shape.get("blocks"), shape.get("width")
    if not (isinstance(blocks, int) and blocks > 0 and isinstance(width, int) and width > 0):
        out.append("the shape needs whole numbers of blocks and width")
        return out
    depths = d["depths"]
    if list(depths) != list(DEPTHS):
        out.append("the depths must be " + ", ".join(DEPTHS) + ", in that order")
    else:
        flat = [b for name in DEPTHS for b in depths[name]]
        if not all(depths[name] for name in DEPTHS) or flat != sorted(set(flat)) or flat[0] < 0 or flat[-1] != blocks - 1:
            out.append("the depths must be rising blocks without repeats, ending at the last block")
    if not isinstance(d["text"].get("thinking"), bool) or not isinstance(d["text"].get("max_tokens"), int):
        out.append("the text part needs thinking (true or false) and max_tokens")
    if not isinstance(d["served_as"], list):
        out.append("served_as must be a list of names")
    if not isinstance(d["qualified"].get("checks"), dict):
        out.append("the qualified part needs its checks")
    for name, lens in (d.get("artifacts", {}).get("lenses") or {}).items():
        if not _SHA.fullmatch(str(lens.get("sha256", ""))) or not lens.get("repo") or not lens.get("path"):
            out.append(f'the lens "{name}" needs a repo, a path and a SHA-256')
    return out
