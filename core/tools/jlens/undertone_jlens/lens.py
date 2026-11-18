"""The lens itself: one map for each block but the last, taking that block's output to the last block.

A block's output, moved by its map and then put through the model's own last step, reads as words: what the model
would go on to say if what it holds at that depth were carried straight to the end. The last block needs no map.

The lens files are made by others and pinned in the model's contract (repository, revision, path, SHA-256). This
module fetches one, proves it is the pinned file, and checks it fits the model before using it.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from undertone_models import Contract


class LensError(ValueError):
    """A lens that cannot be found, is not the pinned file, or does not fit the model."""


@dataclass
class Lens:
    name: str
    #: block -> its map [width, width], in float32
    maps: dict[int, Any]
    #: the block every map leads to: the model's last
    target: int
    #: where the file came from: the contract's pin
    source: dict[str, Any]
    #: what the file says of how it was made
    provenance: dict[str, Any] | None = None

    def transport(self, block: int, residual: Any) -> Any:
        """A block's output as the last block would hold it, in float32. The last block passes through unchanged."""
        x = residual.float()
        if block == self.target:
            return x
        if block not in self.maps:
            raise LensError(f"The {self.name} lens has no map for block {block}.")
        return x @ self.maps[block].T


def fetch(contract: Contract, name: str) -> Path:
    """Download the pinned lens file (about 400 MB) and answer where it is."""
    from huggingface_hub import hf_hub_download

    pin = _pin(contract, name)
    return Path(hf_hub_download(pin["repo"], pin["path"], revision=pin["revision"]))


def load(contract: Contract, name: str = "jpp", path: str | Path | None = None, device: str = "cuda", verify_file: bool = True) -> Lens:
    """A lens the contract pins, ready to use. `path` is a file already in hand; without one it is fetched. The file
    is hashed against the contract first, then checked against the model: a map for every block but the last, each
    [width, width], every number finite."""
    import torch

    pin = _pin(contract, name)
    file = Path(path) if path is not None else fetch(contract, name)
    if not file.is_file():
        raise LensError(f"No lens file at {file}.")
    if verify_file and _sha(file) != pin["sha256"]:
        raise LensError(f"{file.name} is not the pinned {name} lens (its hash differs).")
    data = torch.load(str(file), map_location="cpu", weights_only=True)   # tensors and plain values only: no code is run
    maps = data.get("J") if isinstance(data, dict) else None
    last = contract.blocks - 1
    if not isinstance(maps, dict) or {int(k) for k in maps} != set(range(last)) or data.get("d_model") != contract.width:
        raise LensError(f"The {name} lens does not fit {contract.title}: it needs one map for each of blocks 0 to {last - 1}, of width {contract.width}.")
    out = {}
    for block, matrix in maps.items():
        if list(matrix.shape) != [contract.width, contract.width] or not bool(torch.isfinite(matrix).all()):
            raise LensError(f"The {name} lens's map for block {block} is not a finite {contract.width} by {contract.width} matrix.")
        out[int(block)] = matrix.float().to(device)
    provenance = data.get("provenance") if isinstance(data.get("provenance"), dict) else None
    return Lens(name=name, maps=out, target=last, source={"name": name, **{k: pin[k] for k in ("repo", "revision", "path", "sha256")}}, provenance=provenance)


def _pin(contract: Contract, name: str) -> dict[str, Any]:
    if name not in contract.lenses:
        known = ", ".join(contract.lenses) or "none"
        raise LensError(f"{contract.title}'s contract pins no lens called {name!r}. It has: {known}.")
    return contract.lenses[name]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()
