"""A model's contract, as far as the engine needs it.

The undertone tools keep one `contract.json` per model they support (core/tools/models): what the model is, pinned to
an exact revision. The engine reads three things from it and nothing else: the contract's name (recorded on every
transcript), the tokenizer to rebuild token ids with when the endpoint returns none, and the names the model is
known to be served under. It reads the JSON only, so the engine still runs no model and needs no GPU.
"""
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "undertone.model.v1"


@dataclass(frozen=True)
class ModelContract:
    name: str
    #: the tokenizer's repository and exact revision
    repo: str
    revision: str
    #: whether the tools read this model with thinking on
    thinking: bool
    #: the names a server is known to serve these weights under
    served_as: tuple[str, ...]

    def knows(self, served_model: str) -> bool:
        return served_model.lower() in {name.lower() for name in self.served_as}


def read_contract(ref: str) -> ModelContract:
    """A contract by the path of its `contract.json` (or of its folder), or by its name when the undertone models
    package is installed (`pip install -e core/tools/models`)."""
    path = Path(ref)
    if path.is_dir():
        path = path / "contract.json"
    if not path.is_file():
        try:
            from undertone_models import contract as find
        except ImportError:
            raise ValueError(f"No contract file at {ref!r}. Give the path to a contract.json, or install core/tools/models to name a model.") from None
        try:
            path = Path(find(ref).folder) / "contract.json"
        except ValueError as error:
            raise ValueError(str(error)) from None
    try:
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            raise KeyError("schema")
        return ModelContract(
            name=data["name"],
            repo=data["source"]["repo"],
            revision=data["source"]["revision"],
            thinking=bool(data["text"]["thinking"]),
            served_as=tuple(data["served_as"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        raise ValueError(f'{path} is not a model contract ("{SCHEMA}").') from None
