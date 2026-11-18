"""Model contracts for the undertone tools.

    from undertone_models import contract, adapter

    qwen = contract("qwen3.5-4b")          # the facts: no torch needed
    model = adapter(qwen).load(qwen)       # the model itself: needs the `run` extra and a GPU
    rendering = model.render(transcript)   # a transcript as token ids, with the reply's span
    blocks = model.run(rendering.ids, rendering.reply.positions)

Each model is a folder beside this package holding `contract.json` (the facts), `adapter.py` (the only code that
touches the model's internals) and `check.py` (a self-test against the contract).
"""
import importlib.util
import re
import sys
from types import ModuleType

from .contract import DEPTHS, SCHEMA, Contract, ContractError, contract, names, read
from .rendering import FLAGS, Rendering, Span

__all__ = ["DEPTHS", "FLAGS", "SCHEMA", "Contract", "ContractError", "Rendering", "Span", "adapter", "check", "contract", "names", "read"]

#: what every model's adapter.py must offer
ADAPTER = ("tokenizer", "render", "load")


def _module(c: Contract, file: str) -> ModuleType:
    path = c.folder / file
    if not path.is_file():
        raise ContractError(f"{c.name} has no {file} beside its contract.")
    name = f"undertone_models._{re.sub(r'[^0-9a-z]', '_', c.name.lower())}_{file.removesuffix('.py')}"
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapter(model: Contract | str) -> ModuleType:
    """A model's adapter: `tokenizer(contract)`, `render(contract, tokenizer, transcript)` and `load(contract)`.
    Importing it needs nothing heavy; loading the model does."""
    c = model if isinstance(model, Contract) else contract(model)
    module = _module(c, "adapter.py")
    missing = [f for f in ADAPTER if not callable(getattr(module, f, None))]
    if missing:
        raise ContractError(f"{c.name}'s adapter is missing: {', '.join(missing)}.")
    return module


def check(model: Contract | str) -> ModuleType:
    """A model's self-test module: `check(loaded_model)` answers a report."""
    c = model if isinstance(model, Contract) else contract(model)
    module = _module(c, "check.py")
    if not callable(getattr(module, "check", None)):
        raise ContractError(f"{c.name}'s check.py has no check().")
    return module
