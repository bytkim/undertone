import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeTokenizer
from undertone_engine import cli, read_contract

QWEN = Path(__file__).resolve().parents[2] / "tools" / "models" / "qwen3.5-4b"


def test_the_engine_reads_a_models_contract_by_path():
    for ref in (QWEN / "contract.json", QWEN):
        c = read_contract(str(ref))
        assert (c.name, c.repo, c.thinking) == ("qwen3.5-4b", "Qwen/Qwen3.5-4B", False)
        assert len(c.revision) == 40
    assert c.knows("qwen/qwen3.5-4b") and not c.knows("qwen-q4")


def test_a_contract_by_name_needs_the_models_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "undertone_models", None)   # not installed
    with pytest.raises(ValueError, match="install core/tools/models"):
        read_contract("qwen3.5-4b")
    # installed: the package finds the folder, and the engine still reads the JSON itself
    def find(name):
        if name != "qwen3.5-4b":
            raise ValueError(f"No model contract {name!r}.")
        return SimpleNamespace(folder=QWEN)

    monkeypatch.setitem(sys.modules, "undertone_models", SimpleNamespace(contract=find))
    assert read_contract("qwen3.5-4b").name == "qwen3.5-4b"
    with pytest.raises(ValueError, match="No model contract 'gemma-9'"):
        read_contract("gemma-9")


def test_a_file_that_is_not_a_contract_is_refused(tmp_path):
    for body in ("not json", "{}", json.dumps({"schema": "undertone.model.v1", "name": "x"})):
        (tmp_path / "contract.json").write_text(body)
        with pytest.raises(ValueError, match="is not a model contract"):
            read_contract(str(tmp_path / "contract.json"))


def _args(tmp_path, **more):
    base = dict(base_url="http://localhost:8000/v1", served_model="Qwen/Qwen3.5-4B", token_ids=False, tokenizer=None,
                tokenizer_revision=None, contract=str(QWEN / "contract.json"), transcripts=str(tmp_path))
    return SimpleNamespace(**{**base, **more})


def test_with_a_contract_the_engine_names_the_model_and_uses_its_pinned_tokenizer(tmp_path, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(cli, "hf_tokenizer", lambda name, revision=None: asked.append((name, revision)) or FakeTokenizer())
    engine = cli._engine(_args(tmp_path))[0]
    assert engine.model == "qwen3.5-4b" and isinstance(engine.tokenizer, FakeTokenizer)
    assert asked == [("Qwen/Qwen3.5-4B", read_contract(str(QWEN)).revision)]
    assert capsys.readouterr().err == ""
    # a tokenizer named by hand wins
    cli._engine(_args(tmp_path, tokenizer="some/other", tokenizer_revision="abc"))
    assert asked[-1] == ("some/other", "abc")


def test_without_the_tokenizer_library_the_engine_says_so_and_carries_on(tmp_path, monkeypatch, capsys):
    def missing(name, revision=None):
        raise RuntimeError("Rebuilding token ids needs the tokenizer library")

    monkeypatch.setattr(cli, "hf_tokenizer", missing)
    engine = cli._engine(_args(tmp_path))[0]
    assert engine.model == "qwen3.5-4b" and engine.tokenizer is None
    assert "token ids will not be rebuilt" in capsys.readouterr().err


def test_a_served_name_the_contract_does_not_know_is_pointed_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "hf_tokenizer", lambda name, revision=None: FakeTokenizer())
    cli._engine(_args(tmp_path, served_model="qwen-q4"))
    err = capsys.readouterr().err
    assert 'the endpoint serves "qwen-q4"' in err and "will be flagged" in err
    with pytest.raises(SystemExit, match="No contract file"):
        monkeypatch.setitem(sys.modules, "undertone_models", None)
        cli._engine(_args(tmp_path, contract="nothing-here"))
