import hashlib
import json

import pytest

from undertone_models import DEPTHS, ContractError, adapter, check, contract, names, read
from undertone_models.cli import main


def test_the_qwen_contract_reads_and_says_what_the_tools_need(qwen):
    assert names() == ["qwen3.5-4b"]
    assert (qwen.name, qwen.title, qwen.repo) == ("qwen3.5-4b", "Qwen3.5-4B", "Qwen/Qwen3.5-4B")
    assert len(qwen.revision) == 40
    assert (qwen.blocks, qwen.width, qwen.thinking, qwen.max_tokens) == (32, 2560, False, 2048)
    # the four depths, in order, are rising blocks that end at the last one
    assert list(qwen.depths) == list(DEPTHS)
    assert [(d[0], d[-1]) for d in qwen.depths.values()] == [(8, 15), (16, 23), (24, 30), (31, 31)]
    # every file of the model, and every lens, is pinned to a hash
    assert {"chat_template.jinja", "tokenizer.json", "config.json"} <= set(qwen.files)
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in qwen.files.values())
    assert set(qwen.lenses) == {"jpp", "jlens"}
    assert all(len(lens["revision"]) == 40 and len(lens["sha256"]) == 64 for lens in qwen.lenses.values())
    assert qwen.checks == {"fp32_head_atol": 0.0001, "full_vocab_tv_max": 0.05, "repeat_residual_max_abs": 0.0}


def test_a_contract_is_found_by_name_folder_or_file(qwen):
    assert contract(qwen.folder).data == qwen.data
    assert contract(qwen.folder / "contract.json").data == qwen.data
    with pytest.raises(ContractError, match="Known here: qwen3.5-4b"):
        contract("gemma-9")


@pytest.mark.parametrize(
    "change, complaint",
    [
        ({"schema": "undertone.model.v0"}, "schema"),
        ({"depths": None}, '"depths" is missing'),
        ({"source.revision": "main"}, "40-character revision"),
        ({"source.files": {"config.json": {"sha256": "abc"}}}, "SHA-256"),
        ({"depths.output": [30]}, "ending at the last block"),
        ({"depths.late": [24, 24, 25]}, "without repeats"),
        ({"text.thinking": "off"}, "thinking"),
        ({"artifacts.lenses.jpp.sha256": ""}, 'lens "jpp"'),
    ],
)
def test_a_contract_that_is_not_whole_is_refused(tmp_path, qwen, change, complaint):
    data = json.loads(json.dumps(qwen.data))
    for key, value in change.items():
        node = data
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        if value is None:
            del node[last]
        else:
            node[last] = value
    (tmp_path / "contract.json").write_text(json.dumps(data))
    with pytest.raises(ContractError, match=complaint):
        read(tmp_path / "contract.json")


def test_a_models_adapter_and_self_test_load_from_its_folder(qwen):
    module = adapter(qwen)
    assert all(callable(getattr(module, f)) for f in ("tokenizer", "render", "load", "fetch", "verify"))
    assert adapter("qwen3.5-4b") is module   # loaded once
    assert callable(check(qwen).check)


def test_files_are_proved_against_the_contract(tmp_path, small, qwen_adapter):
    (tmp_path / "snap").mkdir()
    (tmp_path / "snap" / "config.json").write_text("{}")
    c = small(**{"source.files": {"config.json": {"sha256": hashlib.sha256(b"{}").hexdigest(), "bytes": 2}}})
    qwen_adapter.verify(c, tmp_path / "snap")
    (tmp_path / "snap" / "config.json").write_text("{ }")
    with pytest.raises(ContractError, match="not the pinned file"):
        qwen_adapter.verify(c, tmp_path / "snap")
    (tmp_path / "snap" / "config.json").unlink()
    with pytest.raises(ContractError, match="missing"):
        qwen_adapter.verify(c, tmp_path / "snap")


def test_the_command_line_lists_and_shows(capsys):
    main(["list"])
    assert capsys.readouterr().out.startswith("qwen3.5-4b  Qwen3.5-4B  Qwen/Qwen3.5-4B@851bf6e806ef")
    main(["show", "qwen3.5-4b"])
    out = capsys.readouterr().out
    assert "32 blocks, width 2560" in out and "early 8-15; middle 16-23; late 24-30; output 31" in out
    assert "thinking off, up to 2048 tokens" in out and "jpp, jlens" in out and "not qualified:" in out
    with pytest.raises(SystemExit, match="No model contract"):
        main(["show", "nothing"])
