"""Capture on the stand-in model and tokenizer (undertone_models.standin): the logic, not any real model."""
import json

import pytest

from undertone_capture import CaptureError, Nothing, capture, items, read, run, transcripts, verify
from undertone_capture.cli import _blocks, main
from undertone_models import adapter, contract, read as read_contract
from undertone_models.standin import FakeTokenizer, standin_model, transcript

torch = pytest.importorskip("torch")

SAID, REPLY = "What do you see?", "A quiet street under gray rain."


@pytest.fixture
def model(tmp_path):
    """The Qwen adapter around a 4-block stand-in, under a contract cut down to match."""
    qwen = contract("qwen3.5-4b")
    data = json.loads(json.dumps(qwen.data))
    data["shape"].update(blocks=4, width=16)
    data["depths"] = {"early": [0], "middle": [1], "late": [2], "output": [3]}
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "contract.json").write_text(json.dumps(data))
    return adapter(qwen).Model(read_contract(tmp_path / "small" / "contract.json"), standin_model(), FakeTokenizer(), "cpu")


def line(model, said=SAID, reply=REPLY, tid="t1", turn=1, **more):
    return {"schema": "undertone.transcript.v1", **transcript(model.tokenizer, said, reply, **more), "id": tid, "turn": turn}


def test_a_reply_is_captured_at_every_block_with_its_tokens(model):
    t = line(model)
    record, blocks = capture(model, t)
    r = model.render(t)
    assert list(blocks.shape) == [4, len(r.reply.positions), 16] == record["tensor"]["shape"]
    assert torch.equal(blocks, model.run(r.ids, r.reply.positions))
    # the record says what the numbers are of: each token, its text, and that the text is the reply exactly
    assert [tok["position"] for tok in record["tokens"]] == list(r.reply.positions)
    assert "".join(tok["text"] for tok in record["tokens"]) == REPLY and record["joins"] is True
    assert record["model"]["name"] == "qwen3.5-4b" and record["ids"]["input_ids"] == r.ids
    assert (record["select"], record["skip"], record["mean"], record["blocks"]) == ("reply", 0, False, [0, 1, 2, 3])
    assert record["flags"] == ["ids_rebuilt"] and record["qualified"] is True
    assert record["tensor"]["name"] == "blocks" and record["tensor"]["dtype"] == "float32"


def test_the_start_can_be_skipped_the_tokens_averaged_and_blocks_chosen(model):
    t = line(model)
    _, every = capture(model, t)
    record, tail = capture(model, t, skip=3)
    assert torch.equal(tail, every[:, 3:]) and record["joins"] is None and len(record["tokens"]) == every.shape[1] - 3
    record, mean = capture(model, t, skip=3, mean=True, blocks=[1, 3])
    assert list(mean.shape) == [2, 16] and mean.dtype == torch.float32
    assert torch.allclose(mean, every[[1, 3]][:, 3:].float().mean(1))
    assert (record["tensor"]["name"], record["blocks"]) == ("mean", [1, 3])
    with pytest.raises(Nothing, match="6 tokens and 50 are skipped"):
        capture(model, t, skip=50)
    with pytest.raises(Nothing):
        capture(model, line(model, reply=""))
    with pytest.raises(CaptureError, match="blocks 0 to 3"):
        capture(model, t, blocks=[4])
    with pytest.raises(CaptureError, match="reply"):
        capture(model, t, select="messages", skip=1)


def test_every_messages_tokens_can_be_selected_instead(model):
    t = line(model)
    record, blocks = capture(model, t, select="messages")
    r = model.render(t)
    assert [(s["turn"], s["role"]) for s in record["spans"]] == [(0, "user"), (1, "assistant")]
    assert blocks.shape[1] == sum(len(s.positions) for s in r.spans)
    assert "".join(tok["text"] for tok in record["tokens"]) == SAID + REPLY


def test_ids_that_disagree_with_the_template_are_captured_as_given_and_flagged(model):
    t = line(model, ids="thinking", **{"template.thinking": True})
    record, blocks = capture(model, t)
    given = t["tokens"]["prompt_ids"] + t["tokens"]["completion_ids"]
    assert record["ids"]["input_ids"] == given and record["ids"]["source"] == "server"
    assert record["flags"] == ["ids_differ_from_template", "thinking_on"] and record["qualified"] is False
    # the reply's span is everything the server generated; each token's text is what it decodes to
    assert blocks.shape[1] == len(t["tokens"]["completion_ids"]) and record["joins"] is False
    assert "".join(tok["text"] for tok in record["tokens"]).endswith(REPLY + "<|im_end|>")


def test_a_run_keeps_each_capture_as_two_files_and_can_be_read_back(model, tmp_path):
    out = tmp_path / "captures"
    lines = [line(model, tid="a1"), line(model, "And at night?", "Streetlights.", tid="a2", turn=2), line(model, "Hello", "", tid="a3", turn=3),
             {**line(model, tid="a4"), "model": "gemma-9"}, {**line(model, tid="../escape")}]
    seen = []
    result = run(model, lines, out, log=seen.append)
    assert result["captured"] == ["a1", "a2"] and len(seen) == 2
    assert result["skipped"] == [{"transcript": "a3", "reason": "the reply has no tokens"}, {"transcript": "a4", "reason": "made for another model's contract (gemma-9)"}]
    assert result["failed"] == [{"transcript": "../escape", "error": "the transcript has no usable id"}]
    assert sorted(p.name for p in out.iterdir()) == ["a1.json", "a1.safetensors", "a2.json", "a2.safetensors", "capture.json"]
    record, blocks = read(out, "a2")
    assert torch.equal(blocks, capture(model, lines[1])[1]) and record["tensor"]["bytes"] > 0
    assert [r["transcript"] for r in items(out)] == ["a1", "a2"]
    assert verify(out) == {"captures": 2, "passed": True, "problems": []}
    # running again carries on: what is there is left alone
    again = run(model, lines[:2] + [line(model, tid="a5", turn=5)], out)
    assert (again["captured"], again["already_there"]) == (["a5"], ["a1", "a2"])
    assert json.loads((out / "capture.json").read_text())["captured"] == ["a1", "a2", "a5"]
    # other settings do not mix into the same folder
    with pytest.raises(CaptureError, match="other settings"):
        run(model, lines[:1], out, mean=True)


def test_a_capture_that_was_changed_is_caught(model, tmp_path):
    out = tmp_path / "captures"
    run(model, [line(model, tid="a1")], out)
    (out / "a1.safetensors").write_bytes((out / "a1.safetensors").read_bytes()[:-4] + b"\x00\x00\x80\x7f")
    with pytest.raises(CaptureError, match="hash differs"):
        read(out, "a1")
    report = verify(out)
    assert not report["passed"] and report["problems"][0]["transcript"] == "a1"


def test_transcripts_are_read_from_files_and_folders_in_turn_order(model, tmp_path):
    folder = tmp_path / "transcripts"
    folder.mkdir()
    (folder / "c1.jsonl").write_text("\n".join(json.dumps(line(model, tid=f"t{n}", turn=n)) for n in (2, 1)) + "\n")
    (folder / "c2.jsonl").write_text(json.dumps({**line(model, tid="u1"), "conversation": "c2"}) + "\n\n")
    assert [t["id"] for t in transcripts([folder])] == ["t1", "t2", "u1"]
    assert [t["id"] for t in transcripts([folder / "c2.jsonl"])] == ["u1"]
    assert [t["id"] for t in transcripts([folder], conversation="c2")] == ["u1"]
    (folder / "bad.jsonl").write_text('{"schema": "something.else"}\n')
    with pytest.raises(CaptureError, match="is not a transcript"):
        list(transcripts([folder]))
    with pytest.raises(CaptureError, match="No transcripts at"):
        list(transcripts([tmp_path / "nowhere"]))


def test_the_command_line_shows_and_verifies_a_folder(model, tmp_path, capsys):
    out = tmp_path / "captures"
    run(model, [line(model, tid="a1")], out)
    main(["show", str(out)])
    assert capsys.readouterr().out.startswith("a1  turn 1  blocks [4, 6, 16] float32  6 tokens  ids_rebuilt")
    main(["verify", str(out)])
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert _blocks("8-10,31") == [8, 9, 10, 31] and _blocks(None) is None
