"""Corpus to transcripts to captures to vectors to readings, on the stand-in model: the plumbing, not the science."""
import json
import sys

import pytest

from undertone_capture import run as capture_run, transcripts
from undertone_emotion import BuildError, ReadingError, build, families, load_set, read, readings, run, shipped_set, write_transcripts
from undertone_emotion.build import smooth
from undertone_emotion.cli import main
from undertone_models import adapter, contract, read as read_contract
from undertone_models.standin import FakeTokenizer, standin_model, transcript

torch = pytest.importorskip("torch")

WORDS = {"calm": "The water lay still and she breathed slowly beside the open window all evening long",
         "afraid": "He froze at the door while his hands shook and the dark hallway creaked behind him again"}


@pytest.fixture
def world(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "stories").mkdir(parents=True)
    (corpus / "neutral_dialogues").mkdir()
    for author in ("one", "two"):
        rows = [{"story_id": f"{e}.{author}.t{k}.1", "author": author, "emotion": e, "topic_id": f"t{k}", "text": f"{text} in place {k} by {author}."} for e, text in WORDS.items() for k in range(3)]
        (corpus / "stories" / f"{author}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        talk = [{"story_id": f"neutral_dialogue.{author}.t{k}.1", "author": author, "topic_id": f"t{k}",
                 "turns": [{"role": "user", "text": f"List the steps to fix item {k}."}, {"role": "assistant", "text": f"Step one is to check item {k} and step two is to replace the part by {author}."}]} for k in range(4)]
        (corpus / "neutral_dialogues" / f"{author}.jsonl").write_text("\n".join(map(json.dumps, talk)) + "\n")
    (corpus / "manifest.json").write_text("{}")
    qwen = contract("qwen3.5-4b")
    data = json.loads(json.dumps(qwen.data))
    data["shape"].update(blocks=4, width=16)
    data["depths"] = {"early": [0], "middle": [1], "late": [2], "output": [3]}
    # (kept under the model's own name, so a test can have it found by name: see UNDERTONE_MODELS below)
    (tmp_path / "models" / "qwen3.5-4b").mkdir(parents=True)
    (tmp_path / "models" / "qwen3.5-4b" / "contract.json").write_text(json.dumps(data))
    tok = FakeTokenizer()
    model = adapter(qwen).Model(read_contract(tmp_path / "models" / "qwen3.5-4b" / "contract.json"), standin_model(), tok, "cpu")
    counts = write_transcripts(corpus, tmp_path / "ct")
    capture_run(model, transcripts([tmp_path / "ct" / "stories"]), tmp_path / "cap-stories", skip=2, mean=True)
    capture_run(model, transcripts([tmp_path / "ct" / "neutral"]), tmp_path / "cap-neutral", select="messages")
    line = lambda tid, said, reply: {"schema": "undertone.transcript.v1", **transcript(tok, said, reply), "id": tid, "turn": 1}
    capture_run(model, [line("r1", "What do you see?", "He froze and his hands shook in the dark hallway."), line("r2", "And now?", "Still water.")], tmp_path / "cap-replies")
    return {"tmp": tmp_path, "corpus": corpus, "model": model, "counts": counts, "line": line}


def test_the_corpus_becomes_transcripts_with_the_story_as_the_reply(world):
    assert world["counts"] == {"stories": 12, "neutral": 8}
    stories = list(transcripts([world["tmp"] / "ct" / "stories"]))
    assert len(stories) == 12 and {t["request"]["messages"][0]["content"] for t in stories} == {"Write a one-paragraph story."}
    first = next(t for t in stories if t["id"] == "calm.one.t0.1")
    assert first["response"]["content"].startswith("The water lay still") and first["conversation"] == "one.calm"
    talk = next(t for t in transcripts([world["tmp"] / "ct" / "neutral"]) if t["id"] == "neutral_dialogue.two.t1.1")
    assert [m["role"] for m in talk["request"]["messages"]] == ["user"] and talk["response"]["content"].startswith("Step one")
    # every emotion word of the real corpus has a family
    fam = families()
    assert len(fam["emotions"]) == 171 and set(fam["emotions"].values()) == set(fam["families"]) and len(fam["families"]) == 10


def test_a_vector_set_is_built_kept_and_read_back(world, monkeypatch):
    tmp = world["tmp"]
    seen = []
    record = build(tmp / "cap-stories", tmp / "cap-neutral", world["corpus"], tmp / "vectors", blocks=[1, 2], log=seen.append)
    assert record["schema"] == "undertone.emotion.vectors.v1" and record["emotions"] == ["calm", "afraid"] and record["blocks"] == [1, 2]
    assert record["families"] == {"calm": "Peaceful Contentment", "afraid": "Fear and Overwhelm"}
    made = record["made_from"]
    assert (made["stories"], made["authors"], made["situations"], made["neutral_dialogues"], made["neutral_assistant_messages"]) == (12, ["one", "two"], 3, 8, 8)
    assert made["story_prompt"] == "Write a one-paragraph story." and made["story_tokens_skipped"] == 2 and made["corpus_manifest_sha256"]
    assert len(record["neutral_directions_by_block"]) == 2 and all(0.5 <= r <= 1 for r in record["neutral_variance_reached_by_block"])
    assert record["checks"]["held_out"]["stories_tested"] == 4 and set(record["checks"]["authors"]) == {"one", "two"}
    assert record["marks"]["step"] > 0 and seen
    vs = load_set(tmp / "vectors")
    assert list(vs.vectors.shape) == [2, 2, 16] and list(vs.grand_mean.shape) == [2, 16] and len(vs.basis) == 2
    assert set(vs.usual) == {"token", "message", "own"} and list(vs.usual["own"].shape) == [2]
    assert set(record["checks"]["scale"]) >= {"own_emotion_share_quantiles", "any_emotion_share_quantiles", "top1", "top5"}
    assert record["marks"]["step"] == pytest.approx((1 - record["marks"]["floor"]) / 8)
    # the scale's two ends: neutral text reads as 0 on average, and an emotion's own stories as 1
    assert torch.allclose(vs.shares(vs.usual["message"][None], "message"), torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(vs.shares(vs.usual["own"][None], "token"), torch.ones(1, 2), atol=1e-6)
    # by default the band is the model's middle depth
    monkeypatch.setenv("UNDERTONE_MODELS", str(tmp / "models"))
    assert build(tmp / "cap-stories", tmp / "cap-neutral", world["corpus"], tmp / "vectors-middle")["blocks"] == [1]
    # a set is not overwritten, and one whose file was changed is not loaded
    with pytest.raises(BuildError, match="already holds"):
        build(tmp / "cap-stories", tmp / "cap-neutral", world["corpus"], tmp / "vectors")
    (tmp / "vectors-middle" / "vectors.safetensors").write_bytes(b"changed")
    with pytest.raises(BuildError, match="does not hold"):
        load_set(tmp / "vectors-middle")
    # stories captured token by token, or neutral text captured as averages, are the wrong way round
    with pytest.raises(BuildError, match="one average each"):
        build(tmp / "cap-neutral", tmp / "cap-neutral", _relabelled(world, tmp), tmp / "bad1")
    with pytest.raises(BuildError, match="token by token"):
        build(tmp / "cap-stories", tmp / "cap-stories", world["corpus"], tmp / "bad2")


def _relabelled(world, tmp):
    """A corpus whose "stories" are the neutral dialogues' ids, to reach the check on how stories were captured."""
    other = tmp / "corpus-odd"
    (other / "stories").mkdir(parents=True)
    rows = [{"story_id": f"neutral_dialogue.{a}.t{k}.1", "author": a, "emotion": "calm", "topic_id": f"t{k}", "text": "x"} for a in ("one", "two") for k in range(4)]
    (other / "stories" / "one.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    return other


def test_a_reading_says_which_emotions_stand_out_over_the_reply_and_token_by_token(world):
    tmp = world["tmp"]
    build(tmp / "cap-stories", tmp / "cap-neutral", world["corpus"], tmp / "vectors", blocks=[1, 2])
    vs = load_set(tmp / "vectors")
    from undertone_capture import read as read_capture

    record, blocks = read_capture(tmp / "cap-replies", "r1")
    reading = read(vs, record, blocks, top=2)
    assert reading["schema"] == "undertone.emotion.v1" and reading["vectors"]["sha256"] == vs.record["sha256"]
    assert [t["text"] for t in reading["tokens"]] == [t["text"] for t in record["tokens"]]
    # the whole reply: worked by hand
    x = blocks[[1, 2]].float().transpose(0, 1)
    whole = vs.cosines(x.mean(0, keepdim=True))[0]
    share = (whole - vs.usual["message"]) / (vs.usual["own"] - vs.usual["message"])
    order = share.argsort(descending=True).tolist()
    assert [e["name"] for e in reading["overall"]] == [vs.emotions[j] for j in order]
    top = reading["overall"][0]
    assert top["share"] == pytest.approx(float(share[order[0]]), rel=1e-3) and top["level"] == round(max(0, min(1, float(share[order[0]]))) * 100)
    assert top["family"] in ("Peaceful Contentment", "Fear and Overwhelm") and 0 <= top["level"] <= 100
    # token by token: smoothed with the neighbours, marked only from the floor up, strongest first
    st = smooth((vs.cosines(x) - vs.usual["token"]) / (vs.usual["own"] - vs.usual["token"]))
    floor = vs.record["marks"]["floor"]
    for t, row in zip(reading["tokens"], st):
        assert [e["name"] for e in t["emotions"]] == [vs.emotions[j] for j in row.argsort(descending=True).tolist() if float(row[j]) >= floor]
        assert all(1 <= e["strength"] <= 9 and e["share"] >= floor - 1e-3 for e in t["emotions"])
    assert torch.allclose(smooth(torch.tensor([[0.0], [4.0], [0.0], [0.0]])), torch.tensor([[1.0], [2.0], [1.0], [0.0]]))
    assert torch.equal(smooth(torch.ones(2, 3)), torch.ones(2, 3))


def test_a_run_reads_a_folder_and_names_what_it_could_not_read(world, capsys):
    tmp, model = world["tmp"], world["model"]
    build(tmp / "cap-stories", tmp / "cap-neutral", world["corpus"], tmp / "vectors", blocks=[1, 2])
    vs = load_set(tmp / "vectors")
    seen = []
    result = run(vs, tmp / "cap-replies", tmp / "emotion", top=2, log=seen.append)
    assert result["read"] == ["r1", "r2"] and result["left_out"] == [] and len(seen) == 2
    assert sorted(p.name for p in (tmp / "emotion").iterdir()) == ["emotion.json", "r1.json", "r2.json"]
    assert [r["transcript"] for r in readings(tmp / "emotion")] == ["r1", "r2"]
    assert run(vs, tmp / "cap-replies", tmp / "emotion", top=2)["already_there"] == ["r1", "r2"]
    with pytest.raises(ReadingError, match="other vectors or settings"):
        run(vs, tmp / "cap-replies", tmp / "emotion", top=5)
    # averages, and captures without the band's blocks, cannot be read token by token
    assert run(vs, tmp / "cap-stories", tmp / "emotion-stories")["left_out"][0]["reason"].startswith("this capture holds an average")
    capture_run(model, [world["line"]("r3", "Hello", "Still water again.")], tmp / "cap-few", blocks=[0, 3])
    assert "leaves out blocks that are needed" in run(vs, tmp / "cap-few", tmp / "emotion-few")["left_out"][0]["reason"]
    # the command line
    main(["info", str(tmp / "vectors")])
    out = capsys.readouterr().out
    assert out.startswith("qwen3.5-4b  2 emotions  blocks 1-2  width 16") and "held out" in out and "author      one: 6 stories" in out
    main(["show", str(tmp / "emotion")])
    assert capsys.readouterr().out.startswith("r1  turn 1  ")
    main(["show", str(tmp / "emotion"), "--transcript", "r2"])
    assert "' water.'" in capsys.readouterr().out


def test_the_vector_set_that_comes_with_the_tool_is_what_its_record_says():
    qwen = contract("qwen3.5-4b")
    vs = shipped_set({"name": qwen.name, "revision": qwen.revision})
    assert vs is not None, "a vector set for the model's pinned revision comes with the tool"
    record = vs.record
    assert record["schema"] == "undertone.emotion.vectors.v1" and record["model"]["repo"] == qwen.repo
    assert vs.blocks == list(qwen.depths["middle"]) and record["width"] == qwen.data["shape"]["width"]
    fam = families()["emotions"]
    assert len(vs.emotions) == 171 and set(vs.emotions) == set(fam) and record["families"] == {e: fam[e] for e in vs.emotions}
    assert list(vs.vectors.shape) == [171, len(vs.blocks), record["width"]] and list(vs.grand_mean.shape) == [len(vs.blocks), record["width"]]
    assert [int(b.shape[0]) for b in vs.basis] == record["neutral_directions_by_block"]
    for b in vs.basis:   # the neutral directions are orthonormal, and the vectors have none of them left
        assert torch.allclose(b.double() @ b.double().T, torch.eye(b.shape[0], dtype=torch.float64), atol=1e-4)
    assert all(float((vs.vectors[:, k].double() @ b.double().T).abs().max()) < 1e-3 for k, b in enumerate(vs.basis))
    assert set(vs.usual) == {"token", "message", "own"} and all(bool(torch.isfinite(v).all()) and list(v.shape) == [171] for v in vs.usual.values())
    # every emotion's own stories read further along its vector than neutral text does, so the scale has two ends
    assert bool((vs.usual["own"] > torch.maximum(vs.usual["token"], vs.usual["message"])).all())
    assert record["made_from"]["stories"] == 15390 and record["made_from"]["neutral_dialogues"] == 360
    assert 0 < record["marks"]["floor"] < 1 and record["marks"]["step"] == pytest.approx((1 - record["marks"]["floor"]) / 8)
    # another revision of the model is another model
    assert shipped_set({"name": qwen.name, "revision": "0" * 40}) is None and shipped_set({"name": "some-other-model", "revision": qwen.revision}) is None


def test_reading_uses_the_set_that_comes_with_the_tool_unless_one_is_named(world, monkeypatch, capsys):
    tmp = world["tmp"]
    build(tmp / "cap-stories", tmp / "cap-neutral", world["corpus"], tmp / "sets" / "qwen3.5-4b", blocks=[1, 2])
    model = load_set(tmp / "sets" / "qwen3.5-4b").record["model"]
    assert shipped_set(model, roots=[tmp / "nowhere", tmp / "sets"]).record["sha256"] == load_set(tmp / "sets" / "qwen3.5-4b").record["sha256"]
    assert shipped_set({**model, "revision": "another"}, roots=[tmp / "sets"]) is None
    assert shipped_set(model, roots=[tmp / "nowhere"]) is None

    # (the package also offers a function named build, so the module is reached by its full name)
    monkeypatch.setattr(sys.modules["undertone_emotion.build"], "_SHIPPED", (tmp / "sets",))
    main(["read", str(tmp / "cap-replies"), "--out", str(tmp / "emotion-default")])
    said = capsys.readouterr()
    assert "comes with the tool for qwen3.5-4b" in said.err and json.loads(said.out)["read"] == 2
    main(["read", str(tmp / "cap-replies"), "--out", str(tmp / "emotion-named"), "--vectors", str(tmp / "sets" / "qwen3.5-4b")])
    capsys.readouterr()
    assert [r["overall"] for r in readings(tmp / "emotion-default")] == [r["overall"] for r in readings(tmp / "emotion-named")]
    main(["info"])
    assert capsys.readouterr().out.startswith("qwen3.5-4b  2 emotions  blocks 1-2  width 16")

    # no captures, and no set for the captures' model: both are said plainly
    (tmp / "empty").mkdir()
    with pytest.raises(SystemExit, match="No captures"):
        main(["read", str(tmp / "empty"), "--out", str(tmp / "x")])
    monkeypatch.setattr(sys.modules["undertone_emotion.build"], "_SHIPPED", (tmp / "nowhere",))
    with pytest.raises(SystemExit, match="No vector set comes with the tool for qwen3.5-4b"):
        main(["read", str(tmp / "cap-replies"), "--out", str(tmp / "x")])
    with pytest.raises(SystemExit, match="No vector set comes with this install"):
        main(["info"])
    # a set that is there but is not the file its record describes is refused, not passed over
    (tmp / "sets" / "qwen3.5-4b" / "vectors.safetensors").write_bytes(b"changed")
    with pytest.raises(BuildError, match="does not hold"):
        shipped_set(model, roots=[tmp / "sets"])
