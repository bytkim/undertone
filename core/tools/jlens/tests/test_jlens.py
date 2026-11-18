"""The lens on the stand-in model and a made-up lens file: the logic, not any real model or lens."""
import hashlib
import json

import pytest

from undertone_capture import run as capture_run
from undertone_jlens import LensError, Reader, ReadingError, Words, build_baseline, is_word_like, load_baseline, load_lens, readings, run, shipped_baseline
from undertone_jlens.cli import main
from undertone_models import adapter, contract, read as read_contract
from undertone_models.standin import FakeTokenizer, standin_model, transcript

torch = pytest.importorskip("torch")


def lens_file(path, blocks=3, width=16, identity=False, **more):
    torch.manual_seed(1)
    maps = {k: (torch.eye(width) if identity else torch.eye(width) + 0.3 * torch.randn(width, width)).half() for k in range(blocks)}
    torch.save({"J": maps, "d_model": width, "provenance": {"made": "for a test"}, **more}, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path):
    """A 4-block stand-in model under a cut-down contract that pins a made-up lens, and two captured replies."""
    file = tmp_path / "test-lens.pt"
    sha = lens_file(file)
    qwen = contract("qwen3.5-4b")
    data = json.loads(json.dumps(qwen.data))
    data["shape"].update(blocks=4, width=16)
    data["depths"] = {"early": [0], "middle": [1], "late": [2], "output": [3]}
    data["artifacts"]["lenses"] = {"jpp": {"repo": "test/lens", "revision": "0" * 40, "path": "test-lens.pt", "sha256": sha, "bytes": file.stat().st_size}}
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "contract.json").write_text(json.dumps(data))
    small = read_contract(tmp_path / "small" / "contract.json")
    tok = FakeTokenizer()
    model = adapter(qwen).Model(small, standin_model(), tok, "cpu")
    line = lambda tid, turn, said, reply: {"schema": "undertone.transcript.v1", **transcript(tok, said, reply), "id": tid, "turn": turn}
    lines = [line("a1", 1, "What do you see?", "A quiet street under gray rain."), line("a2", 2, "And at night?", "Quiet Streetlights flicker, quiet.")]
    captures = tmp_path / "captures"
    capture_run(model, lines, captures)
    reader = Reader(model.as_head(), load_lens(small, "jpp", path=file, device="cpu"))
    return {"small": small, "model": model, "file": file, "captures": captures, "reader": reader, "lines": lines, "tmp": tmp_path}


def test_words_gather_the_pieces_that_spell_the_same_word():
    assert [is_word_like(p) for p in (" quiet", "Quiet", "don't", "well-known", "<|im_end|>", "<think>", ".", " ", "'s", "x-", "café", "42")] == [True, True, True, True, False, False, False, False, False, False, True, True]
    words = Words([" quiet", ".", "Quiet", " rain", "<|im_end|>", "quiet"])
    assert words.types == ["quiet", "rain"] and words.index.tolist() == [0, -1, 0, 1, -1, 0]
    mass = words.merge(torch.tensor([[0.1, 0.4, 0.2, 0.05, 0.2, 0.05]]))
    assert torch.allclose(mass, torch.tensor([[0.35, 0.05]]))   # what falls on "." and the marker is left out
    assert Words(["Quiet", " rain", "."]).sha256 == words.sha256 and Words([" quiet", " street"]).sha256 != words.sha256


def test_a_lens_must_be_the_pinned_file_and_fit_the_model(world):
    lens = world["reader"].lens
    assert sorted(lens.maps) == [0, 1, 2] and lens.target == 3 and lens.maps[0].dtype == torch.float32
    assert lens.source["sha256"] == world["small"].lenses["jpp"]["sha256"] and lens.provenance == {"made": "for a test"}
    x = torch.randn(5, 16)
    assert torch.equal(lens.transport(3, x), x)                       # the last block needs no map
    assert torch.allclose(lens.transport(1, x), x @ lens.maps[1].T)
    # a file that is not the pinned one, or that does not fit, is refused
    other = world["tmp"] / "other.pt"
    lens_file(other, identity=True)
    with pytest.raises(LensError, match="hash differs"):
        load_lens(world["small"], "jpp", path=other, device="cpu")
    assert load_lens(world["small"], "jpp", path=other, device="cpu", verify_file=False).maps[0].shape == (16, 16)
    for bad in (dict(blocks=2), dict(width=8), dict(d_model=99)):
        lens_file(other, **bad)
        with pytest.raises(LensError, match="does not fit"):
            load_lens(world["small"], "jpp", path=other, device="cpu", verify_file=False)
    with pytest.raises(LensError, match="pins no lens called 'nope'"):
        load_lens(world["small"], "nope", path=other, device="cpu")


def test_a_reading_is_the_lens_then_the_models_last_step_then_words(world):
    reader, model = world["reader"], world["model"]
    from undertone_capture import read as read_capture

    record, blocks = read_capture(world["captures"], "a1")
    reading = reader.read(record, blocks, top=5)
    assert reading["schema"] == "undertone.lens.v1" and reading["ranked_by"] == "probability" and reading["baseline"] is None
    assert [tok["text"] for tok in reading["tokens"]] == [tok["text"] for tok in record["tokens"]]
    assert list(reading["overall"]) == ["early", "middle", "late", "output"] == list(reading["tokens"][0]["lens"])
    # worked by hand for the third token at the middle depth (block 1) and at the output (block 3, no map)
    k = 2
    for depth, block in (("middle", 1), ("output", 3)):
        probs = model.head(reader.lens.transport(block, blocks[block, k : k + 1])).softmax(-1)
        mass = reader.words.merge(probs)[0]
        best = mass.topk(5)
        got = reading["tokens"][k]["lens"][depth]
        assert [w["word"] for w in got] == [reader.words.types[j] for j in best.indices.tolist()]
        assert got[0]["p"] == pytest.approx(float(best.values[0]), rel=1e-3)
        assert [w["p"] for w in got] == sorted((w["p"] for w in got), reverse=True) and sum(w["p"] for w in got) <= 1
    # the whole reply is the average over its tokens
    whole = torch.cat(list(reader.masses(record, blocks)), dim=1).mean(1)
    assert reading["overall"]["late"][0]["word"] == reader.words.types[int(whole[2].argmax())]
    assert all(w["word"] == w["word"].strip().lower() and is_word_like(w["word"]) for t in reading["tokens"] for d in t["lens"].values() for w in d)


def test_a_baseline_ranks_words_by_how_far_they_stand_above_usual(world):
    reader, tmp = world["reader"], world["tmp"]
    summary = build_baseline(reader, world["captures"], tmp / "baseline", log=lambda line: None)
    assert summary["captures"] == 2 and summary["tokens"] == 6 + 4 and summary["word_types"] == len(reader.words)
    baseline = load_baseline(tmp / "baseline")
    assert list(baseline.mass.shape) == [4, len(reader.words)] and float(baseline.mass.sum(1).max()) <= 1.0001
    from undertone_capture import read as read_capture

    record, blocks = read_capture(world["captures"], "a1")
    plain, lifted = reader.read(record, blocks, top=4), reader.read(record, blocks, top=4, baseline=baseline)
    assert lifted["ranked_by"] == "lift over a baseline" and lifted["baseline"]["sha256"] == summary["sha256"]
    words = lifted["tokens"][0]["lens"]["middle"]
    assert all(w["lift"] > 1 for w in words)   # only words that stand above their usual mass are listed
    mass = next(reader.masses(record, blocks))[1, 0]
    score = mass * ((mass + 1e-7) / (baseline.mass[1] + 1e-7)).log()
    assert words[0]["word"] == reader.words.types[int(score.argmax())]
    assert all("lift" not in w for w in plain["tokens"][0]["lens"]["middle"])
    # a baseline is tied to its model, lens and vocabulary, and is not overwritten
    with pytest.raises(ReadingError, match="already holds a baseline"):
        build_baseline(reader, world["captures"], tmp / "baseline")
    other = Reader(reader.head, reader.lens, Words([" other", " words"]))
    with pytest.raises(ReadingError, match="another model, lens or vocabulary"):
        baseline.fits(other)
    (tmp / "baseline" / "baseline.safetensors").write_bytes(b"changed")
    with pytest.raises(ReadingError, match="does not hold the baseline"):
        load_baseline(tmp / "baseline")


def test_a_run_reads_a_folder_and_says_what_it_could_not_read(world):
    reader, tmp, model = world["reader"], world["tmp"], world["model"]
    out = tmp / "lens"
    seen = []
    result = run(reader, world["captures"], out, top=3, log=seen.append)
    assert result["read"] == ["a1", "a2"] and result["left_out"] == [] and len(seen) == 2
    assert sorted(p.name for p in out.iterdir()) == ["a1.json", "a2.json", "lens.json"]
    assert [r["transcript"] for r in readings(out)] == ["a1", "a2"] and len(readings(out)[0]["overall"]["early"]) == 3
    assert run(reader, world["captures"], out, top=3)["already_there"] == ["a1", "a2"]
    with pytest.raises(ReadingError, match="other settings"):
        run(reader, world["captures"], out, top=5)
    # captures that hold averages, or leave out blocks a depth needs, cannot be read; they are named, not fatal
    capture_run(model, world["lines"], tmp / "means", mean=True)
    assert run(reader, tmp / "means", tmp / "lens-means")["left_out"][0]["reason"].startswith("this capture holds an average")
    capture_run(model, world["lines"][:1], tmp / "few", blocks=[2, 3])
    assert "leaves out blocks the depths need (0 to 1)" in run(reader, tmp / "few", tmp / "lens-few")["left_out"][0]["reason"]
    (world["captures"] / "a2.safetensors").write_bytes(b"changed")
    assert "hash differs" in run(reader, world["captures"], tmp / "lens-again")["left_out"][0]["reason"]


def test_the_command_line_shows_a_folder_of_readings(world, capsys):
    out = world["tmp"] / "lens"
    run(world["reader"], world["captures"], out, top=3)
    main(["show", str(out)])
    text = capsys.readouterr().out
    assert text.startswith("a1  turn 1  6 tokens  by probability  ids_rebuilt") and "  middle " in text and "a2  turn 2" in text
    main(["show", str(out), "--transcript", "a2"])
    text = capsys.readouterr().out
    assert "a1  turn 1" not in text and "' flicker,'" in text


def test_a_baseline_that_comes_with_the_tool_is_found_by_model_and_lens(world):
    reader, tmp = world["reader"], world["tmp"]
    assert shipped_baseline(reader, roots=[tmp / "baselines"]) is None   # none there
    summary = build_baseline(reader, world["captures"], tmp / "baselines" / "qwen3.5-4b-jpp", note="two replies, for a test")
    found = shipped_baseline(reader, roots=[tmp / "nowhere", tmp / "baselines"])
    assert found is not None and found.summary["sha256"] == summary["sha256"] and found.summary["note"] == "two replies, for a test"
    # one that does not fit this reader (another vocabulary, say) is passed over, not an error
    assert shipped_baseline(Reader(reader.head, reader.lens, Words([" other", " words"])), roots=[tmp / "baselines"]) is None


def test_the_baseline_in_the_repository_is_the_one_its_record_describes():
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "baselines" / "qwen3.5-4b-jpp"
    baseline = load_baseline(folder)   # the file is checked against the hash in its record
    qwen = contract("qwen3.5-4b")
    s = baseline.summary
    assert s["model"]["revision"] == qwen.revision and s["lens"]["sha256"] == qwen.lenses["jpp"]["sha256"]
    assert s["depths"] == qwen.depths and (s["captures"], s["tokens"]) == (360, 41957) and "neutral dialogues" in s["note"]
    assert list(baseline.mass.shape) == [4, s["word_types"]] and baseline.mass.dtype == torch.float32
    assert bool(torch.isfinite(baseline.mass).all()) and float(baseline.mass.min()) >= 0 and float(baseline.mass.sum(1).max()) <= 1.0001
