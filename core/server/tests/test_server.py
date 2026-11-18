"""The server on the stand-in model, a made-up lens and made-up emotion vectors: the plumbing, not any real model."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import httpx
import pytest

from undertone_capture import capture, run as capture_run
from undertone_emotion import VectorSet, read as read_emotions
from undertone_jlens import Reader, build_baseline, load_baseline, load_lens
from undertone_models import DEPTHS, adapter, contract, read as read_contract
from undertone_models.standin import FakeTokenizer, standin_model, transcript
from undertone_server import Desk, DeskError, NoTranscript, Tools, Transcripts, Unreadable, make_server, pieces, read
from undertone_server.cli import main

torch = pytest.importorskip("torch")

FAMILIES = {"calm": "Peaceful Contentment", "afraid": "Fear and Overwhelm", "amused": "Playful Amusement"}


def vector_set(model: dict, blocks: list[int], width: int = 16, floor: float = 0.3) -> VectorSet:
    """Made-up vectors with a made-up scale: enough for every part of a reading to be filled in."""
    torch.manual_seed(2)
    names = list(FAMILIES)
    record = {"model": model, "emotions": names, "blocks": blocks, "families": FAMILIES, "marks": {"floor": floor, "step": (1 - floor) / 8}, "sha256": "v" * 64}
    usual = {"token": torch.full((3,), -0.1), "message": torch.full((3,), -0.1), "own": torch.full((3,), 0.4)}
    return VectorSet(record, torch.randn(3, len(blocks), width), torch.zeros(len(blocks), width), [torch.zeros(0, width) for _ in blocks], usual)


@pytest.fixture
def world(tmp_path):
    """A 4-block stand-in model under a cut-down contract that pins a made-up lens; a baseline; made-up vectors; and
    a transcripts folder as the engine keeps one."""
    file = tmp_path / "test-lens.pt"
    torch.manual_seed(1)
    torch.save({"J": {k: (torch.eye(16) + 0.3 * torch.randn(16, 16)).half() for k in range(3)}, "d_model": 16}, str(file))
    qwen = contract("qwen3.5-4b")
    data = json.loads(json.dumps(qwen.data))
    data["shape"].update(blocks=4, width=16)
    data["depths"] = {"early": [0], "middle": [1], "late": [2], "output": [3]}
    data["artifacts"]["lenses"] = {"jpp": {"repo": "test/lens", "revision": "0" * 40, "path": "test-lens.pt", "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}}
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "contract.json").write_text(json.dumps(data))
    small = read_contract(tmp_path / "small" / "contract.json")
    tok = FakeTokenizer()
    model = adapter(qwen).Model(small, standin_model(), tok, "cpu")
    line = lambda tid, turn, said, reply, **more: {"schema": "undertone.transcript.v1", **transcript(tok, said, reply, **more), "id": tid, "conversation": "c1", "turn": turn}
    lines = [line("a1", 1, "What do you see?", "A quiet street under gray rain."), line("a2", 2, "And at night?", "Quiet streetlights flicker over the wet street.")]
    capture_run(model, [line("n1", 1, "List the steps.", "Step one is to check the part and step two is to replace it.")], tmp_path / "cap-neutral")
    for t in lines:
        model.render(t)   # (the stand-in tokenizer learns its vocabulary as it reads: the reader must come after)
    reader = Reader(model.as_head(), load_lens(small, "jpp", path=file, device="cpu"))
    build_baseline(reader, tmp_path / "cap-neutral", tmp_path / "baseline")
    vectors = vector_set({"name": small.name, "repo": small.repo, "revision": small.revision}, [1, 2])
    tools = Tools(model=model, reader=reader, baseline=load_baseline(tmp_path / "baseline"), vectors=vectors)
    folder = tmp_path / "transcripts"
    folder.mkdir()
    (folder / "c1.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines))
    return {"tmp": tmp_path, "tools": tools, "model": model, "lines": lines, "line": line, "transcripts": folder}


def is_page_reading(r: dict) -> bool:
    """The page's own test of a reading (frontend/src/readings.ts, isReading)."""
    words = lambda lens: isinstance(lens, dict) and all(isinstance(lens.get(d), list) for d in DEPTHS)
    return (r.get("schema") == "undertone.reading.v1" and isinstance(r["overall"]["emotions"], list) and words(r["overall"]["lens"]) and isinstance(r["tokens"], list)
            and all(isinstance(t["text"], str) and isinstance(t.get("emotions", []), list) and ("lens" not in t or words(t["lens"])) for t in r["tokens"]))


def test_a_reading_is_the_tools_own_numbers_in_the_pages_shape(world):
    tools, t = world["tools"], world["lines"][0]
    reading = read(tools, t)
    assert is_page_reading(reading) and (reading["transcript"], reading["conversation"], reading["turn"], reading["model"]) == ("a1", "c1", 1, "qwen3.5-4b")
    assert tools.blocks == [0, 1, 2, 3]   # the lens's depths and the vectors' band, and nothing else

    # the same capture read by each tool on its own gives the same emotions and the same words
    record, blocks = capture(tools.model, t, blocks=tools.blocks)
    emotions = read_emotions(tools.vectors, record, blocks, top=8)
    lens = tools.reader.read(record, blocks, top=8, baseline=tools.baseline)
    assert [(e["name"], e["family"], e["level"], e["share"]) for e in reading["overall"]["emotions"]] == [(e["name"], e["family"], e["level"], e["share"]) for e in emotions["overall"]]
    assert reading["overall"]["lens"] == {d: [w["word"] for w in lens["overall"][d]] for d in DEPTHS}
    assert any(reading["overall"]["lens"][d] for d in DEPTHS)

    # piece by piece: the pieces are the reply, a marked piece has its emotions and its words, any other only its text
    assert "".join(p["text"] for p in reading["tokens"]) == t["response"]["content"] and reading["fits"] is True
    marked = [k for k, p in enumerate(reading["tokens"]) if "emotions" in p]
    assert marked and len(marked) < len(reading["tokens"]), "the made-up floor should mark some pieces and not others"
    for k, p in enumerate(reading["tokens"]):
        theirs = emotions["tokens"][k]["emotions"]
        if theirs:
            assert [(e["name"], e["strength"]) for e in p["emotions"]] == [(e["name"], e["strength"]) for e in theirs]
            assert p["lens"] == {d: [w["word"] for w in lens["tokens"][k]["lens"][d]] for d in DEPTHS}
        else:
            assert set(p) == {"text"}

    # what it was made with, and what the capture flagged
    made = reading["made_with"]
    assert made["model"]["revision"] == tools.contract.revision and made["vectors"] == "v" * 64 and made["lens"]["name"] == "jpp"
    assert made["baseline"] == tools.baseline.summary["sha256"] and made["lens_ranked_by"] == "lift over a baseline" and made["ids"] == record["ids"]["source"]
    assert reading["flags"] == record["flags"] and reading["qualified"] == record["qualified"]


def test_a_transcript_outside_the_checked_setting_is_read_and_says_so(world):
    reading = read(world["tools"], world["line"]("b1", 1, "Hello", "A quiet street.", **{"endpoint.served_model": "someone-elses-model"}))
    assert "served_model_differs" in reading["flags"] and reading["qualified"] is False and reading["overall"]["emotions"]


def test_a_tool_that_is_not_there_leaves_its_part_empty(world):
    tools, t = world["tools"], world["lines"][0]
    lens_only = read(Tools(model=tools.model, reader=tools.reader), t)
    assert is_page_reading(lens_only) and lens_only["overall"]["emotions"] == [] and all(set(p) == {"text"} for p in lens_only["tokens"])
    assert lens_only["overall"]["lens"]["output"] and lens_only["made_with"]["lens_ranked_by"] == "probability" and lens_only["made_with"]["vectors"] is None
    emotions_only = Tools(model=tools.model, vectors=tools.vectors)
    assert emotions_only.blocks == [1, 2]
    reading = read(emotions_only, t)
    assert is_page_reading(reading) and reading["overall"]["emotions"] and reading["overall"]["lens"] == {d: [] for d in DEPTHS}
    assert all("lens" not in p for p in reading["tokens"]) and reading["made_with"]["lens"] is None
    with pytest.raises(Unreadable, match="no tool"):
        read(Tools(model=tools.model), t)


def test_what_cannot_be_read_says_why(world):
    tools = world["tools"]
    with pytest.raises(Unreadable, match="no tokens"):
        read(tools, world["line"]("e1", 1, "Hello", ""))
    with pytest.raises(Unreadable, match="another model's contract"):
        read(tools, {**world["lines"][0], "model": "some-other-model"})


def test_the_pieces_are_made_to_join_into_the_reply():
    class Bytes:
        """A tokenizer whose second and third tokens are the two halves of one character."""

        def decode(self, ids, skip_special_tokens=False):
            return {(1,): "a", (2,): "�", (3,): "�", (1, 2): "a�", (1, 2, 3): "aé", (1, 2, 3, 4): "aé!"}[tuple(ids)]

    record = {"tokens": [{"id": 1, "text": "a"}, {"id": 2, "text": "�"}, {"id": 3, "text": "�"}, {"id": 4, "text": "!"}]}
    assert pieces(Bytes(), record, "aé!") == ["a", "", "é", "!"]      # the character goes to the token that completes it
    assert pieces(Bytes(), record, "something else") == ["a", "�", "�", "!"]   # not the reply: left as captured
    whole = {"tokens": [{"id": 1, "text": "Still"}, {"id": 2, "text": " water."}]}
    assert pieces(None, whole, "Still water.") == ["Still", " water."]
    assert pieces(None, whole, "\n Still water.\n") == ["\n Still", " water.\n"]   # the edges a template trimmed are put back
    assert pieces(None, {"tokens": []}, "") == []


def test_a_reply_with_trimmed_edges_still_fits_the_page(world):
    t = world["line"]("w1", 1, "Hello", "  A quiet street.\n")
    reading = read(world["tools"], t)
    assert "".join(p["text"] for p in reading["tokens"]) == "  A quiet street.\n" and reading["fits"] is True


def test_the_desk_makes_a_reading_once_and_keeps_it(world, monkeypatch):
    tmp, seen = world["tmp"], []
    desk = Desk(world["tools"], world["transcripts"], tmp / "readings", log=seen.append)
    assert desk.kept("a1") is None and desk.count() == 0
    first = desk.reading("a1")
    assert first["transcript"] == "a1" and json.loads((tmp / "readings" / "a1.json").read_text()) == first and desk.count() == 1
    assert len(seen) == 1 and seen[0].startswith("a1  turn 1  ") and "marked" in seen[0]
    # asked again, the kept file is answered and the model is not run
    monkeypatch.setattr("undertone_server.desk.read", lambda *a: pytest.fail("made twice"))
    assert desk.reading("a1") == first
    monkeypatch.undo()
    # an id that is no transcript's, or is not an id at all
    for bad in ("nothing-here", "../a1", "a1/../a1", ""):
        with pytest.raises(NoTranscript):
            desk.reading(bad)
    assert sorted(p.name for p in (tmp / "readings").iterdir()) == ["a1.json", "readings.json"]
    # a turn the engine adds later is found, and a line still being written is passed over
    with (world["transcripts"] / "c1.jsonl").open("a") as f:
        f.write(json.dumps(world["line"]("a3", 3, "And then?", "Morning light on the wet street.")) + "\n" + '{"schema": "undertone.transcript.v1", "id": "a4", "resp')
    assert desk.reading("a3")["turn"] == 3
    with pytest.raises(NoTranscript):
        desk.reading("a4")
    # every transcript at once: what was made, what was there, what could not be read and why
    (world["transcripts"] / "c2.jsonl").write_text(json.dumps({**world["line"]("e1", 1, "Hello", ""), "conversation": "c2"}) + "\n")
    result = desk.read_all()
    assert result == {"made": ["a2"], "already_there": ["a1", "a3"], "left_out": [{"transcript": "e1", "reason": "the reply has no tokens"}]}
    # the folder holds readings made one way
    assert json.loads((tmp / "readings" / "readings.json").read_text())["settings"] == world["tools"].settings
    with pytest.raises(DeskError, match="another model, lens, baseline or vector set"):
        Desk(Tools(model=world["model"], vectors=world["tools"].vectors), world["transcripts"], tmp / "readings")
    assert Desk(world["tools"], world["transcripts"], tmp / "readings").kept("a1") == first


def test_transcripts_are_found_by_id_in_the_engines_folder(world):
    shelf = Transcripts(world["transcripts"])
    assert shelf.find("a2")["turn"] == 2 and shelf.find("zz") is None and [t["id"] for t in shelf.all()] == ["a1", "a2"]
    assert Transcripts(world["tmp"] / "no-such-folder").find("a1") is None


@pytest.fixture
def service(world):
    desk = Desk(world["tools"], world["transcripts"], world["tmp"] / "readings", log=lambda line: None)
    server = make_server(desk, port=0)
    Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}", timeout=30) as client:
        yield client, desk, world
    server.shutdown()
    server.server_close()


def test_the_service_answers_the_page(service):
    client, desk, world = service
    assert client.get("/health").json() == {"ok": True, "model": "qwen3.5-4b", "revision": desk.tools.contract.revision, "device": "cpu",
                                           "tools": {"emotions": 3, "lens": "jpp", "lens_ranked_by": "lift over a baseline"}, "readings": 0}
    answer = client.get("/readings/a1")
    assert answer.status_code == 200 and is_page_reading(answer.json()) and answer.json() == desk.kept("a1")
    assert client.get("/readings/a1").json() == answer.json() and client.get("/health").json()["readings"] == 1
    assert client.get("/readings/nothing-here").status_code == 404 and client.get("/readings/..%2Fa1").status_code == 404
    assert client.get("/elsewhere").status_code == 404
    (world["transcripts"] / "c2.jsonl").write_text(json.dumps(world["line"]("e1", 1, "Hello", "")) + "\n")
    empty = client.get("/readings/e1")
    assert empty.status_code == 422 and empty.json() == {"error": "the reply has no tokens"}
    # only a page served from this machine may read the answers
    local = client.get("/readings/a1", headers={"Origin": "http://127.0.0.1:5173"})
    assert local.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:5173"
    assert "Access-Control-Allow-Origin" not in client.get("/readings/a1", headers={"Origin": "https://example.com"}).headers
    assert client.options("/readings/a1", headers={"Origin": "http://localhost:5173"}).status_code == 204
    # and a request addressed to another name is not this machine's at all
    assert client.get("/readings/a1", headers={"Host": "elsewhere.example:8788"}).status_code == 403
    assert client.get("/health", headers={"Host": "localhost:8788"}).status_code == 200


def test_a_reading_asked_for_many_times_at_once_is_made_once(service, monkeypatch):
    client, desk, world = service
    made = []
    real = read
    monkeypatch.setattr("undertone_server.desk.read", lambda tools, t: (made.append(t["id"]), real(tools, t))[1])
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(lambda _: client.get("/readings/a2").json(), range(8)))
    assert made == ["a2"] and all(a == answers[0] for a in answers)


def test_a_reading_that_fails_is_answered_plainly_and_the_next_one_is_still_made(service, monkeypatch):
    client, desk, world = service
    real = read
    monkeypatch.setattr("undertone_server.desk.read", lambda tools, t: (_ for _ in ()).throw(RuntimeError("out of memory")) if t["id"] == "a1" else real(tools, t))
    failed = client.get("/readings/a1")
    assert failed.status_code == 500 and failed.json() == {"error": "RuntimeError: out of memory"} and desk.kept("a1") is None
    assert client.get("/readings/a2").status_code == 200


def test_show_prints_what_a_folder_of_readings_holds(world, capsys):
    desk = Desk(world["tools"], world["transcripts"], world["tmp"] / "readings", log=lambda line: None)
    desk.read_all()
    main(["show", str(world["tmp"] / "readings")])
    out = capsys.readouterr().out
    assert out.startswith("a1  turn 1  ") and "  emotions  " in out and "  output    " in out and "a2  turn 2  " in out
    main(["show", str(world["tmp"] / "readings"), "--transcript", "a2"])
    assert "a1  turn 1" not in (shown := capsys.readouterr().out) and "'Quiet'" in shown
