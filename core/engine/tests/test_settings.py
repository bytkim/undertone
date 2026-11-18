"""The engine's endpoint as the chat page's settings see it: asked about, changed, kept, and the key never moved."""
import json
from threading import Thread
from types import SimpleNamespace

import httpx
import pytest

from undertone_engine import Endpoint, EndpointConfig, EndpointSettings, Engine, ModelContract, SettingsError, TranscriptStore, cli
from undertone_engine.service import make_server

from conftest import KEY, FakeServer

QWEN = ModelContract(name="qwen3.5-4b", repo="Qwen/Qwen3.5-4B", revision="abc", thinking=False, served_as=("Qwen/Qwen3.5-4B", "qwen3.5-4b"))


def serve(tmp_path, upstream, *, started=True, key=KEY, contract=None):
    """The service over an engine whose endpoint was given at start (with a key), or over one with no endpoint."""
    transport = httpx.MockTransport(upstream)
    config = EndpointConfig(base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-4B", api_key=key) if started else None
    settings = EndpointSettings(tmp_path / "transcripts", started=config, key_given=bool(key))
    first = settings.initial()
    engine = Engine(Endpoint(first, transport=transport) if first else None, TranscriptStore(tmp_path / "transcripts"))
    server = make_server(engine, port=0, settings=settings, contract=contract, transport=transport)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, engine, httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}")


@pytest.fixture
def running(tmp_path):
    made = []

    def start(upstream, **more):
        made.append(serve(tmp_path, upstream, **more))
        return made[-1][1], made[-1][2]

    yield start
    for server, _, client in made:
        client.close()
        server.shutdown()
        server.server_close()


def test_the_settings_say_which_endpoint_and_what_is_known_of_the_model(running, tmp_path):
    engine, client = running(FakeServer(), contract=QWEN)
    assert client.get("/settings").json() == {
        "endpoint": {"base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3.5-4B", "host": "localhost:8000", "token_ids": False},
        "source": "start", "api_key": "sent",
        "contract": {"name": "qwen3.5-4b", "served_as": ["Qwen/Qwen3.5-4B", "qwen3.5-4b"], "knows_model": True},
        "rebuilds_ids": False, "transcripts": str(tmp_path / "transcripts"), "kept_in": str(tmp_path / "transcripts" / "engine.json"),
    }
    assert KEY not in client.get("/settings").text


def test_the_endpoint_can_be_changed_and_is_kept_for_the_next_start(running, tmp_path):
    upstream = FakeServer(returns_ids=True)
    engine, client = running(upstream, contract=QWEN)
    changed = client.put("/settings", json={"base_url": " http://localhost:8000/v1 ", "model": "another-model", "token_ids": True}).json()
    assert changed["endpoint"] == {"base_url": "http://localhost:8000/v1", "model": "another-model", "host": "localhost:8000", "token_ids": True}
    assert changed["source"] == "page" and changed["contract"]["knows_model"] is False
    assert client.get("/health").json() == {"ok": True, "endpoint": "localhost:8000", "model": "another-model"}
    # the next call goes to it, asking for token ids as told
    t = client.post("/chat", json={"message": "Hello"}).json()
    assert upstream.bodies[-1]["model"] == "another-model" and upstream.bodies[-1]["return_token_ids"] is True
    assert t["endpoint"] == {"host": "localhost:8000", "served_model": "another-model"} and t["tokens"]["source"] == "server"
    # kept as a file, without the key, and used when the engine starts with no endpoint given
    kept = tmp_path / "transcripts" / "engine.json"
    assert json.loads(kept.read_text()) == {"base_url": "http://localhost:8000/v1", "model": "another-model", "token_ids": True}
    again = EndpointSettings(tmp_path / "transcripts")
    config = again.initial()
    assert (config.base_url, config.model, config.ask_token_ids, config.api_key, again.source) == ("http://localhost:8000/v1", "another-model", True, None, "kept")
    # an endpoint given at start takes the kept one's place
    given = EndpointSettings(tmp_path / "transcripts", started=EndpointConfig(base_url="http://localhost:9/v1", model="m"))
    assert given.initial().base_url == "http://localhost:9/v1" and given.source == "start"


def test_the_key_goes_only_to_the_endpoint_the_engine_was_started_with(running):
    upstream = FakeServer()
    engine, client = running(upstream)
    # the same host under another model's name: still the endpoint the key was given for
    assert client.put("/settings", json={"base_url": "http://localhost:8000/v1", "model": "another-model"}).json()["api_key"] == "sent"
    client.post("/chat", json={"message": "Hello"})
    assert upstream.requests[-1].headers["authorization"] == f"Bearer {KEY}"
    # another host: the key is held back, whatever the page asks
    moved = client.put("/settings", json={"base_url": "https://elsewhere.example/v1", "model": "m", "api_key": "from-the-page"}).json()
    assert moved["api_key"] == "held back" and moved["endpoint"]["host"] == "elsewhere.example"
    client.post("/chat", json={"message": "Hello"})
    assert upstream.requests[-1].url.host == "elsewhere.example" and "authorization" not in upstream.requests[-1].headers
    client.post("/endpoint/models", json={"base_url": "https://third.example/v1"})
    assert upstream.requests[-1].url.host == "third.example" and "authorization" not in upstream.requests[-1].headers
    assert KEY not in client.get("/settings").text
    # with no key anywhere there is none to speak of
    assert EndpointSettings("x").key(None) == "none"


def test_settings_that_cannot_be_used_are_refused_and_nothing_changes(running, tmp_path):
    engine, client = running(FakeServer())
    for bad in ({"base_url": "localhost:8000", "model": "m"}, {"base_url": "ftp://localhost/v1", "model": "m"}, {"base_url": "http://user:secret@localhost:8000/v1", "model": "m"},
                {"base_url": "http://localhost:8000/v1", "model": "  "}, {"base_url": "http://localhost:8000/v1"}, {"model": "m"}, {"base_url": 7, "model": "m"}):
        assert client.put("/settings", json=bad).status_code == 400, bad
    assert client.put("/settings", content=b"[]").status_code == 400 and client.put("/elsewhere", json={}).status_code == 404
    assert client.get("/settings").json()["endpoint"]["model"] == "Qwen/Qwen3.5-4B" and not (tmp_path / "transcripts" / "engine.json").exists()
    with pytest.raises(SettingsError):
        EndpointSettings(tmp_path).config("http://localhost:8000/v1", "m" * 201)


def test_an_endpoint_says_which_models_it_has(running):
    upstream = FakeServer()
    engine, client = running(upstream)
    assert client.post("/endpoint/models").json() == {"host": "localhost:8000", "models": ["Qwen/Qwen3.5-4B", "another-model"]}
    assert upstream.requests[-1].url.path == "/v1/models" and upstream.requests[-1].headers["authorization"] == f"Bearer {KEY}"
    tried = client.post("/endpoint/models", json={"base_url": "http://localhost:9000/v1/"})
    assert tried.json()["host"] == "localhost:9000" and upstream.requests[-1].url.path == "/v1/models"
    assert client.post("/endpoint/models", json={"base_url": "not an address"}).status_code == 400
    upstream.status = 500
    unwell = client.post("/endpoint/models")
    assert unwell.status_code == 502 and "answered 500" in unwell.json()["error"]


def test_the_engine_runs_without_an_endpoint_until_one_is_set(running):
    upstream = FakeServer()
    engine, client = running(upstream, started=False, key=None)
    assert client.get("/health").json() == {"ok": True, "endpoint": None, "model": None}
    said = client.get("/settings").json()
    assert said["endpoint"] is None and said["source"] is None and said["api_key"] == "none" and said["contract"] is None
    refused = client.post("/chat", json={"message": "Hello"})
    assert refused.status_code == 502 and "No model endpoint is set" in refused.json()["error"] and upstream.requests == []
    assert client.post("/endpoint/models").status_code == 400 and client.get("/conversations").json() == []
    assert client.put("/settings", json={"base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3.5-4B"}).json()["source"] == "page"
    assert client.post("/chat", json={"message": "Hello"}).json()["response"]["content"] == "reply to 1 messages"


def test_a_key_in_the_environment_with_no_endpoint_at_start_is_never_sent(running):
    upstream = FakeServer()
    engine, client = running(upstream, started=False, key=KEY)
    assert client.put("/settings", json={"base_url": "http://localhost:8000/v1", "model": "m"}).json()["api_key"] == "held back"
    client.post("/chat", json={"message": "Hello"})
    assert "authorization" not in upstream.requests[-1].headers


def test_only_this_machine_may_ask_and_only_its_pages_may_change_anything(running):
    upstream = FakeServer()
    engine, client = running(upstream)
    # a request addressed to another name is not this machine's (a page elsewhere can point its name at 127.0.0.1)
    for path in ("/health", "/settings", "/conversations"):
        assert client.get(path, headers={"Host": "elsewhere.example:8787"}).status_code == 403
    assert client.get("/health", headers={"Host": "localhost:8787"}).status_code == 200
    # a page elsewhere may not change anything, even with a request a browser sends without asking first
    foreign = {"Origin": "https://elsewhere.example", "Content-Type": "text/plain"}
    assert client.post("/chat", content=json.dumps({"message": "Hello"}), headers=foreign).status_code == 403
    assert client.put("/settings", content=json.dumps({"base_url": "https://elsewhere.example/v1", "model": "m"}), headers=foreign).status_code == 403
    assert client.post("/endpoint/models", content=json.dumps({"base_url": "https://elsewhere.example/v1"}), headers=foreign).status_code == 403
    assert upstream.requests == [] and client.get("/settings").json()["endpoint"]["host"] == "localhost:8000"
    # this machine's own page may, and a tool with no page at all may
    local = client.put("/settings", json={"base_url": "http://localhost:8000/v1", "model": "m"}, headers={"Origin": "http://127.0.0.1:5173"})
    assert local.status_code == 200 and "PUT" in local.headers["access-control-allow-methods"]


def test_serve_starts_without_an_endpoint_and_uses_a_kept_one(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("UNDERTONE_API_KEY", raising=False)
    args = lambda **more: SimpleNamespace(**{**dict(base_url=None, served_model=None, token_ids=False, tokenizer=None, tokenizer_revision=None, contract=None, transcripts=str(tmp_path)), **more})
    engine, settings, contract = cli._engine(args(), needs_endpoint=False)
    assert engine.endpoint is None and settings.source is None and contract is None
    with pytest.raises(SystemExit, match="Say which endpoint"):
        cli._engine(args())   # a single chat from the terminal does need one
    with pytest.raises(SystemExit, match="both an address and a model"):
        cli._engine(args(base_url="http://localhost:8000/v1"), needs_endpoint=False)
    settings.keep(settings.config("http://localhost:8000/v1", "kept-model", True))
    engine, settings, _ = cli._engine(args(), needs_endpoint=False)
    assert (engine.endpoint.config.model, engine.endpoint.config.ask_token_ids, settings.source) == ("kept-model", True, "kept")
    # an endpoint given at start wins, and carries the key from the environment
    monkeypatch.setenv("UNDERTONE_API_KEY", "sk-test-do-not-store")
    engine, settings, _ = cli._engine(args(base_url="http://localhost:9000/v1", served_model="given"))
    assert (engine.endpoint.config.model, engine.endpoint.config.api_key, settings.source) == ("given", "sk-test-do-not-store", "start")
