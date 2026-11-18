"""The local service a chat page talks to: chat, history, and who may call it."""
from threading import Thread

import httpx
import pytest

from undertone_engine.service import make_server

from conftest import KEY, FakeServer


@pytest.fixture
def service(make_engine):
    upstream = FakeServer()
    server = make_server(make_engine(upstream), port=0)
    Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}") as client:
        yield client, upstream
    server.shutdown()
    server.server_close()


def test_chat_then_history(service):
    client, upstream = service
    assert client.get("/health").json() == {"ok": True, "endpoint": "localhost:8000", "model": "Qwen/Qwen3.5-4B"}
    assert client.get("/conversations").json() == []

    first = client.post("/chat", json={"message": "Hello", "system": "Be brief.", "params": {"max_tokens": 32}}).json()
    assert first["turn"] == 1 and first["response"]["content"] == "reply to 2 messages"
    assert upstream.bodies[0]["max_tokens"] == 32
    second = client.post("/chat", json={"message": "And then?", "conversation": first["conversation"]}).json()
    assert second["turn"] == 2 and len(upstream.bodies[1]["messages"]) == 4

    listed = client.get("/conversations").json()
    assert [(c["id"], c["turns"], c["title"]) for c in listed] == [(first["conversation"], 2, "Hello")]
    history = client.get(f"/conversations/{first['conversation']}").json()
    assert [t["turn"] for t in history] == [1, 2] and history[0] == first
    assert KEY not in client.get(f"/conversations/{first['conversation']}").text


def test_bad_requests_and_a_failing_endpoint_are_answered_plainly(make_engine):
    server = make_server(make_engine(FakeServer(status=500)), port=0)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}") as client:
            assert client.post("/chat", json={"message": "   "}).status_code == 400
            assert client.post("/chat", json={"message": "Hi", "conversation": 7}).status_code == 400
            assert client.post("/chat", json={"message": "Hi", "conversation": "../x"}).status_code == 400
            assert client.post("/chat", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400
            assert client.get("/conversations/nothing-here").status_code == 404
            assert client.get("/conversations/..%2Fx").status_code == 400
            assert client.get("/elsewhere").status_code == 404
            failed = client.post("/chat", json={"message": "Hi"})
            assert failed.status_code == 502 and "answered 500" in failed.json()["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_only_pages_on_this_machine_may_read_the_answers(service):
    client, _ = service
    local = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert local.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert client.options("/chat", headers={"Origin": "http://127.0.0.1:5173"}).status_code == 204
    for elsewhere in ("https://example.com", "http://localhost.example.com", "http://localhost:5173.example.com"):
        assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": elsewhere}).headers
