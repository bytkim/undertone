"""A stand-in endpoint and tokenizer, so the tests need no server, no network and no model."""
import json

import httpx
import pytest

from undertone_engine import Endpoint, EndpointConfig, Engine, TranscriptStore

KEY = "sk-test-do-not-store"


class FakeServer:
    """An OpenAI-compatible chat endpoint in memory. It echoes how many messages it was sent, and keeps every request."""

    def __init__(self, returns_ids: bool = False, knows_ids_field: bool = True, reasoning: str | None = None, status: int = 200):
        self.returns_ids, self.knows_ids_field, self.reasoning, self.status = returns_ids, knows_ids_field, reasoning, status
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":   # the one thing an endpoint is asked without a body: which models it has
            self.requests.append(request)
            if self.status != 200:
                return httpx.Response(self.status, text="the server is unwell")
            return httpx.Response(200, json={"object": "list", "data": [{"id": "Qwen/Qwen3.5-4B"}, {"id": "another-model"}]})
        body = json.loads(request.content)
        self.requests.append(request)
        self.bodies.append(body)
        if self.status != 200:
            return httpx.Response(self.status, text="the server is unwell")
        if "return_token_ids" in body and not self.knows_ids_field:
            return httpx.Response(400, json={"error": {"message": "Unrecognized request argument: return_token_ids"}})
        message = {"role": "assistant", "content": f"reply to {len(body['messages'])} messages"}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        data = {"id": "x", "choices": [choice], "usage": {"prompt_tokens": 7, "completion_tokens": 5}}
        if self.returns_ids and body.get("return_token_ids"):
            data["prompt_token_ids"] = [1, 2, 3]
            choice["token_ids"] = [4, 5]
        return httpx.Response(200, json=data)


class FakeTokenizer:
    """One id per word, so a test can see what was tokenized."""

    revision = "rev-1"
    template_sha256 = "abc123"

    def prompt_ids(self, messages, thinking):
        return [100 + int(thinking)] + [len(m["content"].split()) for m in messages]

    def completion_ids(self, content, reasoning):
        return None if reasoning is not None else [len(w) for w in content.split()]


@pytest.fixture
def make_engine(tmp_path):
    def make(server: FakeServer, tokenizer=None, ask_token_ids=False, model=None) -> Engine:
        config = EndpointConfig(base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-4B", api_key=KEY, ask_token_ids=ask_token_ids)
        return Engine(Endpoint(config, transport=httpx.MockTransport(server)), TranscriptStore(tmp_path / "transcripts"), tokenizer=tokenizer, model=model)

    return make
