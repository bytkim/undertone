"""The engine: what it sends, what it keeps, and how a transcript says where its token ids came from."""
import json

import pytest

from undertone_engine import SCHEMA, EndpointConfig, EndpointError, Transcript

from conftest import KEY, FakeServer, FakeTokenizer


def test_a_call_is_sent_as_openai_chat_and_kept_as_one_transcript_line(make_engine):
    server = FakeServer()
    engine = make_engine(server, model="qwen3.5-4b")
    t = engine.chat("What do you see?", system="Answer in one line.", temperature=0.2, max_tokens=64)

    # what went to the endpoint: the standard request, with the key as a bearer token
    assert server.requests[0].url.path == "/v1/chat/completions"
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert server.bodies[0] == {
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "system", "content": "Answer in one line."}, {"role": "user", "content": "What do you see?"}],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    # what was kept: the request and the reply in their OpenAI shapes, and where the call went
    assert (t.turn, t.model, t.response.content, t.response.finish_reason) == (1, "qwen3.5-4b", "reply to 2 messages", "stop")
    assert t.endpoint.model_dump() == {"host": "localhost:8000", "served_model": "Qwen/Qwen3.5-4B"}
    assert t.usage == {"prompt_tokens": 7, "completion_tokens": 5}
    line = (engine.store.root / f"{t.conversation}.jsonl").read_text()
    record = json.loads(line)
    assert record["schema"] == SCHEMA
    assert record["request"] == {"messages": server.bodies[0]["messages"], "temperature": 0.2, "max_tokens": 64}
    assert Transcript.parse(line) == t
    # the key is nowhere in what was written
    assert KEY not in line


def test_only_the_host_of_an_endpoint_is_recorded():
    assert EndpointConfig(base_url="https://user:pw@api.example.com/v1", model="m").host == "api.example.com"
    assert EndpointConfig(base_url="http://localhost:8000/v1/", model="m").host == "localhost:8000"


def test_a_conversation_carries_on_from_its_last_call_with_the_answer_only(make_engine):
    server = FakeServer(reasoning="Let me think about the window.")
    engine = make_engine(server)
    first = engine.chat("Hello")
    second = engine.chat("And then?", conversation=first.conversation, system="ignored once a conversation exists")
    assert (second.conversation, second.turn) == (first.conversation, 2)
    # the reasoning is kept on its own call, and is not sent back as history
    assert first.response.reasoning == "Let me think about the window."
    assert server.bodies[1]["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "reply to 1 messages"},
        {"role": "user", "content": "And then?"},
    ]
    assert [t.turn for t in engine.store.conversation(first.conversation)] == [1, 2]
    # another conversation is a file of its own
    other = engine.chat("Hello again")
    assert other.conversation != first.conversation and other.turn == 1


def test_token_ids_come_from_the_server_when_it_can_give_them(make_engine):
    server = FakeServer(returns_ids=True)
    t = make_engine(server, tokenizer=FakeTokenizer(), ask_token_ids=True).chat("Hello")
    assert server.bodies[0]["return_token_ids"] is True
    assert t.tokens.model_dump() == {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5], "source": "server"}
    # the field asked for is the engine's own business: it is not recorded as part of the request
    assert "return_token_ids" not in t.request.model_dump()


def test_token_ids_are_rebuilt_and_flagged_when_the_server_gives_none(make_engine):
    t = make_engine(FakeServer(), tokenizer=FakeTokenizer()).chat("one two three", thinking=True)
    assert t.tokens.source == "rebuilt"
    assert t.tokens.prompt_ids == [101, 3]                    # rebuilt from the messages, with thinking as declared
    assert t.tokens.completion_ids == [5, 2, 1, 8]            # "reply to 1 messages"
    assert t.template.model_dump() == {"tokenizer_revision": "rev-1", "chat_template_sha256": "abc123", "thinking": True}
    # a reply with reasoning: the prompt is rebuilt, the reply's ids are left out rather than guessed
    thought = make_engine(FakeServer(reasoning="hm"), tokenizer=FakeTokenizer()).chat("one two three")
    assert thought.tokens.source == "rebuilt" and thought.tokens.completion_ids is None


def test_a_transcript_without_any_ids_is_still_kept(make_engine):
    engine = make_engine(FakeServer())
    t = engine.chat("Hello")
    assert t.tokens is None and t.template is None
    assert "tokens" not in json.loads(t.line())
    assert engine.store.conversation(t.conversation) == [t]


def test_a_server_that_refuses_the_ids_field_is_asked_again_without_it_and_not_asked_twice(make_engine):
    server = FakeServer(knows_ids_field=False)
    engine = make_engine(server, ask_token_ids=True)
    t = engine.chat("Hello")
    assert t.response.content == "reply to 1 messages" and t.tokens is None
    assert ["return_token_ids" in b for b in server.bodies] == [True, False]
    engine.chat("Again", conversation=t.conversation)
    assert "return_token_ids" not in server.bodies[2] and len(server.bodies) == 3


def test_an_endpoint_error_is_reported_and_nothing_is_kept(make_engine):
    engine = make_engine(FakeServer(status=503))
    with pytest.raises(EndpointError, match="localhost:8000 answered 503"):
        engine.chat("Hello")
    assert engine.store.conversations() == []


def test_conversations_are_listed_newest_first_and_ids_cannot_leave_the_folder(make_engine):
    engine = make_engine(FakeServer())
    a = engine.chat("First   conversation\nwith a long opening " + "word " * 30)
    b = engine.chat("Second conversation")
    engine.chat("more", conversation=a.conversation)
    listed = engine.store.conversations()
    assert {s.id for s in listed} == {a.conversation, b.conversation}
    first = next(s for s in listed if s.id == a.conversation)
    assert first.turns == 2 and first.title.startswith("First conversation with a long opening") and first.title.endswith("…")
    for bad in ("../secrets", "a/b", "", "x" * 65, ".hidden"):
        with pytest.raises(ValueError):
            engine.store.conversation(bad)
    with pytest.raises(ValueError):
        engine.chat("Hello", conversation="../escape")


def test_thinking_that_is_declared_is_recorded_even_without_a_tokenizer(make_engine):
    engine = make_engine(FakeServer(returns_ids=True), ask_token_ids=True)
    assert engine.chat("Hello").template is None
    t = engine.chat("Is seven prime?", thinking=True)
    assert t.template.thinking is True and t.template.tokenizer_revision is None
