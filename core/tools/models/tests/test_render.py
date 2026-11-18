import pytest

from undertone_models.standin import FakeTokenizer, transcript
from undertone_models import FLAGS, ContractError

SAID, REPLY = "What do you see?", "A quiet street under gray rain."


def test_a_transcript_without_ids_is_rebuilt_with_each_messages_span(qwen, qwen_adapter, tok):
    r = qwen_adapter.render(qwen, tok, transcript(tok, SAID, REPLY))
    assert (r.source, r.flags) == ("rebuilt", ["ids_rebuilt"])
    assert r.qualified   # rebuilt ids alone do not put a transcript outside the qualified setting
    # the ids are the pinned template's, thinking off, and each message's own text is placed among them
    assert r.ids == tok.apply_chat_template([{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}])
    assert [(s.turn, s.role) for s in r.spans] == [(0, "user"), (1, "assistant")]
    for s, text in zip(r.spans, (SAID, REPLY)):
        assert r.text[s.char_start : s.char_end] == text
        assert "".join(r.text[a:b] for a, b in r.offsets[s.start : s.stop]) == text
    assert r.reply == r.spans[-1] and list(r.reply.positions) == list(range(r.reply.start, r.reply.stop))
    # template tokens are outside every span
    assert r.reply.stop < len(r.ids) and r.spans[0].start > 0


def test_the_endpoints_ids_are_used_when_they_are_what_the_template_gives(qwen, qwen_adapter, tok):
    t = transcript(tok, SAID, REPLY, ids="server")
    r = qwen_adapter.render(qwen, tok, t)
    assert (r.source, r.flags, r.qualified) == ("server", [], True)
    given = t["tokens"]["prompt_ids"] + t["tokens"]["completion_ids"]
    assert r.ids[: len(given)] == given   # the template only adds what closes the reply
    assert r.reply.start == len(t["tokens"]["prompt_ids"])
    # a server that stops before the closing token is the same reply
    t["tokens"]["completion_ids"] = t["tokens"]["completion_ids"][:-1]
    assert qwen_adapter.render(qwen, tok, t).source == "server"


def test_ids_that_disagree_with_the_template_are_kept_as_they_are_and_flagged(qwen, qwen_adapter, tok):
    t = transcript(tok, SAID, REPLY, ids="thinking", **{"response.reasoning": "Let me look.", "template.thinking": True})
    r = qwen_adapter.render(qwen, tok, t)
    given = t["tokens"]["prompt_ids"] + t["tokens"]["completion_ids"]
    assert r.source == "server" and r.ids == given and r.text is None and r.offsets is None
    assert r.flags == ["ids_differ_from_template", "thinking_on"] and not r.qualified
    # only the reply's span is known: everything the server generated
    assert r.spans == [r.reply] and (r.reply.start, r.reply.stop) == (len(t["tokens"]["prompt_ids"]), len(given))
    assert r.reply.char_start is None


def test_whatever_is_outside_the_qualified_setting_is_flagged_never_refused(qwen, qwen_adapter, tok, small):
    render = lambda t, c=qwen: qwen_adapter.render(c, tok, t).flags
    assert render(transcript(tok, SAID, REPLY, **{"response.reasoning": "Hm."})) == ["ids_rebuilt", "reasoning_dropped", "thinking_on"]
    assert render(transcript(tok, SAID, REPLY, ids="server", **{"endpoint.served_model": "qwen-q4"})) == ["served_model_differs"]
    assert render(transcript(tok, SAID, REPLY, ids="server", **{"endpoint.served_model": "qwen/qwen3.5-4b"})) == []
    assert render(transcript(tok, SAID, REPLY, ids="server", **{"template.tokenizer_revision": "abc"})) == ["tokenizer_revision_differs"]
    assert render(transcript(tok, SAID, REPLY, ids="server", **{"template.tokenizer_revision": qwen.revision, "template.chat_template_sha256": tok.template_sha256})) == []
    assert render(transcript(tok, SAID, REPLY, ids="server", **{"template.chat_template_sha256": "0" * 64})) == ["template_differs"]
    assert render(transcript(tok, SAID, REPLY, ids="server"), small(**{"text.max_tokens": 8})) == ["over_token_bound"]
    empty = qwen_adapter.render(qwen, tok, transcript(tok, SAID, ""))
    assert empty.flags == ["empty_reply", "ids_rebuilt"] and empty.reply is None and len(empty.spans) == 1
    # every flag a rendering can carry is explained
    for t in (transcript(tok, SAID, REPLY, ids="thinking"), transcript(tok, SAID, "")):
        assert set(render(t)) <= set(FLAGS)


def test_a_conversation_keeps_every_turn_in_order(qwen, qwen_adapter, tok):
    t = transcript(tok, "And at night?", "Streetlights.")
    t["request"]["messages"] = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}, *t["request"]["messages"]]
    r = qwen_adapter.render(qwen, tok, t)
    assert [(s.turn, s.role) for s in r.spans] == [(0, "system"), (1, "user"), (2, "assistant"), (3, "user"), (4, "assistant")]
    assert [s.start for s in r.spans] == sorted(s.start for s in r.spans)
    assert r.text[r.reply.char_start : r.reply.char_end] == "Streetlights."


def test_a_message_is_placed_by_the_template_not_by_searching_for_its_words(qwen, qwen_adapter, tok):
    # "a", "user" and "assistant" all occur inside the template's own text; each must still land on the message
    t = transcript(tok, "assistant", "a")
    t["request"]["messages"] = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "user"}, *t["request"]["messages"]]
    r = qwen_adapter.render(qwen, tok, t)
    assert r.flags == ["ids_rebuilt"]
    assert [r.text[s.char_start : s.char_end] for s in r.spans] == ["a", "user", "assistant", "a"]
    for s in r.spans:
        before = r.text[: s.char_start]
        assert before.endswith(f"<|im_start|>{s.role}\n") or before.endswith("</think>\n\n")


def test_a_message_the_template_trims_or_rewrites_is_still_handled(qwen, qwen_adapter):
    class Trimming(FakeTokenizer):
        def render(self, messages, add_generation_prompt=False, enable_thinking=False):
            return super().render([{**m, "content": m["content"].strip()} for m in messages], add_generation_prompt, enable_thinking)

    trimming = Trimming()
    r = qwen_adapter.render(qwen, trimming, transcript(trimming, "  What do you see?\n", "\n\nA quiet street."))
    assert r.flags == ["message_trimmed", "ids_rebuilt"] and r.qualified
    assert [r.text[s.char_start : s.char_end] for s in r.spans] == ["What do you see?", "A quiet street."]

    class Rewriting(FakeTokenizer):
        def render(self, messages, add_generation_prompt=False, enable_thinking=False):
            return super().render([{**m, "content": m["content"].upper() if m["role"] == "user" else m["content"]} for m in messages], add_generation_prompt, enable_thinking)

    rewriting = Rewriting()
    r = qwen_adapter.render(qwen, rewriting, transcript(rewriting, "What do you see?", "A quiet street."))
    # the user's message is not in the text word for word: it has no span, and the reply is still found
    assert r.flags == ["message_not_found", "ids_rebuilt"]
    assert [(s.role, r.text[s.char_start : s.char_end]) for s in r.spans] == [("assistant", "A quiet street.")]


def test_a_tokenizer_that_disagrees_with_its_template_is_an_error_not_a_flag(qwen, qwen_adapter):
    broken = FakeTokenizer(broken=True)
    with pytest.raises(ContractError, match="different token ids"):
        qwen_adapter.render(qwen, broken, transcript(broken, SAID, REPLY))
