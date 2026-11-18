"""The engine: send to the endpoint, keep the transcript.

Everything that talks to a model goes through here (a chat, a story to be written, a tool that needs text), so there
is one record of what was said and one format for it.
"""
from typing import Any
from uuid import uuid4

from .endpoint import Endpoint, EndpointError
from .store import TranscriptStore
from .tokens import ChatTokenizer
from .transcript import Endpoint as EndpointRecord
from .transcript import Message, Request, Response, Template, Tokens, Transcript


class Engine:
    def __init__(self, endpoint: Endpoint | None, store: TranscriptStore, tokenizer: ChatTokenizer | None = None, model: str | None = None):
        """`endpoint` may be None: the engine then keeps its history and says so when asked to chat, until one is set
        (`engine.endpoint = Endpoint(...)`). `tokenizer` rebuilds token ids when the endpoint returns none (optional:
        without it such a transcript is kept without ids). `model` names the model's contract among the undertone
        tools, when the user knows it."""
        self.endpoint = endpoint
        self.store = store
        self.tokenizer = tokenizer
        self.model = model

    def complete(self, messages: list[dict[str, Any]], conversation: str | None = None, thinking: bool = False, **params: Any) -> Transcript:
        """One call with exactly these messages, recorded as the next turn of `conversation` (a new one when not given).

        `thinking` says whether the endpoint serves the model with thinking on: it is recorded, and used when ids are
        rebuilt. It is not sent; a server that needs telling takes its own setting through `params`.
        """
        if self.endpoint is None:
            raise EndpointError("No model endpoint is set. Set one in the chat page's settings, or start the engine with --base-url and --model.")
        conversation = conversation or uuid4().hex
        turn = len(self.store.conversation(conversation)) + 1
        completion = self.endpoint.chat(messages, **params)

        tokens, template = None, None
        if completion.prompt_ids is not None:
            tokens = Tokens(prompt_ids=completion.prompt_ids, completion_ids=completion.completion_ids, source="server")
        elif self.tokenizer is not None:
            tokens = Tokens(prompt_ids=self.tokenizer.prompt_ids(messages, thinking), completion_ids=self.tokenizer.completion_ids(completion.content, completion.reasoning), source="rebuilt")
        if self.tokenizer is not None:
            template = Template(tokenizer_revision=self.tokenizer.revision, chat_template_sha256=self.tokenizer.template_sha256, thinking=thinking)
        elif thinking:
            template = Template(thinking=True)   # declared, so recorded, even with no tokenizer to say the rest

        transcript = Transcript(
            conversation=conversation,
            turn=turn,
            endpoint=EndpointRecord(host=self.endpoint.config.host, served_model=self.endpoint.config.model),
            model=self.model,
            request=Request(messages=[Message(**m) for m in messages], **params),
            response=Response(content=completion.content, reasoning=completion.reasoning, finish_reason=completion.finish_reason),
            tokens=tokens,
            template=template,
            usage=completion.usage,
        )
        self.store.append(transcript)
        return transcript

    def chat(self, message: str, conversation: str | None = None, system: str | None = None, thinking: bool = False, **params: Any) -> Transcript:
        """Say `message` in a conversation and record the reply. A conversation that exists carries on from its last
        call (what was sent, the reply's answer, then this message); a new one may open with a system message."""
        before = self.store.conversation(conversation) if conversation else []
        if before:
            messages = [m.model_dump() for m in before[-1].next_messages(message)]
        else:
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": message}]
        return self.complete(messages, conversation=conversation, thinking=thinking, **params)
