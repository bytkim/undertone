"""The transcript: one record per call to an endpoint.

A call is the unit the model actually saw: the messages sent and the reply that came back. The request and the
response keep the OpenAI chat shapes unchanged, so any endpoint fills them and any other tool can read them. Beside
them sit the token ids, which are what the undertone tools consume (ids are the ground truth; text is not), and how
those ids came to be:

- `tokens.source == "server"`: the endpoint returned the ids it used.
- `tokens.source == "rebuilt"`: the engine tokenized the messages and the reply itself. A tool should check a
  rebuilt transcript before trusting it (the same messages give other tokens under another template).
- `tokens is None`: neither was possible. The transcript is still kept; the ids can be rebuilt later.

A conversation is the calls that share `conversation`, in `turn` order. Transcripts are stored as JSON Lines, one
call a line. An API key is never part of a transcript: only the endpoint's host is.
"""
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "undertone.transcript.v1"


class Message(BaseModel):
    """One chat message, as OpenAI writes it."""

    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class Endpoint(BaseModel):
    """Where the call went: the host (never the key) and the name the server knows the model by."""

    host: str
    served_model: str


class Request(BaseModel):
    """The chat request as sent: the messages, and any sampling settings (temperature, max_tokens, seed...) beside them."""

    model_config = ConfigDict(extra="allow")
    messages: list[Message]


class Response(BaseModel):
    """The reply. A model that thinks first keeps its reasoning apart from its answer: the next prompt is built from
    the answer alone, so a transcript that merged the two could not be rebuilt."""

    content: str
    reasoning: str | None = None
    finish_reason: str | None = None


class Tokens(BaseModel):
    """The token ids of the prompt and of the reply, and where they came from. `completion_ids` may be missing on a
    rebuilt transcript whose reply carries reasoning: how a model lays reasoning out in tokens is that model's own."""

    prompt_ids: list[int]
    completion_ids: list[int] | None = None
    source: Literal["server", "rebuilt"]


class Template(BaseModel):
    """How messages became tokens, where the engine knows: the tokenizer, its chat template, and whether thinking was on."""

    tokenizer_revision: str | None = None
    chat_template_sha256: str | None = None
    thinking: bool | None = None


class Transcript(BaseModel):
    schema_: Literal["undertone.transcript.v1"] = Field(default=SCHEMA, alias="schema")
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    conversation: str
    turn: int
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    endpoint: Endpoint
    #: the model's contract among the undertone tools (for example "qwen3.5-4b"), when the user has said which it is
    model: str | None = None
    request: Request
    response: Response
    tokens: Tokens | None = None
    template: Template | None = None
    #: the endpoint's own token counts, when it reports them
    usage: dict[str, Any] | None = None

    def line(self) -> str:
        """This call as one line of JSON."""
        return self.model_dump_json(by_alias=True, exclude_none=True)

    @classmethod
    def parse(cls, line: str) -> "Transcript":
        return cls.model_validate_json(line)

    def next_messages(self, user: str) -> list[Message]:
        """The messages of the call that follows this one: everything sent, the reply (its answer only), and the new message."""
        return [*self.request.messages, Message(role="assistant", content=self.response.content), Message(role="user", content=user)]
