"""What a model's adapter gives back when it turns a transcript into tokens.

Every model answers with the same shapes, so a tool can read a rendering without knowing which model made it.
"""
from dataclasses import dataclass, field
from typing import Literal

#: Every flag a rendering can carry, and what it means. A flag never stops a tool; it says the transcript lies
#: outside what the model was qualified for, so whatever is read from it should say so too.
FLAGS: dict[str, str] = {
    "ids_rebuilt": "The endpoint returned no token ids, so they were rebuilt with the pinned tokenizer and template.",
    "ids_differ_from_template": "The endpoint's token ids are not what the pinned template gives for these messages (another template, tokenizer or thinking setting). The endpoint's ids are used as they are, and only the reply's span is known.",
    "thinking_on": "The reply was made with thinking on, or carries reasoning. The model was qualified with thinking off.",
    "reasoning_dropped": "The reply's reasoning is not in the rebuilt ids: only its answer is.",
    "template_differs": "The engine recorded another chat template than the pinned one.",
    "tokenizer_revision_differs": "The engine recorded another tokenizer revision than the pinned one.",
    "served_model_differs": "The endpoint served the model under a name the contract does not know, so its weights are not known to be the pinned ones.",
    "over_token_bound": "The text is longer than the model was qualified for.",
    "message_trimmed": "The model's template trims the edges of a message, so its span is the trimmed text and not the transcript's text to the character.",
    "message_not_found": "A message's text is not in the rendered prompt word for word, so it has no span.",
    "empty_reply": "The reply has no text.",
}


@dataclass(frozen=True)
class Span:
    """One message's own text among the tokens: positions `start` up to (not including) `stop`. Template tokens
    around the message are outside it."""

    turn: int
    role: str
    start: int
    stop: int
    #: where the message's text sits in the rendered text; None when the ids did not come from the template
    char_start: int | None = None
    char_end: int | None = None

    @property
    def positions(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True)
class Rendering:
    """A transcript as the model reads it."""

    #: the token ids of the whole exchange: every message sent, then the reply
    ids: list[int]
    #: where the ids came from: the endpoint ("server", checked against the pinned template) or the pinned
    #: tokenizer and template ("rebuilt")
    source: Literal["server", "rebuilt"]
    #: each message's text among the tokens, in order; a message that could not be placed is left out
    spans: list[Span]
    #: the reply's span (the last message), or None when it has no tokens
    reply: Span | None
    #: the rendered text and each token's place in it; None when the endpoint's ids are used as they are
    text: str | None = None
    offsets: list[tuple[int, int]] | None = None
    #: reasons this transcript lies outside what the model was qualified for (keys of FLAGS)
    flags: list[str] = field(default_factory=list)

    @property
    def qualified(self) -> bool:
        """Nothing about this transcript is outside the qualified setting. Rebuilt ids and a trimmed message are
        worth knowing, and are not that."""
        return not [f for f in self.flags if f not in ("ids_rebuilt", "message_trimmed")]
