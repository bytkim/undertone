"""Token ids for a transcript whose endpoint did not return them: rebuilt with the model's tokenizer.

The engine knows nothing about any one model. It takes a `ChatTokenizer`: something that turns messages into the
prompt's ids and a reply into its ids, and says which tokenizer and chat template it used. The undertone tools
supply one per model they support; `hf_tokenizer` below is a general one over a Hugging Face tokenizer (it needs the
tokenizer only, never the weights).

Rebuilt ids are a reconstruction, and a transcript says so (`tokens.source == "rebuilt"`). The prompt depends on
the chat template and on whether thinking is on; a reply usually tokenizes back to what was generated, but not
always. A tool that reads rebuilt ids should check them against the model first.
"""
import hashlib
from typing import Any, Protocol


class ChatTokenizer(Protocol):
    #: the tokenizer's pinned revision, when it has one
    revision: str | None
    #: a hash of the chat template in use
    template_sha256: str | None

    def prompt_ids(self, messages: list[dict[str, Any]], thinking: bool) -> list[int]:
        """The ids of the prompt the model is given for these messages, ready for it to reply."""
        ...

    def completion_ids(self, content: str, reasoning: str | None) -> list[int] | None:
        """The ids of the reply; None where they cannot be rebuilt honestly."""
        ...


class _HuggingFaceTokenizer:
    def __init__(self, tokenizer: Any, revision: str | None):
        self._tokenizer = tokenizer
        self.revision = revision
        template = getattr(tokenizer, "chat_template", None)
        self.template_sha256 = hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None

    def prompt_ids(self, messages: list[dict[str, Any]], thinking: bool) -> list[int]:
        # (a template that knows nothing of thinking ignores the flag)
        ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def completion_ids(self, content: str, reasoning: str | None) -> list[int] | None:
        if reasoning is not None:
            return None   # how reasoning sits in the reply's tokens is the model's own layout
        return list(self._tokenizer.encode(content, add_special_tokens=False))


def hf_tokenizer(name: str, revision: str | None = None) -> ChatTokenizer:
    """A tokenizer by its Hugging Face name or a local path. Needs the `tokens` extra (transformers)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError('Rebuilding token ids needs the tokenizer library: pip install "undertone-engine[tokens]"') from error
    return _HuggingFaceTokenizer(AutoTokenizer.from_pretrained(name, revision=revision), revision)
