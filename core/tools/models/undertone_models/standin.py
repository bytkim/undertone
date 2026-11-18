"""Stand-ins for tests: a tokenizer and a small model with the same layout as a real one, so a tool's tests need no
network, no GPU and no weights. They prove a tool's logic and nothing about any real model.

    from undertone_models.standin import FakeTokenizer, standin_model, transcript
"""
import hashlib
import re
from types import SimpleNamespace
from typing import Any

EPS = 1e-6

_PIECE = re.compile(r"<\|im_start\|>|<\|im_end\|>|</?think>|\n| ?[^\s<]+|\s|<")


class FakeTokenizer:
    """One id per distinct piece of text; offsets as a fast tokenizer gives them. With thinking off the assistant's
    turn opens with an empty thinking block, as Qwen's template does."""

    chat_template = "{# a stand-in ChatML template #}"

    def __init__(self, broken: bool = False):
        self.vocab: dict[str, int] = {}
        self.broken = broken

    def _ids(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets = [], []
        for m in _PIECE.finditer(text):
            ids.append(self.vocab.setdefault(m.group(), len(self.vocab) + 10))
            offsets.append((m.start(), m.end()))
        assert "".join(text[a:b] for a, b in offsets) == text
        return ids, offsets

    def render(self, messages, add_generation_prompt=False, enable_thinking=False) -> str:
        out = ""
        for m in messages:
            opening = "" if m["role"] != "assistant" or enable_thinking else "<think>\n\n</think>\n\n"
            out += f"<|im_start|>{m['role']}\n{opening}{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return out

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False):
        text = self.render(messages, add_generation_prompt, enable_thinking)
        if not tokenize:
            return text
        ids = self._ids(text)[0]
        if self.broken:
            ids = ids[:-1]
        return {"input_ids": ids} if return_dict else ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = self._ids(text)
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)[0]

    def decode(self, ids, skip_special_tokens=False):
        back = {i: piece for piece, i in self.vocab.items()}
        return "".join(back.get(int(i), "") for i in ids)   # (an id no text has used yet decodes to nothing)

    @property
    def template_sha256(self) -> str:
        return hashlib.sha256(self.chat_template.encode()).hexdigest()


def transcript(tok: FakeTokenizer, said: str, reply: str, *, ids: str | None = None, **more):
    """A transcript line as the engine writes it. `ids`: "server" gives the ids a server would return for the pinned
    template (thinking off); "thinking" gives a server's ids with thinking on."""
    messages = [{"role": "user", "content": said}]
    t = {"id": "t1", "conversation": "c1", "turn": 1, "endpoint": {"host": "localhost:8000", "served_model": "Qwen/Qwen3.5-4B"},
         "request": {"messages": messages}, "response": {"content": reply}}
    if ids == "server":
        t["tokens"] = {"prompt_ids": tok.apply_chat_template(messages, add_generation_prompt=True), "completion_ids": tok.encode(reply + "<|im_end|>"), "source": "server"}
    elif ids == "thinking":
        t["tokens"] = {"prompt_ids": tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True),
                       "completion_ids": tok.encode("<think>\nLet me look.\n</think>\n\n" + reply + "<|im_end|>"), "source": "server"}
    for key, value in more.items():
        node = t
        *path, last = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[last] = value
    return t


def standin_model(blocks: int = 4, width: int = 16, vocab: int = 512, tied: bool = True, noisy: bool = False) -> Any:
    """A small torch model laid out like the real one (`model.language_model.layers`, `.norm`, a tied `lm_head`,
    `config.text_config`). `noisy` makes it read differently each time; `tied=False` unties its embeddings."""
    import torch

    nn = torch.nn

    class Norm(nn.Module):
        """RMSNorm scaled by (1 + weight), as the contract says of the real model's."""

        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(width) * 0.1)

        def forward(self, x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (1 + self.weight)


    class Block(nn.Module):
        def __init__(self, width, noisy=False):
            super().__init__()
            self.mix, self.noisy = nn.Linear(width, width), noisy

        def forward(self, h):
            out = h + torch.tanh(self.mix(h))
            if self.noisy:
                out = out + torch.randn_like(out) * 1e-3
            return (out,)


    class StandIn(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            decoder = nn.Module()
            decoder.embed_tokens = nn.Embedding(vocab, width)
            decoder.layers = nn.ModuleList(Block(width, noisy) for _ in range(blocks))
            decoder.norm = Norm(width)
            self.model = nn.Module()
            self.model.language_model = decoder
            self.lm_head = nn.Linear(width, vocab, bias=False)
            if tied:
                self.lm_head.weight = decoder.embed_tokens.weight
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=width, rms_norm_eps=EPS))

        def forward(self, input_ids, use_cache=False, logits_to_keep=None):
            assert use_cache is False
            decoder = self.model.language_model
            h = decoder.embed_tokens(input_ids)
            for block in decoder.layers:
                h = block(h)[0]
            return SimpleNamespace(logits=self.lm_head(decoder.norm(h[:, logits_to_keep])))

    return StandIn().eval()
