"""Qwen3.5-4B for the undertone tools: the only code that touches this model's internals.

Five things, each bound to `contract.json`:

    tokenizer(contract)                     the pinned tokenizer and chat template (no weights)
    render(contract, tokenizer, transcript) a transcript as token ids, with each message's span and the reply's
    fetch(contract) / verify(contract, dir) the pinned files, and proof that a folder holds exactly them
    load(contract)                          the pinned model -> Model
    Model.run(ids, positions)               the output of every block at those positions
    Model.head(residual)                    a block output as logits over the vocabulary, in float32
    load_head(contract)                     only the model's last step -> Head (no blocks: for reading a capture)

`render` and `tokenizer` need only the tokenizer library. `load`, `run` and `head` need torch and a GPU, and are
imported only when called. `load_head` needs torch and about 2.5 GB, on a GPU or not.

How it was qualified (and nothing more than this): Hugging Face Transformers, bfloat16 weights, SDPA attention, no
KV cache, no truncation, TF32 off, thinking off. See README.md beside this file.
"""
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from undertone_models import Contract, ContractError, Rendering, Span


# ── text to tokens ──────────────────────────────────────────────────────────────────────────────────────────────

def tokenizer(contract: Contract, snapshot: str | Path | None = None) -> Any:
    """The pinned tokenizer, with its chat template: from a folder that holds the model's files, or fetched at the
    contract's revision."""
    from transformers import AutoTokenizer

    if snapshot is not None:
        return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    return AutoTokenizer.from_pretrained(contract.repo, revision=contract.revision)


def _template_sha256(tok: Any) -> str | None:
    template = getattr(tok, "chat_template", None)
    return hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None


def render(contract: Contract, tok: Any, transcript: Mapping[str, Any]) -> Rendering:
    """One transcript (an `undertone.transcript.v1` line, as a dict) as this model reads it: every message sent and
    then the reply, through the model's own chat template with thinking off.

    The ids are the endpoint's where it returned them and they are what the pinned template gives; otherwise they
    are rebuilt, or (when the endpoint's ids disagree with the template) the endpoint's are used as they are. A
    transcript is never refused: whatever puts it outside the qualified setting is listed in `flags`.
    """
    request, response = transcript["request"], transcript["response"]
    reply = response.get("content") or ""
    messages = [{"role": m["role"], "content": m.get("content") or ""} for m in request["messages"]] + [{"role": "assistant", "content": reply}]
    flags: list[str] = []

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=contract.thinking)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    native = tok.apply_chat_template(messages, tokenize=True, return_dict=True, add_generation_prompt=False, enable_thinking=contract.thinking)["input_ids"]
    ids = [int(i) for i in encoded["input_ids"]]
    if [int(i) for i in native] != ids:
        # not a property of the transcript: the tokenizer in hand is not the pinned one, or the install is broken
        raise ContractError("The tokenizer and its chat template give different token ids for the same text.")
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    spans: list[Span] = []
    places = _place(tok, messages, text, contract.thinking)
    for turn, message in enumerate(messages):
        if not message["content"]:
            continue
        if places[turn] is None:
            flags.append("message_not_found")
            continue
        lo, hi = places[turn]
        if hi - lo != len(message["content"]):
            flags.append("message_trimmed")
        inside = [k for k, (a, b) in enumerate(offsets) if lo <= a < b <= hi]
        if not inside or inside != list(range(inside[0], inside[-1] + 1)):
            flags.append("message_not_found")
            continue
        spans.append(Span(turn=turn, role=message["role"], start=inside[0], stop=inside[-1] + 1, char_start=lo, char_end=hi))
    last = len(messages) - 1
    reply_span = next((s for s in spans if s.turn == last), None)
    if not reply:
        flags.append("empty_reply")

    # where the ids come from
    tokens = transcript.get("tokens") or {}
    source = "rebuilt"
    if tokens.get("source") == "server" and tokens.get("prompt_ids") is not None and tokens.get("completion_ids") is not None:
        prompt, completion = [int(i) for i in tokens["prompt_ids"]], [int(i) for i in tokens["completion_ids"]]
        given = prompt + completion
        covers = reply_span is None or len(given) >= reply_span.stop
        if ids[: len(given)] == given and covers:
            source = "server"   # the same ids; the template only adds the tokens that close the reply
        else:
            flags.append("ids_differ_from_template")
            whole = Span(turn=last, role="assistant", start=len(prompt), stop=len(given)) if completion else None
            return _finish(contract, tok, transcript, Rendering(ids=given, source="server", spans=[whole] if whole else [], reply=whole, flags=flags))
    else:
        flags.append("ids_rebuilt")
        if response.get("reasoning"):
            flags.append("reasoning_dropped")
    return _finish(contract, tok, transcript, Rendering(ids=ids, source=source, spans=spans, reply=reply_span, text=text, offsets=offsets, flags=flags))


def _place(tok: Any, messages: list[dict[str, str]], text: str, thinking: bool) -> list[tuple[int, int] | None]:
    """Where each message's own text sits in the rendered text: (start, end), or None where it cannot be placed.

    Searching the rendered text for a message is not safe (a short message such as "a" also occurs inside the
    template's own words), so the template is rendered a second time with a marker in place of each message. What
    lies between the markers is the template's own text, and it must be the same in both renderings; each message is
    then exactly where its marker was. A template may trim a message's edges, so the trimmed text is accepted too.
    """
    marks = [f"\u27e6undertone message {k}\u27e7" for k in range(len(messages))]
    marked = [{**m, "content": marks[k] if m["content"] else ""} for k, m in enumerate(messages)]
    blank = tok.apply_chat_template(marked, tokenize=False, add_generation_prompt=False, enable_thinking=thinking)
    places: list[tuple[int, int] | None] = [None] * len(messages)
    at_text = at_blank = 0
    for k, message in enumerate(messages):
        content = message["content"]
        if not content:
            continue
        mark_at = blank.find(marks[k], at_blank)
        between = blank[at_blank:mark_at]
        lo = at_text + len(between)
        if mark_at < 0 or text[at_text:lo] != between:
            break   # the template does not treat this message as plain text: it and what follows cannot be placed
        kept = next((c for c in (content, content.strip("\n"), content.strip()) if c and text.startswith(c, lo)), None)
        if kept is None:
            break
        places[k] = (lo, lo + len(kept))
        at_text, at_blank = lo + len(kept), mark_at + len(marks[k])
    else:
        return places
    # what could not be placed by the template: the reply alone is still safe to find, as the last thing said
    reply = messages[-1]["content"]
    at = text.rfind(reply) if reply else -1
    if places[-1] is None and at >= at_text:
        places[-1] = (at, at + len(reply))
    return places


def _finish(contract: Contract, tok: Any, transcript: Mapping[str, Any], r: Rendering) -> Rendering:
    """The flags that do not depend on how the ids were found."""
    flags = list(r.flags)
    template = transcript.get("template") or {}
    if template.get("thinking") is True or transcript["response"].get("reasoning"):
        flags.append("thinking_on")
    theirs, ours = template.get("chat_template_sha256"), _template_sha256(tok)
    if theirs and ours and theirs != ours:
        flags.append("template_differs")
    if template.get("tokenizer_revision") and template["tokenizer_revision"] != contract.revision:
        flags.append("tokenizer_revision_differs")
    served = (transcript.get("endpoint") or {}).get("served_model")
    if served is not None and served.lower() not in {name.lower() for name in contract.served_as}:
        flags.append("served_model_differs")
    if len(r.ids) > contract.max_tokens:
        flags.append("over_token_bound")
    order = list(dict.fromkeys(flags))
    return Rendering(ids=r.ids, source=r.source, spans=r.spans, reply=r.reply, text=r.text, offsets=r.offsets, flags=order)


# ── the pinned files ────────────────────────────────────────────────────────────────────────────────────────────

def fetch(contract: Contract, cache_dir: str | Path | None = None) -> Path:
    """Download exactly the contract's files at its revision (about 9 GB) and answer the folder they are in."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(contract.repo, revision=contract.revision, allow_patterns=sorted(contract.files), cache_dir=cache_dir))


def verify(contract: Contract, snapshot: str | Path) -> None:
    """Hash every file in `snapshot` against the contract. Raises on a missing or changed file."""
    for name, meta in contract.files.items():
        path = Path(snapshot) / name
        if not path.is_file():
            raise ContractError(f"{contract.title}: {name} is missing from {snapshot}.")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                digest.update(chunk)
        if digest.hexdigest() != meta["sha256"]:
            raise ContractError(f"{contract.title}: {name} is not the pinned file (its hash differs).")


# ── the model ───────────────────────────────────────────────────────────────────────────────────────────────────

def load(contract: Contract, snapshot: str | Path | None = None, device: str = "cuda", verify_files: bool = True) -> "Model":
    """The pinned model, ready to read. `snapshot` is a folder already holding its files; without one they are
    fetched. The files are hashed against the contract first (`verify_files=False` skips that: about 9 GB of
    hashing, and the only proof the weights are the pinned ones).

    Loading turns TF32 off for the whole process, as the model was qualified with it off.
    """
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    folder = Path(snapshot) if snapshot is not None else fetch(contract)
    if verify_files:
        verify(contract, folder)
    numbers = contract.data["numbers"]
    torch.backends.cuda.matmul.allow_tf32 = bool(numbers["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(numbers["tf32"])
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(str(folder), local_files_only=True, dtype=getattr(torch, numbers["weights"]), attn_implementation=numbers["attention"])
    return Model(contract, hf.eval().to(device), AutoTokenizer.from_pretrained(str(folder), local_files_only=True), device)


def _head(residual: Any, norm_weight: Any, embedding: Any, eps: float) -> Any:
    """The model's last step in float32: RMSNorm scaled by (1 + weight), then the tied output embedding."""
    import torch

    with torch.inference_mode():
        x = residual.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        x = x * (1 + norm_weight.float())
        return torch.nn.functional.linear(x, embedding.float())


#: where the two tensors of the last step are kept in the model's files
_NORM, _EMBEDDING = "model.language_model.norm.weight", "model.language_model.embed_tokens.weight"


def load_head(contract: Contract, snapshot: str | Path | None = None, device: str = "cuda", verify_files: bool = True) -> "Head":
    """Only the model's last step (its final norm and its output embedding) and its tokenizer, read straight from
    the pinned files. Enough to turn a captured block output into words; none of the blocks are loaded, so it needs
    about 2.5 GB and works without a GPU (`device="cpu"`)."""
    import json

    from safetensors import safe_open
    from transformers import AutoTokenizer

    folder = Path(snapshot) if snapshot is not None else fetch(contract)
    if verify_files:
        verify(contract, folder)
    where = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((folder / "config.json").read_text())["text_config"]
    tensors = {}
    for name in (_NORM, _EMBEDDING):
        if name not in where:
            raise ContractError(f"{contract.title}'s files have no {name}.")
        with safe_open(str(folder / where[name]), framework="pt", device="cpu") as f:
            tensors[name] = f.get_tensor(name)
    return Head(contract, tensors[_NORM].to(device), tensors[_EMBEDDING].to(device), config["rms_norm_eps"], AutoTokenizer.from_pretrained(str(folder), local_files_only=True), device)


class Head:
    """The model's last step on its own: a block output in, logits over the vocabulary out. What a tool that reads
    captures needs of the model, without the model."""

    def __init__(self, contract: Contract, norm_weight: Any, embedding: Any, eps: float, tok: Any, device: str):
        if list(embedding.shape)[1:] != [contract.width] or list(norm_weight.shape) != [contract.width]:
            raise ContractError(f"This is not {contract.title}'s last step: an embedding of {list(embedding.shape)}.")
        self.contract, self.tokenizer, self.device = contract, tok, device
        # float32 once, here: every read after this is the same arithmetic as Model.head
        self._norm_weight, self._embedding, self._eps = norm_weight.float(), embedding.float(), eps

    @property
    def vocab_size(self) -> int:
        return int(self._embedding.shape[0])

    def head(self, residual: Any) -> Any:
        return _head(residual.to(self.device), self._norm_weight, self._embedding, self._eps)


class Model:
    """The loaded model, as the tools use it. Built by `load`; the model's own layout is read here and nowhere else."""

    def __init__(self, contract: Contract, hf: Any, tok: Any, device: str):
        decoder, config = hf.model.language_model, hf.config.text_config
        if config.hidden_size != contract.width or len(decoder.layers) != contract.blocks:
            raise ContractError(f"This is not {contract.title}'s shape: {len(decoder.layers)} blocks of width {config.hidden_size}.")
        if hf.lm_head.weight.data_ptr() != decoder.embed_tokens.weight.data_ptr():
            raise ContractError("The output embeddings are not tied to the input embeddings.")
        self.contract, self.tokenizer, self.device = contract, tok, device
        self._hf, self._layers, self._norm = hf, decoder.layers, decoder.norm
        self._embedding, self._eps = hf.lm_head.weight, config.rms_norm_eps

    def render(self, transcript: Mapping[str, Any]) -> Rendering:
        return render(self.contract, self.tokenizer, transcript)

    def run(self, ids: Sequence[int], positions: Sequence[int] | None = None, logits_at: Sequence[int] | None = None) -> Any:
        """Read the text once and answer the output of every block at `positions` (all of them when not given): a
        tensor [blocks, positions, width] in the weights' own precision. One text at a time, nothing cached, nothing
        cut short.

        With `logits_at`, also answers the model's own logits at those positions: (blocks, logits).
        """
        import torch

        at = list(range(len(ids))) if positions is None else list(positions)
        kept: dict[int, Any] = {}
        handles = []
        for k, block in enumerate(self._layers):
            def hook(module: Any, args: Any, output: Any, k: int = k) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                kept[k] = hidden[0, at, :].detach().clone()

            handles.append(block.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                # (the model's own logits are only worked out where asked: the last position when nowhere is)
                keep = torch.tensor(list(logits_at) if logits_at is not None else [len(ids) - 1], device=self.device)
                result = self._hf(input_ids=torch.tensor([list(ids)], device=self.device), use_cache=False, logits_to_keep=keep)
        finally:
            for h in handles:
                h.remove()
        blocks = torch.stack([kept[k] for k in range(len(self._layers))])
        return (blocks, result.logits[0].detach()) if logits_at is not None else blocks

    def head(self, residual: Any) -> Any:
        """A block output as logits over the vocabulary, in float32: the model's final norm (RMSNorm scaled by
        1 + weight) and its tied output embedding. This is the model's own last step, done at full precision; a
        lens moves an earlier block's output to the last block and then comes here."""
        return _head(residual, self._norm.weight, self._embedding, self._eps)

    def as_head(self) -> "Head":
        """This model's last step as a `Head`, for a tool that takes one."""
        return Head(self.contract, self._norm.weight.detach(), self._embedding.detach(), self._eps, self.tokenizer, self.device)

    def native_head(self, residual: Any) -> Any:
        """The same step done by the model's own norm module in float32, for checking `head` against."""
        import copy

        import torch

        with torch.inference_mode():
            return torch.nn.functional.linear(copy.deepcopy(self._norm).float()(residual.float()), self._embedding.float())
