# undertone-models

What each model the undertone tools support **is**, written down once.

A tool never hard-codes a layer count, a chat template or a revision. It asks the model's contract. One model is
supported so far: [Qwen3.5-4B](qwen3.5-4b/README.md).

```text
core/tools/models/
  undertone_models/     finds a contract, loads a model's adapter (no torch needed to read a contract);
                        standin.py is a stand-in tokenizer and model for any tool's tests
  qwen3.5-4b/
    contract.json       the facts: pinned files and hashes, shape, depths, template, lens pins, how it was qualified
    adapter.py          the only code that touches the model's internals
    check.py            a self-test of an install against the contract
    README.md
```

Another model is another folder with the same four files.

## Start

Requires Python 3.12 or newer.

```sh
pip install -e core/tools/models            # contracts only: no dependencies
pip install -e "core/tools/models[tokens]"  # plus the tokenizer, to turn transcripts into tokens
pip install -e "core/tools/models[run]"     # plus torch, to load and read the model
```

```sh
undertone-models list
undertone-models show qwen3.5-4b
```

## The contract

`contract.json` is plain JSON (`undertone.model.v1`), so anything can read it without this package: the engine
does, to rebuild token ids with the right tokenizer. It says:

| Part | What it holds |
| --- | --- |
| `source` | the model's repository, the exact revision, and the SHA-256 and size of every file |
| `served_as` | the names a server is known to serve these weights under |
| `shape` | how many blocks, how wide, where the tools read, and how the final step to the vocabulary is made |
| `text` | which chat template, whether thinking is on, and the longest text the model was qualified for |
| `numbers` | the precision of the weights and of the reading, and the attention implementation |
| `depths` | which blocks make up each of the four depths: early, middle, late, output |
| `artifacts` | files made for this model by others or by the tools, each pinned to a revision and a hash (the lenses) |
| `qualified` | the runtime it was checked in, the tolerances it met, and what was **not** checked |

A contract that is not whole is refused when it is read, so no tool has to wonder.

## The adapter

Every model's `adapter.py` offers the same things:

```python
from undertone_models import adapter, contract

qwen = contract("qwen3.5-4b")
model = adapter(qwen).load(qwen)                # the pinned model; its files are hashed against the contract first

rendering = model.render(transcript)            # a transcript (an engine line, as a dict) as token ids
blocks = model.run(rendering.ids, rendering.reply.positions)   # every block's output at the reply's tokens
logits = model.head(blocks[-1])                 # a block output as logits over the vocabulary, in float32
```

- **`tokenizer(contract)`** and **`render(contract, tokenizer, transcript)`** need only the tokenizer, no weights.
- **`load(contract, snapshot=None, device="cuda")`** fetches the pinned files unless `snapshot` is a folder that
  already holds them.
- **`run`** reads one text at a time with nothing cached and nothing cut short, and answers a tensor
  `[blocks, positions, width]`.
- **`head`** is the model's own last step (its final norm and output embedding) done at full precision. A lens moves
  an earlier block's output to the last block and then comes here.
- **`load_head(contract)`** loads only that last step and the tokenizer, straight from the pinned files: about
  2.5 GB, on a GPU or not. It is what a tool that reads captures needs of the model, without running the model.

The adapter does not capture a corpus, apply a lens, build or read emotion vectors, steer, or serve. Those are the
tools.

## A rendering, and its flags

`render` answers the token ids of the whole exchange, each message's span among them, and the reply's span. It
**never refuses a transcript**. Whatever puts one outside the setting the model was qualified in is listed in
`flags`, so whatever is read from it can say so too:

| Flag | Meaning |
| --- | --- |
| `ids_rebuilt` | The endpoint returned no token ids, so they were rebuilt with the pinned tokenizer and template. |
| `ids_differ_from_template` | The endpoint's ids are not what the pinned template gives. They are used as they are, and only the reply's span is known. |
| `thinking_on` | The reply was made with thinking on, or carries reasoning. |
| `reasoning_dropped` | The reply's reasoning is not in the rebuilt ids: only its answer is. |
| `template_differs` | The engine recorded another chat template than the pinned one. |
| `tokenizer_revision_differs` | The engine recorded another tokenizer revision than the pinned one. |
| `served_model_differs` | The endpoint served the model under a name the contract does not know, so its weights are not known to be the pinned ones. |
| `over_token_bound` | The text is longer than the model was qualified for. |
| `message_trimmed` | The model's template trims the edges of a message, so its span is the trimmed text. |
| `message_not_found` | A message's text is not in the rendered prompt word for word, so it has no span. |
| `empty_reply` | The reply has no text. |

`rendering.qualified` is true when there is no flag other than `ids_rebuilt` and `message_trimmed`, which are
worth knowing but are inside the qualified setting.

One thing is an error and not a flag: a tokenizer that disagrees with its own chat template about the ids of the
same text. That is a broken install, not a property of a transcript.

## What a contract cannot promise

**The tools re-run the text; they do not read the server.** A reading comes from the pinned weights in Hugging Face
Transformers, whatever produced the reply. If the reply came from a quantised build, another revision or another
template, the reading describes how the *pinned* model reads that text. `served_model_differs` and
`ids_differ_from_template` are how that shows.

## Checks

```sh
cd core/tools/models
pip install -e ".[dev]"
pytest
```

The tests need no network, no GPU and no model. `run`, `head` and the self-test are exercised on a small stand-in
with the same layout (`undertone_models.standin`, which the other tools' tests use too; skipped where torch is not
installed). That proves the code's logic and nothing about any real model. For that, each model has its own
self-test:

```sh
undertone-models check qwen3.5-4b
```
