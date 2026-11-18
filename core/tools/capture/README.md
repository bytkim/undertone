# undertone-capture

What every block of the model puts out while it reads a transcript, kept as files the other tools read.

A capture is made **after the fact**. Whatever wrote the reply is not asked again. The transcript's text is read once
more by the pinned model ([core/tools/models](../models/README.md)), and the output of each block is kept at the
reply's tokens. That is the raw material a lens or an emotion vector is applied to.

```text
transcripts (from the engine)  ->  undertone-capture  ->  captures/
                                                            capture.json              the run
                                                            <transcript id>.json      what the numbers are of
                                                            <transcript id>.safetensors   the numbers
```

## Start

Requires Python 3.12 or newer and a GPU that holds the model (about 12 GB for Qwen3.5-4B).

```sh
pip install -e "core/tools/models[run]" -e core/tools/capture
```

Chat through the [engine](../../engine/README.md) so there are transcripts, then:

```sh
undertone-capture run transcripts/ --out captures/ --model qwen3.5-4b
undertone-capture show captures/
undertone-capture verify captures/
```

`--model` can be left out when the transcripts name their contract (the engine records it with `--contract`).
The first run fetches the pinned model (about 9.3 GB) and hashes it against its contract; `--snapshot <folder>` uses
files you already have.

## What is kept

For each transcript, two files named by its id:

**`<id>.safetensors`** holds one tensor, `blocks`: `[blocks, tokens, width]` (for Qwen3.5-4B, 32 by the reply's
tokens by 2560), in the weights' own precision (bfloat16). A 90-token reply is about 15 MB.

**`<id>.json`** (`undertone.capture.v1`) says what those numbers are of:

| Field | Meaning |
| --- | --- |
| `transcript`, `conversation`, `turn` | which call this is |
| `model` | the contract's name, and the repository and revision that read the text |
| `ids` | every token id the model read, and whether they came from the endpoint (`server`) or were rebuilt |
| `flags`, `qualified` | whatever puts this transcript outside the setting the model was qualified in (see the [flags](../models/README.md#a-rendering-and-its-flags)) |
| `select`, `skip`, `mean`, `blocks` | how it was captured |
| `spans` | which token positions belong to which message |
| `tokens` | each captured token in order: its position, its id and its text |
| `joins` | whether the tokens' text is the reply to the character (so they can be laid over it) |
| `tensor` | the file, the tensor's name, shape and precision, and the file's SHA-256 |

`capture.json` (`undertone.capture.run.v1`) records the settings, the packages and GPU, what was captured, what was
skipped (a reply with no text) and what failed, with the error.

## Choices

| Option | What it does |
| --- | --- |
| `--select reply` (default) | the reply's own tokens |
| `--select messages` | every message's own tokens, the reply's included. Template tokens are never kept. |
| `--skip N` | leave out the first N tokens of the reply |
| `--mean` | keep the average over the selected tokens in float32, `[blocks, width]`, instead of every token |
| `--blocks 8-31` | keep only some blocks (counted from 0) |
| `--conversation <id>` | capture one conversation |

A folder holds captures made one way, with one model. Running again with the same settings carries on, leaving what
is there alone. Other settings are refused, so two kinds of capture never mix in one folder.

## From Python

```python
from undertone_models import adapter, contract
from undertone_capture import read, run, transcripts

qwen = contract("qwen3.5-4b")
model = adapter(qwen).load(qwen)
run(model, transcripts(["transcripts/"]), "captures/")

record, blocks = read("captures/", "<transcript id>")   # the file is checked against its hash first
middle = blocks[qwen.depths["middle"]]                  # the middle depth's blocks, at every reply token
```

`capture(model, transcript)` reads one transcript and answers `(record, tensor)` without writing anything. A
transcript here is the engine's line as a dict; anything with `request.messages` and `response.content` will do, so
a tool can capture a text that never went through the engine.

## What a capture is, and is not

- **It is the pinned model reading the text.** If the reply came from other weights, a quantised build or another
  template, the capture still describes how the pinned model reads it, and the record's flags say what differed.
- **Nothing is refused for being unusual.** A transcript made with thinking on, or one longer than the model was
  qualified for, is captured and flagged. A reply with no text is skipped; a transcript that fails (too long for the
  memory, say) is recorded with its error and the rest carry on.
- **One text at a time, nothing cached, nothing cut short**, as the model was qualified.
- **It says nothing about emotions or words.** Those are the other tools' to read from it.

## Run on the real model

On 2026-11-18, on one NVIDIA L4 with the packages the model was qualified with, nine transcripts were captured and
verified: three written by hand and six from a real exchange (vLLM 0.34.0 serving the pinned weights, chatted with
through the engine).

- The four transcripts with the endpoint's own token ids were captured with **no flags**: vLLM's ids are what the
  pinned template gives. The one made without ids was rebuilt (`ids_rebuilt`), and one made with thinking on was
  flagged (`ids_differ_from_template`).
- Every capture equals a fresh read of the same text, and equals a read of the whole text taken at those places.
- The captured positions are the right ones: the last block at each token, through the model's head, gives the
  token that follows it a far higher probability (mean log-probability about -0.6) than a token one place either
  side (about -17 and -12). For the replies vLLM generated at temperature 0, its top choice was the next token 95
  to 100 percent of the time.

That last point is evidence that re-reading agrees with the served model here. It is **not** a qualification of
agreement with a serving endpoint: six transcripts, one server, one GPU.

## Checks

```sh
cd core/tools/capture
pip install -e ../models -e ".[dev]" torch
pytest
```

The tests use the stand-in model and tokenizer from `undertone_models.standin`: no network, no GPU, no weights.
