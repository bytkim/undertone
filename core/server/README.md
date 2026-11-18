# undertone-server

Readings for the chat page: for each reply the engine has kept, which emotions stand out in it and which words the
model leans toward at each depth.

It is the last step of the pipeline, and it adds no measurement of its own. It loads the model once, runs the three
tools on a transcript in memory, and hands the page one record:

```text
transcripts/  ->  capture  ->  emotion  ─┐
(the engine's)             ->  jlens    ─┴─>  readings/<transcript id>.json  ->  GET /readings/<transcript id>
```

Every number in a reading is the tools' own, made by the same functions their command lines call.

## Start

Requires Python 3.12 or newer and a GPU that holds the model: a little over 12 GiB for Qwen3.5-4B (the model, its
last step in float32, and the lens).

From the repository's root, on the machine with the GPU:

```sh
pip install -r requirements.txt
```

That installs the engine, the tools and this server in the order they depend on each other. By hand it is:

```sh
pip install -e "core/tools/models[run]" -e core/tools/capture -e core/tools/jlens -e core/tools/emotion -e core/server
```

Then, in the folder where the engine keeps its `transcripts/`:

```sh
undertone-server serve
```

The first start fetches the model's files (about 9 GB) and the lens (about 400 MB) and hashes them against the
model's contract, which takes a minute or two. It then says what it loaded and listens on
`http://127.0.0.1:8788`, which is where the [page](../../frontend/README.md) looks for it.

| Option | Meaning |
| --- | --- |
| `--transcripts` | the folder the engine keeps transcripts in (default `./transcripts`, or `UNDERTONE_TRANSCRIPTS`) |
| `--out` | the folder readings are kept in (default `./readings`, or `UNDERTONE_READINGS`) |
| `--model` | the model's contract (default `qwen3.5-4b`) |
| `--snapshot` | a folder already holding the model's files, instead of fetching them |
| `--lens`, `--lens-file` | which pinned lens (default `jpp`), or `none`; a lens file already in hand |
| `--baseline` | a baseline folder to rank lens words against, or `none` for probability (default: the one that comes with the lens tool) |
| `--vectors` | an emotion vector set's folder, or `none` (default: the set that comes with the emotion tool) |
| `--top` | how many emotions, and how many words a depth, a reading keeps (default 8) |
| `--port`, `--device`, `--skip-hashes` | where to listen; `cuda` or `cpu`; do not hash the model's files first |

Without serving anything:

```sh
undertone-server read               # make the reading of every transcript in the folder now
undertone-server show readings/     # what a folder of readings holds
undertone-server show readings/ --transcript <transcript id>
```

### The GPU is on another machine

The server binds to `127.0.0.1`, refuses a request addressed to any other name, and answers only pages served from
the same machine, as the engine does. It has no
accounts and is meant for one person. To use a GPU elsewhere, run the engine and the server there, side by side, and
forward both ports to the machine with the browser:

```sh
ssh -L 8787:127.0.0.1:8787 -L 8788:127.0.0.1:8788 <the GPU machine>
```

The engine needs no GPU, but the server reads the folder the engine writes, so the two share a machine.

## What it answers

`GET /readings/<transcript id>` answers the reading. The first time, it is made on the spot, which takes a few
seconds (about 50 tokens a second on an NVIDIA L4), and the request waits. After that the kept file is answered.

| Status | When |
| --- | --- |
| 200 | the reading |
| 403 | the request was addressed to another machine's name |
| 404 | no transcript has that id |
| 422 | the transcript cannot be read, with the reason: an empty reply, or a transcript that names another model's contract |
| 500 | making it failed (the reply was too long for the GPU's memory, say); the next reply is still read |

`GET /health` says which model and tools are loaded and how many readings are kept.

The model does one thing at a time. Readings asked for at once are made one after another, and a reading asked for
twice is made once.

## What a reading holds

`readings/<transcript id>.json` (`undertone.reading.v1`), which is also what the page is sent:

| Field | Meaning |
| --- | --- |
| `transcript`, `conversation`, `turn`, `model` | which call this is, and the model's contract |
| `flags`, `qualified` | carried over from the capture: whatever put the transcript outside the setting the model was checked in. The page says so beside the reading. |
| `fits` | whether `tokens`, joined, are the reply to the character. The page lays them over the reply only then. |
| `made_with` | the model revision, the lens file, the baseline and the vector set by their hashes, where the token ids came from, and the floor and step of the token marks |
| `overall.emotions` | the emotions with the greatest share over the whole reply: `name`, `family`, `level` (0 to 100), `share` |
| `overall.lens` | at each of the four depths, the words the model leans toward over the whole reply, most leaned-toward first |
| `tokens` | each token of the reply in order. One where an emotion stands out carries `emotions` (`name`, `family`, `strength` 1 to 9, `share`) and `lens` (the words at each depth there). Any other carries only its `text`. |

What the numbers mean is each tool's to say: the scale of `level`, `share` and `strength` is in
[core/tools/emotion](../tools/emotion/README.md#the-scale), and how the words are found and ranked is in
[core/tools/jlens](../tools/jlens/README.md). A reading keeps the top of each; for every emotion or the full word
masses, run the tools themselves on a capture.

Two things are done here and nowhere else, both to the text and neither to the numbers. A character that is split
over two tokens is given to the token that completes it. And white space a model's template trimmed from the edges
of the reply is put back on the first and last token, so the tokens can be laid over the reply as the page shows it.

## What is kept

- **Readings** are kept, one file each. `readings/readings.json` records the model, lens, baseline and vector set
  they were made with; a server started with others refuses that folder, so readings made two ways are never mixed.
- **Captures are not kept.** A capture is about 120 KB a token at the blocks the tools read, so they are made in
  memory and let go. To keep one, use [undertone-capture](../tools/capture/README.md).
- **Transcripts are only read.** The folder is the engine's.

A tool that cannot be had leaves its part of the reading empty: with `--lens none` there are no words, and for a
model with no vector set there are no emotions and no marked tokens. The server says which when it starts.

## Run on the real model

On 2026-11-18, on one NVIDIA L4 (23 GB), with Qwen3.5-4B, the J++ lens, and the baseline and vector set that come
with the tools:

- **The readings are the tools' own.** For 18 real replies, each reading was held against the output of
  `undertone-emotion read` and `undertone-jlens run` on a capture of the same transcript: the same emotions, levels
  and token marks in all 18, with no difference in any share, and the same words at every depth, overall and on
  every marked token, in the 6 that had a lens reading to compare.
- **The tokens fit the reply** to the character in all 18.
- **Speed:** 1.3 to 4.1 seconds for replies of 60 to 200 tokens; a 10-token reply took 0.4 seconds. Twelve replies
  asked for at once (1,054 tokens) took 22.5 seconds. Readings already kept: all 18 in 0.06 seconds.
- **Memory:** 12.0 GiB once loaded, 12.3 GiB after the 18 readings.
- **With the page:** a message typed into the page went through the engine to vLLM, and the reply's reading came
  back marked on its words about two seconds later. A transcript made with thinking on was read and shown with
  its flag.
- **From a fresh install:** `pip install -r requirements.txt` in a new environment chose torch 2.15.0 and
  transformers 5.23.0, newer than the versions the model was checked with (2.14.0 and 5.20.2). The model's
  self-test passed there, and all 19 readings made again under them came out identical to the first ones.

### One 24 GB GPU for both the model server and this

The model server you chat with holds its own copy of the model, so the two need about 21 GiB between them. On the L4
(22.5 GiB) they fit, with this server started first and vLLM 0.34.0 told to stay small:

```sh
vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --gpu-memory-utilization 0.43 --disable-extensible-kv-cache --kv-cache-memory-bytes 700000000 \
  --max-model-len 2048 --max-num-seqs 2
```

vLLM then took 8.9 GiB, which left about 1.3 GiB spare; it also fitted with vLLM started first. It is one arrangement that worked on one card, with short
conversations; vLLM's defaults did not fit beside this server. With less memory, chat first and read afterwards
(`undertone-server read`), or give each its own GPU.

Eighteen replies and one conversation through the page. This shows the server hands on what the tools make. What
the readings are worth is each tool's README to say.

## What it does not do

- **It does not say the model feels anything.** A reading is a resemblance to stories and a view through a lens.
- **It does not read while the model writes.** Readings are made after the fact, from the transcript, by the pinned
  model reading the text again.
- **It does not check who is asking.** It is for one person on their own machine.
- **It does not read a transcript made for another model's contract**, and one made outside the checked setting is
  read and flagged, not refused.

## From Python

```python
from undertone_server import Desk, Tools, make_server

desk = Desk(Tools(model=model, reader=reader, baseline=baseline, vectors=vectors), "transcripts/", "readings/")
desk.reading("<transcript id>")       # made now if it has not been, then kept
make_server(desk).serve_forever()
```

`read(tools, transcript)` makes one reading without keeping it. `undertone_server.cli.load` shows how the model,
the lens reader, the baseline and the vectors are loaded.

## Files

- `reading.py`: one transcript to one reading, with the tools.
- `desk.py`: the transcripts folder, the readings kept, one reading at a time.
- `service.py`: the two requests.
- `cli.py`: the `undertone-server` command.

## Checks

```sh
cd core/server
pip install -e ../tools/models -e ../tools/capture -e ../tools/jlens -e ../tools/emotion -e ".[dev]" torch
pytest
```

The tests use the stand-in model from `undertone_models.standin`, a made-up lens and made-up emotion vectors: no
network, no GPU, no weights.
