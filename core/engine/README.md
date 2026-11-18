# undertone-engine

Talk to any OpenAI-compatible endpoint and keep what was said as **transcripts**.

The engine is a boundary and nothing more. You bring the inference server (vLLM, llama.cpp, Ollama, a hosted API);
the engine sends it chat requests and writes down each call. It runs no model, manages no weights and needs no GPU.
Everything else in undertone starts from it: the chat page talks to it, and the tools and the readings server read
the transcripts it keeps, so there is one record of what was said and one format for it.

## Start

Requires Python 3.12 or newer.

```sh
cd core/engine
pip install -e .
```

Point it at an endpoint and say something:

```sh
export UNDERTONE_BASE_URL=http://localhost:8000/v1
export UNDERTONE_MODEL=Qwen/Qwen3.5-4B
undertone-engine chat "Imagine you are looking out of your window. What do you see?"
```

The reply is printed, and the call is kept in `./transcripts/` (one file per conversation). Carry a conversation on
with `--conversation <id>`; see what is kept with `undertone-engine list` and `undertone-engine show <id>`.

If the endpoint needs a key, set `UNDERTONE_API_KEY`. The key is read from the environment only and is never written
to a transcript.

From Python:

```python
from undertone_engine import Endpoint, EndpointConfig, Engine, TranscriptStore

engine = Engine(
    Endpoint(EndpointConfig(base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-4B")),
    TranscriptStore("transcripts"),
)
first = engine.chat("Hello")
second = engine.chat("And then?", conversation=first.conversation)
```

`engine.complete(messages, ...)` sends exactly the messages you give it, for callers that build their own prompts.

## The transcript

One JSON line per call to the endpoint, because a call is the unit the model actually saw:

```json
{
  "schema": "undertone.transcript.v1",
  "id": "…", "conversation": "…", "turn": 2, "created": "2026-11-18T09:12:04+00:00",
  "endpoint": { "host": "localhost:8000", "served_model": "Qwen/Qwen3.5-4B" },
  "model": "qwen3.5-4b",
  "request":  { "messages": [ … ], "temperature": 0.7, "max_tokens": 512 },
  "response": { "content": "…", "reasoning": "…", "finish_reason": "stop" },
  "tokens":   { "prompt_ids": [ … ], "completion_ids": [ … ], "source": "server" },
  "template": { "tokenizer_revision": "…", "chat_template_sha256": "…", "thinking": false }
}
```

- `request` and `response` are the OpenAI chat shapes, unchanged.
- `tokens` is what the undertone tools read. Token ids are the ground truth; text is not, because the same messages
  become different tokens under a different chat template.
- `model` names the model's contract among the undertone tools, apart from whatever name the server uses. It is
  set by `--contract` (see below).
- A reply's `reasoning` (from a model that thinks first) is kept apart from its `content`. The next prompt in a
  conversation is built from the content alone.
- A conversation is the calls that share `conversation`, in `turn` order.

### Where the token ids come from

| `tokens` | Meaning |
| --- | --- |
| `"source": "server"` | The endpoint returned the ids it used. Ask for them with `--token-ids`. |
| `"source": "rebuilt"` | The endpoint returned none, and the engine tokenized the messages and the reply itself (with the contract's tokenizer, or `--tokenizer <name>`). |
| absent | Neither. The transcript is still kept, and the ids can be rebuilt later. |

Every transcript is accepted; the source is the flag. **Rebuilt ids are a reconstruction.** A tool that reads them
should check them against the model before trusting them.

- Asking for ids is off by default: it is not part of the OpenAI API, and a strict server rejects a field it does
  not know. With `--token-ids` the engine sends vLLM's `return_token_ids`; if the server refuses the request, the
  engine asks once more without it and stops asking that server.
- Rebuilding needs the tokenizer only, never the weights: `pip install -e ".[tokens]"`.
- A rebuilt transcript whose reply has reasoning keeps the prompt's ids and leaves the reply's out. How a model
  lays its reasoning out in tokens is that model's own, and belongs to its contract, not to the engine.
- `thinking` is what you declare about the endpoint (is the model served with thinking on?). It is recorded and used
  when rebuilding; it is not sent to the server.

### A model's contract

The undertone tools keep one `contract.json` for each model they support
([core/tools/models](../tools/models/README.md)): what the model is, pinned to an exact revision. Give the engine
one and it does three things with it:

```sh
undertone-engine serve --contract core/tools/models/qwen3.5-4b/contract.json
```

- every transcript records the contract's name as its `model`;
- ids the server does not return are rebuilt with the contract's own tokenizer at its pinned revision, so you do
  not name a tokenizer by hand (this still needs the `tokens` extra; without it the engine says so and keeps
  transcripts without ids);
- if the server's name for the model is not one the contract knows, the engine says so when it starts, since the
  tools will flag readings of those transcripts.

`--contract qwen3.5-4b` (the name alone) works when `core/tools/models` is installed. The engine reads the JSON
only: it still runs no model and needs no GPU.

## The local service

A chat page cannot call an endpoint itself (the key would sit in the page, and most endpoints refuse a browser), so
it talks to the engine, and the engine talks to the endpoint:

```sh
undertone-engine serve --port 8787
```

| Request | Answer |
| --- | --- |
| `GET /health` | the endpoint's host and model (both null while no endpoint is set) |
| `GET /conversations` | the conversations kept, most recently continued first |
| `GET /conversations/{id}` | every call of one conversation, as transcripts |
| `POST /chat` with `{"message", "conversation"?, "system"?, "params"?}` | the transcript of the call |
| `GET /settings` | which endpoint the engine talks to, where that came from, and what it knows of the model |
| `PUT /settings` with `{"base_url", "model", "token_ids"?}` | points the engine at another endpoint; the settings as they now are |
| `POST /endpoint/models` with `{"base_url"?}` | the names that endpoint (or the one in use) knows its models by |

### The endpoint can be set from the page

`serve` needs no endpoint to start. With none, the engine keeps its history and tells the page it has no endpoint,
and the page's settings set one. An endpoint comes from one of three places, in this order:

1. **Given at start:** `--base-url` and `--model`, or `UNDERTONE_BASE_URL` and `UNDERTONE_MODEL`.
2. **Kept:** `engine.json` in the transcripts folder, written whenever the endpoint is set through `PUT /settings`.
3. **None**, until one is set.

So a change made from the page lasts across restarts, and an endpoint given on the command line takes its place.
The contract and the tokenizer are chosen at start only.

**The API key is never a setting.** It is read from the environment, belongs to the endpoint the engine was started
with, and is not written anywhere. An endpoint set through `PUT /settings` is sent the key only if it is that same
host and port; any other is called without it, and `GET /settings` says so (`"api_key": "held back"`). Nothing a page
does can carry the key to another address. To use a key with a different endpoint, start the engine with that
endpoint.

### Who may call it

It is for one person on their own machine, and has no accounts. It listens on 127.0.0.1 only. It refuses a request
addressed to any other name (a page elsewhere can point its own name at 127.0.0.1), answers only pages served from
this machine, and refuses a request that changes anything when it comes from a page elsewhere, even one a browser
would send without asking first.

## Files

- `transcript.py`: the record.
- `endpoint.py`: one call to an OpenAI-compatible endpoint.
- `tokens.py`: what a tokenizer must offer to rebuild ids, and one over a Hugging Face tokenizer.
- `contract.py`: the few facts the engine reads from a model's contract.
- `store.py`: transcripts as files, one per conversation.
- `settings.py`: which endpoint the engine talks to: given at start, kept, or set from the page.
- `engine.py`: send, then keep.
- `service.py`: the local HTTP service.
- `cli.py`: the `undertone-engine` command.

## What is not here yet

- **Streaming.** A reply arrives whole.
- **A way to tell the server about thinking.** `thinking` is only recorded. A server that needs telling takes its
  own setting (with vLLM, `--default-chat-template-kwargs '{"enable_thinking": false}'` when it is started).
- **Checking the endpoint against the contract.** The engine compares the served name only. Nothing can tell from
  outside whether a server holds the pinned weights.

## Run against a real server

On 2026-11-18 the engine was run against vLLM 0.34.0 serving Qwen3.5-4B, with the Qwen contract:

- `--token-ids` works: vLLM returned the ids it used, for single and multi-turn conversations and with a system
  message, and they are exactly what the model's pinned template gives.
- Without `--token-ids`, the ids were rebuilt with the contract's tokenizer, and agree with what the tools rebuild.
- With thinking on and no reasoning parser on the server, the reasoning arrives inside `content`. The engine keeps
  what the server sent; the tools flag such a transcript.

One server, one model, a handful of calls.

## Tests

```sh
pip install -e ".[dev]"
pytest
```

They use an in-memory endpoint: no server, no network and no model.
