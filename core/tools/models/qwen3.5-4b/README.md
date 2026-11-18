# Qwen3.5-4B

The first model the undertone tools support. Everything here is bound to one exact copy of it:

| | |
| --- | --- |
| Model | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Files | 11, about 9.3 GB, each pinned to a SHA-256 in `contract.json` |
| Shape | 32 blocks, width 2560: three Gated DeltaNet blocks, then one full-attention block, repeated |
| Read at | each block's output |
| Last step | RMSNorm scaled by (1 + weight), then the output embedding, which is tied to the input embedding |
| Text | the model's own chat template, **thinking off**, up to 2,048 tokens |
| Precision | bfloat16 weights; float32 for reading; SDPA attention; no KV cache; TF32 off |

## Depths

The four depths the tools and the page speak of are these blocks, counted from 0:

| Depth | Blocks |
| --- | --- |
| early | 8 to 15 |
| middle | 16 to 23 |
| late | 24 to 30 |
| output | 31 |

They are coarse bands chosen before any result was looked at. Blocks 0 to 7 are in none of them.

## Lenses

`contract.json` pins two lens files made for this model by others. [core/tools/jlens](../../jlens/README.md) reads
captures through them; the contract only says which files they are.

| Name | File | Size |
| --- | --- | --- |
| `jpp` | [J++ lens](https://huggingface.co/neuronpedia/jacobian-lens/blob/d2527c3cdbaa60dd4b65cbca1604921f044e989a/qwen3.5-4b/jpp/Salesforce-wikitext/CREDIT.md) | 406 MB |
| `jlens` | Jacobian lens (1,000 samples), kept as a comparison | 406 MB |

Both carry one map for each of blocks 0 to 30, taking that block's output to block 31, stored in float16 and read in
float32. The J++ maps were fitted by a
[contributor](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses/blob/c1b9747f3861977df31482adc142eb3b38811546/provenance/fit-recipe.json)
on this same model revision and are published by Neuronpedia. The contributor reports 57.2% recall at 10 for them.
That figure is theirs and has not been reproduced here.

## How it was qualified

On one NVIDIA A40 with Python 3.12.3, torch 2.14.0, transformers 5.20.2, safetensors 0.8.0, accelerate 1.13.0 and
huggingface-hub 2.4.0, reading 36 stories once each and checking three positions a story:

| Check | Limit | Worst seen |
| --- | --- | --- |
| The template and the tokenizer give the same token ids | exact | exact |
| `head` agrees with the model's own norm and head in float32 | 0.0001 | 0.0 |
| `head`'s distribution against the model's native bfloat16 logits (total variation) | 0.05 | 0.0215 |
| Reading the same text twice gives the same block outputs | exact | exact |

`check.py` runs the same four checks, on one short fixed exchange, against whatever is installed:

```sh
pip install -e "core/tools/models[run]"
undertone-models check qwen3.5-4b                      # fetches the pinned files (about 9.3 GB), hashes them, loads, tests
undertone-models check qwen3.5-4b --snapshot <folder>  # a folder that already holds them
```

It prints a report and which packages differ from the ones above. It needs a GPU with room for the model (the
qualifying runs peaked at about 13.5 GB).

## What was not qualified

- **Agreement with any serving endpoint.** The tools re-run the text through this model; they do not read the
  server's own activations. Nothing here says a vLLM, llama.cpp or Ollama build of the model computes the same
  numbers.
- **Any other runtime, package version, quantisation or weights.**
- **Thinking on.** A transcript made with thinking on is read and flagged, not refused.
- **Tool calls.**
- **Texts longer than 2,048 tokens.** A long conversation passes that; it is read and flagged.
- **How well emotions are recognised, how good a lens is, or steering.** Those belong to the tools.

## Run from this repository

The numbers above come from the development work the adapter was ported from. The adapter itself was then run on
2026-11-18, on one NVIDIA L4 with the same package versions:

- `undertone-models check qwen3.5-4b` passed. All 11 files matched the contract's hashes, and on the fixed exchange
  `head` matched the model's own float32 head exactly, with a total variation of 0.0029 against its native logits
  (limit 0.05). Two reads gave identical block outputs.
- `render` was run against the real tokenizer on single and multi-turn exchanges, a system message, one-letter
  messages, non-English text and an empty reply: every message landed on its own text.
- **Qwen's template trims the edges of a message.** A reply stored as `"\n\nHello there.  "` is read as
  `"Hello there."`. `render` places the trimmed text and says so (`message_trimmed`).
- `load_head` gives exactly the same logits as the full model's `head` on every captured token tried.
- With vLLM 0.34.0 serving these weights (thinking off), the token ids it returned were exactly what the pinned
  template gives, so those transcripts carry no flags. With thinking on they differ, and are flagged.
- With the newest packages `pip` chose that day (torch 2.15.0, transformers 5.23.0, accelerate 1.16.0) the self-test
  passed with the same figures, and 19 readings made under them were identical to the ones made under the
  qualified versions.

That is one machine and a handful of texts. It shows the port works; it does not widen what was qualified.
