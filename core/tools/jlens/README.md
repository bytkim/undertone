# undertone-jlens

The words the model leans toward at each depth, token by token, read from a capture.

A capture ([core/tools/capture](../capture/README.md)) holds what every block of the model put out at each token of
a reply. On its own that is 2,560 numbers a block. A Jacobian lens makes it readable: it carries a block's output
to the end of the model, where the model's own last step turns it into words. What comes out is what the model
would go on to say if what it holds at that depth were all it had.

```text
captures/  ->  undertone-jlens  ->  lens/
                                      lens.json               the run
                                      <transcript id>.json    the words, token by token and for the whole reply
```

## Start

Requires Python 3.12 or newer. A GPU makes it quick; it is not needed (`--device cpu`), because the model itself is
never run here.

```sh
pip install -e "core/tools/models[run]" -e core/tools/capture -e core/tools/jlens
```

```sh
undertone-jlens run captures/ --out lens/
undertone-jlens show lens/
undertone-jlens show lens/ --transcript <transcript id>
```

The first run fetches the lens the model's contract pins (about 400 MB) and proves it is that file by its hash. The
model's files are needed too (for its last step and its tokenizer); `--snapshot <folder>` uses a copy you already
have, such as the one `undertone-capture` fetched.

## How a reading is made

For one captured token and one depth:

1. Each block of that depth is carried to the last block by the lens's map for it. The last block needs no map.
2. The model's own last step (its final norm and output embedding, in float32) turns that into a probability for
   every piece of the vocabulary.
3. The pieces that spell the same word are added up. " quiet", "quiet" and " Quiet" are one word.
4. The depth's blocks are averaged.

The whole reply is the average over its tokens. The four depths are the model's contract's: for Qwen3.5-4B, early is
blocks 8 to 15, middle 16 to 23, late 24 to 30, and output is block 31, the model's actual next-word prediction.

### What counts as a word

A piece of the vocabulary counts when it is letters and digits only (an apostrophe or hyphen may stand inside it),
and pieces are one word when they match once trimmed and lower-cased. For Qwen3.5-4B that makes 169,939 word types
out of 248,320 pieces. It is a rule about spelling, not a dictionary: fragments of longer words ("noct", "ness")
and words of other languages are word types too, and they do turn up.

## What a reading holds

`<transcript id>.json` (`undertone.lens.v1`):

| Field | Meaning |
| --- | --- |
| `transcript`, `conversation`, `turn` | which call this is |
| `model`, `lens` | the model revision, and the lens file by repository, revision and hash |
| `capture` | the hash of the capture that was read |
| `flags`, `qualified` | carried over from the capture: whatever put the transcript outside the qualified setting |
| `ranked_by`, `baseline` | how the words were ranked (below) |
| `overall` | for the whole reply: at each depth, the top words, most leaned-toward first |
| `tokens` | for each token in order: its text, and the top words at each depth |

A word is `{"word", "p"}`: `p` is its mass, the share of the model's probability that falls on that word (0 to 1).
With a baseline it also has `lift`.

## Ranking: against a baseline, or by probability

Words can be ranked two ways, and a reading says which (`ranked_by`).

**Against a baseline** (the default, where one comes with the tool): a word ranks by how far it stands above its
usual mass, weighted by its mass, `mass × log(mass / usual mass)`. The usual mass is the average over many tokens
of ordinary text. This says what the model leans toward *here* more than it usually does.

**By probability** (`--baseline none`): the raw reading, what the model would most likely say. At the early and
middle depths it is usually on topic already. At the output depth it is the model's next-word prediction, so it is
led by "the", "a" and "of".

```sh
undertone-jlens run captures/ --out lens/                     # against the baseline that comes with the tool
undertone-jlens run captures/ --out lens-raw/ --baseline none # by probability
```

### The baseline that comes with the tool

[`baselines/qwen3.5-4b-jpp/`](baselines/qwen3.5-4b-jpp/baseline.json) is for Qwen3.5-4B with the J++ lens: the
word masses averaged over all 41,957 message tokens of the repository's 360 neutral dialogues
([stories/corpus/neutral_dialogues](../../../stories/README.md)), made on 2026-11-18 on one NVIDIA L4. It is one
tensor, `[4 depths, 169,939 word types]` in float32 (2.7 MB), with a record of the model revision, the lens file
and the vocabulary it belongs to.

It is tied to those three. With another model revision, another lens file, or an install whose tokenizer spells the
vocabulary differently, it does not fit. The tool then says so and ranks by probability; a baseline that does not fit is never used.
"Usual" here means matter-of-fact questions and answers written by three other models. That is a choice, and not
the only reasonable one.

### A baseline of your own

```sh
undertone-jlens baseline captures-of-ordinary-text/ --out mine/ --note "what these captures are of"
undertone-jlens run captures/ --out lens-mine/ --baseline mine/
```

What "usual" means is whatever you captured for it. The one above was made from the neutral dialogues: each
becomes a transcript whose reply is its last assistant turn, and
`undertone-capture run ... --select messages --blocks 8-31` captures every message's tokens at the blocks the
depths need.

```python
import json, pathlib

out = pathlib.Path("transcripts-neutral"); out.mkdir(exist_ok=True)
for file in sorted(pathlib.Path("stories/corpus/neutral_dialogues").glob("*.jsonl")):
    lines = []
    for row in map(json.loads, file.read_text().splitlines()):
        turns = [{"role": t["role"], "content": t["text"]} for t in row["turns"]]
        lines.append({"schema": "undertone.transcript.v1", "id": row["story_id"], "conversation": file.stem,
                      "turn": len(lines) + 1, "request": {"messages": turns[:-1]}, "response": {"content": turns[-1]["content"]}})
    (out / file.name).write_text("".join(json.dumps(line) + "\n" for line in lines))
```

A baseline you name is refused if it was made with another model revision, lens file or vocabulary. A folder of
readings holds one ranking; another ranking goes in another folder.

## What a reading is, and is not

- **It is a way of looking, not a measurement of what the model thinks.** A lens is a fitted approximation of how
  the model carries a block's output forward. The J++ lens was fitted by a contributor on other text; they report
  57.2% recall at 10 for it, which has not been reproduced here.
- **Only the output depth is the model's own answer.** There no map is used: it is the model's real next-word
  prediction. The other three depths pass through the lens.
- **Words are a narrow window.** What falls on punctuation, spaces and markers is left out, so a token's words add
  up to less than 1, sometimes much less.
- **It reads a capture, so it inherits the capture's flags.** A reading of a transcript outside the qualified
  setting says so.
- **It says nothing about emotions.** That is another tool's.

## Run on the real model

On 2026-11-18, on one NVIDIA L4, the J++ lens read six real replies (Qwen3.5-4B served by vLLM, chatted with
through the engine, then captured).

- Both lens files matched the hashes in the contract.
- `load_head` gave exactly the same logits as the full model on every captured token.
- **The output depth is the model's next word.** Where the next token said was a word, the output depth's first
  word was that word 352 times out of 358.
- The words are on topic. For the reply to "And at night?" ("At night, the scene transforms into a study in
  contrasts and quiet mystery..."), by probability:

  | Depth | Whole reply |
  | --- | --- |
  | early | night, street, light, sky, lighting, lights, hurried, color |
  | middle | night, sky, 夜空, street, replaced, headlights, twilight, darkness |
  | late | replaced, light, street, the, sky, casts, headlights, casting |
  | output | the, a, by, of, replaced, sky, pavement, that |

  and token by token, at the middle depth: "At" reads *night, nighttime, nights* before the word is said; " scene"
  reads *scene, shifts, changes*; " study" reads *of, contrasts, study*.

- **The baseline does what it should, with one side effect.** Against the baseline that comes with the tool, the output depth of the same reply reads *pavement, pools, replaced,
  contrasts, casting* in place of *the, a, by, of*. It also brings up fragments (*igo*, *ety*, *rying*): the ends
  of words being written, which are rare in ordinary text and so stand far above usual. At the early and middle
  depths the two rankings mostly agree.
- In that baseline, the share of probability that falls on words at all is 0.81 early, 0.89 middle, 0.87 late and
  0.69 at the output.
- Speed: about 50 tokens a second on the L4, in float32. The six replies (510 tokens) took 10 seconds; the baseline
  took about 13 minutes.
- Readings repeat, but not to the last digit on a GPU: two runs over the same captures gave the same words in the
  same order in all 2,064 lists, and one probability that differed in its fourth digit.

Six replies, one lens, one GPU. This shows the tool works and the lens reads sensibly on this model. It is not an
evaluation of the lens.

## From Python

```python
from undertone_models import adapter, contract
from undertone_jlens import Reader, load_lens, run, shipped_baseline

qwen = contract("qwen3.5-4b")
reader = Reader(adapter(qwen).load_head(qwen), load_lens(qwen, "jpp"))
run(reader, "captures/", "lens/", baseline=shipped_baseline(reader))   # baseline=None ranks by probability
```

`reader.read(record, blocks)` reads one capture and answers the reading without writing it.
`reader.masses(record, blocks)` gives the full word masses, a few tokens at a time, for a tool that wants more than
the top words.

## Checks

```sh
cd core/tools/jlens
pip install -e ../models -e ../capture -e ".[dev]" torch
pytest
```

The tests use the stand-in model from `undertone_models.standin` and a made-up lens file: no network, no GPU, no
weights.
