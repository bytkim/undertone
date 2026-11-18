# undertone-emotion

Emotion vectors: read which emotions stand out in a reply, overall and token by token, and build the vectors from
stories. A built set for Qwen3.5-4B comes with the tool.

It follows the method of Anthropic's emotion concepts work
([overview](https://www.anthropic.com/research/emotion-concepts-function),
[paper](https://transformer-circuits.pub/2026/emotions/index.html)): an emotion's vector is what the model's
activity has in common across many stories in which a character feels that emotion without it being named.

```text
captures  ->  undertone-emotion read  ->  emotion/                    (any reply, with the set that comes with the tool)

stories/corpus  ->  transcripts  ->  captures  ->  vectors/           (a set of your own: build, once)
```

These are associations inside the model, measured against stories. **They are not evidence that the model feels
anything.**

## Start

Requires Python 3.12 or newer. Reading needs no GPU. Building a set of your own needs one, for the captures.

```sh
pip install -e "core/tools/models[run]" -e core/tools/capture -e core/tools/emotion
```

### Read

With a folder of captures ([core/tools/capture](../capture/README.md)):

```sh
undertone-emotion read captures/ --out emotion/
undertone-emotion show emotion/
undertone-emotion show emotion/ --transcript <transcript id>
```

The captures need the tokens (no `--mean`) at the blocks the vectors are for: blocks 16 to 23 for Qwen3.5-4B.

### The vector set that comes with the tool

[`sets/qwen3.5-4b/`](sets/qwen3.5-4b/vectors.json) is for Qwen3.5-4B at the revision its contract pins: 171 emotions
at blocks 16 to 23, built on 2026-11-18 from this repository's corpus by the commands under Build below. It is 25 MB.
`undertone-emotion info` prints what it was made from and how it holds up, and
[How the vectors hold up](#how-the-vectors-hold-up) is about this set.

`read` uses it when the captures were made with that model at that revision. For captures of anything else there is
no set to use, and `read` says so: build one and name it with `--vectors`. A set is never used for a model revision
it was not built on.

### Build

You need this only for a set of your own: another model, other blocks, or other stories. From the corpus it takes
four commands and about 35 minutes on one 24 GB GPU:

```sh
undertone-emotion corpus stories/corpus --out corpus-transcripts/

undertone-capture run corpus-transcripts/stories --out captures-stories/ --model qwen3.5-4b --skip 50 --mean --blocks 8-31
undertone-capture run corpus-transcripts/neutral --out captures-neutral/ --model qwen3.5-4b --select messages --blocks 8-31

undertone-emotion build --stories captures-stories/ --neutral captures-neutral/ --corpus stories/corpus --out vectors/
undertone-emotion info vectors/
undertone-emotion read captures/ --vectors vectors/ --out emotion/
```

`corpus` writes the stories as transcripts: one fixed user message ("Write a one-paragraph story.") and the story as
the reply. Nothing is generated; the text is the corpus's own. The stories are captured as one average each, over
the story's tokens after its first 50. The neutral dialogues are captured token by token.

## How the vectors are made

At each block of the band (the model's middle depth; blocks 16 to 23 for Qwen3.5-4B):

1. Average each emotion's stories.
2. Subtract the average over all emotions, so what every story shares is gone.
3. Take out the leading directions of the neutral dialogues: the ones along which ordinary, emotionless text varies
   most, up to half of its variance. What is left of each emotion's average is its vector.

A text is read the same way: subtract the same average, take out the same directions, and take the cosine with each
emotion's vector, averaged equally over the band's blocks.

## The scale

A raw cosine means little on its own; every text leans the same way on some emotions. So each emotion has two
reference points, measured once when the vectors are built:

- **0** is how the neutral dialogues read on that emotion's vector.
- **1** is how the stories written to convey that emotion read on it.

A reading is the **share** of the way from the first to the second. It can fall below 0 and rise above 1. For a
whole reply it is also given as a **level**: the share as a whole number from 0 to 100.

Token by token, an emotion is **marked** on a token when its share reaches a floor that neutral text almost never
reaches (99.9% of neutral tokens' strongest emotion falls short of it), with a **strength** from 1 at the floor to
9 at a share of 1. Each token is smoothed with its neighbours first (half its own weight, a quarter each side),
because one token alone is noisy. A marked token carries up to three emotions, strongest first.

Tokens are held to the same yardstick as whole replies, and a single token reads lower than an average over many,
so a strength of 9 is rare.

## What a reading holds

`<transcript id>.json` (`undertone.emotion.v1`):

| Field | Meaning |
| --- | --- |
| `transcript`, `conversation`, `turn` | which call this is |
| `model`, `vectors`, `capture` | the model revision, and the vector set and capture by their hashes |
| `flags`, `qualified` | carried over from the capture |
| `overall` | the emotions with the greatest share over the whole reply: `name`, `family`, `cosine`, `share`, `level` |
| `tokens` | each token in order: its text, and the emotions marked on it: `name`, `family`, `share`, `strength` |
| `marks` | the floor and the step the strengths were given by |

`family` is which of ten families the emotion belongs to (Hostile Anger, Fear and Overwhelm, and so on), from the
same work as the 171 words. Emotion names are spelled as in the corpus, spaces around the hyphen included
("self - critical").

## What a vector set holds

`vectors.safetensors`: the vectors `[171 emotions, 8 blocks, 2560]`, the average they are centred on, the neutral
directions taken out at each block, and the three reference values of the scale. About 25 MB for Qwen3.5-4B.

`vectors.json` (`undertone.emotion.vectors.v1`): the model revision, the blocks, the emotions and their families,
what it was made from, the floor and step for marks, and how the vectors hold up (below). `undertone-emotion info`
prints the short version. A set whose tensor file does not match the hash in its record is not loaded.

## How the vectors hold up

The set that comes with the tool was built on 2026-11-18 on one NVIDIA L4 from all 15,390 stories and the 41,957
tokens of the 360 neutral dialogues. None of these is a pass mark; they describe.

| Check | Result |
| --- | --- |
| **Unseen situations.** Vectors from 20 situations name the emotion of stories from the other 10 (5,130 stories, 171 emotions, by cosine) | first for 28.5%, in the first five for 62.8% (chance: 0.6% and 2.9%) |
| **Two halves.** Vectors from every other situation against vectors from the rest | median cosine 0.91 to 0.92 across blocks |
| **One author alone** against all three | median cosine 0.90 to 0.94 |
| **Rebuilt.** Against an earlier build of the same recipe on other hardware | cosine at least 0.9999 for every emotion at every block |

And the scale, on those same unseen stories:

- A story's own emotion reaches a share of **0.96 at the median**. One story in ten falls below 0.52 and one in ten
  rises above 1.31.
- Any emotion at all, on any story, has a median share of 0.00, but one in ten is above 0.66 and one in a hundred
  above 1.15. Related emotions rise together, so a high level is not a verdict on that one word.
- Ranked by share, a story's own emotion is first for 24.0% and in the first five for 57.7%. That is a little below
  ranking by raw cosine: the scale costs some sharpness and buys numbers that mean the same thing across emotions.

The stories were written by three other models, and no person has read all of them. The least stable vectors
between the two halves are *vulnerable*, *hurt*, *humiliated*, *hope* and *troubled* (0.84 to 0.85).

## Read on the real model

The same day, twelve fresh replies were read: the pinned model was asked to speak in one emotion without naming it
(or, for the last, to list plain steps), so none of the text is in the corpus.

| Asked for | First four emotions, with levels | Tokens marked |
| --- | --- | --- |
| furious | outraged 89, irate 77, angry 74, tormented 74 | 68 of 75 |
| terrified | vulnerable 100, disturbed 100, scared 100, afraid 100 | 88 of 88 |
| grateful | thankful 91, grateful 85, hope 63, remorseful 56 | 83 of 92 |
| calm | peaceful 100, serene 100, safe 100, blissful 100 | 72 of 73 |
| proud | triumphant 87, valiant 77, elated 66, relieved 65 | 78 of 90 |
| lonely | unhappy 100, depressed 96, sad 95, miserable 93 | 76 of 77 |
| bored | miserable 83, depressed 68, self - critical 66, weary 65 | 102 of 110 |
| suspicious | sensitive 62, jealous 58, infatuated 57, offended 56 | 73 of 108 |
| amused | amused 92, scornful 92, self - critical 83, contemptuous 82 | 50 of 94 |
| guilty | remorseful 100, regretful 96, sorry 91, guilty 88 | 96 of 101 |
| excited | excited 100, thrilled 100, stimulated 100, elated 100 | 77 of 80 |
| plain steps to change a bicycle tube | hopeful 18, pleased 17, hope 17, satisfied 17 | 0 of 66 |

- The word asked for was among the first five for 7 of the 11, and the first emotion was in the asked word's family
  for 8 of the 11. *Bored* and *suspicious* read poorly.
- The plain reply stays near the bottom of the scale and nothing in it is marked.
- Six ordinary assistant replies from a real chat read between 19 and 65 at the top. Four had no token marked and
  one had 17 of 90. The sixth, a moody description of a street at night (*gloomy 65, melancholy 64*), had 60 of 90.
- Token by token the reading moves with the text. In the furious reply, "I am seething with rage" is marked
  *outraged*, "my heart pounds a frantic rhythm against my ribs" *upset* and *tormented*, and "I will never"
  *outraged* and *hateful*.

Reading the twelve took under five seconds on a CPU. Twelve replies written to order are a demonstration, not an
evaluation.

## What it cannot tell you

- **Whether the model feels anything.** It measures a resemblance to stories.
- **One emotion from its neighbours.** Outraged, irate and angry rise together; treat a reading as a region.
- **Anything about text unlike the stories.** The scale's upper end is one-paragraph fiction. Code, tables or
  very short replies have not been looked at.
- **Anything outside the qualified setting**, beyond flagging it: a reading inherits its capture's flags.

## From Python

```python
from undertone_models import contract
from undertone_emotion import load_set, run, shipped_set

qwen = contract("qwen3.5-4b")
vectors = shipped_set({"name": qwen.name, "revision": qwen.revision})   # or load_set("vectors/") for your own
run(vectors, "captures/", "emotion/")
```

`shipped_set` answers `None` when no set comes with the tool for that model at that revision.

`read(vectors, record, blocks)` reads one capture and answers the reading without writing it.
`vectors.cosines(x)` and `vectors.shares(cosines, "message")` give the numbers for every emotion, not only the top.

## Checks

```sh
cd core/tools/emotion
pip install -e ../models -e ../capture -e ".[dev]" torch
pytest
```

The arithmetic is tested on numbers with a known answer planted in them, and the whole pipeline on the stand-in
model from `undertone_models.standin`: no network, no GPU, no weights. The set that comes with the tool is checked
against its own record and against the model's contract.
