# stories

The prompts and the stories undertone was developed with, as plain files.

There is no code here and nothing in this repository writes stories for you. Read the corpus as it is, or send the
prompts to any model you like and make your own.

```text
stories/
  prompts/
    emotion_story.txt       the prompt for one emotion story
    neutral_dialogue.txt    the prompt for four emotionless dialogues
    emotions.json           the 171 emotion words
    topics.json             the 30 situations
    task_types.json         the five kinds of request a neutral dialogue is about
  corpus/
    LICENSE                 CC0 1.0: the corpus is given to the public domain
    manifest.json           every file's hash and count, and how each author was called
    stories/                15,390 emotion stories: one file an author, 5,130 each
    neutral_dialogues/      360 emotionless dialogues: one file an author, 120 each
```

## What it is for

The stories are the raw material for emotion vectors in the manner of Anthropic's emotion concepts work
([overview](https://www.anthropic.com/research/emotion-concepts-function),
[paper](https://transformer-circuits.pub/2026/emotions/index.html)): many short stories in which a character feels
one emotion without the emotion being named, so that what the stories for one emotion share inside a model is that
emotion and not a word. The neutral dialogues are for the denoising step: what emotionless text has in common is
removed from the vectors.

The 171 emotion words are the paper's list. The prompts and the 30 situations were written for this project in the
style the paper describes; they are not copied from it.

## Prompts

`emotion_story.txt` has four places to fill:

| Place | Filled with |
| --- | --- |
| `{n}` | how many stories to ask for. The corpus used `1`. |
| `{topic}` | a situation's `text` from `topics.json` |
| `{emotion}` | a word from `emotions.json`, exactly as written there |
| `{narration}` | `first` or `third`. The corpus chose one for each emotion and situation, about half each, separately for each author. |

`neutral_dialogue.txt` has three: `{n}` (the corpus used `4`), `{topic}`, and `{task_types}`, a line that gives each
dialogue one of the five kinds in `task_types.json`, written as
`dialogue 1: <kind>; dialogue 2: <kind>; dialogue 3: <kind>; dialogue 4: <kind>`.

Both ask for JSON back: `{"stories": ["..."]}` and `{"dialogues": [{"turns": [{"role", "text"}]}]}`.

Filling one in takes a few lines in any language. In Python:

```python
import json, pathlib

prompts = pathlib.Path("stories/prompts")
template = (prompts / "emotion_story.txt").read_text().rstrip("\n")
emotions = json.loads((prompts / "emotions.json").read_text())
topics = json.loads((prompts / "topics.json").read_text())

prompt = (template.replace("{n}", "1").replace("{topic}", topics[0]["text"])
          .replace("{emotion}", emotions[0]).replace("{narration}", "third"))
```

171 emotions by 30 situations is 5,130 prompts. Each was sent as a single user message.

These are the prompts the corpus answered, to the character: every row records the SHA-256 of the prompt it was
written for (`generation.prompt_sha256`), and filling the template with that row's own `topic`, `emotion` and
`narration` gives that hash for all 15,390 stories and all 360 dialogues.

A few of the emotion words are spelled with spaces around the hyphen (`grief - stricken`, `self - confident`,
`self - conscious`, `self - critical`), as in the list they come from. They went into the prompts that way and are
kept that way in the corpus.

## Corpus

Three models each wrote the whole grid, so no emotion rests on one author's habits:

| File | Author, as recorded when it was called | Called through | Written |
| --- | --- | --- | --- |
| `gpt-6-luna.jsonl` | `gpt-6-luna` | OpenAI's chat completions API | 2026-11-11 |
| `claude-sonnet-5-5.jsonl` | `claude-sonnet-5-5` | the Claude Code command line, with no tools or settings | 2026-11-12 |
| `gemini-3.8-flash.jsonl` | `gemini-3.8-flash` | Google's OpenAI-compatible endpoint | 2026-11-12 |

Each line of a file is one JSON object. A story:

| Field | Meaning |
| --- | --- |
| `story_id` | `{emotion}.{version}.{topic_id}.{n}`. The version names the author: `v1` is GPT, `sonnet-v1` Claude, `gemini-v1` Gemini. Spaces in the emotion are `_` here. |
| `author` | the model that wrote it |
| `emotion` | the emotion word, as in `emotions.json` |
| `topic_id`, `topic` | the situation, as in `topics.json` |
| `narration` | `first` or `third` |
| `text` | the story: one paragraph |
| `words` | its length in words (101 to 187 across the corpus; about 150 is typical) |
| `other_emotion_words` | words from the 171 that appear in the text, other than the story's own. Recorded, not forbidden. Matching is by word stem, so it over-counts ("alarm" counts as "alarmed"). |
| `generation` | `seed`, `prompt_sha256`, and `attempts` (2 where the first try failed a check) |

A neutral dialogue has `story_id`, `author`, `topic_id`, `topic`, `words`, `other_emotion_words` and `generation`
as above, and:

| Field | Meaning |
| --- | --- |
| `turns` | the dialogue: `{role, text}` in order, `user` first and `assistant` last |
| `text` | the same dialogue as one string, each turn on a line beginning `User:` or `Assistant:` |
| `task_type` | which of the five kinds of request it is |

The stories are in the order of `emotions.json`, and within an emotion in the order of `topics.json`. To read one
author:

```python
import json

stories = [json.loads(line) for line in open("stories/corpus/stories/gpt-6-luna.jsonl")]
```

### What was checked when they were written

A story was kept only if it was one paragraph, 60 to 220 words long, and did not use its own emotion word or a
simple form of it.
One that failed was asked for once more. That happened for 24 of GPT's stories, 64 of Claude's and 35 of Gemini's.
Nothing checked whether a story conveys its emotion well; nobody has read all of them.

A neutral dialogue was kept only if its turns alternate from user to assistant, it runs to 1 to 6 exchanges and 20
to 450 words, it uses none of the 171 emotion words in their exact forms, apart from four with an everyday sense
(content, alert, safe, patient), and it has none of a short list of stock pleasantries ("thanks", "happy to help",
"great question" and the like).

### Things to know before relying on it

- **The authors were not called in quite the same way.** GPT and Gemini got the prompt alone. Claude was reached
  through the command line, which also needs a system message: "You are a fiction writer. Follow the request
  exactly and return only the requested JSON." Sampling settings differ too. `manifest.json` has each author's.
- **Only GPT was sent the seed** each row records. Gemini's endpoint and the Claude command line take none, so for
  those two the seed is a label, and no author's stories can be reproduced by asking again.
- **One story answers a different situation.** Gemini refused `aroused` with "a parent waiting outside a school at
  pickup time" every time, so `aroused.gemini-v1.t015.1` was written for "a person waiting outside a train station
  to meet someone off the evening train". Its `topic` says so and its `topic_id` is still `t015`;
  `manifest.json` lists it under `substituted_topics`.
- **The neutral dialogues are short and uneven between authors.** Most of GPT's are a single exchange of about 40
  words; Claude's are mostly two exchanges of about 120.
- **Many stories name some other emotion.** Roughly half of each author's stories contain a word from the list
  that is not their own, by the generous stem matching above.
- **All of it is written by models.** No person wrote or edited any story.

## Terms

The corpus (`corpus/`) is given to the public domain under [CC0 1.0](corpus/LICENSE). To the extent the author of
this repository holds any rights in these texts, they are waived: you may copy, change and use them for any purpose
without asking.

That is all it can promise. Every story and dialogue was written by a language model and no person wrote or edited
any of them, so there may be no copyright in them to waive in the first place. The terms of each model's provider
applied to the account that generated them; this repository passes none of them on, and says nothing about what
they allow. The texts come as they are, with no warranty and no claim that any of them is accurate or original.

The prompts (`prompts/`) were written for this project and are under the repository's
[Apache License 2.0](../LICENSE). `emotions.json` is the list of 171 emotion words from Anthropic's emotion concepts
work, credited in [`NOTICE`](../NOTICE).

## Checking the files

`corpus/manifest.json` holds the SHA-256 and row count of every corpus file and the SHA-256 of every prompt file.

```sh
shasum -a 256 stories/corpus/stories/*.jsonl stories/corpus/neutral_dialogues/*.jsonl stories/prompts/*
```
