"""Build a set of emotion vectors from captures of the story corpus, and keep it as two files.

What goes in:

    captures of the stories          one average per story (undertone-capture ... --skip 50 --mean)
    captures of the neutral dialogues every message's tokens (undertone-capture ... --select messages)
    the corpus itself                 which emotion, situation and author each story is

What comes out, in one folder:

    vectors.safetensors   the vectors, the average they are centred on, the neutral directions taken out,
                          and how each emotion usually reads on neutral text
    vectors.json          what they were made from and how, and how well they hold up (undertone.emotion.vectors.v1)

The last of those is what gives a reading its scale. A raw cosine means little on its own: every text leans the
same way on some emotions. So two reference points are measured here, once, for each emotion: how neutral text reads
on its vector, and how the stories written to convey that emotion read on it. A reading is then a share of the way
from the first to the second.
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from undertone_capture import items as capture_items, read as read_capture
from undertone_models import contract as find_contract

from . import vectors as V
from .corpus import STORY_PROMPT, labels as corpus_labels

SCHEMA = "undertone.emotion.vectors.v1"
#: a token's emotions are smoothed with its neighbours before anything is read off them: one token is noisy
SMOOTH = (0.25, 0.5, 0.25)


class BuildError(ValueError):
    """Captures that cannot be made into vectors, or a vector set that is not what its record says."""


@dataclass
class VectorSet:
    """A set of emotion vectors, loaded."""

    record: dict[str, Any]
    vectors: Any       # [E, B, d]
    grand_mean: Any    # [B, d]
    basis: list[Any]   # per block [k, d]
    usual: dict[str, Any]   # token, message, own: each [E] (see `shares`)

    @property
    def emotions(self) -> list[str]:
        return self.record["emotions"]

    @property
    def blocks(self) -> list[int]:
        return self.record["blocks"]

    def cosines(self, x: Any) -> Any:
        """x [N, B, d] at this set's blocks -> the band's cosine with each emotion [N, E], in float32."""
        return V.band_similarity(x, self.vectors, self.grand_mean, self.basis).float()

    def shares(self, cosines: Any, of: str) -> Any:
        """Cosines [N, E] as shares of the way from neutral text (0) to that emotion's own stories (1). `of` says
        what the cosines are of, and so which neutral value they start from: "message" (an average over a whole
        reply) or "token" (single tokens, which read differently from averages)."""
        neutral, own = self.usual[of], self.usual["own"]
        return (cosines - neutral) / (own - neutral).clamp_min(1e-6)


def families() -> dict[str, Any]:
    """Which of the ten families each emotion word belongs to."""
    return json.loads((Path(__file__).parent / "families.json").read_text())


def smooth(z: Any) -> Any:
    """z [tokens, E] -> each token with half the weight and its neighbours a quarter each (the ends repeat
    themselves). Fewer than three tokens are left as they are."""
    import torch

    if z.shape[0] < 3:
        return z
    pad = torch.cat([z[:1], z, z[-1:]])
    return SMOOTH[0] * pad[:-2] + SMOOTH[1] * pad[1:-1] + SMOOTH[2] * pad[2:]


def build(stories: str | Path, neutral: str | Path, corpus: str | Path, out: str | Path, blocks: list[int] | None = None,
          variance: float = V.VARIANCE, mark_quantile: float = 0.999, log: Any = None) -> dict[str, Any]:
    """Make a vector set in the folder `out` and answer its record.

    blocks         which blocks to build for (default: the model's middle depth, which is the band read with)
    variance       the share of neutral variance whose directions are taken out
    mark_quantile  a token is marked when its strongest emotion stands further toward its stories than this share
                   of neutral tokens' strongest do: neutral text is then almost never marked
    """
    import torch
    from safetensors.torch import save_file

    say = log or (lambda line: None)
    out = Path(out)
    if (out / "vectors.json").exists():
        raise BuildError(f"{out} already holds a vector set.")
    story_records = {r["transcript"]: r for r in capture_items(stories)}
    if not story_records:
        raise BuildError(f"No captures in {stories}.")
    first = next(iter(story_records.values()))
    model = first["model"]
    band = list(blocks) if blocks is not None else list(find_contract(model["name"]).depths["middle"])

    # the stories: one average each, in the corpus's own order
    labelled = corpus_labels(corpus)
    ids = [tid for tid in labelled if tid in story_records]
    if not ids:
        raise BuildError("None of the captured stories is in the corpus.")
    rows_of = _rows(first, band)
    X = torch.empty(len(ids), len(band), first["tensor"]["shape"][-1])
    for n, tid in enumerate(ids):
        record = story_records[tid]
        if not record["mean"] or record["select"] != "reply" or record["skip"] != first["skip"] or record["model"] != model or record["blocks"] != first["blocks"]:
            raise BuildError(f"{tid} was not captured like the others: stories are captured as one average each (--mean), all the same way.")
        _, tensor = read_capture(stories, tid)
        X[n] = tensor[rows_of].float()
        if (n + 1) % 2000 == 0:
            say(f"stories read: {n + 1} of {len(ids)}")
    emotion = [labelled[t]["emotion"] for t in ids]
    topic = [labelled[t]["topic"] for t in ids]
    author = [labelled[t]["author"] for t in ids]

    # the neutral text: its leading directions at each block, gathered a dialogue at a time
    neutral_records = capture_items(neutral)
    grams = [V.Gram(X.shape[-1]) for _ in band]
    for record in neutral_records:
        if record["mean"] or record["model"] != model:
            raise BuildError(f"{record['transcript']}: neutral text is captured token by token, with the same model as the stories.")
        _, tensor = read_capture(neutral, record["transcript"])
        for b, row in enumerate(_rows(record, band)):
            grams[b].add(tensor[row])
    if not neutral_records:
        raise BuildError(f"No captures in {neutral}.")
    found = [g.basis(variance) for g in grams]
    basis, explained = [f[0] for f in found], [f[1] for f in found]
    say(f"neutral directions taken out at each block: {[int(b.shape[0]) for b in basis]}")

    fit = V.extract(X, emotion, basis)
    vs = VectorSet({"emotions": fit["emotions"], "blocks": band}, fit["vectors"], fit["grand_mean"], basis, {})

    # a second set of vectors from two thirds of the situations, to check the scale on stories they have not seen
    several = len(set(topic)) > 1
    train = set(sorted(set(topic))[: max(1, (2 * len(set(topic))) // 3)])
    seen = [i for i, t in enumerate(topic) if t in train]
    unseen = [i for i, t in enumerate(topic) if t not in train]
    index = torch.tensor([fit["emotions"].index(e) for e in emotion])
    part = V.extract(X, emotion, basis, rows=seen) if several else None
    vs_part = VectorSet({"emotions": fit["emotions"], "blocks": band}, part["vectors"], part["grand_mean"], basis, {}) if several and part["emotions"] == fit["emotions"] else None

    # the two reference points. Neutral: how each emotion reads on the neutral text, token by token and message by
    # message. Own: how the stories written to convey an emotion read on its vector.
    token_cos, message_cos, part_message_cos, spans_of = [], [], [], []
    for record in neutral_records:
        _, tensor = read_capture(neutral, record["transcript"])
        x = tensor[_rows(record, band)].float().transpose(0, 1)   # [tokens, B, d]
        token_cos.append(vs.cosines(x))
        at = 0
        for span in record["spans"]:
            n = span["stop"] - span["start"]
            spans_of.append((len(token_cos) - 1, at, at + n))
            if span["role"] == "assistant":
                whole = x[at : at + n].mean(0, keepdim=True)
                message_cos.append(vs.cosines(whole))
                if vs_part is not None:
                    part_message_cos.append(vs_part.cosines(whole))
            at += n
    if not message_cos:
        raise BuildError("The neutral text has no assistant message to take a reply's usual reading from.")
    tokens, messages = torch.cat(token_cos), torch.cat(message_cos)
    story_cos = vs.cosines(X)
    own = torch.stack([story_cos[index == e, e].mean() for e in range(len(fit["emotions"]))])
    vs.usual = {"token": tokens.mean(0), "message": messages.mean(0), "own": own}
    strongest = torch.cat([smooth(vs.shares(token_cos[i][a:b], "token")).max(-1).values for i, a, b in spans_of])
    quantiles = {str(q): float(strongest.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)}
    floor = float(strongest.quantile(mark_quantile))
    step = max(1e-6, (1.0 - floor) / 8)

    say("checking how well the vectors hold up")
    full = {"emotions": fit["emotions"], "vectors": fit["vectors"]}
    scale = None
    if vs_part is not None:
        # the same scale built from the seen situations only, applied to stories of the unseen ones
        part_cos = vs_part.cosines(X)
        vs_part.usual = {"message": torch.cat(part_message_cos).mean(0), "own": torch.stack([part_cos[seen][index[seen] == e, e].mean() for e in range(len(fit["emotions"]))])}
        share = vs_part.shares(part_cos[unseen], "message")
        truth = index[unseen]
        theirs = share[torch.arange(len(unseen)), truth]
        ranks = share.topk(min(5, share.shape[1]), dim=-1).indices
        at = lambda t, qs: {str(q): float(t.quantile(q)) for q in qs}
        scale = {"what": "unseen-situation stories on the scale built from the seen ones: the share their own emotion reaches, the share any emotion reaches, and how often their own ranks first or in the first five by share",
                 "own_emotion_share_quantiles": at(theirs, (0.1, 0.25, 0.5, 0.75, 0.9)), "any_emotion_share_quantiles": at(share.flatten(), (0.1, 0.5, 0.9, 0.99)),
                 "top1": float((ranks[:, 0] == truth).double().mean()), "top5": float((ranks == truth[:, None]).any(-1).double().mean())}
    checks = {
        "what": "descriptive, not a pass mark: held-out situations, vectors from half the situations against the other half, each author's stories alone against all, and the reading scale on unseen stories",
        "held_out": V.heldout(X, emotion, topic, train, basis) if several else None,
        "split_half": V.split_half(X, emotion, topic, basis) if several else None,
        "authors": V.group_agreement(X, emotion, author, basis, full),
        "scale": scale,
    }
    usual = vs.usual

    out.mkdir(parents=True, exist_ok=True)
    tensors = {"vectors": fit["vectors"].contiguous(), "grand_mean": fit["grand_mean"].contiguous(), **{f"basis.{block}": b.float().contiguous() for block, b in zip(band, basis)},
               **{f"usual.{k}": v.float().contiguous() for k, v in usual.items()}}
    save_file(tensors, str(out / "vectors.safetensors"))
    fam = families()["emotions"]
    manifest = Path(corpus) / "manifest.json"
    record = {
        "schema": SCHEMA,
        "model": model,
        "blocks": band,
        "width": int(X.shape[-1]),
        "emotions": fit["emotions"],
        "families": {e: fam.get(e) for e in fit["emotions"]},
        "made_from": {
            "stories": len(ids),
            "stories_left_out": len(story_records) - len(ids),
            "authors": sorted(set(author)),
            "situations": len(set(topic)),
            "story_prompt": STORY_PROMPT,
            "story_tokens_skipped": first["skip"],
            "neutral_dialogues": len(neutral_records),
            "neutral_tokens": int(tokens.shape[0]),
            "neutral_assistant_messages": int(messages.shape[0]),
            "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        },
        "method": "average each emotion's stories, subtract the average over emotions, take out the leading directions of neutral text at each block; "
                  "read by the cosine at each block, averaged over the blocks",
        "neutral_variance_taken_out": variance,
        "neutral_directions_by_block": [int(b.shape[0]) for b in basis],
        "neutral_variance_reached_by_block": explained,
        "scale": {"what": "a reading is a share of the way from how neutral text reads on an emotion's vector (0) to how that emotion's own stories read on it (1). "
                          "The three reference values are in vectors.safetensors: usual.token and usual.message (the neutral dialogues, per token and per assistant message), and usual.own (each emotion's stories).",
                  "smallest_distance_from_neutral_to_own": float((own - torch.maximum(usual["token"], usual["message"])).min()),
                  "neutral_tokens_strongest_share_quantiles": quantiles},
        "marks": {"floor": floor, "step": step, "what": f"a token is marked when its strongest emotion reaches a share of at least `floor` (more than {mark_quantile:.1%} of neutral tokens' strongest do not); strength 1 at the floor to 9 at a share of 1, in steps of `step`"},
        "checks": checks,
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": _sha(out / "vectors.safetensors"),
    }
    (out / "vectors.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


def load_set(folder: str | Path, device: str = "cpu") -> VectorSet:
    """A vector set from its folder. The tensor file is checked against the hash in its record first."""
    from safetensors.torch import load_file

    folder = Path(folder)
    if not (folder / "vectors.json").is_file():
        raise BuildError(f"No vector set in {folder}.")
    record = json.loads((folder / "vectors.json").read_text())
    file = folder / "vectors.safetensors"
    if record.get("schema") != SCHEMA or not file.is_file() or _sha(file) != record["sha256"]:
        raise BuildError(f"{folder} does not hold the vector set its record describes.")
    t = {k: v.to(device) for k, v in load_file(str(file)).items()}
    return VectorSet(record, t["vectors"], t["grand_mean"], [t[f"basis.{b}"] for b in record["blocks"]], {k.removeprefix("usual."): v for k, v in t.items() if k.startswith("usual.")})


#: where the vector sets that come with the tool are kept: beside the package in a checkout, inside it in a wheel
_SHIPPED = (Path(__file__).resolve().parent.parent / "sets", Path(__file__).resolve().parent / "sets")


def shipped_folders(roots: Any = None) -> dict[str, Path]:
    """The folders of the vector sets that come with the tool, by the name of the model each is for."""
    out: dict[str, Path] = {}
    for root in roots or _SHIPPED:
        if Path(root).is_dir():
            for folder in sorted(Path(root).iterdir()):
                if (folder / "vectors.json").is_file():
                    out.setdefault(folder.name, folder)
    return out


def shipped_set(model: Mapping[str, Any], device: str = "cpu", roots: Any = None) -> VectorSet | None:
    """The vector set that comes with the tool for a model, or None when there is none for it. `model` is the
    model as a capture records it: its name and the revision of its files. A set made at another revision is not
    this model's, and is not answered. One that is there but is not the file its record describes raises."""
    folder = shipped_folders(roots).get(str(model.get("name")))
    if folder is None:
        return None
    made_with = json.loads((folder / "vectors.json").read_text()).get("model", {})
    if made_with.get("name") != model.get("name") or made_with.get("revision") != model.get("revision"):
        return None
    return load_set(folder, device=device)


def _rows(record: dict[str, Any], band: list[int]) -> list[int]:
    """Where the band's blocks are in a capture's tensor."""
    at = {int(b): k for k, b in enumerate(record["blocks"])}
    missing = [b for b in band if b not in at]
    if missing:
        raise BuildError(f"{record['transcript']} leaves out blocks that are needed ({missing[0]} to {missing[-1]}).")
    return [at[b] for b in band]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()
