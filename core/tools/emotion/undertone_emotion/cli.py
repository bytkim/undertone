"""Emotion vectors from a terminal.

    undertone-emotion read captures/ --out emotion/                       which emotions stand out in each capture,
                                                                          with the vector set that comes with the tool
    undertone-emotion show emotion/
    undertone-emotion info                                                what that vector set is and how it holds up

A vector set of your own:

    undertone-emotion corpus stories/corpus --out corpus-transcripts/     the story corpus as transcripts, to capture
    undertone-emotion build --stories captures-stories/ --neutral captures-neutral/ --corpus stories/corpus --out vectors/
    undertone-emotion info vectors/
    undertone-emotion read captures/ --vectors vectors/ --out emotion/
"""
import argparse
import json
import sys
from typing import Any

from undertone_capture import CaptureError, items as capture_items
from undertone_models import ContractError

from .build import BuildError, VectorSet, build, load_set, shipped_folders, shipped_set
from .corpus import CorpusError, write_transcripts
from .reading import ReadingError, readings, run


def _blocks(text: str | None) -> list[int] | None:
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _shipped(captures: str, device: str, log: Any) -> VectorSet:
    """The vector set that comes with the tool for the model a folder of captures was made with."""
    models = {(r["model"]["name"], r["model"]["revision"]) for r in capture_items(captures)}
    if not models:
        raise SystemExit(f"No captures in {captures}.")
    if len(models) != 1:
        raise SystemExit("The captures there were made with more than one model. Name a vector set with --vectors.")
    name, revision = models.pop()
    vs = shipped_set({"name": name, "revision": revision}, device=device)
    if vs is None:
        raise SystemExit(f"No vector set comes with the tool for {name} at the revision these captures were made with. Build one (undertone-emotion build) and name it with --vectors.")
    log(f"Reading with the vector set that comes with the tool for {name}.")
    return vs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="undertone-emotion", description="Emotion vectors: build them from stories, and read which emotions stand out in a capture.")
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("corpus", help="write the story corpus as transcripts, ready to capture")
    c.add_argument("corpus", help="the corpus folder (stories/corpus)")
    c.add_argument("--out", required=True)
    b = sub.add_parser("build", help="build a vector set from captures of the corpus")
    b.add_argument("--stories", required=True, help="captures of the stories: one average each (--skip 50 --mean)")
    b.add_argument("--neutral", required=True, help="captures of the neutral dialogues: every message's tokens (--select messages)")
    b.add_argument("--corpus", required=True, help="the corpus folder, for each story's emotion, situation and author")
    b.add_argument("--out", required=True)
    b.add_argument("--blocks", help='which blocks to build for, for example "16-23" (default: the model\'s middle depth)')
    i = sub.add_parser("info", help="what a vector set is and how it holds up")
    i.add_argument("vectors", nargs="?", help="a vector set's folder (default: the sets that come with the tool)")
    r = sub.add_parser("read", help="read every capture in a folder")
    r.add_argument("captures")
    r.add_argument("--vectors", help="a vector set's folder (default: the set that comes with the tool for the captures' model)")
    r.add_argument("--out", required=True)
    r.add_argument("--top", type=int, default=8, help="how many emotions to keep for the whole reply (default: 8)")
    r.add_argument("--device", default="cpu", help='"cpu" (the default: this is light work) or "cuda"')
    s = sub.add_parser("show", help="what a folder of readings holds")
    s.add_argument("out")
    s.add_argument("--transcript", help="show one reading token by token")
    a = ap.parse_args(argv)
    log = lambda line: print(line, file=sys.stderr)

    try:
        if a.command == "corpus":
            print(json.dumps(write_transcripts(a.corpus, a.out)))
        elif a.command == "build":
            record = build(a.stories, a.neutral, a.corpus, a.out, blocks=_blocks(a.blocks), log=log)
            _info(record)
        elif a.command == "info":
            folders = [a.vectors] if a.vectors else list(shipped_folders().values())
            if not folders:
                raise SystemExit("No vector set comes with this install. Name a vector set's folder.")
            for folder in folders:
                _info(load_set(folder).record)
        elif a.command == "read":
            vs = load_set(a.vectors, device=a.device) if a.vectors else _shipped(a.captures, a.device, log)
            result = run(vs, a.captures, a.out, top=a.top, log=log)
            print(json.dumps({k: (len(v) if k in ("read", "already_there") else v) for k, v in result.items() if k != "settings"}, indent=1))
            if result["left_out"]:
                raise SystemExit(1)
        else:
            for reading in readings(a.out):
                if a.transcript and reading["transcript"] != a.transcript:
                    continue
                marked = [t for t in reading["tokens"] if t["emotions"]]
                print(f"{reading['transcript']}  turn {reading['turn']}  {len(marked)} of {len(reading['tokens'])} tokens marked  {', '.join(reading['flags']) or 'no flags'}")
                print("  " + ", ".join(f"{e['name']} {e['level']}" for e in reading["overall"]))
                if a.transcript:
                    for t in reading["tokens"]:
                        print(f"  {t['text']!r:16}" + ", ".join(f"{e['name']} {e['strength']}" for e in t["emotions"]))
    except (BuildError, ReadingError, CorpusError, CaptureError, ContractError) as error:
        raise SystemExit(str(error))


def _info(record: dict) -> None:
    made, checks = record["made_from"], record["checks"]
    print(f"{record['model']['name']}  {len(record['emotions'])} emotions  blocks {record['blocks'][0]}-{record['blocks'][-1]}  width {record['width']}")
    print(f"  made from   {made['stories']} stories by {len(made['authors'])} authors on {made['situations']} situations; {made['neutral_tokens']} tokens of {made['neutral_dialogues']} neutral dialogues")
    print(f"  taken out   {record['neutral_directions_by_block']} neutral directions a block ({record['neutral_variance_taken_out']:.0%} of neutral variance)")
    if checks.get("held_out"):
        h = checks["held_out"]
        top = next(k for k in h if k.startswith("band_top") and k != "band_top1")
        print(f"  held out    the right emotion is named first for {h['band_top1']:.1%} of {h['stories_tested']} unseen-situation stories, and in the first {top[8:]} for {h[top]:.1%} (chance {h['chance_top1']:.1%})")
        print(f"  split half  median cosine between vectors from two halves of the situations: {min(checks['split_half']['median_by_block']):.3f} to {max(checks['split_half']['median_by_block']):.3f} across blocks")
    for name, g in checks["authors"].items():
        m = g["median_cosine_to_all_by_block"]
        print(f"  author      {name}: {g['stories']} stories, median cosine to all authors' vectors {min(m):.3f} to {max(m):.3f}")
    if checks.get("scale"):
        q = checks["scale"]["own_emotion_share_quantiles"]
        print(f"  scale       an unseen story's own emotion reaches a share of {q['0.5']:.2f} at the median ({q['0.1']:.2f} to {q['0.9']:.2f} for the middle 80%); ranked by share it is first for {checks['scale']['top1']:.1%}, in the first five for {checks['scale']['top5']:.1%}")
    print(f"  marks       a token is marked from a share of {record['marks']['floor']:.2f}, in steps of {record['marks']['step']:.2f}")


if __name__ == "__main__":
    main()
