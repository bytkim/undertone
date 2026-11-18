"""The lens from a terminal.

    undertone-jlens run captures/ --out lens/                        ranked against the baseline that comes with the tool
    undertone-jlens run captures/ --out lens/ --baseline none        ranked by probability
    undertone-jlens baseline captures-of-ordinary-text/ --out mine/  make a baseline of your own
    undertone-jlens run captures/ --out lens/ --baseline mine/
    undertone-jlens show lens/
"""
import argparse
import json
import sys

from undertone_capture import CaptureError, items as capture_items
from undertone_models import ContractError, adapter, contract

from .lens import LensError, load as load_lens
from .reading import Reader, ReadingError, build_baseline, load_baseline, readings, run, shipped_baseline


def _reader(a: argparse.Namespace) -> Reader:
    records = capture_items(a.captures)
    if not records:
        raise SystemExit(f"No captures in {a.captures}.")
    names = {r["model"]["name"] for r in records}
    if len(names) != 1:
        raise SystemExit("The captures there were made with more than one model.")
    c = contract(names.pop())
    head = adapter(c).load_head(c, snapshot=a.snapshot, device=a.device, verify_files=not a.skip_hashes)
    return Reader(head, load_lens(c, a.lens, path=a.lens_file, device=a.device))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="undertone-jlens", description="Read a capture through a Jacobian lens: the words the model leans toward at each depth.")
    sub = ap.add_subparsers(dest="command", required=True)

    def reading_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("captures", help="a folder of captures (from undertone-capture)")
        p.add_argument("--out", required=True)
        p.add_argument("--lens", default="jpp", help="which of the lenses the model's contract pins (default: jpp)")
        p.add_argument("--lens-file", help="the lens file, if you already have it (default: fetch it)")
        p.add_argument("--snapshot", help="a folder already holding the model's files (default: fetch them)")
        p.add_argument("--device", default="cuda", help='"cuda" or "cpu" (the lens needs no GPU, only patience)')
        p.add_argument("--skip-hashes", action="store_true", help="do not hash the model's files against its contract first")

    r = sub.add_parser("run", help="read every capture in a folder")
    reading_options(r)
    r.add_argument("--top", type=int, default=8, help="how many words to keep at each depth (default: 8)")
    r.add_argument("--baseline", help='a baseline folder to rank words against, or "none" to rank by probability (default: the baseline that comes with the tool for this model and lens, if there is one)')
    b = sub.add_parser("baseline", help="make a baseline from captures of ordinary text")
    reading_options(b)
    b.add_argument("--note", help="what the captures are of, kept in the baseline's record")
    s = sub.add_parser("show", help="what a folder of readings holds")
    s.add_argument("out")
    s.add_argument("--transcript", help="show one reading token by token")
    a = ap.parse_args(argv)

    try:
        if a.command == "show":
            for reading in readings(a.out):
                if a.transcript and reading["transcript"] != a.transcript:
                    continue
                print(f"{reading['transcript']}  turn {reading['turn']}  {len(reading['tokens'])} tokens  by {reading['ranked_by']}  {', '.join(reading['flags']) or 'no flags'}")
                for depth, words in reading["overall"].items():
                    print(f"  {depth:7}" + ", ".join(w["word"] for w in words))
                if a.transcript:
                    for t in reading["tokens"]:
                        print(f"  {t['text']!r:16}" + "  |  ".join(" ".join(w["word"] for w in t["lens"][d][:3]) for d in reading["overall"]))
            return
        reader = _reader(a)
        log = lambda line: print(line, file=sys.stderr)
        if a.command == "baseline":
            summary = build_baseline(reader, a.captures, a.out, note=a.note, log=log)
            print(json.dumps({k: v for k, v in summary.items() if k != "depths"}, indent=1))
            return
        if a.baseline == "none":
            baseline = None
        elif a.baseline:
            baseline = load_baseline(a.baseline, device=a.device)
        else:
            baseline = shipped_baseline(reader)
            log("Ranking against the baseline that comes with the tool." if baseline is not None else "No baseline comes with the tool for this model and lens (or it does not fit this install): ranking by probability.")
        result = run(reader, a.captures, a.out, top=a.top, baseline=baseline, log=log)
    except (ReadingError, LensError, CaptureError, ContractError) as error:
        raise SystemExit(str(error))
    print(json.dumps({k: (len(v) if k in ("read", "already_there") else v) for k, v in result.items() if k != "settings"}, indent=1))
    if result["left_out"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
