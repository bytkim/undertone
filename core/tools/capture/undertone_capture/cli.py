"""Capture from a terminal.

    undertone-capture run transcripts/ --out captures/ --model qwen3.5-4b
    undertone-capture show captures/
    undertone-capture verify captures/
"""
import argparse
import json
import sys

from undertone_models import ContractError, adapter, contract

from .capture import CaptureError, items, run, transcripts, verify


def _blocks(text: str | None) -> list[int] | None:
    """"8-31" or "16,17,20-23" as block numbers."""
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="undertone-capture", description="Capture what a model's blocks put out when it reads a transcript.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="read transcripts with the pinned model and keep each block's output")
    r.add_argument("transcripts", nargs="+", help="transcript files (.jsonl) or folders of them, as the engine keeps them")
    r.add_argument("--out", required=True, help="the folder to keep the captures in")
    r.add_argument("--model", help="the model's contract (default: the one the transcripts name)")
    r.add_argument("--conversation", help="capture this conversation only")
    r.add_argument("--select", choices=("reply", "messages"), default="reply", help="the reply's tokens (default), or every message's")
    r.add_argument("--skip", type=int, default=0, help="leave out the first N tokens of the reply")
    r.add_argument("--mean", action="store_true", help="keep the average over the selected tokens instead of every token")
    r.add_argument("--blocks", help='which blocks to keep, for example "8-31" (default: all)')
    r.add_argument("--snapshot", help="a folder already holding the model's files (default: fetch them)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--skip-hashes", action="store_true", help="do not hash the model's files against its contract first")
    s = sub.add_parser("show", help="what a capture folder holds")
    s.add_argument("out")
    v = sub.add_parser("verify", help="read every capture back: hash, shape, precision, finite")
    v.add_argument("out")
    a = ap.parse_args(argv)

    try:
        if a.command == "show":
            for record in items(a.out):
                t = record["tensor"]
                print(f"{record['transcript']}  turn {record['turn']}  {t['name']} {t['shape']} {t['dtype']}  {len(record['tokens'])} tokens  {', '.join(record['flags']) or 'no flags'}")
            return
        if a.command == "verify":
            report = verify(a.out)
            print(json.dumps(report, indent=1))
            if not report["passed"]:
                raise SystemExit(1)
            return
        found = list(transcripts(a.transcripts, a.conversation))
        if not found:
            raise SystemExit("No transcripts there.")
        named = {t.get("model") for t in found} - {None}
        name = a.model or (named.pop() if len(named) == 1 else None)
        if not name:
            raise SystemExit("Say which model to read them with: --model (the transcripts do not name one contract).")
        c = contract(name)
        model = adapter(c).load(c, snapshot=a.snapshot, device=a.device, verify_files=not a.skip_hashes)
        result = run(model, found, a.out, select=a.select, skip=a.skip, mean=a.mean, blocks=_blocks(a.blocks), log=lambda line: print(line, file=sys.stderr))
    except (CaptureError, ContractError) as error:
        raise SystemExit(str(error))
    print(json.dumps({k: (len(v) if k in ("captured", "already_there") else v) for k, v in result.items() if k != "settings"}, indent=1))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
