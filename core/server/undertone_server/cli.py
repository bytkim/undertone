"""The readings server from a terminal.

    undertone-server serve                        answer the chat page: a reading for each reply the engine has kept
    undertone-server read                         make the reading of every transcript now, without serving
    undertone-server show readings/               what a folder of readings holds

`serve` and `read` load the model the transcripts were made with (--model, default qwen3.5-4b) on a GPU, with the
lens its contract pins and the baseline and emotion vectors that come with the tools. Both read the engine's
transcripts folder (--transcripts, default ./transcripts or UNDERTONE_TRANSCRIPTS) and keep readings in --out
(default ./readings or UNDERTONE_READINGS).
"""
import argparse
import json
import os
import sys

from undertone_capture import CaptureError
from undertone_models import ContractError, adapter, contract

from .desk import Desk, DeskError
from .reading import SCHEMA, Tools
from .service import health, make_server


def load(a: argparse.Namespace, log) -> Tools:
    """The model and the tools, loaded once. A tool that cannot be had is said and left out; the model is needed."""
    from undertone_emotion import load_set, shipped_set
    from undertone_jlens import Reader, load_baseline, load_lens, shipped_baseline

    c = contract(a.model)
    log(f"Loading {c.title} on {a.device}" + ("" if a.skip_hashes else " (its files are hashed against the contract first)") + "…")
    model = adapter(c).load(c, snapshot=a.snapshot, device=a.device, verify_files=not a.skip_hashes)

    reader = baseline = None
    if a.lens != "none":
        log(f"Loading the {a.lens} lens…")
        reader = Reader(model.as_head(), load_lens(c, a.lens, path=a.lens_file, device=a.device))
        if a.baseline == "none":
            log("Lens words are ranked by probability.")
        elif a.baseline:
            baseline = load_baseline(a.baseline, device=a.device)
            baseline.fits(reader)
        else:
            baseline = shipped_baseline(reader)
            log("Lens words are ranked against the baseline that comes with the tool." if baseline is not None
                else "No baseline comes with the tool for this model and lens (or it does not fit this install): lens words are ranked by probability.")

    if a.vectors == "none":
        vectors = None
    elif a.vectors:
        vectors = load_set(a.vectors, device=a.device)
        made_with = vectors.record["model"]
        if made_with.get("name") != c.name or made_with.get("revision") != c.revision:
            raise SystemExit(f"The vector set in {a.vectors} was built on another model or another revision of it.")
    else:
        vectors = shipped_set({"name": c.name, "revision": c.revision}, device=a.device)
        log(f"Emotions are read with the vector set that comes with the tool ({len(vectors.emotions)} emotions)." if vectors is not None
            else f"No vector set comes with the tool for {c.name}: readings will have no emotions. Build one and name it with --vectors.")
    if reader is None and vectors is None:
        raise SystemExit("There is nothing to read with: no lens and no vector set.")
    return Tools(model=model, reader=reader, baseline=baseline, vectors=vectors, top=a.top)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="undertone-server", description="Readings for the chat page: one reply's emotions and lens words, made from its transcript.")
    sub = ap.add_subparsers(dest="command", required=True)

    def options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--transcripts", default=os.environ.get("UNDERTONE_TRANSCRIPTS", "transcripts"), help="the folder the engine keeps transcripts in (default: ./transcripts)")
        p.add_argument("--out", default=os.environ.get("UNDERTONE_READINGS", "readings"), help="the folder to keep readings in (default: ./readings)")
        p.add_argument("--model", default="qwen3.5-4b", help="the model's contract (default: qwen3.5-4b)")
        p.add_argument("--snapshot", help="a folder already holding the model's files (default: fetch them)")
        p.add_argument("--device", default="cuda", help='"cuda" (the default) or "cpu"')
        p.add_argument("--skip-hashes", action="store_true", help="do not hash the model's files against its contract first")
        p.add_argument("--lens", default="jpp", help='which of the lenses the model\'s contract pins, or "none" (default: jpp)')
        p.add_argument("--lens-file", help="the lens file, if you already have it (default: fetch it)")
        p.add_argument("--baseline", help='a baseline folder to rank lens words against, or "none" for probability (default: the one that comes with the tool)')
        p.add_argument("--vectors", help='an emotion vector set\'s folder, or "none" (default: the set that comes with the tool)')
        p.add_argument("--top", type=int, default=8, help="how many emotions, and how many words a depth, a reading keeps (default: 8)")

    serve = sub.add_parser("serve", help="answer the chat page's requests for readings on this machine")
    serve.add_argument("--port", type=int, default=8788)
    options(serve)
    read = sub.add_parser("read", help="make the reading of every transcript now, without serving")
    options(read)
    show = sub.add_parser("show", help="what a folder of readings holds")
    show.add_argument("out")
    show.add_argument("--transcript", help="show one reading token by token")
    a = ap.parse_args(argv)
    log = lambda line: print(line, file=sys.stderr, flush=True)

    if a.command == "show":
        return _show(a.out, a.transcript)
    try:
        from undertone_emotion import BuildError
        from undertone_jlens import LensError, ReadingError

        try:
            desk = Desk(load(a, log), a.transcripts, a.out, log=log)
        except (ContractError, CaptureError, LensError, ReadingError, BuildError, DeskError) as error:
            raise SystemExit(str(error))
    except ImportError as error:   # torch, most likely: the server runs the model
        raise SystemExit(f"The server needs the tools and what they run on ({error}). See core/server/README.md.")
    if a.command == "read":
        result = desk.read_all()
        print(json.dumps({"made": len(result["made"]), "already_there": len(result["already_there"]), "left_out": result["left_out"]}, indent=1))
        if result["left_out"]:
            raise SystemExit(1)
        return
    server = make_server(desk, port=a.port)
    said = health(desk)["tools"]
    print(f"undertone-server on http://127.0.0.1:{a.port}  {desk.tools.contract.name} on {a.device}  "
          f"emotions: {said['emotions'] or 'none'}  lens: {said['lens'] or 'none'}  transcripts in {a.transcripts}/  readings in {a.out}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _show(out: str, transcript: str | None) -> None:
    from pathlib import Path

    from undertone_models import DEPTHS

    for path in sorted(Path(out).glob("*.json")):
        if path.name == "readings.json":
            continue
        r = json.loads(path.read_text())
        if r.get("schema") != SCHEMA or (transcript and r["transcript"] != transcript):
            continue
        marked = sum(1 for t in r["tokens"] if t.get("emotions"))
        print(f"{r['transcript']}  turn {r['turn']}  {marked} of {len(r['tokens'])} tokens marked  {', '.join(r['flags']) or 'no flags'}")
        print("  emotions  " + (", ".join(f"{e['name']} {e['level']}" for e in r["overall"]["emotions"]) or "none"))
        for depth in DEPTHS:
            print(f"  {depth:8}  " + ", ".join(r["overall"]["lens"][depth]))
        if transcript:
            for t in r["tokens"]:
                print(f"  {t['text']!r:16}" + ", ".join(f"{e['name']} {e['strength']}" for e in t.get("emotions", [])))


if __name__ == "__main__":
    main()
