"""Model contracts from a terminal.

    undertone-models list                  the models that have a contract here
    undertone-models show qwen3.5-4b       one contract's facts
    undertone-models check qwen3.5-4b      load the pinned model and run its self-test (needs a GPU)
"""
import argparse
import json
import sys

from . import ContractError, adapter, check, contract, names


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="undertone-models", description="The model contracts the undertone tools read.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the models that have a contract here")
    show = sub.add_parser("show", help="one contract's facts")
    show.add_argument("model")
    run = sub.add_parser("check", help="load the pinned model and run its self-test")
    run.add_argument("model")
    run.add_argument("--snapshot", help="a folder already holding the model's files (default: fetch them)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--skip-hashes", action="store_true", help="do not hash the files against the contract first")
    a = ap.parse_args(argv)

    try:
        if a.command == "list":
            for name in names():
                c = contract(name)
                print(f"{c.name}  {c.title}  {c.repo}@{c.revision[:12]}")
            return
        c = contract(a.model)
        if a.command == "show":
            q = c.data["qualified"]
            print(f"{c.title}  ({c.name})")
            print(f"  source     {c.repo} at {c.revision}")
            print(f"  shape      {c.blocks} blocks, width {c.width}, read at each block's output")
            print("  depths     " + "; ".join(f"{d} {b[0]}-{b[-1]}" if len(b) > 1 else f"{d} {b[0]}" for d, b in c.depths.items()))
            print(f"  text       the model's own chat template, thinking {'on' if c.thinking else 'off'}, up to {c.max_tokens} tokens")
            print("  lenses     " + (", ".join(c.lenses) or "none"))
            print("  qualified  " + ", ".join(f"{p} {v}" for p, v in q["runtime"]["packages"].items()))
            print("  not qualified:")
            for line in q.get("not_qualified", []):
                print(f"    - {line}")
            return
        model = adapter(c).load(c, snapshot=a.snapshot, device=a.device, verify_files=not a.skip_hashes)
        report = check(c).check(model)
    except ContractError as error:
        raise SystemExit(str(error))
    print(json.dumps(report, indent=1))
    for item in report["checks"]:
        print(("pass  " if item["passed"] else "FAIL  ") + item["name"], file=sys.stderr)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
