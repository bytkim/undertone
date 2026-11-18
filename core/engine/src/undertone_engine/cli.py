"""The engine from a terminal: say something, list what was said, or serve the chat page's API.

    undertone-engine chat "What do you see from your window?" --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B
    undertone-engine list
    undertone-engine show <conversation>
    undertone-engine serve

With a model's contract the transcripts name it, and token ids the server does not return are rebuilt with the
contract's pinned tokenizer:

    undertone-engine serve --contract core/tools/models/qwen3.5-4b/contract.json

The endpoint can also come from the environment: UNDERTONE_BASE_URL, UNDERTONE_MODEL, UNDERTONE_API_KEY (the key is
read from the environment only, so it never lands in a shell history), and UNDERTONE_TRANSCRIPTS for the folder.

`serve` needs no endpoint to start: one can be set from the chat page's settings, and is then kept in engine.json in
the transcripts folder for the next start. An endpoint given here takes its place.
"""
import argparse
import os
import sys

from .contract import ModelContract, read_contract
from .endpoint import Endpoint, EndpointConfig, EndpointError
from .engine import Engine
from .service import make_server
from .settings import EndpointSettings
from .store import TranscriptStore
from .tokens import hf_tokenizer


def _engine(a: argparse.Namespace, needs_endpoint: bool = True) -> tuple[Engine, EndpointSettings, ModelContract | None]:
    """The engine, where its endpoint came from, and the model's contract when one was named. The endpoint is the
    one given here, else the one kept from the chat page's settings; `serve` may start with none."""
    key = os.environ.get("UNDERTONE_API_KEY")
    if bool(a.base_url) != bool(a.served_model):
        raise SystemExit("An endpoint is both an address and a model's name: --base-url and --model (or UNDERTONE_BASE_URL and UNDERTONE_MODEL).")
    started = EndpointConfig(base_url=a.base_url, model=a.served_model, api_key=key, ask_token_ids=a.token_ids) if a.base_url else None
    settings = EndpointSettings(a.transcripts, started=started, key_given=bool(key))
    config = settings.initial()
    if config is None and needs_endpoint:
        raise SystemExit("Say which endpoint: --base-url and --model (or UNDERTONE_BASE_URL and UNDERTONE_MODEL).")
    try:
        contract = read_contract(a.contract) if a.contract else None
    except ValueError as error:
        raise SystemExit(str(error))
    tokenizer = None
    if a.tokenizer:
        tokenizer = hf_tokenizer(a.tokenizer, a.tokenizer_revision)
    elif contract:
        # the contract's own tokenizer, at its pinned revision. Without it the engine still works: a transcript the
        # server gives no ids for is then kept without ids, which the tools accept and flag.
        try:
            tokenizer = hf_tokenizer(contract.repo, contract.revision)
        except (RuntimeError, OSError) as error:
            print(f"Note: token ids will not be rebuilt, because {contract.name}'s tokenizer could not be loaded ({error}).", file=sys.stderr)
    if contract and config and not contract.knows(config.model):
        known = " or ".join(f'"{name}"' for name in contract.served_as)
        print(f'Note: the endpoint serves "{config.model}", and the {contract.name} contract knows its weights as {known}. Readings of these transcripts will be flagged.', file=sys.stderr)
    engine = Engine(Endpoint(config) if config else None, TranscriptStore(a.transcripts), tokenizer=tokenizer, model=contract.name if contract else None)
    return engine, settings, contract


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="undertone-engine", description="Talk to any OpenAI-compatible endpoint and keep what was said as transcripts.")
    ap.add_argument("--transcripts", default=os.environ.get("UNDERTONE_TRANSCRIPTS", "transcripts"), help="the folder transcripts are kept in (default: ./transcripts)")
    sub = ap.add_subparsers(dest="command", required=True)

    def endpoint_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", default=os.environ.get("UNDERTONE_BASE_URL"), help='the API base, for example "http://localhost:8000/v1"')
        p.add_argument("--model", dest="served_model", default=os.environ.get("UNDERTONE_MODEL"), help="the name the server knows the model by")
        p.add_argument("--contract", help="the model's contract among the undertone tools: the path to its contract.json, or its name (for example qwen3.5-4b) when core/tools/models is installed")
        p.add_argument("--token-ids", action="store_true", help="ask the server for the token ids it used (for servers that can return them)")
        p.add_argument("--tokenizer", help="rebuild token ids with this tokenizer (a Hugging Face name or a path) when the server returns none (default: the contract's, when one is given)")
        p.add_argument("--tokenizer-revision")

    chat = sub.add_parser("chat", help="say something and print the reply")
    chat.add_argument("message")
    chat.add_argument("--conversation", help="carry on this conversation (default: start a new one)")
    chat.add_argument("--system", help="a system message, when starting a conversation")
    chat.add_argument("--max-tokens", type=int)
    chat.add_argument("--temperature", type=float)
    endpoint_options(chat)

    serve = sub.add_parser("serve", help="serve the chat page's API on this machine")
    serve.add_argument("--port", type=int, default=8787)
    endpoint_options(serve)

    sub.add_parser("list", help="list the conversations kept")
    show = sub.add_parser("show", help="print a conversation")
    show.add_argument("conversation")

    a = ap.parse_args(argv)
    if a.command == "list":
        for s in TranscriptStore(a.transcripts).conversations():
            print(f"{s.id}  {s.turns:>3} turn{'s' if s.turns != 1 else ' '}  {s.updated}  {s.title}")
    elif a.command == "show":
        try:
            calls = TranscriptStore(a.transcripts).conversation(a.conversation)
        except ValueError as error:
            raise SystemExit(str(error))
        if not calls:
            raise SystemExit("No such conversation.")
        for t in calls:
            print(f"[{t.turn}] you: {t.request.messages[-1].content}\n[{t.turn}] model: {t.response.content}\n")
    elif a.command == "chat":
        params = {k: v for k, v in (("max_tokens", a.max_tokens), ("temperature", a.temperature)) if v is not None}
        try:
            t = _engine(a)[0].chat(a.message, conversation=a.conversation, system=a.system, **params)
        except (EndpointError, ValueError) as error:
            raise SystemExit(str(error))
        print(t.response.content)
        ids = "no token ids" if t.tokens is None else f"token ids {t.tokens.source}"
        print(f"\nconversation {t.conversation} · turn {t.turn} · {ids}", file=sys.stderr)
    elif a.command == "serve":
        engine, settings, contract = _engine(a, needs_endpoint=False)
        server = make_server(engine, port=a.port, settings=settings, contract=contract)
        config = engine.endpoint.config if engine.endpoint else None
        where = f"{config.base_url} ({config.model})" + (", as kept from the page's settings" if settings.source == "kept" else "") if config else "no endpoint yet: set one in the chat page's settings"
        print(f"undertone-engine on http://127.0.0.1:{a.port} → {where}; transcripts in {a.transcripts}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
