"""A small local HTTP service over the engine, for a chat page.

A browser page cannot call an arbitrary endpoint itself (the key would sit in the page, and most endpoints refuse
requests from a browser), so the page talks to this and this talks to the endpoint. It is also where history comes
from: the conversations it lists are the transcripts the engine has kept.

    GET  /health                    {"ok": true, "endpoint": host, "model": served model}   (both null with no endpoint)
    GET  /conversations             the conversations, most recently continued first
    GET  /conversations/{id}        every call of one conversation, as transcripts
    POST /chat                      {"message": "...", "conversation"?: id, "system"?: "...", "params"?: {...}}
                                    -> the transcript of the call
    GET  /settings                  which endpoint the engine talks to, and what it knows of the model
    PUT  /settings                  {"base_url": "...", "model": "...", "token_ids"?: bool} -> the settings as they now are
    POST /endpoint/models           {"base_url"?: "..."} -> {"models": [...]}: the names that endpoint (or, with no
                                    address, the one in use) knows its models by

It is for one person on their own machine: it binds to 127.0.0.1, answers only requests addressed to this machine
and only pages served from it, and has no accounts. The API key never passes through it: the key stays in the
engine's environment and goes only to the endpoint the engine was started with (settings.py). It is built on the
standard library so the engine stays two dependencies.
"""
import json
import re
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

from .contract import ModelContract
from .endpoint import Endpoint, EndpointError
from .engine import Engine
from .settings import EndpointSettings, SettingsError

MAX_BODY = 1_000_000
LOCAL_ORIGIN = re.compile(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?")
LOCAL_HOST = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:\d+)?")
#: how long an endpoint is given to say which models it has
LOOK_TIMEOUT = 10.0


def make_server(engine: Engine, host: str = "127.0.0.1", port: int = 8787, settings: EndpointSettings | None = None,
                contract: ModelContract | None = None, transport: Any = None) -> ThreadingHTTPServer:
    """`settings` says where the engine's endpoint came from and where a new one is kept (default: the endpoint the
    engine has now, kept beside its transcripts). `contract` is the model's contract, when one was named.
    `transport` is for tests: what a new endpoint is reached through."""
    turn = Lock()   # one call at a time: a conversation's turns are numbered by what is already stored
    if settings is None:
        settings = EndpointSettings(engine.store.root, started=engine.endpoint.config if engine.endpoint is not None else None)
        settings.source = "start" if engine.endpoint is not None else None

    def described() -> dict[str, Any]:
        config = engine.endpoint.config if engine.endpoint is not None else None
        return {
            "endpoint": None if config is None else {"base_url": config.base_url, "model": config.model, "host": config.host, "token_ids": config.ask_token_ids},
            "source": settings.source if config is not None else None,
            "api_key": settings.key(config),
            "contract": None if contract is None else {"name": contract.name, "served_as": list(contract.served_as), "knows_model": contract.knows(config.model) if config is not None else None},
            "rebuilds_ids": engine.tokenizer is not None,
            "transcripts": str(engine.store.root),
            "kept_in": str(settings.file),
        }

    class Handler(BaseHTTPRequestHandler):
        server_version = "undertone-engine"

        def log_message(self, *args) -> None:   # quiet: the transcripts are the record
            pass

        def _send(self, status: int, body: object | None = None) -> None:
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            origin = self.headers.get("Origin")
            if origin and LOCAL_ORIGIN.fullmatch(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _local(self, changes: bool = False) -> bool:
            """Whether the request is this machine's to make. One addressed to another name is not (a page elsewhere
            can point its own name at 127.0.0.1), and one that changes anything must not come from a page elsewhere,
            which a browser will send here even though it may not read the answer."""
            named = self.headers.get("Host")
            origin = self.headers.get("Origin")
            if (named and not LOCAL_HOST.fullmatch(named)) or (changes and origin and not LOCAL_ORIGIN.fullmatch(origin)):
                self._send(403, {"error": "This service answers this machine only."})
                return False
            return True

        def _body(self) -> dict[str, Any] | None:
            """The request's JSON object, or None once the fault has been answered."""
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, {"error": "The request needs a JSON body of at most 1 MB."})
                return None
            try:
                body = json.loads(self.rfile.read(length)) if length > 0 else None
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                self._send(400, {"error": "The request needs a JSON object as its body."})
                return None
            return body

        def do_OPTIONS(self) -> None:
            self._send(204)

        def do_GET(self) -> None:
            if not self._local():
                return
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/health":
                config = engine.endpoint.config if engine.endpoint is not None else None
                self._send(200, {"ok": True, "endpoint": config.host if config else None, "model": config.model if config else None})
            elif path == "/settings":
                self._send(200, described())
            elif path == "/conversations":
                self._send(200, [asdict(s) for s in engine.store.conversations()])
            elif path.startswith("/conversations/"):
                try:
                    calls = engine.store.conversation(path.removeprefix("/conversations/"))
                except ValueError:
                    return self._send(400, {"error": "Not a conversation id."})
                if not calls:
                    return self._send(404, {"error": "No such conversation."})
                self._send(200, [json.loads(t.line()) for t in calls])
            else:
                self._send(404, {"error": "Not found."})

        def do_PUT(self) -> None:
            if not self._local(changes=True):
                return
            if self.path.split("?", 1)[0].rstrip("/") != "/settings":
                return self._send(404, {"error": "Not found."})
            body = self._body()
            if body is None:
                return
            try:
                config = settings.config(body.get("base_url"), body.get("model"), bool(body.get("token_ids")))
            except SettingsError as error:
                return self._send(400, {"error": str(error)})
            with turn:   # not in the middle of a call
                before = engine.endpoint
                engine.endpoint = Endpoint(config, transport=transport)
                settings.keep(config)
                if before is not None:
                    before.close()
            self._send(200, described())

        def do_POST(self) -> None:
            if not self._local(changes=True):
                return
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/endpoint/models":
                return self._models()
            if path != "/chat":
                return self._send(404, {"error": "Not found."})
            body = self._body()
            if body is None:
                return
            try:
                message = body["message"]
                params = body.get("params") or {}
                if not isinstance(message, str) or not message.strip() or not isinstance(params, dict):
                    raise ValueError
                if not all(isinstance(body.get(k), str | None) for k in ("conversation", "system")):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": 'Send {"message": "..."} with an optional "conversation", "system" and "params".'})
            try:
                with turn:
                    transcript = engine.chat(message, conversation=body.get("conversation"), system=body.get("system"), **params)
            except ValueError as error:   # a conversation id that is not one
                return self._send(400, {"error": str(error)})
            except EndpointError as error:
                return self._send(502, {"error": str(error)})
            self._send(200, json.loads(transcript.line()))

        def _models(self) -> None:
            """Ask an endpoint which models it has: the one in use, or one at an address being tried out."""
            body = self._body() if int(self.headers.get("Content-Length") or 0) > 0 else {}
            if body is None:
                return
            if body.get("base_url") is not None:
                try:
                    config = replace(settings.config(body["base_url"], "-"), timeout=LOOK_TIMEOUT)
                except SettingsError as error:
                    return self._send(400, {"error": str(error)})
            elif engine.endpoint is not None:
                config = replace(engine.endpoint.config, timeout=LOOK_TIMEOUT)
            else:
                return self._send(400, {"error": "Say which endpoint to ask: no endpoint is set."})
            looked = Endpoint(config, transport=transport)
            try:
                self._send(200, {"host": config.host, "models": looked.models()})
            except EndpointError as error:
                self._send(502, {"error": str(error)})
            finally:
                looked.close()

    return ThreadingHTTPServer((host, port), Handler)
