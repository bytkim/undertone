"""The readings over HTTP, for the chat page.

    GET /health                    {"ok": true, "model": ..., "tools": {...}, "readings": how many are kept}
    GET /readings/{transcript id}  the reading (undertone.reading.v1), made now if it has not been
                                   404 when there is no such transcript; 422 when it cannot be read, with the reason

A reading that has to be made takes a few seconds; the request waits for it. Like the engine's service this is for
one person on their own machine: it binds to 127.0.0.1, answers only pages served from this machine, and has no
accounts. A request addressed to any other name is refused. Reach a GPU machine's by forwarding the port
(ssh -L 8788:127.0.0.1:8788 ...).
"""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .desk import Desk, NoTranscript
from .reading import Unreadable

LOCAL_ORIGIN = re.compile(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?")
LOCAL_HOST = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:\d+)?")


def health(desk: Desk) -> dict:
    tools = desk.tools
    return {
        "ok": True,
        "model": tools.contract.name,
        "revision": tools.contract.revision,
        "device": str(tools.model.device),
        "tools": {
            "emotions": len(tools.vectors.emotions) if tools.vectors is not None else None,
            "lens": tools.reader.lens.name if tools.reader is not None else None,
            "lens_ranked_by": None if tools.reader is None else "lift over a baseline" if tools.baseline is not None else "probability",
        },
        "readings": desk.count(),
    }


def make_server(desk: Desk, host: str = "127.0.0.1", port: int = 8788) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "undertone-server"

        def log_message(self, *args) -> None:   # quiet: the desk says what it made
            pass

        def _send(self, status: int, body: object | None = None) -> None:
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            origin = self.headers.get("Origin")
            if origin and LOCAL_ORIGIN.fullmatch(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):   # the page went away while its reading was being made
                pass

        def do_OPTIONS(self) -> None:
            self._send(204)

        def do_GET(self) -> None:
            # a request addressed to another name is not this machine's (a page elsewhere can point its own name
            # at 127.0.0.1, and a reading holds the reply's words)
            named = self.headers.get("Host")
            if named and not LOCAL_HOST.fullmatch(named):
                return self._send(403, {"error": "This service answers this machine only."})
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/health":
                return self._send(200, health(desk))
            if not path.startswith("/readings/"):
                return self._send(404, {"error": "Not found."})
            try:
                self._send(200, desk.reading(path.removeprefix("/readings/")))
            except NoTranscript:
                self._send(404, {"error": "No such transcript."})
            except Unreadable as error:
                self._send(422, {"error": str(error)})
            except Exception as error:   # the model ran out of memory, say: this reply has no reading, the next may
                desk.log(f"{path.removeprefix('/readings/')}  failed: {type(error).__name__}: {error}"[:500])
                self._send(500, {"error": f"{type(error).__name__}: {error}"[:500]})

    return ThreadingHTTPServer((host, port), Handler)
