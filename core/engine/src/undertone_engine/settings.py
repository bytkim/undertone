"""Which endpoint the engine talks to, as something that can be asked about and changed while it runs.

An endpoint comes from one of three places, in this order:

    given at start   --base-url and --model (or UNDERTONE_BASE_URL and UNDERTONE_MODEL)
    kept             engine.json in the transcripts folder, written when the endpoint is set from the chat page
    none             the engine runs, keeps its history, and says it has no endpoint until one is set

The API key is another matter. It is read from the environment only and belongs to the endpoint the engine was
started with. An endpoint set from the page is never sent the key unless it is that same host, so nothing a page
does can carry the key somewhere else.
"""
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .endpoint import EndpointConfig

FILE = "engine.json"


class SettingsError(ValueError):
    """Settings that cannot be used: not a URL, or no model name."""


class EndpointSettings:
    def __init__(self, root: str | Path, started: EndpointConfig | None = None, key_given: bool | None = None):
        """`root` is the transcripts folder. `started` is the endpoint given when the engine started, with its key.
        `key_given` says a key is in the environment even when no endpoint was given with it."""
        self.file = Path(root) / FILE
        self.started = started
        self.key_given = bool(started and started.api_key) if key_given is None else key_given
        self.source: str | None = None   # where the endpoint in use came from: "start", "kept" or "page"

    def initial(self) -> EndpointConfig | None:
        """The endpoint to begin with: the one given at start, else the one kept, else none."""
        if self.started is not None:
            self.source = "start"
            return self.started
        try:
            kept = json.loads(self.file.read_text(encoding="utf-8"))
            config = self.config(kept.get("base_url"), kept.get("model"), bool(kept.get("token_ids")))
        except (OSError, ValueError, AttributeError):   # no file, or one that is not usable settings
            return None
        self.source = "kept"
        return config

    def config(self, base_url: Any, model: Any, token_ids: bool = False) -> EndpointConfig:
        """An endpoint from what a person typed. The key goes with it only when it is the host the engine was
        started with."""
        if not isinstance(base_url, str) or not isinstance(model, str):
            raise SettingsError('Send {"base_url": "...", "model": "..."}.')
        base_url, model = base_url.strip(), model.strip()
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc or "@" in parts.netloc:
            raise SettingsError('The endpoint is an address like "http://localhost:8000/v1", with no name or password in it.')
        if not model or len(model) > 200:
            raise SettingsError("Say the name the endpoint knows the model by.")
        config = EndpointConfig(base_url=base_url, model=model, ask_token_ids=bool(token_ids))
        if self.started is not None and self.started.api_key and config.host == self.started.host:
            config = EndpointConfig(base_url=base_url, model=model, ask_token_ids=bool(token_ids), api_key=self.started.api_key, timeout=self.started.timeout)
        return config

    def keep(self, config: EndpointConfig) -> None:
        """Write the endpoint down for the next start. The key is never written."""
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps({"base_url": config.base_url, "model": config.model, "token_ids": config.ask_token_ids}, indent=1) + "\n", encoding="utf-8")
        self.source = "page"

    def key(self, config: EndpointConfig | None) -> str:
        """What becomes of the API key with this endpoint: "sent", "held back" (there is a key, and this is not
        the endpoint it was given for), or "none"."""
        if config is not None and config.api_key:
            return "sent"
        return "held back" if self.key_given else "none"
