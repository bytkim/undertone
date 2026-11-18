"""The boundary: one call to an OpenAI-compatible chat endpoint.

Any server that speaks `POST {base_url}/chat/completions` works: a local one (vLLM, llama.cpp, Ollama) or a hosted
one. The engine runs no model and manages no weights.

Token ids. Some servers can return the ids they used when asked (vLLM: `return_token_ids`, giving
`prompt_token_ids` on the response and `token_ids` on the choice). That is not part of the OpenAI API, and a strict
server rejects a field it does not know, so the engine asks only when told to (`ask_token_ids`), and when a server
refuses the request it asks once more without the field and does not ask that server again.
"""
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

#: the request field that asks a server for the ids it used, where the server has one
TOKEN_IDS_FIELD = "return_token_ids"


class EndpointError(RuntimeError):
    """The endpoint could not be reached, or answered with an error."""


@dataclass(frozen=True)
class EndpointConfig:
    #: the API's base, for example "http://localhost:8000/v1"
    base_url: str
    #: the name the server knows the model by
    model: str
    #: sent as a bearer token when given; never written to a transcript
    api_key: str | None = None
    timeout: float = 120.0
    ask_token_ids: bool = False

    @property
    def host(self) -> str:
        """The endpoint's host and port, as recorded in a transcript (no path, no credentials)."""
        parts = urlsplit(self.base_url)
        return parts.netloc.rsplit("@", 1)[-1] or self.base_url


@dataclass(frozen=True)
class Completion:
    """What came back from one call, read out of the server's JSON."""

    content: str
    reasoning: str | None
    finish_reason: str | None
    prompt_ids: list[int] | None
    completion_ids: list[int] | None
    usage: dict[str, Any] | None


def _ids(value: Any) -> list[int] | None:
    return list(value) if isinstance(value, list) and all(isinstance(v, int) for v in value) else None


class Endpoint:
    def __init__(self, config: EndpointConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._asks_ids = config.ask_token_ids
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self._client = httpx.Client(base_url=config.base_url.rstrip("/") + "/", headers=headers, timeout=config.timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post("chat/completions", json=body)
        except httpx.HTTPError as error:
            raise EndpointError(f"Could not reach {self.config.host}: {error}") from error

    def models(self) -> list[str]:
        """The names the server knows its models by (`GET {base_url}/models`), for someone choosing one."""
        try:
            response = self._client.get("models")
        except httpx.HTTPError as error:
            raise EndpointError(f"Could not reach {self.config.host}: {error}") from error
        if response.status_code >= 400:
            raise EndpointError(f"{self.config.host} answered {response.status_code}: {response.text[:300]}")
        try:
            return [str(m["id"]) for m in response.json()["data"]]
        except (ValueError, KeyError, TypeError) as error:
            raise EndpointError(f"{self.config.host} did not answer with a list of models") from error

    def chat(self, messages: list[dict[str, Any]], **params: Any) -> Completion:
        """Send the messages (and any sampling settings) and read the reply."""
        body = {"model": self.config.model, "messages": messages, **params}
        asked = self._asks_ids and TOKEN_IDS_FIELD not in body
        response = self._post({**body, TOKEN_IDS_FIELD: True} if asked else body)
        if asked and response.status_code in (400, 422):
            # most likely the server does not know the field: ask plainly, and stop asking this server
            self._asks_ids = False
            response = self._post(body)
        if response.status_code >= 400:
            raise EndpointError(f"{self.config.host} answered {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise EndpointError(f"{self.config.host} did not answer with a chat completion") from error
        return Completion(
            content=message.get("content") or "",
            # servers name a thinking model's reasoning either way
            reasoning=message.get("reasoning_content") or message.get("reasoning") or None,
            finish_reason=choice.get("finish_reason"),
            prompt_ids=_ids(data.get("prompt_token_ids")),
            completion_ids=_ids(choice.get("token_ids")),
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
        )
