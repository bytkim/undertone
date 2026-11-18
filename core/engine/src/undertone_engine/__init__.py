"""undertone-engine: talk to any OpenAI-compatible endpoint and keep what was said as transcripts."""
from .contract import ModelContract, read_contract
from .endpoint import Endpoint, EndpointConfig, EndpointError
from .engine import Engine
from .settings import EndpointSettings, SettingsError
from .store import Summary, TranscriptStore
from .tokens import ChatTokenizer, hf_tokenizer
from .transcript import SCHEMA, Message, Request, Response, Template, Tokens, Transcript

__all__ = [
    "SCHEMA", "ChatTokenizer", "Endpoint", "EndpointConfig", "EndpointError", "EndpointSettings", "Engine", "Message", "ModelContract",
    "Request", "Response", "SettingsError", "Summary", "Template", "Tokens", "Transcript", "TranscriptStore", "hf_tokenizer", "read_contract",
]
