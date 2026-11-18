"""Capture what a model's blocks put out when it reads a transcript.

    from undertone_models import adapter, contract
    from undertone_capture import run, read, transcripts

    qwen = contract("qwen3.5-4b")
    model = adapter(qwen).load(qwen)
    run(model, transcripts(["transcripts/"]), "captures/")
    record, blocks = read("captures/", "<transcript id>")    # blocks: [32, reply tokens, 2560]
"""
from .capture import RUN_SCHEMA, SCHEMA, CaptureError, Nothing, capture, items, read, run, transcripts, verify

__all__ = ["RUN_SCHEMA", "SCHEMA", "CaptureError", "Nothing", "capture", "items", "read", "run", "transcripts", "verify"]
