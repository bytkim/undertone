"""Emotion vectors: build them from stories, and read which emotions stand out in a capture.

    from undertone_emotion import load_set, run

    run(load_set("vectors/"), "captures/", "emotion/")

`shipped_set(model)` answers the vector set that comes with the tool for a model, in place of one you built.
"""
from .build import SCHEMA as VECTORS_SCHEMA
from .build import BuildError, VectorSet, build, families, load_set, shipped_set
from .corpus import STORY_PROMPT, CorpusError, write_transcripts
from .reading import RUN_SCHEMA, SCHEMA, ReadingError, read, readings, run

__all__ = ["RUN_SCHEMA", "SCHEMA", "STORY_PROMPT", "VECTORS_SCHEMA", "BuildError", "CorpusError", "ReadingError", "VectorSet",
           "build", "families", "load_set", "read", "readings", "run", "shipped_set", "write_transcripts"]
