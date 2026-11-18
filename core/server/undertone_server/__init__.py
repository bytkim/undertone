"""Readings for a chat page: one reply's emotions and lens words, made from its transcript and served over HTTP.

    from undertone_server import Desk, Tools, make_server

    desk = Desk(Tools(model, reader, baseline, vectors), "transcripts/", "readings/")
    desk.reading("<transcript id>")          # made now if it has not been, then kept
    make_server(desk).serve_forever()        # GET /readings/<transcript id>
"""
from .desk import FOLDER_SCHEMA, Desk, DeskError, NoTranscript, Transcripts
from .reading import SCHEMA, Tools, Unreadable, pieces, read
from .service import make_server

__all__ = ["FOLDER_SCHEMA", "SCHEMA", "Desk", "DeskError", "NoTranscript", "Tools", "Transcripts", "Unreadable", "make_server", "pieces", "read"]
