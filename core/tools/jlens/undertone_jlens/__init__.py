"""Read a capture through a Jacobian lens: the words the model leans toward at each depth.

    from undertone_models import adapter, contract
    from undertone_jlens import Reader, load_lens, run

    qwen = contract("qwen3.5-4b")
    reader = Reader(adapter(qwen).load_head(qwen), load_lens(qwen, "jpp"))
    run(reader, "captures/", "lens/", baseline=shipped_baseline(reader))
"""
from .lens import Lens, LensError
from .lens import load as load_lens
from .reading import BASELINE_SCHEMA, RUN_SCHEMA, SCHEMA, Baseline, Reader, ReadingError, build_baseline, load_baseline, readings, run, shipped_baseline
from .words import Words, is_word_like

__all__ = ["BASELINE_SCHEMA", "RUN_SCHEMA", "SCHEMA", "Baseline", "Lens", "LensError", "Reader", "ReadingError", "Words",
           "build_baseline", "is_word_like", "load_baseline", "load_lens", "readings", "run", "shipped_baseline"]
