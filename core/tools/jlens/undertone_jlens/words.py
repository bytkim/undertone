"""Words: the model's vocabulary gathered into word types.

A model's vocabulary is pieces of text, and one word is usually several of them (" quiet", "quiet", " Quiet"). To say
which *words* the model leans toward, the probability of every piece that spells the same word is added up.

A piece counts as a word when it is letters and digits only (an apostrophe or a hyphen may stand inside it). Pieces
that spell the same thing once trimmed and lower-cased are one word type. This is a rule about spelling, not a
dictionary: fragments of longer words ("ing") and words of any language are word types too.
"""
import hashlib
import unicodedata
from typing import Any


def is_word_like(piece: str) -> bool:
    word = piece.strip()
    if not word or "<|" in word or (word.startswith("<") and word.endswith(">")):
        return False
    return all(unicodedata.category(c)[0] in ("L", "N") or (0 < i < len(word) - 1 and c in "'-’") for i, c in enumerate(word))


def word_key(piece: str) -> str:
    return piece.replace("▁", " ").strip().lower()


class Words:
    """Every piece of the vocabulary mapped to its word type (or to none)."""

    def __init__(self, pieces: list[str], device: str = "cpu"):
        import torch

        index, types, seen = [], [], {}
        for piece in pieces:
            key = word_key(piece) if is_word_like(piece) else ""
            if not key:
                index.append(-1)
                continue
            if key not in seen:
                seen[key] = len(types)
                types.append(key)
            index.append(seen[key])
        self.types = types
        self.index = torch.tensor(index, dtype=torch.long, device=device)
        self._valid = self.index >= 0
        self._to = self.index[self._valid]
        #: a fingerprint of the word types, so a baseline made with one vocabulary is not used with another
        self.sha256 = hashlib.sha256("\n".join(types).encode()).hexdigest()

    @classmethod
    def of(cls, tokenizer: Any, vocab_size: int, device: str = "cpu") -> "Words":
        """The word types of a tokenizer's vocabulary: each id decoded on its own."""
        ids = [[i] for i in range(vocab_size)]
        if hasattr(tokenizer, "batch_decode"):
            pieces = tokenizer.batch_decode(ids, skip_special_tokens=False)
        else:
            pieces = [tokenizer.decode(i, skip_special_tokens=False) for i in ids]
        return cls(list(pieces), device)

    def __len__(self) -> int:
        return len(self.types)

    def merge(self, probs: Any) -> Any:
        """Probabilities over the vocabulary [n, vocabulary] as mass per word type [n, types]. What falls on pieces
        that are not words (punctuation, spaces, markers) is left out, so a row adds up to at most 1."""
        import torch

        out = torch.zeros(probs.shape[0], len(self.types), dtype=probs.dtype, device=probs.device)
        return out.index_add_(1, self._to, probs[:, self._valid])
