"""
nanomind/tokenizer/char.py — Character-level tokenizer for NanoMind.

Maps every unique character in the training corpus to an integer ID.
This is the simplest possible tokenizer and works well for small datasets.
"""

from __future__ import annotations

from typing import List

from nanomind.tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.

    Each unique character in the training text gets a unique integer ID.
    Special tokens occupy the lowest IDs so they are always present.
    """

    # ── Special tokens ────────────────────────────────────────────────────────
    PAD = "<PAD>"   # Padding (ID 0)
    UNK = "<UNK>"   # Unknown character (ID 1)
    BOS = "<BOS>"   # Beginning of sequence (ID 2)
    EOS = "<EOS>"   # End of sequence (ID 3)
    SPECIAL_TOKENS: list[str] = [PAD, UNK, BOS, EOS]

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._built: bool = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, text: str) -> "CharTokenizer":
        """
        Build the character vocabulary from a raw text string.

        Special tokens are always prepended so they have the lowest IDs:
        PAD=0, UNK=1, BOS=2, EOS=3.

        Args:
            text: The full training corpus as a single string.

        Returns:
            Self (for method chaining).
        """
        vocab = self._build_vocab(text)
        self._char_to_id = {ch: i for i, ch in enumerate(vocab)}
        self._id_to_char = {i: ch for i, ch in enumerate(vocab)}
        self._built = True
        return self

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_vocab(self, text: str) -> list[str]:
        """Return ordered list of tokens: special tokens first, then sorted chars."""
        unique_chars = sorted(set(text) - set(self.SPECIAL_TOKENS))
        return self.SPECIAL_TOKENS + unique_chars

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of integer token IDs.

        Unknown characters (not seen during build) are mapped to UNK (ID 1).

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        self._require_built()
        unk_id = self._char_to_id[self.UNK]
        return [self._char_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        raise NotImplementedError

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call .build(text) or .load(path) first.")
