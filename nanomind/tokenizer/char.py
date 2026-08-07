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

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._built: bool = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, text: str) -> "CharTokenizer":
        """Build vocabulary from a raw text string."""
        raise NotImplementedError("Coming next commit")

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

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
