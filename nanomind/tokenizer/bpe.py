"""
nanomind/tokenizer/bpe.py — Byte-Pair Encoding (BPE) tokenizer for NanoMind.

BPE is a data-driven subword tokenization algorithm. It starts with a
character-level vocabulary and repeatedly merges the most frequent adjacent
symbol pair until a desired vocabulary size is reached.

Reference: Sennrich et al., 2016 — https://arxiv.org/abs/1508.07909
"""

from __future__ import annotations

from typing import List

from nanomind.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding tokenizer built from scratch.

    Workflow:
        1. ``tok = BPETokenizer()``
        2. ``tok.train(text, vocab_size=500)``   — learn merge rules
        3. ``tok.encode("hello world")``          — encode string
        4. ``tok.decode([...])``                  — decode back
        5. ``tok.save("bpe_vocab.json")``         — persist
        6. ``tok = BPETokenizer.load(...)``       — restore
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SPECIAL_TOKENS: list[str] = [PAD, UNK, BOS, EOS]
    WORD_END = "</w>"          # Marks the end of a word during BPE training

    def __init__(self) -> None:
        self._merges: list[tuple[str, str]] = []   # Ordered merge rules
        self._vocab: dict[str, int] = {}            # token -> id
        self._id_to_token: dict[int, str] = {}      # id -> token
        self._trained: bool = False

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Call .train(text) or .load(path) first.")

    # ── Abstract method stubs (implemented in later commits) ──────────────────

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        raise NotImplementedError
