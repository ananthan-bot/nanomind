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


    # ── Vocabulary building helpers ───────────────────────────────────────────

    @staticmethod
    def _get_word_freqs(text: str) -> dict[str, int]:
        """
        Count word frequencies in the corpus.

        Each word is represented as a space-separated sequence of characters
        with a special end-of-word marker on the last character.

        Example:
            "hello hello world" ->
            {"h e l l o</w>": 2, "w o r l d</w>": 1}

        Args:
            text: Raw training corpus.

        Returns:
            Dict mapping space-separated character sequences to frequencies.
        """
        word_freqs: dict[str, int] = {}
        for word in text.split():
            # Convert each word to space-separated chars + end-of-word marker
            chars = list(word[:-1]) + [word[-1] + BPETokenizer.WORD_END]
            key = " ".join(chars)
            word_freqs[key] = word_freqs.get(key, 0) + 1
        return word_freqs

    @staticmethod
    def _get_pairs(word_freqs: dict[str, int]) -> dict[tuple[str, str], int]:
        """
        Count all adjacent symbol pair frequencies across all words.

        Args:
            word_freqs: Output of :meth:`_get_word_freqs`.

        Returns:
            Dict mapping ``(symbol_a, symbol_b)`` pairs to their total count.

        Example:
            {"h e l l o</w>": 2} ->
            {("h","e"):2, ("e","l"):2, ("l","l"):2, ("l","o</w>"):2}
        """
        pairs: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            symbols = word.split()
            for a, b in zip(symbols[:-1], symbols[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + freq
        return pairs

    @staticmethod
    def _merge_pair(
        pair: tuple[str, str],
        word_freqs: dict[str, int],
    ) -> dict[str, int]:
        """
        Apply a single BPE merge rule to all words.

        Replaces all occurrences of ``pair[0] + " " + pair[1]`` with
        the merged token ``pair[0] + pair[1]`` in every word.

        Args:
            pair:       The ``(a, b)`` symbol pair to merge.
            word_freqs: Current word frequency table.

        Returns:
            Updated word frequency table with the merge applied.
        """
        import re
        a, b = pair
        pattern = re.compile(r"(?<![\S])" + re.escape(a) + r" " + re.escape(b) + r"(?![\S])")
        merged = a + b
        new_freqs: dict[str, int] = {}
        for word, freq in word_freqs.items():
            new_word = pattern.sub(merged, word)
            new_freqs[new_word] = freq
        return new_freqs
