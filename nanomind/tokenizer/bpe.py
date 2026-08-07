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
        """
        Encode a string into a list of BPE token IDs.

        Words are split on whitespace; each word is tokenized with
        :meth:`_tokenize_word`. Unknown subwords map to UNK.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        self._require_trained()
        unk_id = self._vocab.get(self.UNK, 1)
        ids: list[int] = []
        for word in text.split():
            for subword in self._tokenize_word(word):
                ids.append(self._vocab.get(subword, unk_id))
            # Add a space token between words if it exists in the vocab
            if " " in self._vocab:
                ids.append(self._vocab[" "])
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of BPE token IDs back to a string.

        End-of-word markers (``</w>``) are replaced with spaces.
        Special tokens (PAD, BOS, EOS) are stripped.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        self._require_trained()
        skip_ids = {
            self._vocab.get(self.PAD, -1),
            self._vocab.get(self.BOS, -2),
            self._vocab.get(self.EOS, -3),
        }
        tokens = [self._id_to_token.get(i, self.UNK) for i in ids if i not in skip_ids]
        text = "".join(tokens)
        text = text.replace(self.WORD_END, " ")
        return text.strip()

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the BPE vocabulary."""
        self._require_trained()
        return len(self._vocab)

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

    def train(self, text: str, vocab_size: int = 500) -> "BPETokenizer":
        """
        Learn BPE merge rules from a text corpus.

        Starts with a character-level vocabulary and iteratively merges
        the most frequent adjacent symbol pair until ``vocab_size`` is reached.

        Args:
            text:       Raw training corpus.
            vocab_size: Target vocabulary size (including special tokens).

        Returns:
            Self (for method chaining).
        """
        # Step 1: Build initial character vocab + special tokens
        chars = sorted(set("".join(text.split())))
        base_vocab = self.SPECIAL_TOKENS + chars + [c + self.WORD_END for c in chars]
        vocab = {tok: i for i, tok in enumerate(dict.fromkeys(base_vocab))}

        # Step 2: Get initial word frequencies
        word_freqs = self._get_word_freqs(text)
        merges: list[tuple[str, str]] = []

        # Step 3: Iteratively merge most frequent pair
        n_merges = max(0, vocab_size - len(vocab))
        for _ in range(n_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            word_freqs = self._merge_pair(best, word_freqs)
            merged_token = best[0] + best[1]
            if merged_token not in vocab:
                vocab[merged_token] = len(vocab)
            merges.append(best)

        self._merges = merges
        self._vocab = vocab
        self._id_to_token = {i: t for t, i in vocab.items()}
        self._trained = True
        return self

    def _tokenize_word(self, word: str) -> list[str]:
        """
        Apply learned BPE merge rules to a single word.

        The word is first split into individual characters (with an end-of-word
        marker on the last one), then merge rules are applied in order.

        Args:
            word: A single word string (no spaces).

        Returns:
            List of BPE subword tokens.
        """
        if not word:
            return []
        # Initialise as character sequence with end-of-word marker
        symbols = list(word[:-1]) + [word[-1] + self.WORD_END]

        # Apply each merge rule in training order
        for a, b in self._merges:
            i = 0
            new_symbols: list[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols
