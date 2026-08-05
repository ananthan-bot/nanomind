"""
tokenizer.py - Character-level tokenizer for NanoMind
"""

from typing import List


class CharTokenizer:
    """Maps every unique character in the training corpus to an integer ID."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self._built = False

    def build(self, text: str) -> "CharTokenizer":
        """Build vocabulary from a string of text."""
        chars = sorted(set(text))
        vocab = self.SPECIAL_TOKENS + chars
        self.char_to_id = {ch: i for i, ch in enumerate(vocab)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self._built = True
        return self

    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of integer token IDs."""
        self._check_built()
        unk = self.char_to_id[self.UNK_TOKEN]
        return [self.char_to_id.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        self._check_built()
        return "".join(self.id_to_char.get(i, self.UNK_TOKEN) for i in ids)

    @property
    def vocab_size(self) -> int:
        self._check_built()
        return len(self.char_to_id)

    def _check_built(self):
        if not self._built:
            raise RuntimeError("Call .build(text) first.")

    def __repr__(self) -> str:
        size = self.vocab_size if self._built else "?"
        return f"CharTokenizer(vocab_size={size})"
