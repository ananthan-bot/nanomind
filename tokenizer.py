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


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Save the vocabulary to a JSON file."""
        import json
        from pathlib import Path as _P
        self._check_built()
        _P(path).write_text(
            json.dumps({"char_to_id": self.char_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        """Load a previously saved vocabulary from a JSON file."""
        import json
        from pathlib import Path as _P
        data = json.loads(_P(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(i): ch for ch, i in tok.char_to_id.items()}
        tok._built = True
        return tok

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK_TOKEN]

    def _check_built(self):
        if not self._built:
            raise RuntimeError("Call .build(text) first.")

    def __repr__(self) -> str:
        size = self.vocab_size if self._built else "?"
        return f"CharTokenizer(vocab_size={size})"
