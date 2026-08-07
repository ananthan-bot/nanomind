"""
nanomind/tokenizer/base.py — Abstract base class for all NanoMind tokenizers.

Defines the interface every tokenizer must implement so downstream code
(data pipeline, generation) can work with any tokenizer interchangeably.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Abstract interface shared by all NanoMind tokenizers."""

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of integer token IDs."""
        ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the tokenizer state to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """Load a tokenizer from a previously saved file."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of strings. Returns a list of token ID lists."""
        return [self.encode(t) for t in texts]

    def batch_decode(self, id_lists: List[List[int]]) -> List[str]:
        """Decode a list of token ID lists. Returns a list of strings."""
        return [self.decode(ids) for ids in id_lists]
