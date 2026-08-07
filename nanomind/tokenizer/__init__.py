"""NanoMind tokenizer sub-package.

Available tokenizers:
    - ``CharTokenizer`` — character-level, simplest, great for small datasets
    - ``BPETokenizer``  — byte-pair encoding, subword, better compression

Use :func:`get_tokenizer` to look up a tokenizer by name.
"""

from nanomind.tokenizer.base import BaseTokenizer
from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.bpe import BPETokenizer
from nanomind.tokenizer.factory import get_tokenizer, list_tokenizers

__all__ = [
    "BaseTokenizer",
    "CharTokenizer",
    "BPETokenizer",
    "get_tokenizer",
    "list_tokenizers",
]
