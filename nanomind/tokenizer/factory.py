"""
nanomind/tokenizer/factory.py — Tokenizer factory for NanoMind.

Provides a single get_tokenizer() function that returns the right tokenizer
by name, so downstream code doesn't need to import specific classes.
"""

from __future__ import annotations

from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.bpe import BPETokenizer
from nanomind.tokenizer.bpe import BPETokenizer
from nanomind.tokenizer.base import BaseTokenizer

_REGISTRY: dict[str, type[BaseTokenizer]] = {
    "char": CharTokenizer,
    "bpe":  BPETokenizer,
    "bpe":  BPETokenizer,
}


def get_tokenizer(name: str) -> type[BaseTokenizer]:
    """
    Return a tokenizer *class* (not instance) by name.

    Args:
        name: Tokenizer name. Currently supported: ``"char"``.

    Returns:
        The tokenizer class. Call ``.build(text)`` or ``.load(path)`` on it.

    Raises:
        ValueError: If the tokenizer name is not recognised.

    Example::

        TokenizerClass = get_tokenizer("char")
        tok = TokenizerClass().build(text)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown tokenizer '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_tokenizers() -> list[str]:
    """Return a sorted list of all registered tokenizer names."""
    return sorted(_REGISTRY)
