"""
nanomind/data/dataset.py — PyTorch Dataset for language model training.

Implements a sliding-window dataset where each sample is a pair
(x, y) of consecutive token sequences of length `block_size`.
The model learns to predict y[t] given x[0..t].
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from nanomind.tokenizer.base import BaseTokenizer


class TextDataset(Dataset):
    """
    Sliding-window character/token language modelling dataset.

    Each sample ``(x, y)`` satisfies:
        - ``x = tokens[i : i + block_size]``
        - ``y = tokens[i + 1 : i + block_size + 1]``

    so the model learns to predict the *next* token at every position.

    Args:
        tokens:     1-D integer tensor of all token IDs.
        block_size: Context window length in tokens.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        assert len(tokens) > block_size, (
            f"Dataset too small: {len(tokens)} tokens <= block_size {block_size}"
        )
        self._tokens = tokens
        self._block_size = block_size

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def block_size(self) -> int:
        """Context window length."""
        return self._block_size

    @property
    def num_tokens(self) -> int:
        """Total number of tokens in the dataset."""
        return len(self._tokens)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of valid (x, y) pairs in the dataset."""
        return len(self._tokens) - self._block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the (input, target) pair at position ``idx``.

        Args:
            idx: Starting position in the token stream.

        Returns:
            Tuple ``(x, y)`` where both are 1-D LongTensors of length
            ``block_size``.
        """
        chunk = self._tokens[idx : idx + self._block_size + 1]
        x = chunk[:-1].clone()   # input  tokens
        y = chunk[1:].clone()    # target tokens (shifted by 1)
        return x, y

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_string(
        cls,
        text: str,
        tokenizer: "BaseTokenizer",
        block_size: int,
    ) -> "TextDataset":
        """
        Build a :class:`TextDataset` from a raw text string.

        Args:
            text:       The training corpus as a string.
            tokenizer:  A fitted tokenizer (CharTokenizer or BPETokenizer).
            block_size: Context window length.

        Returns:
            A :class:`TextDataset` ready for use with a DataLoader.
        """
        ids = tokenizer.encode(text)
        tokens = torch.tensor(ids, dtype=torch.long)
        return cls(tokens, block_size)
