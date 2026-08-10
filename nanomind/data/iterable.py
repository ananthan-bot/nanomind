"""
nanomind/data/iterable.py — Streaming dataset for large text corpora.

For corpora too large to fit in RAM, IterableTextDataset streams
fixed-size chunks from disk without loading the full file.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from nanomind.tokenizer.base import BaseTokenizer


class IterableTextDataset(IterableDataset):
    """
    Streaming language model dataset for very large text files.

    Reads the file line-by-line, encodes on-the-fly, and yields
    ``(x, y)`` sliding-window pairs without holding everything in RAM.

    Args:
        path:       Path to the text file.
        tokenizer:  A fitted tokenizer.
        block_size: Context window length.
        encoding:   File text encoding.
    """

    def __init__(
        self,
        path: str,
        tokenizer: BaseTokenizer,
        block_size: int,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._encoding = encoding

    def __iter__(self):
        buffer: list[int] = []
        bs = self._block_size

        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                buffer.extend(self._tokenizer.encode(line))
                while len(buffer) >= bs + 1:
                    chunk = buffer[: bs + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield x, y
                    buffer = buffer[bs:]  # Advance by block_size (non-overlapping)

    def __repr__(self) -> str:
        return (
            f"IterableTextDataset("
            f"path='{self._path}', "
            f"block_size={self._block_size})"
        )
