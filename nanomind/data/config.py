"""
nanomind/data/config.py — Data pipeline configuration.

Efficient training on large text corpora requires:
  1. Streaming       — read data without loading everything into RAM
  2. Online tokenization — tokenize on-the-fly with caching
  3. Document packing    — pack multiple docs into one block_size chunk
                          (avoids wasting padding tokens)
  4. Data mixing         — blend multiple datasets with configurable weights
  5. Sharding            — split data into shards for multi-worker loading
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """
    Configuration for the NanoMind data pipeline.

    Attributes:
        block_size:      Context window length — all samples are exactly this long.
        batch_size:      Samples per batch.
        pack_documents:  If True, concatenate documents and chunk into block_size
                         windows instead of padding short documents. Maximises
                         GPU utilisation (no wasted padding tokens).
        eos_token_id:    End-of-document token inserted between packed documents.
        stride:          Sliding window stride when packing (block_size = no overlap).
        num_workers:     DataLoader worker processes.
        prefetch:        Number of batches to prefetch per worker.
        seed:            Random seed for shuffling.
        split_ratio:     (train, val) split ratio if single file provided.
        sources:         List of (path_or_name, weight) tuples for data mixing.
    """

    block_size:      int              = 512
    batch_size:      int              = 32
    pack_documents:  bool             = True
    eos_token_id:    int              = 0
    stride:          int | None       = None    # None = block_size (no overlap)
    num_workers:     int              = 2
    prefetch:        int              = 2
    seed:            int              = 42
    split_ratio:     tuple[float, float] = (0.9, 0.1)
    sources:         list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.block_size > 0
        assert self.batch_size > 0
        assert abs(sum(self.split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1"
        if self.stride is None:
            self.stride = self.block_size
        if self.sources:
            weights = [w for _, w in self.sources]
            assert all(w > 0 for w in weights), "source weights must be positive"
