"""NanoMind data pipeline sub-package.

Efficient streaming tokenization, document packing, sharded loading,
and multi-source data mixing for large-scale language model training.

Key features:
  - Document packing: zero padding waste — pack multiple docs into block_size
  - Data mixing:      blend multiple corpora with configurable weights
  - Shard loading:    distribute large datasets across multiple files
  - Token throughput: benchmark tokens/sec through your pipeline

Primary exports:
    - :class:`DataPipeline`         — high-level train/val DataLoader builder
    - :class:`DataConfig`           — block_size, packing, mixing, stride config
    - :class:`TextFileDataset`      — text file → tokenize → pack → Dataset
    - :class:`InMemoryTokenDataset` — fast sliding-window dataset from token array
    - :class:`MixedDataset`         — blend multiple datasets by weight
    - :class:`ShardedDataset`       — load and concatenate multiple shard files
    - :func:`pack_documents`        — concatenate and chunk documents
    - :func:`make_input_target_pairs` — (input, target) pairs from chunks
    - :func:`dataset_stats`         — summary statistics for a dataset
    - :func:`print_dataset_report`  — pretty-print dataset summary
"""

from nanomind.data.config import DataConfig
from nanomind.data.packing import pack_documents, make_input_target_pairs
from nanomind.data.text_dataset import TextFileDataset
from nanomind.data.token_dataset import InMemoryTokenDataset
from nanomind.data.mixed_dataset import MixedDataset
from nanomind.data.sharded import ShardedDataset
from nanomind.data.pipeline import DataPipeline
from nanomind.data.stats import dataset_stats, print_dataset_report, estimate_tokens_per_second

__all__ = [
    "DataConfig",
    "DataPipeline",
    "TextFileDataset",
    "InMemoryTokenDataset",
    "MixedDataset",
    "ShardedDataset",
    "pack_documents",
    "make_input_target_pairs",
    "dataset_stats",
    "print_dataset_report",
    "estimate_tokens_per_second",
]
