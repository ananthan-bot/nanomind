"""NanoMind data pipeline sub-package.

Primary entry points:
    - :func:`get_dataloaders`             — build loaders from text + tokenizer
    - :func:`get_dataloaders_from_config` — build loaders from DataConfig
    - :class:`TextDataset`                — map-style sliding window dataset
    - :class:`IterableTextDataset`        — streaming dataset for large files
    - :class:`DataConfig`                 — configuration dataclass
"""

from nanomind.data.config import DataConfig
from nanomind.data.dataset import TextDataset
from nanomind.data.iterable import IterableTextDataset
from nanomind.data.split import split_dataset
from nanomind.data.loader import get_dataloaders, get_dataloaders_from_config
from nanomind.data.stats import dataset_stats, print_stats
from nanomind.data.prefetch import PrefetchLoader

__all__ = [
    "DataConfig",
    "TextDataset",
    "IterableTextDataset",
    "split_dataset",
    "get_dataloaders",
    "get_dataloaders_from_config",
    "dataset_stats",
    "print_stats",
    "PrefetchLoader",
]
