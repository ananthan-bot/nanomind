"""
nanomind/data/split.py — Train/validation split utilities.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, random_split, Subset


def split_dataset(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Split a dataset into train and validation subsets.

    Uses a reproducible random split (seeded via a dedicated Generator)
    so the split is identical across runs.

    Args:
        dataset:      The full dataset to split.
        val_fraction: Fraction of samples for validation (default: 10%).
        seed:         Random seed for reproducibility.

    Returns:
        ``(train_subset, val_subset)`` — both are :class:`torch.utils.data.Subset`.

    Example::

        train_ds, val_ds = split_dataset(dataset, val_fraction=0.1, seed=42)
    """
    n = len(dataset)  # type: ignore[arg-type]
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds
