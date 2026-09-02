"""
nanomind/data/stats.py — Dataset statistics and inspection utilities.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


def dataset_stats(dataset: Dataset, max_samples: int = 1000) -> dict:
    """
    Compute basic statistics for a token dataset.

    Args:
        dataset:     Token dataset yielding (x, y) pairs.
        max_samples: Maximum samples to scan.

    Returns:
        Dict with ``n_samples``, ``block_size``, ``token_coverage``,
        ``mean_target``, ``unique_tokens``.
    """
    n   = min(len(dataset), max_samples)
    all_tokens: list[torch.Tensor] = []

    for i in range(n):
        x, _ = dataset[i]
        all_tokens.append(x)

    data         = torch.stack(all_tokens)       # (n, block_size)
    unique       = data.unique().numel()
    mean_val     = data.float().mean().item()

    return {
        "n_samples":      len(dataset),
        "block_size":     data.shape[1],
        "scanned":        n,
        "unique_tokens":  unique,
        "mean_token_id":  mean_val,
        "min_token_id":   data.min().item(),
        "max_token_id":   data.max().item(),
    }


def estimate_tokens_per_second(
    loader:     DataLoader,
    n_batches:  int = 10,
) -> float:
    """
    Estimate throughput: tokens per second through the data pipeline.

    Args:
        loader:    DataLoader to benchmark.
        n_batches: Number of batches to time.

    Returns:
        Estimated tokens per second.
    """
    import time
    total_tokens = 0
    t0 = time.perf_counter()
    for i, (x, _) in enumerate(loader):
        total_tokens += x.numel()
        if i + 1 >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    return total_tokens / max(elapsed, 1e-9)


def print_dataset_report(dataset: Dataset) -> None:
    """Pretty-print a dataset summary."""
    stats = dataset_stats(dataset)
    print("=" * 50)
    print("Dataset Report")
    print("=" * 50)
    print(f"  Samples         : {stats['n_samples']:,}")
    print(f"  Block size      : {stats['block_size']}")
    print(f"  Unique tokens   : {stats['unique_tokens']:,}")
    print(f"  Token ID range  : [{stats['min_token_id']}, {stats['max_token_id']}]")
    print(f"  Mean token ID   : {stats['mean_token_id']:.1f}")
    print("=" * 50)
