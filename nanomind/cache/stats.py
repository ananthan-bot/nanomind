"""
nanomind/cache/stats.py — KV cache memory and utilisation reporting.
"""

from __future__ import annotations

from nanomind.cache.config import KVCacheConfig
from nanomind.cache.manager import KVCacheManager


def print_cache_report(cache: KVCacheManager) -> None:
    """Pretty-print KV cache utilisation statistics."""
    s = cache.stats()
    print("=" * 50)
    print("KV Cache Report")
    print("=" * 50)
    print(f"  Layers          : {s['n_layers']}")
    print(f"  Tokens cached   : {s['current_len']} / {s['max_seq_len']}")
    print(f"  Fill ratio      : {s['fill_ratio']:.1%}")
    print(f"  Memory used     : {s['memory_mb']:.2f} MB")
    print(f"  Config limit    : {s['config_mb']:.2f} MB")
    print("=" * 50)


def estimate_cache_memory(cfg: KVCacheConfig) -> dict:
    """
    Estimate KV cache memory requirements before allocation.

    Args:
        cfg: KV cache configuration.

    Returns:
        Dict with ``bytes``, ``mb``, ``gb``, and a human-readable ``summary``.
    """
    b   = cfg.cache_size_bytes
    mb  = b / (1024 ** 2)
    gb  = b / (1024 ** 3)
    return {
        "bytes":   b,
        "mb":      mb,
        "gb":      gb,
        "summary": (
            f"{cfg.n_layers} layers × 2 (K+V) × "
            f"batch={cfg.max_batch_size} × "
            f"seq={cfg.max_seq_len} × "
            f"heads={cfg.n_heads} × "
            f"dim={cfg.head_dim} × "
            f"{cfg.dtype} = {mb:.1f} MB"
        ),
    }
