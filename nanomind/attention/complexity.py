"""
nanomind/attention/complexity.py — Attention memory and compute complexity analysis.
"""

from __future__ import annotations


def attention_memory_bytes(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 4,
    window_size: int | None = None,
) -> dict:
    """
    Estimate attention memory usage in bytes.

    Args:
        seq_len:     Sequence length T.
        n_heads:     Number of attention heads H.
        head_dim:    Dimension per head D_h.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).
        window_size: If provided, compute SWA memory instead of full O(T²).

    Returns:
        Dict with:
        - ``attn_matrix_bytes`` : bytes for attention score matrix
        - ``kv_cache_bytes``    : bytes for K and V cache
        - ``total_mb``          : total attention memory in MB
    """
    W = window_size if window_size is not None else seq_len

    # Attention score matrix: (H, T, W) per batch
    attn_matrix = n_heads * seq_len * W * dtype_bytes

    # KV cache: 2 × (H, T, D_h) for K and V
    kv_cache = 2 * n_heads * seq_len * head_dim * dtype_bytes

    total     = attn_matrix + kv_cache
    total_mb  = total / (1024 ** 2)

    return {
        "attn_matrix_bytes": attn_matrix,
        "kv_cache_bytes":    kv_cache,
        "total_bytes":       total,
        "total_mb":          total_mb,
        "window_size":       W,
        "seq_len":           seq_len,
    }


def print_complexity_comparison(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    window_size: int,
    dtype_bytes: int = 2,
) -> None:
    """
    Print a side-by-side memory comparison: full attention vs. SWA.
    """
    full = attention_memory_bytes(seq_len, n_heads, head_dim, dtype_bytes)
    swa  = attention_memory_bytes(seq_len, n_heads, head_dim, dtype_bytes, window_size)
    savings = 1 - swa["total_bytes"] / full["total_bytes"]

    print("=" * 55)
    print(f"Attention Memory: T={seq_len}, H={n_heads}, D_h={head_dim}")
    print("=" * 55)
    print(f"  Full attention : {full['total_mb']:.1f} MB")
    print(f"  SWA (W={window_size:4d})  : {swa['total_mb']:.1f} MB")
    print(f"  Memory savings : {savings:.1%}")
    print("=" * 55)
