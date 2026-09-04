"""
nanomind/flash/memory.py — Memory footprint analysis for standard vs Flash Attention.
"""

from __future__ import annotations

import math


def standard_attention_memory(
    batch:    int,
    heads:    int,
    seq_len:  int,
    head_dim: int,
    dtype_bytes: int = 4,
) -> dict:
    """
    Compute theoretical memory usage of standard attention.

    Materialises a full (N × N) attention weight matrix per head.

    Args:
        batch, heads, seq_len, head_dim: Model dimensions.
        dtype_bytes: Bytes per element (4=float32, 2=float16).

    Returns:
        Dict with ``qkv_bytes``, ``attn_matrix_bytes``, ``output_bytes``, ``total_bytes``, ``total_mb``.
    """
    qkv_bytes    = 3 * batch * heads * seq_len * head_dim * dtype_bytes
    attn_bytes   = batch * heads * seq_len * seq_len * dtype_bytes     # N×N matrix
    output_bytes = batch * heads * seq_len * head_dim * dtype_bytes
    total        = qkv_bytes + attn_bytes + output_bytes
    return {
        "qkv_bytes":         qkv_bytes,
        "attn_matrix_bytes": attn_bytes,
        "output_bytes":      output_bytes,
        "total_bytes":       total,
        "total_mb":          total / (1024 ** 2),
    }


def flash_attention_memory(
    batch:    int,
    heads:    int,
    seq_len:  int,
    head_dim: int,
    block_kv: int = 64,
    dtype_bytes: int = 4,
) -> dict:
    """
    Compute theoretical memory usage of Flash Attention.

    Only materialises two tiles (Q block + KV block) at a time,
    plus the O(N) output buffer.

    Args:
        batch, heads, seq_len, head_dim: Model dimensions.
        block_kv: KV tile size.
        dtype_bytes: Bytes per element.

    Returns:
        Dict with ``qkv_bytes``, ``tile_bytes``, ``output_bytes``, ``total_bytes``, ``total_mb``.
    """
    qkv_bytes    = 3 * batch * heads * seq_len * head_dim * dtype_bytes
    # SRAM: one Q tile + one KV tile at a time
    tile_bytes   = batch * heads * block_kv * head_dim * dtype_bytes * 3   # Q + K + V tiles
    output_bytes = batch * heads * seq_len * head_dim * dtype_bytes
    total        = qkv_bytes + tile_bytes + output_bytes
    return {
        "qkv_bytes":   qkv_bytes,
        "tile_bytes":  tile_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total,
        "total_mb":    total / (1024 ** 2),
    }


def memory_comparison_report(
    seq_len: int,
    batch:   int = 1,
    heads:   int = 8,
    head_dim:int = 64,
) -> str:
    """
    Generate a string report comparing standard vs Flash Attention memory.

    Args:
        seq_len:  Sequence length to analyse.
        batch:    Batch size.
        heads:    Number of attention heads.
        head_dim: Head dimension.

    Returns:
        Formatted report string.
    """
    std   = standard_attention_memory(batch, heads, seq_len, head_dim)
    flash = flash_attention_memory(batch, heads, seq_len, head_dim)
    ratio = std["total_bytes"] / max(flash["total_bytes"], 1)

    lines = [
        f"Memory Comparison  (N={seq_len}, B={batch}, H={heads}, Dh={head_dim})",
        "-" * 60,
        f"  Standard attention  : {std['total_mb']:>8.2f} MB  "
        f"  (attn matrix: {std['attn_matrix_bytes']/(1024**2):.2f} MB)",
        f"  Flash attention     : {flash['total_mb']:>8.2f} MB  "
        f"  (tile buffer: {flash['tile_bytes']/(1024**2):.4f} MB)",
        f"  Memory saving       : {ratio:.1f}× less memory with Flash Attention",
    ]
    return "
".join(lines)
