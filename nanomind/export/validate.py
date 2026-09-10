"""
nanomind/export/validate.py — Export validation: compare original vs exported outputs.

After exporting, we verify that the exported model produces the same output
as the original PyTorch model (within floating-point tolerance).

This guards against:
  - Numerical precision changes from fp32 → fp16 conversion
  - Operator differences between PyTorch and ONNX Runtime
  - TorchScript compilation bugs (rare but possible)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.utils.logger import get_logger

log = get_logger("export.validate")


def validate_torchscript(
    original:  nn.Module,
    ts_path:   str | Path,
    n_inputs:  int   = 4,
    seq_len:   int   = 16,
    rtol:      float = 1e-3,
    atol:      float = 1e-4,
) -> dict:
    """
    Validate TorchScript export vs original PyTorch model.

    Args:
        original:  Original PyTorch model.
        ts_path:   Path to the TorchScript ``.pt`` file.
        n_inputs:  Number of random inputs to test.
        seq_len:   Sequence length for test inputs.
        rtol:      Relative tolerance for output comparison.
        atol:      Absolute tolerance for output comparison.

    Returns:
        Dict with ``passed``, ``max_abs_diff``, ``max_rel_diff``.
    """
    original.eval()
    ts_model = torch.jit.load(str(ts_path))

    emb  = next(m for m in original.modules() if isinstance(m, nn.Embedding))
    V    = emb.num_embeddings
    diffs: list[float] = []

    for _ in range(n_inputs):
        x         = torch.randint(0, V, (1, seq_len))
        with torch.no_grad():
            out_orig, _ = original(x)
            out_ts      = ts_model(x)
            if isinstance(out_ts, tuple):
                out_ts = out_ts[0]
        diffs.append((out_orig - out_ts).abs().max().item())

    max_diff = max(diffs)
    passed   = max_diff <= atol + rtol * out_orig.abs().max().item()
    log.info(f"TorchScript validation: passed={passed}, max_diff={max_diff:.2e}")
    return {"passed": passed, "max_abs_diff": max_diff, "n_inputs": n_inputs}


def model_size_report(model: nn.Module, name: str = "Model") -> dict:
    """
    Compute model size statistics.

    Args:
        model: PyTorch model.
        name:  Model display name.

    Returns:
        Dict with ``n_params``, ``trainable_params``, ``size_mb``, ``name``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = sum(p.nbytes for p in model.parameters()) / (1024 ** 2)
    return {
        "name":             name,
        "n_params":         total,
        "trainable_params": trainable,
        "frozen_params":    total - trainable,
        "size_mb":          size_mb,
        "size_mb_fp16":     size_mb / 2,
        "size_mb_int8":     size_mb / 4,
    }


def format_size_report(report: dict) -> str:
    """Format a model size report as a readable string."""
    return (
        f"Model: {report['name']}
"
        f"  Parameters : {report['n_params']:>12,}
"
        f"  Trainable  : {report['trainable_params']:>12,}
"
        f"  Size (fp32): {report['size_mb']:>10.2f} MB
"
        f"  Size (fp16): {report['size_mb_fp16']:>10.2f} MB
"
        f"  Size (int8): {report['size_mb_int8']:>10.2f} MB
"
    )
