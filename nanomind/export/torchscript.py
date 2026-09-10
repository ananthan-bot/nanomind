"""
nanomind/export/torchscript.py — TorchScript model export.

TorchScript compiles a PyTorch model to a static IR (Intermediate Representation)
that can run without Python — used for mobile, C++ inference, and edge deployment.

Two modes:
  torch.jit.script:  Analyses Python source code and compiles to IR.
                     Handles control flow (if/for) but requires type annotations.

  torch.jit.trace:   Runs the model with example inputs and records operations.
                     Simpler, but doesn't handle dynamic control flow.

NanoMind uses tracing (trace mode) since transformers have static control flow.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.utils.logger import get_logger

log = get_logger("export.torchscript")


def export_torchscript(
    model:      nn.Module,
    cfg:        ExportConfig,
    example_input: torch.Tensor | None = None,
) -> Path:
    """
    Export a NanoMind model to TorchScript via tracing.

    Args:
        model:         PyTorch model to export.
        cfg:           Export configuration.
        example_input: Example input tensor ``(B, T)``.
                       Generated automatically if not provided.

    Returns:
        Path to the saved ``.pt`` TorchScript file.
    """
    model.eval()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if example_input is None:
        # Get vocab size from embedding layer
        emb = next(m for m in model.modules() if isinstance(m, nn.Embedding))
        V   = emb.num_embeddings
        example_input = torch.randint(0, V, (1, 32))

    log.info(f"Tracing model with input shape {tuple(example_input.shape)}...")
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_input,))

    out_path = cfg.filename("torchscript")
    traced.save(str(out_path))
    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"TorchScript saved to {out_path} ({size_mb:.2f} MB)")
    return out_path


def load_torchscript(path: str | Path) -> torch.jit.ScriptModule:
    """
    Load a saved TorchScript model.

    Args:
        path: Path to the ``.pt`` TorchScript file.

    Returns:
        Loaded :class:`torch.jit.ScriptModule`.
    """
    return torch.jit.load(str(path))
