"""
nanomind/export/quantize_export.py — Post-training quantisation for export.

Reduces model size 4× with minimal quality loss by converting float32
weights to int8 using PyTorch's dynamic quantisation.

Dynamic quantisation:
  - Quantises weights to int8 at load time (offline)
  - Activations quantised per-batch at runtime
  - No calibration dataset needed (unlike static quantisation)
  - Ideal for NLP models (Linear layers dominate computation)

Size comparison (NanoMind 128d, 4L):
  float32 : 4 bytes/param → ~20 MB
  float16 : 2 bytes/param → ~10 MB
  int8    : 1 byte/param  →  ~5 MB
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.export.torchscript import export_torchscript
from nanomind.utils.logger import get_logger

log = get_logger("export.quantize")


def quantize_dynamic(
    model:      nn.Module,
    dtype:      torch.dtype = torch.qint8,
    layer_types: set | None = None,
) -> nn.Module:
    """
    Apply dynamic INT8 quantisation to a model.

    Args:
        model:       PyTorch model to quantise.
        dtype:       Quantisation dtype (default: qint8).
        layer_types: Layer types to quantise. Defaults to ``{nn.Linear}``.

    Returns:
        Quantised model (in-place and returned).
    """
    if layer_types is None:
        layer_types = {nn.Linear}
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(), layer_types, dtype=dtype
    )
    n_orig = sum(p.numel() for p in model.parameters())
    log.info(f"Dynamic INT8 quantisation applied to {n_orig:,} params")
    return quantized


def export_quantized_torchscript(
    model:         nn.Module,
    cfg:           ExportConfig,
    example_input: torch.Tensor | None = None,
) -> Path:
    """
    Quantise a model and export as TorchScript.

    Equivalent to:
      quantize_dynamic(model) → export_torchscript(...)

    Args:
        model:         Original float32 model.
        cfg:           Export configuration.
        example_input: Example input for tracing.

    Returns:
        Path to quantised TorchScript file.
    """
    quantized = quantize_dynamic(model)
    orig_cfg  = cfg.format
    cfg.format = "torchscript"
    # Save with "_int8" suffix
    orig_name  = cfg.model_name
    cfg.model_name = orig_name + "_int8"
    path = export_torchscript(quantized, cfg, example_input)
    cfg.model_name = orig_name
    cfg.format     = orig_cfg
    return path
