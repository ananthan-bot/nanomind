"""
nanomind/export/onnx_export.py — ONNX model export.

ONNX (Open Neural Network Exchange) is the industry-standard portable format
for deploying neural networks across frameworks and hardware:

  PyTorch → ONNX → ONNX Runtime (CPU/GPU, 2-10× faster than PyTorch)
                 → TensorRT     (NVIDIA GPU, 5-30× faster)
                 → OpenVINO     (Intel CPU/VPU)
                 → CoreML       (Apple Silicon)
                 → ONNX.js      (browser inference)
                 → AWS, Azure ML serving

Dynamic axes allow the exported model to handle variable batch sizes
and sequence lengths at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.utils.logger import get_logger

log = get_logger("export.onnx")


def export_onnx(
    model:         nn.Module,
    cfg:           ExportConfig,
    example_input: torch.Tensor | None = None,
) -> Path:
    """
    Export a NanoMind model to ONNX format.

    Args:
        model:         PyTorch model to export.
        cfg:           Export configuration.
        example_input: Example input tensor ``(B, T)``.

    Returns:
        Path to the saved ``.onnx`` file.
    """
    model.eval()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if example_input is None:
        emb = next(m for m in model.modules() if isinstance(m, nn.Embedding))
        V   = emb.num_embeddings
        example_input = torch.randint(0, V, (1, 32))

    # Build dynamic axes mapping
    dynamic_axes: dict = {"input_ids": {}}
    if cfg.dynamic_batch:
        dynamic_axes["input_ids"][0] = "batch_size"
        dynamic_axes["logits"] = {0: "batch_size"}
    if cfg.dynamic_seq:
        dynamic_axes["input_ids"][1] = "seq_len"
        if "logits" not in dynamic_axes:
            dynamic_axes["logits"] = {}
        dynamic_axes["logits"][1] = "seq_len"

    out_path = cfg.filename("onnx")
    log.info(f"Exporting to ONNX (opset={cfg.opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (example_input,),
            str(out_path),
            opset_version=cfg.opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes if (cfg.dynamic_batch or cfg.dynamic_seq) else None,
            do_constant_folding=True,
            export_params=True,
        )

    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"ONNX saved to {out_path} ({size_mb:.2f} MB)")
    return out_path
