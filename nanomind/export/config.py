"""
nanomind/export/config.py — Model export configuration.

## Why Export?

Training happens in PyTorch (research-friendly, dynamic graphs, autograd).
Deployment happens in optimised runtimes:

  TorchScript:    PyTorch's own compiled IR — deploy without Python.
                  Used by: mobile apps (iOS/Android via LibTorch),
                           C++ inference servers, edge devices.

  ONNX:           Open Neural Network Exchange — portable format.
                  Deploy to: ONNX Runtime, TensorRT, OpenVINO, CoreML,
                             ONNX.js, cloud ML services (Azure, AWS SageMaker).

  SafeTensors:    HuggingFace's safe, zero-copy weight format.
                  Replaces pickle-based .pt files — immune to code injection.
                  Used by all modern HuggingFace models.

Export pipeline:
  NanoMind (PyTorch) → export() → Optimised Runtime
                                       ↓
                               fast, safe, portable inference

Reference:
  ONNX: https://onnx.ai/
  SafeTensors: https://github.com/huggingface/safetensors
  TorchScript: https://pytorch.org/docs/stable/jit.html
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportConfig:
    """
    Configuration for model export.

    Attributes:
        format:         Export format: ``"torchscript"``, ``"onnx"``, or
                        ``"safetensors"``.
        output_path:    Directory to write exported files.
        opset_version:  ONNX opset version (default: 17).
        dynamic_batch:  If True, export with dynamic batch dimension.
        dynamic_seq:    If True, export with dynamic sequence length.
        validate:       If True, validate exported model against original.
        rtol:           Relative tolerance for output validation.
        atol:           Absolute tolerance for output validation.
        model_name:     Name used in exported filenames.
    """

    format:         str   = "torchscript"
    output_path:    str   = "exported_models"
    opset_version:  int   = 17
    dynamic_batch:  bool  = True
    dynamic_seq:    bool  = True
    validate:       bool  = True
    rtol:           float = 1e-3
    atol:           float = 1e-5
    model_name:     str   = "nanomind"

    def __post_init__(self) -> None:
        assert self.format in ("torchscript", "onnx", "safetensors")
        assert self.opset_version >= 11
        assert self.rtol > 0
        assert self.atol > 0

    @property
    def output_dir(self) -> Path:
        return Path(self.output_path)

    def filename(self, suffix: str) -> Path:
        ext = {"torchscript": ".pt", "onnx": ".onnx", "safetensors": ".safetensors"}
        return self.output_dir / f"{self.model_name}{ext.get(suffix, suffix)}"
