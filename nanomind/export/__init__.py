"""NanoMind Export sub-package — model export to TorchScript, ONNX, SafeTensors.

Supported formats:
  torchscript — PyTorch JIT compiled model, runs without Python
  onnx        — Open Neural Network Exchange, deploy to any runtime
  safetensors — HuggingFace safe weight format, zero-code, memory-mapped

Primary exports:
    - :class:`ModelExporter`          — unified export() + export_all() + size_report()
    - :class:`ExportConfig`           — format, output_path, opset, validate
    - :func:`export_torchscript`      — JIT trace export
    - :func:`load_torchscript`        — load a TorchScript .pt file
    - :func:`export_onnx`             — ONNX export with dynamic axes
    - :func:`export_safetensors`      — SafeTensors weight export
    - :func:`save_safetensors`        — raw dict→safetensors writer
    - :func:`load_safetensors`        — raw safetensors→dict reader
    - :func:`validate_torchscript`    — output correctness check
    - :func:`model_size_report`       — param count + MB statistics
    - :func:`quantize_dynamic`        — INT8 dynamic quantisation
    - :func:`export_quantized_torchscript` — quantise + export
"""

from nanomind.export.config import ExportConfig
from nanomind.export.exporter import ModelExporter
from nanomind.export.torchscript import export_torchscript, load_torchscript
from nanomind.export.onnx_export import export_onnx
from nanomind.export.safetensors_export import (
    export_safetensors, save_safetensors, load_safetensors
)
from nanomind.export.validate import validate_torchscript, model_size_report, format_size_report
from nanomind.export.quantize_export import quantize_dynamic, export_quantized_torchscript

__all__ = [
    "ExportConfig", "ModelExporter",
    "export_torchscript", "load_torchscript",
    "export_onnx",
    "export_safetensors", "save_safetensors", "load_safetensors",
    "validate_torchscript", "model_size_report", "format_size_report",
    "quantize_dynamic", "export_quantized_torchscript",
]
