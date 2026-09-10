"""
nanomind/export/exporter.py — Unified model exporter.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.export.torchscript import export_torchscript
from nanomind.export.onnx_export import export_onnx
from nanomind.export.safetensors_export import export_safetensors
from nanomind.export.validate import validate_torchscript, model_size_report
from nanomind.utils.logger import get_logger

log = get_logger("export.exporter")


class ModelExporter:
    """
    Unified model export interface for NanoMind.

    Supports TorchScript, ONNX, and SafeTensors formats with
    optional post-export validation.

    Args:
        model: PyTorch model to export.
        cfg:   Export configuration.

    Example::

        exporter = ModelExporter(model, ExportConfig(format="torchscript"))
        path     = exporter.export()
        report   = exporter.size_report()
        print(exporter.format_summary())
    """

    def __init__(
        self,
        model: nn.Module,
        cfg:   ExportConfig | None = None,
    ) -> None:
        self.model = model.eval()
        self.cfg   = cfg or ExportConfig()

    def export(
        self,
        example_input: torch.Tensor | None = None,
    ) -> Path:
        """
        Export the model in the configured format.

        Args:
            example_input: Optional example input tensor for tracing.

        Returns:
            Path to the exported file.
        """
        fmt = self.cfg.format
        log.info(f"Exporting model as {fmt}...")

        if fmt == "torchscript":
            path = export_torchscript(self.model, self.cfg, example_input)
            if self.cfg.validate:
                result = validate_torchscript(
                    self.model, path,
                    rtol=self.cfg.rtol, atol=self.cfg.atol,
                )
                if result["passed"]:
                    log.info("Validation passed ✓")
                else:
                    log.warning(f"Validation failed! max_diff={result['max_abs_diff']:.2e}")
        elif fmt == "onnx":
            path = export_onnx(self.model, self.cfg, example_input)
        elif fmt == "safetensors":
            path = export_safetensors(self.model, self.cfg)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        log.info(f"Export complete: {path}")
        return path

    def export_all(self, example_input: torch.Tensor | None = None) -> dict[str, Path]:
        """Export to all three formats."""
        results = {}
        for fmt in ("torchscript", "onnx", "safetensors"):
            self.cfg.format = fmt
            try:
                results[fmt] = self.export(example_input)
            except Exception as e:
                log.warning(f"{fmt} export failed: {e}")
                results[fmt] = None
        return results

    def size_report(self) -> dict:
        """Return model size statistics."""
        return model_size_report(self.model, self.cfg.model_name)

    def format_summary(self) -> str:
        """Return a formatted export summary string."""
        rep = self.size_report()
        lines = [
            f"Export Summary: {rep['name']}",
            f"  Format     : {self.cfg.format}",
            f"  Output dir : {self.cfg.output_dir}",
            f"  Parameters : {rep['n_params']:,}",
            f"  Size fp32  : {rep['size_mb']:.2f} MB",
            f"  Size fp16  : {rep['size_mb_fp16']:.2f} MB",
        ]
        return "
".join(lines)
