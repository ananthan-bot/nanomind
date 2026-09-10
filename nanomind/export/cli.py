"""
nanomind/export/cli.py — CLI entry point: ``nanomind export``.

Usage::

    python -m nanomind.export.cli --format torchscript --output exported/
    python -m nanomind.export.cli --format onnx --opset 17
    python -m nanomind.export.cli --format safetensors
    python -m nanomind.export.cli --all   # export to all formats
"""

from __future__ import annotations

import argparse
import torch
from nanomind.export.config import ExportConfig
from nanomind.export.exporter import ModelExporter
from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanomind export",
        description="Export a NanoMind model to TorchScript, ONNX, or SafeTensors",
    )
    p.add_argument("--format",    default="torchscript",
                   choices=["torchscript", "onnx", "safetensors"],
                   help="Export format")
    p.add_argument("--all",       action="store_true", help="Export all formats")
    p.add_argument("--output",    default="exported_models", help="Output directory")
    p.add_argument("--opset",     type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-validate", action="store_true", help="Skip validation")
    p.add_argument("--model-name", default="nanomind", help="Model filename prefix")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    CORPUS    = "hello world " * 20
    tokenizer = CharTokenizer().build(CORPUS)
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32,
                            d_model=64, n_layers=2, n_heads=4, dropout=0.0)
    model = NanoMind(model_cfg)

    cfg = ExportConfig(
        format=args.format,
        output_path=args.output,
        opset_version=args.opset,
        validate=not args.no_validate,
        model_name=args.model_name,
    )
    exporter = ModelExporter(model, cfg)
    print(exporter.format_summary())

    if args.all:
        results = exporter.export_all()
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")
    else:
        path = exporter.export()
        print(f"  Saved to: {path}")


if __name__ == "__main__":
    main()
