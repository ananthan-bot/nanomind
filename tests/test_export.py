"""
tests/test_export.py — Tests for NanoMind model export.
"""
import json, struct, pytest, tempfile
from pathlib import Path

import torch

from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.export import (
    ExportConfig, ModelExporter,
    export_torchscript, load_torchscript,
    export_onnx,
    export_safetensors, save_safetensors, load_safetensors,
    validate_torchscript, model_size_report, format_size_report,
    quantize_dynamic,
)

CORPUS = "abcde " * 10
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
T      = 16

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=32,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── ExportConfig ──────────────────────────────────────────────────────────────

class TestExportConfig:
    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.format == "torchscript"
        assert cfg.opset_version == 17

    def test_invalid_format(self):
        with pytest.raises(AssertionError):
            ExportConfig(format="pickle")

    def test_invalid_opset(self):
        with pytest.raises(AssertionError):
            ExportConfig(opset_version=9)

    def test_filename_torchscript(self):
        cfg = ExportConfig(model_name="test")
        assert cfg.filename("torchscript").suffix == ".pt"

    def test_filename_onnx(self):
        cfg = ExportConfig(model_name="test")
        assert cfg.filename("onnx").suffix == ".onnx"

    def test_filename_safetensors(self):
        cfg = ExportConfig(model_name="test")
        assert cfg.filename("safetensors").suffix == ".safetensors"
