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


# ── TorchScript ───────────────────────────────────────────────────────────────

class TestTorchScript:
    def test_export_creates_file(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        assert path.exists()
        assert path.suffix == ".pt"

    def test_export_file_nonzero(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        assert path.stat().st_size > 0

    def test_load_torchscript(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        ts    = load_torchscript(path)
        x     = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            out = ts(x)
        assert isinstance(out, tuple) or isinstance(out, torch.Tensor)

    def test_output_matches_original(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        ts    = load_torchscript(path)
        x     = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            orig, _ = model(x)
            ts_out  = ts(x)
            if isinstance(ts_out, tuple): ts_out = ts_out[0]
        assert torch.allclose(orig, ts_out, atol=1e-4)
