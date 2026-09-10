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


# ── SafeTensors ───────────────────────────────────────────────────────────────

class TestSafeTensors:
    def test_save_creates_file(self, tmp_path):
        tensors = {"a": torch.randn(4, 4), "b": torch.randn(8)}
        path    = tmp_path / "test.safetensors"
        save_safetensors(tensors, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip(self, tmp_path):
        tensors = {"w1": torch.randn(16, 8), "b1": torch.randn(16)}
        path    = tmp_path / "rt.safetensors"
        save_safetensors(tensors, path)
        loaded  = load_safetensors(path)
        assert set(loaded.keys()) == set(tensors.keys())
        for k in tensors:
            assert torch.allclose(tensors[k], loaded[k], atol=1e-6)

    def test_model_export(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(format="safetensors",
                              output_path=str(tmp_path), model_name="test")
        path  = export_safetensors(model, cfg)
        assert path.exists()
        loaded = load_safetensors(path)
        assert len(loaded) == len(list(model.state_dict()))


# ── model_size_report + validate ──────────────────────────────────────────────

class TestSizeReport:
    def test_keys(self):
        model = tiny_model()
        rep   = model_size_report(model, "Test")
        for k in ("n_params", "trainable_params", "size_mb", "size_mb_fp16"):
            assert k in rep

    def test_params_positive(self):
        model = tiny_model()
        rep   = model_size_report(model)
        assert rep["n_params"] > 0
        assert rep["size_mb"]  > 0.0

    def test_fp16_half_of_fp32(self):
        model = tiny_model()
        rep   = model_size_report(model)
        assert abs(rep["size_mb_fp16"] - rep["size_mb"] / 2) < 1e-6

    def test_format_is_string(self):
        model = tiny_model()
        rep   = model_size_report(model, "TestModel")
        s     = format_size_report(rep)
        assert isinstance(s, str)
        assert "TestModel" in s

    def test_validate_torchscript_passes(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="val")
        path  = export_torchscript(model, cfg)
        result = validate_torchscript(model, path, n_inputs=2, seq_len=T)
        assert result["passed"]
        assert result["max_abs_diff"] < 1e-3


# ── ModelExporter ─────────────────────────────────────────────────────────────

class TestModelExporter:
    def _exporter(self, tmp_path, fmt="torchscript"):
        model = tiny_model()
        cfg   = ExportConfig(format=fmt, output_path=str(tmp_path),
                              model_name="test", validate=False)
        return ModelExporter(model, cfg)

    def test_export_torchscript(self, tmp_path):
        exp  = self._exporter(tmp_path, "torchscript")
        path = exp.export()
        assert path.exists()

    def test_export_safetensors(self, tmp_path):
        exp  = self._exporter(tmp_path, "safetensors")
        path = exp.export()
        assert path.exists()

    def test_size_report(self, tmp_path):
        exp = self._exporter(tmp_path)
        rep = exp.size_report()
        assert "n_params" in rep

    def test_format_summary(self, tmp_path):
        exp     = self._exporter(tmp_path)
        summary = exp.format_summary()
        assert "torchscript" in summary

    def test_export_all(self, tmp_path):
        exp     = self._exporter(tmp_path)
        results = exp.export_all()
        # At least safetensors should work (no ONNX library required)
        assert results.get("safetensors") is not None


# ── quantize_dynamic ──────────────────────────────────────────────────────────

class TestQuantizeDynamic:
    def test_returns_model(self):
        model = tiny_model()
        q     = quantize_dynamic(model)
        assert q is not None

    def test_inference_still_works(self):
        model = tiny_model()
        q     = quantize_dynamic(model)
        x     = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            logits, _ = q(x)
        assert logits.shape == (1, T, VOCAB)

    def test_output_shape_unchanged(self):
        model = tiny_model()
        q     = quantize_dynamic(model)
        x     = torch.randint(0, VOCAB, (2, T))
        with torch.no_grad():
            orig_logits, _ = model(x)
            q_logits, _    = q(x)
        assert orig_logits.shape == q_logits.shape
