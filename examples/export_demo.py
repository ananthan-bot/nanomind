"""
examples/export_demo.py — NanoMind model export demo.

Demonstrates exporting a trained model to all three formats:
  1. TorchScript (.pt)   — deploy without Python
  2. ONNX (.onnx)       — deploy to any runtime
  3. SafeTensors (.safetensors) — safe weight storage

Usage:
    python examples/export_demo.py
"""
import torch
from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.export import (
    ExportConfig, ModelExporter,
    model_size_report, format_size_report,
    save_safetensors, load_safetensors,
    quantize_dynamic,
)

# ── Build model ───────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog " * 10
tokenizer = CharTokenizer().build(CORPUS)
V         = tokenizer.vocab_size

torch.manual_seed(0)
cfg   = ModelConfig(vocab_size=V, block_size=32, d_model=64,
                    n_layers=2, n_heads=4, dropout=0.0)
model = NanoMind(cfg)

print("=" * 55)
print("NanoMind Model Export Demo")
print("=" * 55)

# ── Size report ───────────────────────────────────────────────────────────────
rep = model_size_report(model, "NanoMind-64d-2L")
print(f"
{format_size_report(rep)}")

# ── Export to all formats ─────────────────────────────────────────────────────
export_cfg = ExportConfig(
    output_path="exported_models",
    model_name="nanomind_demo",
    validate=True,
    dynamic_batch=True,
    dynamic_seq=True,
)
exporter = ModelExporter(model, export_cfg)
print(exporter.format_summary())
print()

results = exporter.export_all()
for fmt, path in results.items():
    if path and path.exists():
        size = path.stat().st_size / (1024 ** 2)
        print(f"  ✅ {fmt:<15} → {path.name}  ({size:.2f} MB)")
    else:
        print(f"  ⚠️  {fmt:<15} → skipped")

# ── SafeTensors roundtrip ─────────────────────────────────────────────────────
print("
── SafeTensors roundtrip ──")
st_path = "exported_models/roundtrip_test.safetensors"
save_safetensors(dict(model.state_dict()), st_path)
loaded  = load_safetensors(st_path)
first   = list(loaded.keys())[0]
match   = torch.allclose(model.state_dict()[first], loaded[first], atol=1e-6)
print(f"  Saved {len(loaded)} tensors, roundtrip match: {match}")

# ── Quantised model ───────────────────────────────────────────────────────────
print("
── INT8 Quantisation ──")
q_model = quantize_dynamic(model)
q_rep   = model_size_report(q_model, "NanoMind-INT8")
print(f"  Original : {rep['size_mb']:.2f} MB")
print(f"  INT8     : {q_rep['size_mb']:.2f} MB  ({q_rep['size_mb']/rep['size_mb']:.1%} of original)")
print("
Export demo complete!")
