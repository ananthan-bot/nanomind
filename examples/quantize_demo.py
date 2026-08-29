"""
examples/quantize_demo.py — INT8 post-training quantization demo.

Shows how to quantize a NanoMind model and compare size + accuracy.

Usage:
    python examples/quantize_demo.py
"""

import copy
import torch
from nanomind import NanoMind, ModelConfig
from nanomind.quant import (
    QuantConfig,
    quantize_model,
    print_quantization_report,
    save_quantized_checkpoint,
)

# ── 1. Build a float32 model ──────────────────────────────────────────────────
cfg   = ModelConfig(vocab_size=256, block_size=64, d_model=128,
                    n_layers=4, n_heads=4, dropout=0.0)
model = NanoMind(cfg)
print(f"Original: {model.num_parameters():,} parameters")

# ── 2. Quantize to INT8 ───────────────────────────────────────────────────────
quant_cfg    = QuantConfig(mode="weight_only", granularity="per_channel")
quant_model  = copy.deepcopy(model)
quantize_model(quant_model, quant_cfg)

# ── 3. Compare sizes ──────────────────────────────────────────────────────────
print_quantization_report(model, quant_model)

# ── 4. Check output consistency ───────────────────────────────────────────────
idx = torch.randint(0, 256, (1, 16))
with torch.no_grad():
    logits_fp32, _ = model(idx)
    logits_int8, _ = quant_model(idx)

mse = ((logits_fp32 - logits_int8) ** 2).mean().item()
print(f"
Logit MSE (fp32 vs int8): {mse:.6f}")
print("Quantization complete — model ready for deployment!")

# ── 5. Save quantized checkpoint ─────────────────────────────────────────────
save_quantized_checkpoint(quant_model, "checkpoints/model_int8.pt",
                          metadata={"bits": 8, "granularity": "per_channel"})
