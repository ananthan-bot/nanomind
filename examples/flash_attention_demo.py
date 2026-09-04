"""
examples/flash_attention_demo.py — Flash Attention demo.

Demonstrates:
  1. Memory comparison: standard vs Flash Attention
  2. Numerical equivalence of tiled vs torch.sdpa output
  3. NanoMindFlash forward pass

Usage:
    python examples/flash_attention_demo.py
"""

import torch
from nanomind.model.config import ModelConfig
from nanomind.flash import (
    FlashConfig, FlashAttention, NanoMindFlash,
    tiled_flash_attention, memory_comparison_report,
    standard_attention_memory, flash_attention_memory,
)

# ── 1. Memory analysis ────────────────────────────────────────────────────────
for N in [512, 2048, 8192]:
    print(memory_comparison_report(N, batch=1, heads=8, head_dim=64))
    print()

# ── 2. Numerical equivalence ──────────────────────────────────────────────────
torch.manual_seed(42)
B, H, N, Dh = 1, 4, 64, 32
q = torch.randn(B, H, N, Dh)
k = torch.randn(B, H, N, Dh)
v = torch.randn(B, H, N, Dh)
scale = Dh ** -0.5

# Standard SDPA
import torch.nn.functional as F
std_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

# Tiled reference
tile_out = tiled_flash_attention(q, k, v, block_q=16, block_kv=16,
                                  causal=True, scale=scale)

max_diff = (std_out - tile_out).abs().max().item()
print(f"Max diff (tiled vs torch.sdpa): {max_diff:.2e}  (should be < 1e-4)")

# ── 3. NanoMindFlash forward ──────────────────────────────────────────────────
model_cfg = ModelConfig(vocab_size=256, block_size=64, d_model=64,
                        n_layers=2, n_heads=4, dropout=0.0)

# torch.sdpa backend
model_fast = NanoMindFlash(model_cfg, FlashConfig(use_torch_sdpa=True))
# Tiled reference backend
model_tile = NanoMindFlash(model_cfg, FlashConfig(use_torch_sdpa=False))

idx = torch.randint(0, 256, (1, 32))
with torch.no_grad():
    logits_fast, _ = model_fast(idx)
    logits_tile, _ = model_tile(idx)

print(f"
NanoMindFlash (torch_sdpa) logits shape: {logits_fast.shape}")
print(f"NanoMindFlash (tiled)      logits shape: {logits_tile.shape}")
print(f"Total params: {model_fast.num_parameters():,}")
