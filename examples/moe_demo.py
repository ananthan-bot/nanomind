"""
examples/moe_demo.py — Mixture of Experts demo.

Shows how to build a NanoMindMoE model and inspect expert utilization.

Usage:
    python examples/moe_demo.py
"""

import torch
from nanomind.model.config import ModelConfig
from nanomind.moe import MoEConfig, NanoMindMoE, get_all_router_stats, print_moe_utilization

# ── Build model ───────────────────────────────────────────────────────────────
model_cfg = ModelConfig(
    vocab_size=256, block_size=64, d_model=128,
    n_layers=4, n_heads=4, dropout=0.0,
)
moe_cfg   = MoEConfig(
    num_experts=8,
    top_k=2,
    load_balance_coef=0.01,
    activation="swiglu",
)
model = NanoMindMoE(model_cfg, moe_cfg)
print(model)
print(f"Total params : {model.num_parameters():,}")

# Dense equivalent params (for comparison)
dense_ffn = 2 * model_cfg.d_model * (4 * model_cfg.d_model) * model_cfg.n_layers
moe_ffn   = moe_cfg.num_experts * 2 * model_cfg.d_model * (4 * model_cfg.d_model) * model_cfg.n_layers
print(f"Dense FFN params  : {dense_ffn:,}")
print(f"MoE FFN params    : {moe_ffn:,}  ({moe_ffn/dense_ffn:.1f}x more)")
print(f"Active per token  : top-{moe_cfg.top_k} of {moe_cfg.num_experts} experts")

# ── Forward pass ──────────────────────────────────────────────────────────────
idx = torch.randint(0, 256, (2, 32))
with torch.no_grad():
    logits, aux_loss = model(idx)
print(f"
Logits shape : {logits.shape}")
print(f"Aux loss     : {aux_loss.item():.4f}")

# ── Training step with combined loss ─────────────────────────────────────────
targets = torch.randint(0, 256, (2, 32))
logits, total_loss = model(idx, targets)
print(f"Total loss   : {total_loss.item():.4f}  (CE + {moe_cfg.load_balance_coef}×aux)")

# ── Expert utilization diagnostics ───────────────────────────────────────────
stats = get_all_router_stats(model, idx)
print_moe_utilization(stats)
