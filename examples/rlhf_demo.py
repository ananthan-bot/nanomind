"""
examples/rlhf_demo.py — RLHF demo: Reward Model training + PPO signals.

Demonstrates:
  1. Building a RewardModel from a NanoMind backbone
  2. Training on synthetic preference pairs
  3. Computing PPO signals (KL penalty + advantages)

Usage:
    python examples/rlhf_demo.py
"""

import torch
from torch.utils.data import DataLoader

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.rlhf import (
    RewardModelConfig, PPOConfig,
    RewardModel, RewardModelTrainer,
    PreferenceDataset, preference_accuracy, reward_stats,
    ValueHead, compute_gae,
    token_kl_divergence, AdaptiveKLController,
    ppo_total_loss,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog. " * 20
tokenizer = CharTokenizer().build(CORPUS)
V         = tokenizer.vocab_size

model_cfg = ModelConfig(vocab_size=V, block_size=32, d_model=64,
                        n_layers=2, n_heads=4, dropout=0.0)
backbone  = NanoMind(model_cfg)

# ── Step 2: Reward Model ──────────────────────────────────────────────────────
rm_cfg = RewardModelConfig(pooling="last", dropout=0.1)
rm     = RewardModel(backbone, model_cfg, rm_cfg)

# Synthetic preference pairs
pairs = [
    ("the quick", " brown fox", " slow turtle"),
    ("jumps over", " the lazy", " a tree"),
] * 8

ds      = PreferenceDataset(pairs, tokenizer, max_length=32)
loader  = DataLoader(ds, batch_size=4, collate_fn=PreferenceDataset.collate_fn)
opt_rm  = torch.optim.Adam(rm.parameters(), lr=1e-3)
trainer = RewardModelTrainer(rm, opt_rm, device="cpu")

print("Training Reward Model...")
for epoch in range(3):
    m = trainer.train_epoch(loader)
    print(f"  Epoch {epoch+1}: loss={m['loss']:.4f}, acc={m['accuracy']:.2%}")

# ── Step 3: PPO Signals ───────────────────────────────────────────────────────
value_head = ValueHead(d_model=64)
kl_ctrl    = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0)

# Simulate a rollout
torch.manual_seed(42)
T_gen = 10
old_log_probs = torch.randn(T_gen).log_softmax(dim=0)
cur_log_probs = old_log_probs + 0.1 * torch.randn(T_gen)
advantages    = torch.randn(T_gen)
values        = torch.randn(T_gen)
returns       = advantages + values
logits        = torch.randn(T_gen, V)

total_loss, info = ppo_total_loss(
    cur_log_probs, old_log_probs, advantages,
    values, returns, logits,
    clip_ratio=0.2, value_coef=0.1, entropy_coef=0.01
)
print(f"
PPO loss    : {info['total_loss']:.4f}")
print(f"Clip fraction: {info['clip_fraction']:.2%}")
print(f"Entropy      : {info['entropy']:.4f}")

new_kl_coef = kl_ctrl.update(current_kl=7.0)
print(f"
Adaptive KL coef: {new_kl_coef:.4f} (target KL=6.0, observed KL=7.0)")
