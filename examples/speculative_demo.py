"""
examples/speculative_demo.py — Speculative decoding demo.

Shows how to pair a large target model with a small draft model
for faster generation, and compares speed vs. standard autoregressive.

Usage:
    python examples/speculative_demo.py
"""

import torch
from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.speculative import (
    SpeculativeConfig,
    SpeculativeGenerator,
    print_speculative_report,
    benchmark_speculative_vs_autoregressive,
)

# ── 1. Build two models: large (target) + small (draft) ──────────────────────
CORPUS = "the quick brown fox jumps over the lazy dog " * 50

tokenizer = CharTokenizer().build(CORPUS)
VOCAB     = tokenizer.vocab_size
BLOCK     = 32

# Target model: larger and more accurate
target_cfg = ModelConfig(
    vocab_size=VOCAB, block_size=BLOCK,
    d_model=128, n_layers=4, n_heads=4, dropout=0.0,
)
target_model = NanoMind(target_cfg)

# Draft model: smaller and faster (same vocabulary!)
draft_cfg = ModelConfig(
    vocab_size=VOCAB, block_size=BLOCK,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0,
)
draft_model = NanoMind(draft_cfg)

print(f"Target model: {target_model.num_parameters():,} params")
print(f"Draft  model: {draft_model.num_parameters():,} params")
print(f"Draft/Target ratio: {draft_model.num_parameters()/target_model.num_parameters():.1%}")

# ── 2. Speculative generation ─────────────────────────────────────────────────
device = torch.device("cpu")
gen = SpeculativeGenerator(target_model, draft_model, tokenizer, device)

cfg = SpeculativeConfig(
    n_draft=5,
    max_new_tokens=50,
    temperature=1.0,
)

text, stats = gen.generate("the ", cfg)
print(f"
Prompt: 'the '
Generated: {text}")
print_speculative_report(stats)

# ── 3. Speed comparison ───────────────────────────────────────────────────────
prompt_ids = torch.tensor([tokenizer.encode("the ")], dtype=torch.long)
print("
Benchmarking...")
results = benchmark_speculative_vs_autoregressive(
    target_model, draft_model, prompt_ids,
    n_tokens=30, n_draft=5, n_runs=2,
)
print(f"  Autoregressive : {results['autoregressive_ms']:.2f} ms/token")
print(f"  Speculative    : {results['speculative_ms']:.2f} ms/token")
print(f"  Speedup        : {results['speedup']:.2f}x")
print(f"  Acceptance rate: {results['acceptance_rate']:.2%}")
