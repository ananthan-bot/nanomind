# NanoMind 🧠

> A GPT-style language model built **layer by layer** — from raw text to full training and generation.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-purple.svg)](pyproject.toml)

---

## Overview

NanoMind is a clean, fully-documented GPT-style transformer LM built from scratch in PyTorch.
It is designed to be **readable**, **modular**, and **hackable** — every layer added incrementally
with 20 atomic commits per day across a **14-day build**.

```python
from nanomind import NanoMind, ModelConfig

cfg   = ModelConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
model = NanoMind(cfg)

logits, loss = model(idx, targets)
```

---

## Features

| Feature | Details |
|---|---|
| **Tokenizers** | Char-level + BPE |
| **Attention** | SDPA, CausalSelfAttention, KV-Cache, Flash Attention |
| **Position** | Learned, RoPE (LLaMA-style), ALiBi (BLOOM-style) |
| **Attention** | MHA, GQA (Llama 2/Mistral), MQA (Falcon), GQA+RoPE |
| **Fine-tuning** | LoRA (rank, alpha, target modules, merge, save/load) |
| **Inference** | Speculative decoding (2-4x speedup, exact target distribution) |
| **Long context** | Sliding Window Attention — O(T·W) vs O(T²), Mistral-style |
| **Blocks** | TransformerBlock (Pre/Post-LN), SwiGLU / GELU FFN, RMSNorm |
| **Model** | Weight tying, GPT-2 init, `generate()` with top-k/p/beam |
| **Training** | Trainer, AMP, grad accumulation, grad clip, early stopping |
| **Optimizers** | AdamW + WarmupCosine schedule (and 4 other schedules) |
| **Checkpoints** | Atomic save/load, best tracking, auto-resume, inference ckpts |
| **Generation** | Greedy, temperature, top-k, top-p, min-p, beam search |
| **Evaluation** | PPL, BPC, accuracy, top-K, generation quality (TTR, distinct-N) |
| **CLI** | `nanomind train / generate / eval / info` |

---

## Quick Start

```bash
pip install -e ".[dev]"
python examples/train_tiny.py
```

### Train from config
```bash
nanomind train --config configs/small.yaml --data data/corpus.txt
```

### Generate text
```bash
nanomind generate --checkpoint checkpoints/best.pt --prompt "Once upon a time"
```

### Evaluate
```bash
nanomind eval --checkpoint checkpoints/best.pt --data data/val.txt
```

---

## Architecture

```
Input IDs (B, T)
    │
Token Embedding  +  Positional Embedding
    │
[TransformerBlock × N]
    │  ├─ Pre-LN / Post-LN
    │  ├─ CausalSelfAttention (SDPA / Flash Attention)
    │  └─ FeedForward (GELU or SwiGLU)
    │
Final LayerNorm
    │
LM Head  (weight-tied to token embedding)
    │
Logits (B, T, vocab_size)
```

---

## 14-Day Build Log

| Day | Layer | Status |
|---|---|---|
| 1 | Project scaffold & tooling | ✅ Done — 20 commits |
| 2 | Character-level tokenizer | ✅ Done — 20 commits |
| 3 | BPE tokenizer | ✅ Done — 20 commits |
| 4 | Data pipeline | ✅ Done — 20 commits |
| 5 | Attention mechanism | ✅ Done — 20 commits |
| 6 | Transformer blocks | ✅ Done — 20 commits |
| 7 | Full model | ✅ Done — 20 commits |
| 8 | Training infrastructure | ✅ Done — 20 commits |
| 9 | Optimizers & LR scheduling | ✅ Done — 20 commits |
| 10 | Checkpointing & resumption | ✅ Done — 20 commits |
| 11 | Text generation strategies | ✅ Done — 20 commits |
| 12 | Evaluation & metrics | ✅ Done — 20 commits |
| 13 | CLI & configuration | ✅ Done — 20 commits |
| 14 | Polish & v1.0.0 release | ✅ Done — 20 commits |

**Total: 380 commits across 19 days.**

---

## License

MIT — see [LICENSE](LICENSE).
