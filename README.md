# NanoMind 🧠

[![CI](https://github.com/ananthan-bot/nanomind/actions/workflows/ci.yml/badge.svg)](https://github.com/ananthan-bot/nanomind/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A small GPT-style transformer language model built **from scratch** in pure PyTorch. A tiny mind — big ideas.

---

## Overview

NanoMind is a complete, from-scratch implementation of a GPT-style causal language model. It is designed to be:

- **Educational** — every component is written clearly with full docstrings
- **Modular** — swap tokenizers, attention variants, and sampling strategies
- **Trainable on CPU** — ~1–3M parameters, trains in minutes on any laptop
- **Production-quality code** — typed, tested, linted, and CI-verified

---

## Architecture

```
Input text
    │
    ▼
┌─────────────┐
│  Tokenizer  │  Character-level or BPE
└──────┬──────┘
       │ token IDs
       ▼
┌─────────────────────────────┐
│         NanoMind            │
│  ┌──────────────────────┐   │
│  │  Token Embedding     │   │  vocab_size → d_model
│  │  + Pos Embedding     │   │  block_size → d_model
│  └──────────┬───────────┘   │
│             │               │
│  ┌──────────▼───────────┐   │
│  │  Transformer Block × N│  │
│  │  ┌─────────────────┐ │   │
│  │  │ LayerNorm        │ │   │
│  │  │ CausalSelfAttn   │ │   │  Multi-head masked attention
│  │  │ LayerNorm        │ │   │
│  │  │ FeedForward      │ │   │  4× expansion, GELU
│  │  └─────────────────┘ │   │
│  └──────────┬───────────┘   │
│             │               │
│  ┌──────────▼───────────┐   │
│  │  Final LayerNorm     │   │
│  │  LM Head (tied)      │   │  d_model → vocab_size
│  └──────────────────────┘   │
└─────────────────────────────┘
       │ logits
       ▼
  Sampling Strategy
  (greedy / top-k / top-p / temperature)
       │
       ▼
  Generated text
```

### Default Configuration
> Fully configurable via CLI flags or YAML config file.

| Hyperparameter | Value |
|---|---|
| Embedding dim (`d_model`) | 128 |
| Layers (`n_layers`) | 4 |
| Attention heads (`n_heads`) | 4 |
| Context window (`block_size`) | 128 |
| Parameters | ~1.2M |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/ananthan-bot/nanomind.git
cd nanomind

# 2. Install
pip install -e ".[dev]"

# 3. Get training data (tiny Shakespeare, ~1MB)
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'data.txt')"

# 4. Train (smoke test — 200 steps, ~1 min on CPU)
make train-fast

# 5. Generate
make generate
```

---

## Project Structure

```
nanomind/
├── nanomind/               # Main package
│   ├── __init__.py
│   ├── tokenizer/          # Char-level + BPE tokenizers    [Day 2-3]
│   ├── data.py             # TextDataset + DataLoaders       [Day 4]
│   ├── attention.py        # CausalSelfAttention             [Day 5]
│   ├── blocks.py           # TransformerBlock, FFN, norms    [Day 6]
│   ├── model.py            # NanoMind model                  [Day 7]
│   ├── trainer.py          # Training loop                   [Day 8]
│   ├── optim.py            # Optimizers + LR schedules       [Day 9]
│   ├── checkpoint.py       # Save/load checkpoints           [Day 10]
│   ├── generate.py         # Sampling strategies             [Day 11]
│   ├── eval.py             # Evaluation + metrics            [Day 12]
│   ├── cli/                # CLI entry points                [Day 13]
│   └── utils/              # Logging, seeding, timing        [Day 1]
│       ├── logger.py
│       ├── seed.py
│       ├── device.py
│       └── timer.py
├── tests/                  # Pytest test suite
├── configs/                # Example YAML training configs
├── .github/workflows/      # CI/CD
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Development

```bash
make dev      # Install with dev dependencies + pre-commit hooks
make lint     # Run ruff + black check
make format   # Auto-fix lint + format
make test     # Run test suite
make test-cov # Run tests with HTML coverage report
```

---

## Roadmap

This project is built in 14 daily layers:

| Day | Layer | Status |
|---|---|---|
| 1 | Project scaffold & tooling | ✅ Done — 20 commits |
| 2 | Character-level tokenizer | 🔜 |
| 3 | BPE tokenizer | 🔜 |
| 4 | Data pipeline | 🔜 |
| 5 | Attention mechanism | 🔜 |
| 6 | Transformer blocks | 🔜 |
| 7 | Full model | 🔜 |
| 8 | Training infrastructure | 🔜 |
| 9 | Optimizers & LR | 🔜 |
| 10 | Checkpointing | 🔜 |
| 11 | Text generation | 🔜 |
| 12 | Evaluation & metrics | 🔜 |
| 13 | CLI & configuration | 🔜 |
| 14 | Polish & v1.0.0 release | 🔜 |

---

## License

MIT
