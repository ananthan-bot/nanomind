# Changelog

All notable changes to NanoMind are documented here.

---

## [1.2.0] — 2024 — GQA & MQA

### Added
- `GroupedQueryAttention` — GQA with configurable n_kv_heads
- `MultiQueryAttention` — MQA (n_kv_heads=1)
- `GQARoPEAttention` — Llama 2 / Mistral exact attention
- `repeat_kv()` — expand KV heads to match query heads
- `n_kv_heads` field in `ModelConfig` (None = standard MHA)
- `pos_type`: ``"gqa"``, ``"mqa"``, ``"gqa_rope"`` in factory
- `configs/mistral_style.yaml` and `configs/llama2_style.yaml`

---

## [1.1.0] — 2024 — RoPE & ALiBi

### Added
- RoPE (Rotary Position Embeddings) — `pos_type="rope"` in ModelConfig
- ALiBi (Attention with Linear Biases) — `pos_type="alibi"` in ModelConfig
- `get_attention()` factory to swap positional embedding type via config
- `configs/rope.yaml` — LLaMA-style config with RoPE + RMSNorm + SwiGLU
- NanoMind now skips learned pos_emb when pos_type is rope or alibi

---

## [1.0.0] — 2024 — Initial Release 🎉

### Added — 14-Day Build (280 commits)

- **Day 1** — Project scaffold, CI/CD, utils (`logger`, `seed`, `device`, `timer`, `format`)
- **Day 2** — Character-level tokenizer (`BaseTokenizer`, `CharTokenizer`, factory)
- **Day 3** — BPE tokenizer (merge learning, encode/decode, factory registration)
- **Day 4** — Data pipeline (`DataConfig`, `TextDataset`, `IterableTextDataset`, `PrefetchLoader`)
- **Day 5** — Attention mechanism (SDPA, `CausalSelfAttention`, `KVCache`, Flash Attention dispatch)
- **Day 6** — Transformer blocks (`TransformerBlock` Pre/Post-LN, `FeedForward` GELU/SwiGLU, `RMSNorm`)
- **Day 7** — Full NanoMind model (embeddings, N blocks, weight tying, GPT-2 init, `generate()`, `ModelConfig`)
- **Day 8** — Training infrastructure (`Trainer`, AMP, gradient accumulation, gradient clipping, early stop)
- **Day 9** — Optimizers & LR scheduling (AdamW factory, param groups, `WarmupCosine`/`Cosine`/`Linear` schedules)
- **Day 10** — Checkpointing (atomic save/load, `CheckpointManager`, best tracking, `auto_resume`, inference ckpts)
- **Day 11** — Text generation (greedy, temperature, top-k, top-p, min-p, beam search, `Generator`, `stream()`)
- **Day 12** — Evaluation & metrics (PPL, BPC, accuracy, top-K, `Evaluator`, benchmark, generation quality)
- **Day 13** — CLI (`nanomind train/generate/eval/info`, `NanoMindConfig`, JSON/YAML config I/O)
- **Day 14** — Polish & v1.0.0 release (public API, pyproject.toml, LICENSE, integration tests, full README)

### Architecture

- GPT-style causal transformer with configurable depth, width, and attention heads
- Pre-Norm and Post-Norm variants
- SwiGLU and GELU feed-forward options
- RMSNorm and LayerNorm support
- Tied token embedding / LM head weights
- KV-Cache for efficient autoregressive inference
- Flash Attention dispatch for PyTorch 2.0+

### Testing

- 200+ unit tests across all modules
- End-to-end integration tests: tokenize → train → checkpoint → generate → evaluate
