# Changelog

All notable changes to NanoMind are documented here.

---

## [1.9.0] — 2024 — Mixture of Experts (MoE)

### Added
- `NanoMindMoE` — full transformer model with SparseMoE FFN in every block
- `SparseMoELayer` — top-K router + N expert FFNs with weighted blending
- `TopKRouter` — linear gate, top-K selection, softmax routing weights
- `Expert` — single FFN expert (gelu/relu/swiglu activations)
- `MoETransformerBlock` — attention + MoE FFN with pre-norm residuals
- `MoEConfig` — num_experts, top_k, load_balance_coef, expert_capacity
- `load_balance_loss()` — Switch Transformer auxiliary load balancing loss
- `expert_utilization()` — per-expert token fraction statistics
- `get_all_router_stats()` — hook-based routing diagnostics across all layers
- `examples/moe_demo.py` — NanoMindMoE forward, aux loss, utilization report

---

## [1.8.0] — 2024 — Beam Search & Diverse Beam Search

### Added
- `beam_search()` — standard beam search with length penalty and no-repeat-ngram
- `diverse_beam_search()` — G-group diverse beam search (Vijayakumar et al. 2016)
- `BeamConfig` — num_beams, length_penalty, num_beam_groups, diversity_penalty
- `BeamHypothesis` / `BeamHypotheses` — hypothesis container with scored sorting
- `BeamSearchGenerator` — high-level generate() API for beam and diverse beam
- `_block_repeat_ngrams()` — no-repeat-ngram constraint for beam search
- `beam_decode()` added to generate strategies pipeline
- `examples/beam_search_demo.py` — greedy vs beam vs diverse beam comparison

---

## [1.7.0] — 2024 — Training Logging

### Added
- `TrainingLogger` — multiplex logger: fans out to all enabled backends
- `ConsoleLogger` — formatted one-liner training metrics to stdout
- `TensorBoardLogger` — scalar, histogram, hparam logging (graceful fallback)
- `WandbLogger` — W&B integration with graceful fallback
- `LogConfig` — backend, log_dir, project, run_name, log_interval config
- `MetricsBuffer` — step-level metric accumulation and averaging
- `build_loggers()` — factory to build backends from LogConfig
- `ActivationCalibrator` (Day 20 — already in quant package)
- `Trainer` now accepts an optional `TrainingLogger`
- `examples/train_with_logging.py` — full training + logging demo

---

## [1.6.0] — 2024 — INT8 Quantization

### Added
- `QuantizedLinear` — INT8 weight storage, float32 dequantize-on-forward
- `DynamicQuantizedLinear` — runtime activation + offline weight quantization
- `quantize_model()` — replace `nn.Linear` with quantized equivalents in-place
- `QuantConfig` — mode, granularity, skip_modules configuration
- `quantize_tensor()` / `dequantize_tensor()` — per-tensor and per-channel ops
- `quantization_stats()` / `quantization_error()` — size and MSE analysis
- `save_quantized_checkpoint()` / `load_quantized_checkpoint()` — INT8 I/O
- `ActivationCalibrator` — hook-based activation range collection
- `examples/quantize_demo.py` — full quantization workflow demo

---

## [1.5.0] — 2024 — Sliding Window Attention

### Added
- `SlidingWindowAttention` — O(T·W) causal local attention with window mask
- `SWARoPEAttention` — SWA + RoPE (Mistral 7B exact attention)
- `build_sliding_window_mask()` — causal + local window boolean mask
- `window_size` field in `ModelConfig` / `BlockConfig`
- `pos_type`: ``"swa"`` and ``"swa_rope"`` in get_attention() factory
- `attention_memory_bytes()` — O(T²) vs O(T·W) memory comparison
- `configs/mistral_swa.yaml` — full Mistral-style SWA config

---

## [1.4.0] — 2024 — Speculative Decoding

### Added
- `speculative_decode()` — full draft-verify-accept generation loop
- `generate_draft()` — K draft tokens + probabilities from small model
- `verify_draft()` — target model verifies all drafts in one forward pass
- `rejection_sample()` — token accept/reject with guaranteed exact distribution
- `SpeculativeGenerator` — high-level generate() API with stats
- `SpeculativeConfig` — n_draft, temperature, top_k/p, max_new_tokens
- `SpeculativeStats` — running acceptance rate tracker
- `benchmark_speculative_vs_autoregressive()` — speedup measurement
- `examples/speculative_demo.py` — target+draft pair demo with benchmarks

---

## [1.3.0] — 2024 — LoRA Fine-tuning

### Added
- `LoRALinear` — drop-in frozen linear + trainable low-rank A/B matrices
- `LoRAModel` — high-level wrapper: inject, freeze, train, merge, save/load
- `LoRAConfig` — rank, alpha, dropout, target_modules, bias config
- `inject_lora()` — replace target `nn.Linear` with `LoRALinear`
- `merge_all_lora()` / `unmerge_all_lora()` — zero-overhead inference
- `save_lora_checkpoint()` — save only A/B matrices (tiny files)
- `load_lora_checkpoint()` — load LoRA weights into injected model
- `finetune_with_lora()` — one-call fine-tuning convenience function
- `examples/lora_finetune.py` — end-to-end LoRA demo

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
