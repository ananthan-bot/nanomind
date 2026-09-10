## [3.3.0] — 2024 — Prompt Templates & Chat Format

### Added
- `PromptManager` — stateful build()/add_assistant()/reset() multi-turn
- `ChatMLTemplate` — OpenAI <|im_start|>/<|im_end|> format
- `LLaMA2Template` — Meta [INST]/<<SYS>> format
- `LLaMA3Template` — Meta <|start_header_id|> format
- `AlpacaTemplate` — Stanford ### Instruction/Response format
- `CompletionTemplate` — plain Human:/Assistant: format
- `FewShotBuilder` — k-shot prompt construction with any template
- `PromptConfig` — template, system_prompt, max_history_turns
- `get_template()` / `list_templates()` / `register_template()` registry
- `Role`, `Message`, `Conversation` — chat primitive types
- `examples/prompt_demo.py` — all templates + few-shot + PromptManager

---

## [3.2.0] — 2024 — Model Export: TorchScript, ONNX, SafeTensors

### Added
- `ModelExporter` — unified export() + export_all() + size_report()
- `ExportConfig` — format, output_path, opset, dynamic_batch/seq, validate
- `export_torchscript()` — JIT trace export + `load_torchscript()`
- `export_onnx()` — ONNX export with dynamic axes and constant folding
- `export_safetensors()` — SafeTensors weight export (pure Python)
- `save_safetensors()` / `load_safetensors()` — raw tensor I/O
- `validate_torchscript()` — output correctness check post-export
- `model_size_report()` — fp32/fp16/int8 size statistics
- `quantize_dynamic()` — INT8 dynamic quantisation
- `export_quantized_torchscript()` — quantise + export pipeline
- `nanomind/export/cli.py` — CLI: nanomind export --format onnx
- `examples/export_demo.py` — full export + roundtrip demo

---

## [3.1.0] — 2024 — Benchmarking & Evaluation Suite

### Added
- `EvalRunner` — unified run() + format_report() + compare() interface
- `BenchmarkConfig` — batch_size, seq_len, n_trials, top_k_values
- `compute_perplexity()` — NLL-based PPL with bits-per-char over datasets
- `perplexity_from_logits()` — PPL directly from logits
- `benchmark_prefill()` — tokens/sec timing with warmup + N trials
- `benchmark_memory()` — parameter + buffer memory in MB
- `full_benchmark_report()` — formatted speed + memory string
- `top_k_accuracy()` / `multi_k_accuracy()` — token prediction accuracy
- `evaluate_accuracy()` — full dataset top-K accuracy evaluation
- `examples/benchmark_demo.py` — large vs small model comparison

---

## [3.0.0] — 2024 — Grand Finale: DPO + Knowledge Distillation

### Added
- `DPOTrainer` — frozen reference model + DPO/IPO loss + train_step/epoch
- `DPODataset` — (prompt, chosen, rejected) pairs with completion masks
- `DPOConfig` — beta, label_smoothing, loss_type (sigmoid/ipo)
- `dpo_loss()` — DPO + IPO loss with reward margin/accuracy metrics
- `compute_log_probs()` — masked sequence log-probabilities
- `reference_free_dpo_loss()` — DPO without reference model
- `DistillTrainer` — frozen teacher, soft+hard KD, compression_ratio()
- `DistillConfig` — temperature, alpha, feature_distill
- `distillation_loss()` — combined CE + KL divergence (soft labels)
- `soft_cross_entropy()` — soft-label KL divergence with temperature
- `feature_distillation_loss()` — hidden state MSE matching
- `scripts/project_stats.py` — codebase statistics
- `examples/grand_finale_demo.py` — DPO + distillation showcase

### Changed
- Complete README overhaul with full package table and quickstart guide

---

# Changelog

All notable changes to NanoMind are documented here.

---

## [2.5.0] — 2024 — Model Serving: REST API Inference Server

### Added
- `ModelServer` — HTTP inference server with start/stop/background lifecycle
- `InferenceEngine` — model + tokenizer wrapper with stop-string detection
- `NanoMindClient` — stdlib-only HTTP client (/health /info /generate /tokenize)
- `ServeConfig` — host, port, max_tokens, temperature defaults, log_requests
- `GenerateRequest` / `GenerateResponse` — typed request/response schemas
- `TokenizeRequest` / `TokenizeResponse` — tokenize endpoint schemas
- `HealthResponse` / `InfoResponse` / `ErrorResponse` — endpoint schemas
- `TokenBucketRateLimiter` — thread-safe token bucket rate limiting
- `nanomind/serve/cli.py` — CLI entry point: `nanomind serve`
- `examples/serve_demo.py` — background server + client request demo

---

## [2.4.0] — 2024 — RLHF: Reward Model + PPO

### Added
- `RewardModel` — transformer backbone + scalar head for preference scoring
- `RewardModelTrainer` — train_step() + evaluate() on preference pairs
- `PreferenceDataset` — (prompt, chosen, rejected) dataset with collate_fn
- `RewardModelConfig` / `PPOConfig` — RLHF hyperparameter dataclasses
- `preference_loss()` — Bradley-Terry pairwise ranking loss
- `preference_accuracy()` / `reward_stats()` — RM evaluation metrics
- `ValueHead` — per-token value estimates for actor-critic PPO
- `compute_gae()` — Generalised Advantage Estimation (λ-returns)
- `token_kl_divergence()` / `approx_token_kl()` — KL penalty computation
- `AdaptiveKLController` — dynamic β controller (Ziegler et al. 2019)
- `PPORolloutBuffer` — rollout collection with GAE + advantage normalisation
- `ppo_total_loss()` — clip + value + entropy combined PPO objective
- `examples/rlhf_demo.py` — reward model training + PPO signals demo

---

## [2.3.0] — 2024 — Mixed Precision Training & Gradient Checkpointing

### Added
- `AMPTrainer` — autocast + GradScaler + accumulation + grad clipping in one loop
- `AMPConfig` — dtype, grad_scaler, grad_accum_steps, checkpoint_layers, clip_grad_norm
- `NanoGradScaler` — float16 loss scaling (scale/step/update/unscale)
- `GradAccumulator` — micro-batch counter with should_step() and loss_scale
- `mixed_precision_context()` — device-aware autocast context manager
- `CheckpointedTransformerBlock` — block with activation checkpointing
- `checkpointed_forward()` — run any module with gradient checkpointing
- `apply_gradient_checkpointing()` — patch all blocks in-place
- `estimate_activation_memory()` — memory estimate with/without checkpointing
- `memory_tracker()` — context manager for GPU memory profiling
- `model_parameter_memory_mb()` — parameter memory breakdown
- `examples/amp_training_demo.py` — bfloat16 AMP + grad accum demo

---

## [2.2.0] — 2024 — Flash Attention

### Added
- `NanoMindFlash` — transformer model using FlashAttention in every block
- `FlashAttention` — drop-in SDPA with torch.sdpa + tiled fallback backends
- `FlashTransformerBlock` — RMSNorm + FlashAttention + SwiGLU FFN block
- `FlashConfig` — block_q, block_kv, causal, use_torch_sdpa
- `tiled_flash_attention()` — pure-PyTorch O(N) memory reference implementation
- `OnlineSoftmaxState` — streaming max/sum accumulator (core of Flash Attention)
- `standard_attention_memory()` / `flash_attention_memory()` — memory analysis
- `memory_comparison_report()` — N×N vs tile memory savings report
- `examples/flash_attention_demo.py` — memory analysis and equivalence demo

---

## [2.1.0] — 2024 — KV Cache for Fast Inference

### Added
- `NanoMindCached` — transformer with `prefill()` + `decode_step()` KV cache API
- `CachedGenerator` — high-level `generate()` with temperature/top-K/top-P
- `KVCacheManager` — multi-layer cache manager with `stats()` and memory tracking
- `LayerKVCache` — per-layer pre-allocated K/V storage with overflow check
- `KVCacheConfig` — `cache_size_bytes` / `cache_size_mb` pre-allocation estimation
- `CachedSelfAttention` — attention module with cache-aware prefill/decode modes
- `CachedTransformerBlock` — transformer block with cache passthrough
- `estimate_cache_memory()` — pre-allocation memory estimate by config
- `print_cache_report()` — pretty-print cache utilisation
- `examples/cached_generation_demo.py` — speed, prefill, decode demo

---

## [2.0.0] — 2024 — Streaming Data Pipeline

### Added
- `DataPipeline` — high-level train/val DataLoader builder with source mixing
- `DataConfig` — block_size, packing, mixing sources, stride, split_ratio
- `TextFileDataset` — text file → tokenize → pack → Dataset
- `InMemoryTokenDataset` — zero-overhead sliding-window dataset from token array
- `MixedDataset` — blend multiple datasets with configurable sampling weights
- `ShardedDataset` — load and concatenate multiple shard files into one dataset
- `pack_documents()` — concatenate docs with EOS and chunk into blocks
- `make_input_target_pairs()` — (input, target) pairs from chunks
- `dataset_stats()` / `print_dataset_report()` — dataset inspection
- `estimate_tokens_per_second()` — data pipeline throughput benchmark
- `examples/data_pipeline_demo.py` — packing, mixing, and throughput demo

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
