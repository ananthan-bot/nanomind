# 🧠 NanoMind v3.0.0

> A production-grade language model library built from scratch in 30 days — 605 commits.

**NanoMind** is a comprehensive PyTorch library implementing every major component of modern
large language models: from byte-pair tokenization to Mixture of Experts, from Flash Attention
to RLHF, and from quantization to REST API serving.

## 📦 Sub-Packages

| Package | Feature | Version |
|---------|---------|---------|
| `nanomind.tokenizer` | Char, BPE tokenizers | v1.0 |
| `nanomind.model` | Transformer (attention, blocks, norms) | v1.0 |
| `nanomind.trainer` | Training loop, optimizer, scheduler | v1.0 |
| `nanomind.generate` | Greedy, top-K/P, beam search, diverse beam | v1.0 |
| `nanomind.pos` | RoPE, ALiBi, SWA positional encodings | v1.1 |
| `nanomind.attention` | MHA, GQA, MQA, SWA | v1.2 |
| `nanomind.lora` | LoRA fine-tuning, merge/unmerge | v1.3 |
| `nanomind.speculative` | Speculative decoding with draft model | v1.4 |
| `nanomind.quant` | INT8 post-training quantization | v1.6 |
| `nanomind.logging` | TensorBoard + W&B training logging | v1.7 |
| `nanomind.moe` | Sparse Mixture of Experts (Switch Transformer) | v1.9 |
| `nanomind.data` | Streaming data pipeline, document packing, mixing | v2.0 |
| `nanomind.cache` | KV Cache — prefill + O(1) decode | v2.1 |
| `nanomind.flash` | Flash Attention — O(N) memory tiled SDPA | v2.2 |
| `nanomind.amp` | AMP, grad checkpointing, grad accumulation | v2.3 |
| `nanomind.rlhf` | RLHF — reward model + PPO + KL penalty | v2.4 |
| `nanomind.serve` | REST API server — /generate /health /info | v2.5 |
| `nanomind.dpo` | DPO — Direct Preference Optimization | v3.0 |
| `nanomind.distill` | Knowledge distillation — soft labels + features | v3.0 |

## 🚀 Quickstart

```python
from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer

tokenizer = CharTokenizer().build("your training text here")
model     = NanoMind(ModelConfig(vocab_size=tokenizer.vocab_size))

# Train
logits, loss = model(input_ids, targets)

# Serve
from nanomind.serve import ModelServer, ServeConfig
with ModelServer(model, tokenizer, ServeConfig(port=8080)) as server:
    ...  # curl http://localhost:8080/generate

# Align with DPO
from nanomind.dpo import DPOTrainer, DPOConfig
trainer = DPOTrainer(model, optimizer, DPOConfig(beta=0.1))

# Distill
from nanomind.distill import DistillTrainer, DistillConfig
d = DistillTrainer(teacher, student, optimizer, DistillConfig(temperature=4.0))
```

## 📊 Project Stats

- **30 days** of continuous development
- **605 commits** with atomic, meaningful messages
- **21 sub-packages** covering the complete modern LLM stack
- **Zero mandatory external dependencies** (serve works with stdlib only)
- Full test suite across all packages

## 📝 License
MIT
