"""
examples/cached_generation_demo.py — KV Cache generation demo.

Compares generation speed with and without KV cache, and demonstrates
the prefill + decode API.

Usage:
    python examples/cached_generation_demo.py
"""

import time, torch
from nanomind.model.config import ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.cache import (
    KVCacheConfig, NanoMindCached, KVCacheManager,
    CachedGenerator, estimate_cache_memory, print_cache_report,
)

CORPUS    = "the quick brown fox jumps over the lazy dog. " * 40
tokenizer = CharTokenizer().build(CORPUS)
VOCAB     = tokenizer.vocab_size
BLOCK     = 64

model_cfg = ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                        d_model=64, n_layers=4, n_heads=4, dropout=0.0)
cache_cfg = KVCacheConfig(
    max_batch_size=1, max_seq_len=BLOCK,
    n_layers=model_cfg.n_layers,
    n_heads=model_cfg.n_heads,
    head_dim=model_cfg.d_model // model_cfg.n_heads,
)

# ── Memory estimate ───────────────────────────────────────────────────────────
mem = estimate_cache_memory(cache_cfg)
print(f"Cache estimate: {mem['summary']}")

model = NanoMindCached(model_cfg, cache_cfg)

# ── CachedGenerator.generate() ───────────────────────────────────────────────
gen   = CachedGenerator(model, tokenizer)
prompt = "the quick"
MAX_NEW = 30

t0    = time.perf_counter()
text  = gen.generate(prompt, max_new_tokens=MAX_NEW, temperature=0.8, top_k=10)
t1    = time.perf_counter()
print(f"
Prompt   : {prompt!r}")
print(f"Generated: {text!r}")
print(f"Time     : {(t1-t0)*1000:.1f}ms for {MAX_NEW} tokens")

# ── Manual prefill + decode ───────────────────────────────────────────────────
ids    = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
cache  = model.new_cache()

logits = model.prefill(ids, cache)
print(f"
Prefill : logits {logits.shape}")

next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
logits   = model.decode_step(next_tok, cache)
print(f"1 decode: logits {logits.shape}, cache len={cache.current_len}")

print_cache_report(cache)
