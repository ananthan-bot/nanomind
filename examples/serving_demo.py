"""
examples/serving_demo.py — NanoMind Continuous Batching & Serving demo.

Demonstrates:
  1. KV Cache block manager (PagedAttention)
  2. Inference request lifecycle
  3. Continuous batching scheduler
  4. Prefix cache (RadixAttention)
  5. Full LLM Engine: submit, step, run_until_done
  6. Throughput benchmark: continuous vs static batching

Usage:
    python examples/serving_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.serving import (
    KVBlock, KVCacheManager,
    InferenceRequest, SamplingParams, RequestStatus,
    ContinuousBatchingScheduler, SchedulerOutput,
    PrefixCache,
    LLMEngine, EngineConfig, EngineStats,
    BenchmarkConfig, BenchmarkResult, run_benchmark, make_requests,
)

V = 64   # small vocab for demo

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind Continuous Batching & Serving Demo")
print("=" * 60)

# ── KV Cache Manager ──────────────────────────────────────────────────────────
print("
── KV Cache Manager (PagedAttention) ──")
kv = KVCacheManager(n_blocks=64, block_size=8, n_layers=2, n_kv_heads=1, d_head=16)
print(f"  Pool: {kv.n_blocks} blocks × {kv.block_size} tokens")
print(f"  Memory: {kv.stats()['memory_gb']:.3f} GB")

kv.allocate("req_1", n_tokens=16)
kv.allocate("req_2", n_tokens=8)
print(f"  After 2 allocs: {kv.stats()}")
kv.free("req_1")
print(f"  After free req_1: free_blocks={kv.n_free_blocks}")

# Copy-on-write (beam search fork)
kv.allocate("req_3", n_tokens=8)
kv.copy_on_write("req_3", "req_3_fork")
print(f"  After CoW fork: n_seqs={kv.stats()['n_sequences']}")

# ── Inference Request ──────────────────────────────────────────────────────────
print("
── Inference Request Lifecycle ──")
sp  = SamplingParams(max_new_tokens=4, temperature=0.8, top_p=0.9)
req = InferenceRequest("req_test", [1, 2, 3, 4, 5], sp)
print(f"  Status: {req.status.value}, prompt_len={req.prompt_len}")
req.status = RequestStatus.RUNNING
req.add_token(42); req.add_token(7); req.add_token(99); req.add_token(11)
print(f"  Generated: {req.generated_tokens}")
print(f"  Finished: {req.is_finished} (max_new_tokens=4)")
print(f"  Output: {req.output_ids[:8]}...")

# ── Continuous Batching Scheduler ─────────────────────────────────────────────
print("
── Continuous Batching Scheduler ──")
kv2 = KVCacheManager(n_blocks=128, block_size=8, n_layers=1, n_kv_heads=1, d_head=8)
sched = ContinuousBatchingScheduler(kv2, max_batch_size=4)
for i in range(5):
    r = InferenceRequest(f"req_{i}", list(range(i+2)),
                          SamplingParams(max_new_tokens=4))
    sched.add_request(r)
print(f"  Queued: {sched.queue_len()} requests")
out = sched.step()
print(f"  After 1 step: running={sched.running_len()}, queue={sched.queue_len()}")
print(f"  Batched tokens: {out.n_batched_tokens}")
print(f"  Scheduler stats: {sched.stats()['running']} running, {sched.stats()['waiting']} waiting")

# ── Prefix Cache ──────────────────────────────────────────────────────────────
print("
── Prefix Cache (RadixAttention) ──")
cache = PrefixCache()
sys_prompt = [10, 11, 12, 13, 14, 15]   # system prompt tokens
cache.insert(sys_prompt, block_ids=[0, 1])
# Request with same system prompt
req_tokens = sys_prompt + [20, 21, 22]
n_cached, cached_blocks = cache.lookup(req_tokens)
print(f"  System prompt len: {len(sys_prompt)}")
print(f"  Cache hit: {n_cached} tokens, blocks={cached_blocks}")
print(f"  Cache stats: {cache.stats()}")
# Request without cache hit
n2, b2 = cache.lookup([99, 88, 77])
print(f"  Miss: {n2} tokens cached")

# ── LLM Engine (full continuous batching) ─────────────────────────────────────
print("
── LLM Engine: Continuous Batching ──")
cfg    = EngineConfig(
    max_batch_size=4, n_kv_blocks=64, block_size=8,
    n_layers=1, n_kv_heads=1, d_head=16, vocab_size=V,
)
engine = LLMEngine(TinyLM(), cfg)

reqs = []
for i in range(6):
    r = engine.submit(f"r{i}", [i % V, (i+1) % V, (i+2) % V],
                       SamplingParams(max_new_tokens=6))
    reqs.append(r)

print(f"  Submitted {len(reqs)} requests")
finished = engine.run_until_done()
print(f"  Finished: {len(finished)} requests")
for r in finished[:3]:
    print(f"    {r.request_id}: {r.n_generated} tokens generated")
print(f"  Engine stats: {engine.engine_stats()['n_tokens_generated']} tokens")

# ── Benchmark ──────────────────────────────────────────────────────────────────
print("
── Throughput Benchmark ──")
bench_cfg = BenchmarkConfig(
    n_requests=10, min_prompt_len=2, max_prompt_len=6,
    min_output_len=3, max_output_len=8, vocab_size=V,
)
bench_engine = LLMEngine(TinyLM(), cfg)
result = run_benchmark(bench_engine, bench_cfg)
print(f"  Result: {result.to_dict()}")

print("
Continuous batching demo complete!")
