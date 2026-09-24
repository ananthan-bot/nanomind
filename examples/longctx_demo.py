"""
examples/longctx_demo.py — NanoMind Long-Context & Efficient Attention demo.

Demonstrates:
  1. RoPE: rotary position embeddings, linear/NTK scaling
  2. ALiBi: linear attention bias, no positional encodings
  3. Sliding Window Attention: O(T×W) with attention sinks
  4. Linear Attention: O(T) kernel attention
  5. RetNetDecay: decayed linear attention
  6. Grouped Query Attention (GQA/MQA)
  7. Chunked Attention: FlashAttention-style memory efficiency
  8. LongContextLM: full LM with all variants

Usage:
    python examples/longctx_demo.py
"""
import math
import torch
from nanomind.longctx import (
    RotaryEmbedding, ALiBi, SlidingWindowAttention,
    LinearAttention, RetNetDecay, GroupedQueryAttention,
    ChunkedAttention, LongContextConfig, LongContextLM,
)

V = 64
print("=" * 60)
print("NanoMind Long-Context & Efficient Attention Demo")
print("=" * 60)

# ── RoPE ──────────────────────────────────────────────────────────────────────
print("
── Rotary Position Embedding (RoPE) ──")
rope = RotaryEmbedding(dim=32, base=10000.0, max_seq=128)
q    = torch.randn(2, 4, 16, 32)   # (B, H, T, D_head)
k    = torch.randn(2, 4, 16, 32)
q_r, k_r = rope.apply(q, k, seq_len=16)
print(f"  Q_rotated shape: {tuple(q_r.shape)}")
# Verify: relative position property
print(f"  RoPE config: {rope.to_dict()}")
# NTK scaling for long contexts
rope_ntk = RotaryEmbedding(dim=32, max_seq=512, scale_factor=4.0, scaling_type="ntk")
q_r2, _ = rope_ntk.apply(q, k, seq_len=16)
print(f"  NTK-scaled output shape: {tuple(q_r2.shape)}")

# ── ALiBi ────────────────────────────────────────────────────────────────────
print("
── ALiBi: Attention with Linear Biases ──")
alibi  = ALiBi(n_heads=4, max_seq=64)
bias   = alibi.bias(seq_len=16)
print(f"  Bias shape: {tuple(bias.shape)}")
print(f"  Slopes: {[round(s,4) for s in alibi.slopes.tolist()]}")
# Apply to dummy scores
scores = torch.randn(2, 4, 16, 16)
biased = alibi.apply_to_scores(scores, seq_len=16)
print(f"  Biased scores shape: {tuple(biased.shape)}")
print(f"  Causal mask applied: future positions = -inf? "
      f"{biased[0, 0, 0, 1].item() == float('-inf')}")

# ── Sliding Window ────────────────────────────────────────────────────────────
print("
── Sliding Window Attention (Mistral-style) ──")
swa = SlidingWindowAttention(d_model=32, n_heads=4, window_size=8, n_sinks=2)
x   = torch.randn(2, 16, 32)
out, kv = swa(x)
print(f"  SWA output: {tuple(out.shape)}")
print(f"  Effective context (4 layers): {swa.effective_context(4)}")

# ── Linear Attention ──────────────────────────────────────────────────────────
print("
── Linear Attention: O(T) complexity ──")
la  = LinearAttention(d_model=32, n_heads=4, feature="elu")
x   = torch.randn(2, 8, 32)
out = la(x)
print(f"  Linear attn output: {tuple(out.shape)}")
print(f"  Complexity: {la.complexity}")

# ── RetNetDecay ───────────────────────────────────────────────────────────────
print("
── RetNet Decay Attention ──")
retnet = RetNetDecay(d_model=32, n_heads=4, gamma_min=0.9, gamma_max=0.999)
out    = retnet(x)
print(f"  RetNet output: {tuple(out.shape)}")
print(f"  Gamma range: [{retnet.gammas.min():.3f}, {retnet.gammas.max():.3f}]")

# ── GQA ───────────────────────────────────────────────────────────────────────
print("
── Grouped Query Attention (GQA/MQA) ──")
# MHA: 4 Q heads, 4 KV heads
mha = GroupedQueryAttention(32, n_heads=4, n_kv_heads=4)
# GQA: 4 Q heads, 2 KV heads
gqa = GroupedQueryAttention(32, n_heads=4, n_kv_heads=2)
# MQA: 4 Q heads, 1 KV head
mqa = GroupedQueryAttention(32, n_heads=4, n_kv_heads=1)
x   = torch.randn(2, 8, 32)
with torch.no_grad():
    o_mha, _ = mha(x); o_gqa, _ = gqa(x); o_mqa, _ = mqa(x)
print(f"  MHA output: {tuple(o_mha.shape)}  KV factor: {mha.kv_cache_factor:.2f}")
print(f"  GQA output: {tuple(o_gqa.shape)}  KV factor: {gqa.kv_cache_factor:.2f}")
print(f"  MQA output: {tuple(o_mqa.shape)}  KV factor: {mqa.kv_cache_factor:.2f}")
print(f"  GQA info: {gqa.to_dict()}")

# ── Chunked Attention ──────────────────────────────────────────────────────────
print("
── Chunked Attention (FlashAttention-style) ──")
ca  = ChunkedAttention(d_model=32, n_heads=4, chunk_size=4, causal=True)
x   = torch.randn(2, 16, 32)
out = ca(x)
print(f"  Chunked output: {tuple(out.shape)}")

# ── LongContextLM ─────────────────────────────────────────────────────────────
print("
── LongContextLM ──")
for attn_type, pos_enc in [("gqa", "rope"), ("sliding", "alibi"), ("linear", "none")]:
    cfg = LongContextConfig(
        vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq=32, window_size=8, attn_type=attn_type, pos_encoding=pos_enc,
    )
    model = LongContextLM(cfg)
    ids   = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        logits, _ = model(ids)
    eff_ctx = model.effective_context()
    print(f"  [{attn_type:8}+{pos_enc:5}] logits={tuple(logits.shape)} "
          f"params={model.n_params:,} eff_ctx={eff_ctx}")

print("
Long-context demo complete!")
