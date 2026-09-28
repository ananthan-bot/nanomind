"""
examples/specd_demo.py — NanoMind Speculative Decoding & Fast Inference demo.

Demonstrates:
  1. N-gram draft model
  2. Small model draft (neural)
  3. Speculative sampler: verify (accept/reject)
  4. Speculative decoder: full speculation loop + stats
  5. Naive vs speculative comparison
  6. Medusa heads: multi-lookahead prediction
  7. Token tree: candidate exploration
  8. Lookahead decoding: Jacobi iteration

Usage:
    python examples/specd_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.specd import (
    NgramDraftModel, SmallModelDraft,
    SpeculativeSampler, SpeculativeResult,
    SpeculativeDecoder, GenerationStats,
    MedusaHead, MedusaModel,
    TokenTree, TreeNode,
    LookaheadDecoder,
)

V = 32   # small vocab for demo

# ── Tiny target model (for demo — normally a large LM) ────────────────────────
class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

target_model = TinyLM()

print("=" * 60)
print("NanoMind Speculative Decoding & Fast Inference Demo")
print("=" * 60)

# ── N-gram Draft Model ────────────────────────────────────────────────────────
print("
── N-gram Draft Model ──")
ngram = NgramDraftModel(vocab_size=V, n=2)
corpus = [list(range(V)), list(range(V-1, -1, -1))]
ngram.train_ngrams(corpus)
ids   = torch.tensor([[1, 2, 3]])
draft_ids, draft_logits = ngram.draft(ids, n_tokens=4)
print(f"  Context: {ids.tolist()}")
print(f"  Draft tokens (4): {draft_ids.tolist()}")
print(f"  Draft logits shape: {tuple(draft_logits.shape)}")

# ── Small Model Draft ─────────────────────────────────────────────────────────
print("
── Small Model Draft (neural) ──")
small_draft = SmallModelDraft(TinyLM())
input_ids   = torch.randint(0, V, (1, 6))
d_ids, d_lp = small_draft.draft(input_ids, n_tokens=4)
print(f"  Draft shape: {tuple(d_ids.shape)}, logits: {tuple(d_lp.shape)}")

# ── Speculative Sampler ───────────────────────────────────────────────────────
print("
── Speculative Sampler (verify) ──")
sampler     = SpeculativeSampler(temperature=1.0)
K           = 4
draft_ids_  = torch.randint(0, V, (2, K))
draft_lp_   = torch.randn(2, K, V)
target_lp_  = torch.randn(2, K + 1, V)
result      = sampler.verify(draft_ids_, draft_lp_, target_lp_)
print(f"  Accepted shape:    {tuple(result.accepted_ids.shape)}")
print(f"  n_accepted:        {result.n_accepted}")
print(f"  Acceptance rate:   {result.acceptance_rate:.2%}")
print(f"  Efficiency gain:   {result.efficiency_gain(K):.2f}x")

# Greedy verify
greedy_res = sampler.greedy_verify(draft_ids_, target_lp_)
print(f"  Greedy accepted: {greedy_res.n_accepted} tokens")

# ── SpeculativeDecoder ────────────────────────────────────────────────────────
print("
── SpeculativeDecoder (full loop) ──")
engine    = SpeculativeDecoder(target_model, ngram, k=4, temperature=1.0)
ctx       = torch.randint(0, V, (1, 8))
generated = engine.generate(ctx, max_new_tokens=12)
stats     = engine.stats
print(f"  Generated: {tuple(generated.shape)}")
print(f"  Stats: {stats.to_dict()}")

# Compare with naive
naive_out = engine.generate_naive(ctx, max_new_tokens=12)
print(f"  Naive output shape: {tuple(naive_out.shape)}")

# ── Medusa Model ──────────────────────────────────────────────────────────────
print("
── Medusa Self-Speculation ──")
medusa = MedusaModel(TinyLM(), n_heads=3, d_model=V, vocab_size=V)
ids    = torch.randint(0, V, (2, 8))
base_logits, medusa_logits = medusa(ids)
print(f"  Base logits: {tuple(base_logits.shape)}")
print(f"  Medusa heads: {len(medusa_logits)} × {tuple(medusa_logits[0].shape)}")
print(f"  Total params:  {medusa.n_params:,}")
print(f"  Medusa params: {medusa.n_medusa_params:,} ({medusa.n_medusa_params/medusa.n_params:.1%})")

base_t, medusa_t = medusa.speculate(ids, top_k_each=2)
print(f"  Base predicted token: {tuple(base_t.shape)}")
print(f"  Medusa tokens per head: {[tuple(m.shape) for m in medusa_t]}")

# Medusa loss
labels     = torch.randint(0, V, (2, 8))
base_loss, med_losses = medusa.medusa_loss(ids, labels)
print(f"  Base loss: {base_loss.item():.4f}")
print(f"  Medusa losses: {[round(l.item(),4) for l in med_losses]}")

# ── Token Tree ────────────────────────────────────────────────────────────────
print("
── Token Tree (candidate exploration) ──")
tree   = TokenTree(max_depth=3, branching=2)
cands  = [[5, 7], [3, 8], [1, 6]]   # depth 0,1,2 candidates
probs  = [[0.4, 0.3], [0.5, 0.3], [0.6, 0.2]]
root   = tree.build(cands, probs)
paths  = tree.all_paths(root)
n_nodes = tree.n_nodes(root)
print(f"  Tree nodes: {n_nodes}")
print(f"  All paths: {paths}")
# Verify against target
target_seq = [5, 3, 6]   # what target model would predict
best_path  = tree.verify_paths(root, target_seq)
print(f"  Best matching path: {best_path}")

# ── Lookahead Decoding ────────────────────────────────────────────────────────
print("
── Lookahead Decoding (Jacobi) ──")
la_decoder = LookaheadDecoder(target_model, window_size=4, n_iters=2)
ctx2       = torch.randint(0, V, (1, 6))
out        = la_decoder.generate(ctx2, max_new_tokens=8)
print(f"  Input: {tuple(ctx2.shape)} → Output: {tuple(out.shape)}")
print(f"  Generated {out.shape[1] - ctx2.shape[1]} new tokens via Jacobi iteration")

print("
Speculative decoding demo complete!")
