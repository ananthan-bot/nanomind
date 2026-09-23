"""
examples/moe_v2_demo.py — NanoMind Advanced MoE (MoE++) demo.

Demonstrates:
  1. MoEConfig: configure n_experts, top_k, capacity_factor
  2. Expert / ExpertBank: GELU and SwiGLU FFNs
  3. TopKRouter: noisy top-K, expert utilisation
  4. ExpertChoiceRouter: expert selects tokens
  5. HashRouter: deterministic routing
  6. CapacityBuffer: token overflow analysis
  7. MoELayer: full dispatch/combine pass
  8. SparseMoETransformer: full LM, n_active_params
  9. Auxiliary losses: load_balance, z_loss, combined

Usage:
    python examples/moe_v2_demo.py
"""
import torch
import torch.nn.functional as F
from nanomind.moe_v2 import (
    MoEConfig, Expert, ExpertBank,
    TopKRouter, ExpertChoiceRouter, HashRouter,
    CapacityBuffer, MoELayer,
    SparseMoETransformer,
    load_balance_loss, z_loss, entropy_loss, combined_moe_loss,
)

V = 64
print("=" * 60)
print("NanoMind Advanced MoE (MoE++) Demo")
print("=" * 60)

# ── MoEConfig ─────────────────────────────────────────────────────────────────
print("
── MoEConfig ──")
cfg = MoEConfig(n_experts=8, top_k=2, d_model=64, d_ff=256, capacity_factor=1.25)
print(f"  {cfg.to_dict()}")
print(f"  Active ratio: {cfg.active_ratio:.2%} of parameters used per token")
print(f"  Total param estimate: {cfg.total_params_estimate:,}")

# ── Expert (GELU vs SwiGLU) ───────────────────────────────────────────────────
print("
── Expert FFN variants ──")
x = torch.randn(4, 64)
gelu_exp   = Expert(64, 256, variant="gelu")
swiglu_exp = Expert(64, 256, variant="swiglu")
print(f"  GELU expert output:   {tuple(gelu_exp(x).shape)}")
print(f"  SwiGLU expert output: {tuple(swiglu_exp(x).shape)}")

# ── ExpertBank ────────────────────────────────────────────────────────────────
print("
── ExpertBank ──")
cfg_shared = MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128, shared_experts=1)
bank = ExpertBank(cfg_shared)
print(f"  n_experts: {bank.n_experts}, shared: {len(bank.shared)}")
print(f"  Params per expert: {bank.n_params_per_expert:,}")
print(f"  Total expert params: {bank.n_total_params:,}")

# ── Routers ───────────────────────────────────────────────────────────────────
print("
── Routing Strategies ──")
N, D = 16, 64
h    = torch.randn(N, D)
cfg8 = MoEConfig(n_experts=8, top_k=2, d_model=D, d_ff=256)

# TopK
topk_router = TopKRouter(cfg8)
topk_router.train()
r = topk_router(h)
print(f"  TopK: indices={tuple(r.indices.shape)} weights={tuple(r.weights.shape)}")
util = topk_router.expert_utilisation(r.router_probs)
print(f"  Expert counts: {util['expert_counts']}")
print(f"  Router entropy: {util['entropy']:.4f}")

# ExpertChoice
ec_router = ExpertChoiceRouter(cfg8, capacity=4)
r_ec = ec_router(h)
print(f"  ExpertChoice: indices={tuple(r_ec.indices.shape)}")

# Hash
hash_router = HashRouter(cfg8)
r_h = hash_router(h)
print(f"  Hash: indices={r_h.indices.flatten().tolist()}")

# ── CapacityBuffer ────────────────────────────────────────────────────────────
print("
── Capacity Buffer ──")
cap_buf = CapacityBuffer(n_experts=8, capacity_factor=1.0)
print(f"  Capacity for 64 tokens: {cap_buf.capacity(64)} per expert")
stats   = cap_buf.stats(r.indices, N)
print(f"  Stats: {stats.to_dict()}")

# ── MoELayer ─────────────────────────────────────────────────────────────────
print("
── MoELayer (dispatch/combine) ──")
layer = MoELayer(cfg8)
x_3d  = torch.randn(2, 8, D)
out, aux = layer(x_3d)
print(f"  Output shape: {tuple(out.shape)}")
print(f"  Aux (LB) loss: {aux.item():.6f}")
rstats = layer.routing_stats(x_3d)
print(f"  Routing stats: overflow_frac={rstats['capacity']['overflow_frac']:.2%}")

# ── SparseMoETransformer ──────────────────────────────────────────────────────
print("
── Sparse MoE Transformer ──")
model = SparseMoETransformer(
    vocab_size=V, d_model=64, n_layers=2, n_heads=4, max_seq=16, moe_cfg=cfg8
)
ids    = torch.randint(0, V, (2, 8))
with torch.no_grad():
    logits, _, aux_loss = model(ids)
print(f"  Logits: {tuple(logits.shape)}")
print(f"  Total params:  {model.n_params:,}")
print(f"  Active params: {model.n_active_params:,}  "
      f"({model.n_active_params/model.n_params:.1%} of total)")

# With targets
ids2 = torch.randint(0, V, (2, 8))
logits, loss, aux = model(ids, ids2)
total_loss = loss + aux
print(f"  Task loss: {loss.item():.4f}, Aux loss: {aux.item():.6f}")

# ── Auxiliary Losses ──────────────────────────────────────────────────────────
print("
── Auxiliary Losses ──")
probs   = torch.softmax(torch.randn(32, 8), dim=-1)
logits_ = torch.randn(32, 8)
idx_    = probs.topk(2, dim=-1).indices

lb   = load_balance_loss(probs, idx_, n_experts=8)
zl   = z_loss(logits_)
el   = entropy_loss(probs)
comb = combined_moe_loss(probs, logits_, idx_, n_experts=8)
print(f"  Load balance loss: {lb.item():.6f}")
print(f"  Z-loss:            {zl.item():.6f}")
print(f"  Entropy loss:      {el.item():.6f}")
print(f"  Combined:          {comb.item():.6f}")
print("
MoE++ demo complete!")
