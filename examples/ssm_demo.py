"""
examples/ssm_demo.py — NanoMind SSM (Mamba/S4) demo.

Usage:
    python examples/ssm_demo.py
"""
import torch
import torch.nn.functional as F
from nanomind.ssm import (
    SSMConfig, make_hippo_matrix, DiscretizedSSM,
    SelectiveSSM, MambaBlock, MambaLM,
    S4Layer, S4Block, S4Model,
    LinearAttnConfig, LinearAttention, RetentiveLayer,
    flops_comparison, parameter_count_comparison,
    compute_ssm_impulse_response, effective_memory_length,
    HybridConfig, HybridSSMTransformer,
    ARCHITECTURE_COMPLEXITIES,
)

V = 64
T = 32
B = 2

print("=" * 60)
print("NanoMind SSM (Mamba/S4/Linear Attention) Demo")
print("=" * 60)

# ── HiPPO matrix ──────────────────────────────────────────────────────────────
print("
── HiPPO-LegS Matrix ──")
A = make_hippo_matrix(8)
print(f"  A shape: {A.shape}")
print(f"  A diagonal: {A.diagonal().tolist()}")
print(f"  A is lower-triangular: {(A.triu(diagonal=1) == 0).all().item()}")

# ── Discretized SSM ───────────────────────────────────────────────────────────
print("
── Discretized SSM (S4 core) ──")
ssm = DiscretizedSSM(d_state=8)
x   = torch.randn(B, T, 1)

# Convolutional mode (parallel, training)
y_conv = ssm.forward_conv(x)
print(f"  Conv mode:      input {tuple(x.shape)} → {tuple(y_conv.shape)}")

# Recurrent mode (sequential, inference)
h, y_rec = ssm.forward_recurrent(x)
print(f"  Recurrent mode: input {tuple(x.shape)} → {tuple(y_rec.shape)}, "
      f"state {tuple(h.shape)}")

# Both modes should give same output
max_diff = (y_conv - y_rec).abs().max().item()
print(f"  Conv vs Recurrent max diff: {max_diff:.6f}")

# Impulse response
A_bar, B_bar = ssm._discretise()
C = ssm.C[..., 0] + 1j * ssm.C[..., 1]
K = compute_ssm_impulse_response(A_bar, B_bar.real, C.real, T=16)
eff_len = effective_memory_length(K)
print(f"  Impulse response length: {len(K)}, effective memory: {eff_len}")

# ── Mamba Block ───────────────────────────────────────────────────────────────
print("
── Mamba Block (Selective SSM) ──")
cfg   = SSMConfig(d_model=32, d_state=8, d_conv=4, expand=2, dt_min=0.001)
block = MambaBlock(cfg)
x     = torch.randn(B, T, 32)
y     = block(x)
print(f"  Input:  {tuple(x.shape)}")
print(f"  Output: {tuple(y.shape)}")

# ── Mamba LM ──────────────────────────────────────────────────────────────────
print("
── Mamba Language Model ──")
mamba_lm = MambaLM(vocab_size=V, cfg=cfg, n_layers=3)
ids      = torch.randint(0, V, (B, T))
logits   = mamba_lm(ids)
print(f"  MambaLM: {mamba_lm.n_parameters():,} params")
print(f"  Logits: {tuple(logits.shape)}")
# Loss
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
print(f"  Cross-entropy loss: {loss.item():.4f}")

# ── S4 ────────────────────────────────────────────────────────────────────────
print("
── S4 Layer ──")
s4_layer = S4Layer(d_model=16, d_state=4)
x_s4     = torch.randn(B, 16, 16)
y_s4     = s4_layer(x_s4)
print(f"  S4Layer: {tuple(x_s4.shape)} → {tuple(y_s4.shape)}")

s4_model = S4Model(vocab_size=V, d_model=16, d_state=4, n_layers=2, d_ff=64)
logits_s4 = s4_model(ids[:, :16])
print(f"  S4Model: {s4_model.n_parameters():,} params, "
      f"logits={tuple(logits_s4.shape)}")

# ── Linear Attention ──────────────────────────────────────────────────────────
print("
── Linear Attention (O(T)) ──")
la_cfg = LinearAttnConfig(d_model=32, n_heads=4, d_head=8)
la     = LinearAttention(la_cfg)
x_la   = torch.randn(B, T, 32)
y_la   = la(x_la)
print(f"  LinearAttention: {tuple(x_la.shape)} → {tuple(y_la.shape)}")

# Recurrent inference step
S0 = torch.zeros(B, la_cfg.n_heads, la_cfg.d_head, la_cfg.d_head)
z0 = torch.zeros(B, la_cfg.n_heads, la_cfg.d_head)
y_step, S1, z1 = la.recurrent_step(x_la[:, 0, :], S0, z0)
print(f"  Recurrent step: input {tuple(x_la[:, 0, :].shape)} → {tuple(y_step.shape)}")

# ── Retentive Networks ────────────────────────────────────────────────────────
print("
── Retentive Networks ──")
ret  = RetentiveLayer(d_model=32, n_heads=4)
y_r  = ret(x_la)
print(f"  RetentiveLayer: {tuple(x_la.shape)} → {tuple(y_r.shape)}")

# ── Hybrid Model ──────────────────────────────────────────────────────────────
print("
── Hybrid SSM-Transformer (Jamba-style) ──")
h_cfg   = HybridConfig(d_model=32, n_heads=4, d_state=8, n_layers=8,
                         attn_every=4, vocab_size=V)
hybrid  = HybridSSMTransformer(h_cfg)
print(f"  Layer summary: {hybrid.layer_type_summary()}")
print(f"  Parameters: {hybrid.n_parameters():,}")
logits_h = hybrid(ids)
print(f"  Logits: {tuple(logits_h.shape)}")

# ── Complexity comparison ─────────────────────────────────────────────────────
print("
── Architecture Complexity Comparison ──")
for c in ARCHITECTURE_COMPLEXITIES:
    d = c.to_dict()
    print(f"  {d['architecture']:20s}: train={d['train_FLOPs']}, "
          f"infer={d['infer_FLOPs']}")

flops = flops_comparison(2, 1024, 512, 16)
print(f"
  FLOPs at T=1024, D=512, N=16:")
for k, v in flops.items():
    if k != "note":
        print(f"    {k}: {v}")

params = parameter_count_comparison(512, 16, 1024, 50000, 12)
print(f"
  Parameter counts (D=512, L=12, V=50k):")
for k, v in params.items():
    print(f"    {k}: {v}")

print("
SSM demo complete!")
