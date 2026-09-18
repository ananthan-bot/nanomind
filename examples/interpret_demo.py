"""
examples/interpret_demo.py — NanoMind Interpretability demo.

Demonstrates:
  1. Attention weight extraction + entropy
  2. Gradient saliency (vanilla + grad×input)
  3. Linear probing at each layer
  4. Logit Lens: what does each layer predict?
  5. Head ablation: which heads are critical?
  6. Occlusion attribution: which tokens matter?

Usage:
    python examples/interpret_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.interpret import (
    AttentionExtractor, GradientSaliency, LinearProbe,
    LogitLens, HeadAblator, OcclusionAttributor, ShapleyAttributor,
    LayerwiseProber,
)

# ── Tiny model ────────────────────────────────────────────────────────────────
class TinyTF(nn.Module):
    def __init__(self, V=16, D=32, T=8, H=2, L=2):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(D, H, batch_first=True)
            for _ in range(L)
        ])
        self.lns  = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.ln   = nn.LayerNorm(D)
        self.lm   = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(S))
        for attn, ln in zip(self.blocks, self.lns):
            r, _ = attn(h, h, h)
            h    = ln(h + r)
        h = self.ln(h)
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, 16), t.view(-1)) if t is not None else None
        return logits, loss

V = 16
model = TinyTF(V=V)
model.eval()

print("=" * 60)
print("NanoMind Interpretability Demo")
print("=" * 60)

# ── AttentionExtractor ────────────────────────────────────────────────────────
print("
── Attention Extraction ──")
x   = torch.randint(0, V, (1, 6))
ext = AttentionExtractor(model)
maps = ext.extract(x, tokens=["t0", "t1", "t2", "t3", "t4", "t5"])
for m in maps:
    print(f"  Layer {m.layer}: shape={tuple(m.weights.shape)} "
          f"entropy={m.entropy().tolist()}")
if maps:
    print(f"  Mean attention (layer 0):
  {maps[0].mean_head().round(decimals=3)}")

# ── Gradient Saliency ─────────────────────────────────────────────────────────
print("
── Gradient Saliency ──")
model.train()   # need grad
sal = GradientSaliency(model)
x_g = torch.randint(0, V, (1, 5))
sal_vanilla = sal.vanilla(x_g, target_pos=-1,
                           tokens=["a", "b", "c", "d", "e"])
sal_gxi     = sal.grad_times_input(x_g, target_pos=-1,
                                    tokens=["a", "b", "c", "d", "e"])
print(f"  Vanilla top-3: {sal_vanilla.top_k_tokens(3)}")
print(f"  Grad×Input top-3: {sal_gxi.top_k_tokens(3)}")
model.eval()

# ── Linear Probing ────────────────────────────────────────────────────────────
print("
── Linear Probing ──")
X      = torch.randn(20, 32)
labels = torch.randint(0, 3, (20,))
probe  = LinearProbe(d_model=32, n_classes=3)
result = probe.fit(X[:15], labels[:15], X[15:], labels[15:],
                   layer=0, task="test_probe", epochs=30)
print(f"  Layer 0 probe: accuracy={result.accuracy:.2%} loss={result.loss:.4f}")

# ── Logit Lens ────────────────────────────────────────────────────────────────
print("
── Logit Lens ──")
model.eval()
lens   = LogitLens(model)
x_ll   = torch.randint(0, V, (1, 4))
result = lens.analyse(x_ll)
print(f"  Layers analysed: {result.n_layers}")
if result.n_layers > 0:
    print(f"  Layer 0 top tokens: {result.get_layer_top_tokens(0, k=3)}")
    print(f"  Prediction changes: {result.prediction_change()}")

# ── Head Ablation ─────────────────────────────────────────────────────────────
print("
── Head Ablation ──")
x_a  = torch.randint(0, V, (1, 4))
y_a  = torch.randint(0, V, (1, 4))
abl  = HeadAblator(model)
results = abl.ablate_all(x_a, y_a)
results.sort(key=lambda r: r.importance, reverse=True)
print(f"  Most important head: layer={results[0].layer} "
      f"head={results[0].head} importance={results[0].importance:.4f}")

# ── Occlusion Attribution ─────────────────────────────────────────────────────
print("
── Occlusion Attribution ──")
x_o  = torch.randint(0, V, (1, 5))
attr = OcclusionAttributor(model)
attrib = attr.attribute(x_o, target_pos=-1, target_class=3,
                         tokens=["w0", "w1", "w2", "w3", "w4"])
print(f"  Top-3 by occlusion: {attrib.top_k(3)}")

# ── Shapley Attribution ───────────────────────────────────────────────────────
print("
── Shapley Attribution ──")
shap = ShapleyAttributor(model, n_samples=5)
s_attr = shap.attribute(x_o, target_pos=-1, target_class=3,
                          tokens=["w0", "w1", "w2", "w3", "w4"])
print(f"  Shapley top-3: {s_attr.top_k(3)}")
print("
Interpretability demo complete!")
