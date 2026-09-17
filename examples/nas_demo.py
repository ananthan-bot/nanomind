"""
examples/nas_demo.py — NanoMind Neural Architecture Search demo.

Demonstrates:
  1. SearchSpace definition and sampling
  2. ProxyEvaluator: param score, synflow, loss proxy
  3. RandomSearch: 10 random architectures
  4. EvolutionarySearch: 3 generations × 5 population
  5. ProgressiveScheduler: shrinking search space
  6. Supernet: weight-sharing one-shot forward
  7. Pareto front: accuracy vs parameter count

Usage:
    python examples/nas_demo.py
"""
import torch
from nanomind.nas import (
    ArchConfig, SearchSpace, ProxyEvaluator,
    RandomSearch, EvolutionarySearch,
    ProgressiveScheduler, WarmRestartScheduler,
    Supernet, pareto_front, efficiency_score,
)

print("=" * 60)
print("NanoMind Neural Architecture Search Demo")
print("=" * 60)

# ── SearchSpace ───────────────────────────────────────────────────────────────
space = SearchSpace(
    d_model_choices   = [32, 64, 128],
    n_layers_choices  = [1, 2, 4],
    n_heads_choices   = [1, 2, 4],
    ffn_ratio_choices = [2, 4],
    dropout_choices   = [0.0, 0.1],
)
print(f"
Search space size: {space.size} valid configs")
cfg = space.random_sample()
print(f"Random sample: d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} "
      f"ffn={cfg.ffn_ratio} drop={cfg.dropout}")
print(f"Estimated params: {cfg.n_params_estimate:,}")

# ── ProxyEvaluator ────────────────────────────────────────────────────────────
ev     = ProxyEvaluator(vocab_size=32, proxy_steps=3)
scores = ev.evaluate(cfg)
print(f"
ProxyEvaluator for d={cfg.d_model} L={cfg.n_layers}:")
for k, v in scores.items():
    print(f"  {k}: {v}")

# ── RandomSearch ──────────────────────────────────────────────────────────────
print("
── Random Search (n=10) ──")
rs   = RandomSearch(space, ev, n_samples=10)
best = rs.run()
print(f"  Best: d={best.config.d_model} L={best.config.n_layers} "
      f"H={best.config.n_heads}  score={best.composite:.4f}")
summary = rs.summary()
print(f"  Best score={summary['best_score']:.4f} "
      f"Mean={summary['mean_score']:.4f}")

# ── EvolutionarySearch ────────────────────────────────────────────────────────
print("
── Evolutionary Search (pop=5, gen=3) ──")
es   = EvolutionarySearch(space, ev, population=5, generations=3, top_k=3)
best_e = es.run()
print(f"  Best: d={best_e.config.d_model} L={best_e.config.n_layers} "
      f"score={best_e.composite:.4f}")
print(f"  Gen bests: {es.generation_bests()}")

# ── ProgressiveScheduler ──────────────────────────────────────────────────────
print("
── Progressive Scheduler (4 rounds) ──")
sched = ProgressiveScheduler(space, n_rounds=4, shrink_ratio=0.6)
for r in range(4):
    s = sched.get_space(r)
    t = sched.temperature(r)
    print(f"  Round {r}: space_size={s.size} temp={t:.3f}")

# ── WarmRestartScheduler ──────────────────────────────────────────────────────
print("
── Warm Restart (T_0=3, T_mult=2) ──")
wr = WarmRestartScheduler(T_0=3, T_mult=2)
temps = [round(wr.step(), 3) for _ in range(8)]
print(f"  Temperatures: {temps}")

# ── Supernet ──────────────────────────────────────────────────────────────────
print("
── Supernet (weight sharing) ──")
supernet = Supernet(space, vocab_size=32)
subnet   = supernet.sample_subnet()
print(f"  Sampled subnet: d={subnet.d_model} L={subnet.n_layers} H={subnet.n_heads}")
x = torch.randint(0, 32, (1, 8))
y = torch.randint(0, 32, (1, 8))
with torch.no_grad():
    logits, loss = supernet(x, y, subnet_cfg=subnet)
print(f"  Supernet forward: logits={tuple(logits.shape)} loss={loss.item():.4f}")

# ── Pareto Front ──────────────────────────────────────────────────────────────
print("
── Pareto Front (accuracy vs params) ──")
front = pareto_front(rs.results)
print(f"  {len(front)}/{len(rs.results)} architectures on Pareto front")
for p in front[:3]:
    print(f"  d={p.result.config.d_model} L={p.result.config.n_layers} "
          f"score={p.accuracy:.4f} params={p.params:,}")
eff = efficiency_score(best, target_params=50000)
print(f"  Efficiency score (target=50K params): {eff:.4f}")
print("
NAS demo complete!")
