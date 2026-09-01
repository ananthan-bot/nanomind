"""
bonus5_commits.py — 5 bonus commits to polish Day 23 MoE.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== BONUS 5 COMMITS — MoE Polish ===\n")

# ── BONUS 1 — Router jitter noise (Switch Transformer trick) ─────────────────
src = read("nanomind/moe/router.py")
src = src.replace(
    "        logits   = self.gate(x_flat)               # (tokens, N)",
    "        logits   = self.gate(x_flat)               # (tokens, N)\n\n"
    "        # Multiplicative jitter noise during training (Switch Transformer)\n"
    "        # helps prevent router collapse and encourages exploration\n"
    "        if self.training and self.jitter_noise > 0.0:\n"
    "            noise   = torch.empty_like(logits).uniform_(1 - self.jitter_noise,\n"
    "                                                        1 + self.jitter_noise)\n"
    "            logits  = logits * noise"
)
src = src.replace(
    "        self.gate        = nn.Linear(d_model, num_experts, bias=False)",
    "        self.gate        = nn.Linear(d_model, num_experts, bias=False)\n"
    "        self.jitter_noise = 0.01   # multiplicative noise for training stability"
)
write("nanomind/moe/router.py", src)
commit("feat: add multiplicative jitter noise to TopKRouter — prevents router collapse")

# ── BONUS 2 — Mixtral-inspired config YAML ──────────────────────────────────
write("configs/mixtral_moe.yaml", '''\
# NanoMind Mixtral-inspired MoE configuration
# Sliding Window Attention + RoPE + 8 experts top-2 + RMSNorm + SwiGLU

run_name: nanomind_mixtral_moe

model:
  vocab_size: 32000
  block_size: 512
  d_model: 256
  n_layers: 8
  n_heads: 8
  n_kv_heads: 2         # GQA 4:1
  window_size: 128      # SWA
  dropout: 0.0
  norm_type: rmsnorm
  pos_type: swa_rope
  weight_tying: false
  bias: false

moe:
  num_experts: 8
  top_k: 2
  load_balance_coef: 0.01
  expert_dropout: 0.0
  d_ff_expert: null     # defaults to 4 x d_model
  activation: swiglu

train:
  max_iters: 25000
  eval_interval: 500
  grad_clip: 1.0
  use_amp: false
  device: auto
  out_dir: checkpoints/mixtral_moe
''')
commit("feat: add configs/mixtral_moe.yaml — SWA+RoPE+GQA+MoE Mixtral-inspired config")

# ── BONUS 3 — expert_collapse_rate() diagnostic helper ──────────────────────
src = read("nanomind/moe/load_balance.py")
src += '''

def expert_collapse_rate(
    router_logits_history: list[torch.Tensor],
    num_experts: int,
    threshold: float = 0.01,
) -> float:
    """
    Estimate router collapse: fraction of experts receiving < threshold of tokens.

    A collapsed MoE will route nearly all tokens to 1-2 experts, leaving the rest
    unused. This diagnostic tracks collapse over a sequence of training steps.

    Args:
        router_logits_history: List of router logit tensors from recent steps.
        num_experts:           Total number of experts.
        threshold:             Fraction below which an expert is considered collapsed.

    Returns:
        Collapse rate: fraction of experts below the threshold (0 = healthy, 1 = full collapse).
    """
    if not router_logits_history:
        return 0.0

    all_logits = torch.cat(router_logits_history, dim=0)  # (total_tokens, N)
    probs      = torch.softmax(all_logits, dim=-1)
    mean_probs = probs.mean(dim=0)                         # (N,)
    collapsed  = (mean_probs < threshold).float().mean().item()
    return collapsed
'''
write("nanomind/moe/load_balance.py", src)
commit("feat: add expert_collapse_rate() — diagnose router collapse over training history")

# ── BONUS 4 — test: jitter noise and collapse rate ──────────────────────────
src = read("tests/test_moe.py")
src += '''

# ── Jitter noise and collapse rate ────────────────────────────────────────────

class TestRouterJitter:
    def test_jitter_in_training_mode(self):
        """Router with jitter should still return valid shapes in train mode."""
        router = TopKRouter(D, N_EXP, TOP_K)
        router.train()
        x = torch.randn(B, T, D)
        indices, weights, logits = router(x)
        assert indices.shape == (B * T, TOP_K)
        assert weights.isfinite().all()

    def test_jitter_disabled_in_eval(self):
        """In eval mode, output should be deterministic (no noise)."""
        router = TopKRouter(D, N_EXP, TOP_K)
        router.eval()
        x = torch.randn(B, T, D)
        idx1, w1, _ = router(x)
        idx2, w2, _ = router(x)
        assert torch.equal(idx1, idx2)
        assert torch.allclose(w1, w2)


class TestExpertCollapseRate:
    def test_zero_history(self):
        from nanomind.moe.load_balance import expert_collapse_rate
        rate = expert_collapse_rate([], N_EXP)
        assert rate == 0.0

    def test_uniform_routing_low_collapse(self):
        from nanomind.moe.load_balance import expert_collapse_rate
        # Uniform logits → uniform probs → no collapse
        history = [torch.zeros(32, N_EXP) for _ in range(5)]
        rate    = expert_collapse_rate(history, N_EXP, threshold=0.01)
        assert rate == 0.0

    def test_collapsed_routing_high_rate(self):
        from nanomind.moe.load_balance import expert_collapse_rate
        # Route all tokens to expert 0 → collapse rate should be high
        logits  = torch.full((32, N_EXP), -1e9)
        logits[:, 0] = 0.0
        history = [logits for _ in range(5)]
        rate    = expert_collapse_rate(history, N_EXP, threshold=0.1)
        assert rate > 0.5   # most experts collapsed
'''
write("tests/test_moe.py", src)
commit("test: add router jitter train/eval and expert_collapse_rate diagnostic tests")

# ── BONUS 5 — NanoMindMoE.num_active_params() ────────────────────────────────
src = read("nanomind/moe/model.py")
src = src.replace(
    "    def num_parameters(self) -> int:\n"
    "        return sum(p.numel() for p in self.parameters())",
    "    def num_parameters(self) -> int:\n"
    "        return sum(p.numel() for p in self.parameters())\n\n"
    "    def num_active_parameters(self) -> int:\n"
    "        \"\"\"\n"
    "        Compute the number of parameters *active* per forward token.\n\n"
    "        Unlike ``num_parameters()`` which counts all weights including\n"
    "        the unused expert FFNs, this returns the effective parameter count\n"
    "        that actually contributes to each token's computation:\n"
    "        non-MoE params + top_k/num_experts × MoE params.\n"
    "        \"\"\"\n"
    "        from nanomind.moe.layer import SparseMoELayer\n"
    "        total_moe, total_non_moe = 0, 0\n"
    "        for name, module in self.named_modules():\n"
    "            if isinstance(module, SparseMoELayer):\n"
    "                # Expert params: each expert is equally sized\n"
    "                expert_params = sum(\n"
    "                    p.numel() for e in module.experts for p in e.parameters()\n"
    "                )\n"
    "                router_params = sum(p.numel() for p in module.router.parameters())\n"
    "                # Only top_k experts activate per token\n"
    "                active_expert = int(expert_params * self.moe_cfg.top_k / self.moe_cfg.num_experts)\n"
    "                total_moe    += active_expert + router_params\n"
    "            elif not any(isinstance(module, SparseMoELayer)\n"
    "                         for module in module.modules()):\n"
    "                pass\n"
    "        non_moe = sum(\n"
    "            p.numel() for n, p in self.named_parameters()\n"
    "            if 'experts.' not in n\n"
    "        )\n"
    "        return non_moe + total_moe"
)
write("nanomind/moe/model.py", src)
commit("feat: add NanoMindMoE.num_active_parameters() — compute effective active param count per token")

# ── Push ─────────────────────────────────────────────────────────────────────
print("\n=== Pushing 5 bonus commits to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-5")
print(f"\n=== Last 5 commits ===\n{log.stdout}")
print("=== BONUS 5 COMPLETE! ===")
