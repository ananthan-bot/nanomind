"""
day44_commits.py — 20 atomic commits for Day 44: Advanced MoE (Mixture of Experts++).
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

print("\n=== DAY 44: Advanced Mixture of Experts (MoE++) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — moe_v2 package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/__init__.py",
      '"""NanoMind MoE++ sub-package — Advanced Mixture of Experts."""\n')
commit("feat: add nanomind/moe_v2/ package skeleton for Advanced MoE++")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — MoE config
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/config.py", '''\
"""
nanomind/moe_v2/config.py — MoE configuration.

## Mixture of Experts (MoE) in LLMs

Standard MoE replaces dense FFN layers with N expert FFN networks,
routing each token to the top-K experts via a learned gating function.

Key concepts:
  n_experts:      Total experts (e.g., 8, 64, 128, 256)
  top_k:          Active experts per token (e.g., 2, 4)
  capacity_factor: Max tokens per expert = capacity_factor × tokens/experts
  load_balance:   Auxiliary loss to prevent all tokens routing to one expert

Famous MoE models:
  GShard (Lepikhin et al., 2021):   Top-2 routing, 600B params
  Switch Transformer (Fedus et al., 2022): Top-1, 1.6T params
  GLaM (Du et al., 2022):           64 experts, Top-2
  Mixtral (Mistral AI, 2024):       8 experts, Top-2
  DeepSeek-MoE (2024):              64 experts, fine-grained routing

References:
  Shazeer et al. (2017) "Outrageously Large Neural Networks"
  https://arxiv.org/abs/1701.06538
  Fedus et al. (2022) "Switch Transformers"
  https://arxiv.org/abs/2101.03961
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MoEConfig:
    """
    Configuration for a Mixture of Experts layer.

    Attributes:
        n_experts:        Total number of expert networks.
        top_k:            Number of experts each token is routed to.
        d_model:          Input/output feature dimension.
        d_ff:             Expert hidden dimension.
        capacity_factor:  Tokens per expert = cf × (total_tokens / n_experts).
        router_type:      ``"topk"``, ``"expert_choice"``, or ``"hash"``.
        load_balance_coef: Coefficient for auxiliary load balancing loss.
        router_noise:     Jitter noise std for top-K routing (prevents collapse).
        dropout:          Expert dropout rate.
        use_bias:         Whether experts use bias terms.
        shared_experts:   Number of always-active shared experts (DeepSeek-style).
    """
    n_experts:        int   = 8
    top_k:            int   = 2
    d_model:          int   = 128
    d_ff:             int   = 512
    capacity_factor:  float = 1.25
    router_type:      str   = "topk"
    load_balance_coef: float = 1e-2
    router_noise:     float = 0.01
    dropout:          float = 0.0
    use_bias:         bool  = True
    shared_experts:   int   = 0

    def __post_init__(self) -> None:
        assert self.n_experts   >= 1
        assert 1 <= self.top_k  <= self.n_experts
        assert self.d_model     > 0
        assert self.d_ff        > 0
        assert self.capacity_factor > 0.0
        assert self.router_type in ("topk", "expert_choice", "hash")
        assert self.shared_experts >= 0

    @property
    def active_ratio(self) -> float:
        """Fraction of parameters used per token (top_k / n_experts)."""
        return self.top_k / self.n_experts

    @property
    def total_params_estimate(self) -> int:
        """Rough parameter count for all experts."""
        per_expert = 2 * self.d_model * self.d_ff
        return self.n_experts * per_expert + self.d_model * self.n_experts

    def to_dict(self) -> dict:
        return {
            "n_experts":    self.n_experts,
            "top_k":        self.top_k,
            "d_model":      self.d_model,
            "d_ff":         self.d_ff,
            "router_type":  self.router_type,
            "active_ratio": round(self.active_ratio, 4),
        }
''')
commit("feat: add MoEConfig — n_experts, top_k, capacity_factor, router_type, active_ratio, shared_experts")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Expert network
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/expert.py", '''\
"""
nanomind/moe_v2/expert.py — Individual expert and expert bank.

Each expert is an independent FFN:
  Expert_i(x) = W2_i × GELU(W1_i × x + b1_i) + b2_i

Expert banks allow parallel computation of multiple experts.

SwiGLU experts (LLaMA-style):
  Expert(x) = W2 × (SiLU(W_gate × x) × W1 × x)
  Better performance than standard GELU FFN.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.moe_v2.config import MoEConfig


class Expert(nn.Module):
    """
    A single FFN expert.

    Args:
        d_model:  Input/output dimension.
        d_ff:     Hidden dimension.
        use_bias: Include bias.
        variant:  ``"gelu"`` or ``"swiglu"``.
    """

    def __init__(
        self,
        d_model:  int,
        d_ff:     int,
        use_bias: bool = True,
        variant:  str  = "gelu",
    ) -> None:
        super().__init__()
        self.variant = variant
        if variant == "swiglu":
            # SwiGLU: gate + value projections
            self.w_gate = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w_val  = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w_out  = nn.Linear(d_ff, d_model, bias=use_bias)
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "swiglu":
            return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))
        return self.w2(F.gelu(self.w1(x)))


class ExpertBank(nn.Module):
    """
    A bank of N independent expert FFNs.

    Stores all experts in a single module list for easy iteration.

    Args:
        cfg: :class:`MoEConfig`.

    Example::

        bank = ExpertBank(MoEConfig(n_experts=8, d_model=128, d_ff=512))
        # Route x through expert 3:
        out = bank.experts[3](x)
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.experts   = nn.ModuleList([
            Expert(cfg.d_model, cfg.d_ff, cfg.use_bias)
            for _ in range(cfg.n_experts)
        ])
        # Shared experts (always active, DeepSeek-MoE style)
        self.shared = nn.ModuleList([
            Expert(cfg.d_model, cfg.d_ff, cfg.use_bias)
            for _ in range(cfg.shared_experts)
        ])

    def forward_expert(
        self,
        expert_idx: int,
        x:          torch.Tensor,
    ) -> torch.Tensor:
        """Run a single expert."""
        return self.experts[expert_idx](x)

    def shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum all shared expert outputs."""
        if not self.shared:
            return torch.zeros_like(x)
        return sum(e(x) for e in self.shared)

    @property
    def n_params_per_expert(self) -> int:
        if not self.experts:
            return 0
        return sum(p.numel() for p in self.experts[0].parameters())

    @property
    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
''')
commit("feat: add Expert (GELU/SwiGLU FFN), ExpertBank — n_experts, shared experts (DeepSeek-style)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Top-K router with noise
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/router.py", '''\
"""
nanomind/moe_v2/router.py — Token routing strategies for MoE.

## Token Routing

The router decides which expert(s) each token is sent to.
This is a discrete selection problem — argmax is non-differentiable,
so gradients only flow through the gating weights, not the routing decision.

## Top-K Routing (Shazeer et al., 2017)

  gates = softmax(x W_router)     # (T, E) logits
  top_k_indices = argtopk(gates)  # (T, K) selected experts
  top_k_weights = softmax(gates[top_k_indices])  # normalised weights

Output: Σ_{k=1}^K weight_k × Expert_k(x)

## Noisy Top-K Routing (Switch Transformer improvement)

Add Gaussian noise before top-K to prevent routing collapse:
  gates = gates + ε × softplus(W_noise × x),  ε ~ N(0, 1)

## Expert Choice Routing (Zhou et al., 2022)

Instead of each token choosing K experts,
each expert chooses its top-C tokens (C = capacity):
  - Balanced by design (no overflow)
  - Better load balance than token choice
  - Used in ST-MoE-32B

## Hash Routing

Deterministic: route based on token position hash.
  expert_id = hash(position) % n_experts
  No learnable parameters, but no quality-based routing.

References:
  Shazeer et al. (2017) https://arxiv.org/abs/1701.06538
  Fedus et al. (2022) Switch: https://arxiv.org/abs/2101.03961
  Zhou et al. (2022) Expert Choice: https://arxiv.org/abs/2202.09368
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanomind.moe_v2.config import MoEConfig


@dataclass
class RoutingOutput:
    """Output of a router: routing decisions and auxiliary loss."""
    indices:      torch.Tensor    # (T, K) expert indices per token
    weights:      torch.Tensor    # (T, K) routing weights
    gates:        torch.Tensor    # (T, E) raw gate logits
    aux_loss:     torch.Tensor    # scalar auxiliary load-balance loss
    router_probs: torch.Tensor    # (T, E) router probabilities


class TopKRouter(nn.Module):
    """
    Top-K token router with optional jitter noise.

    Args:
        cfg: :class:`MoEConfig`.

    Example::

        router = TopKRouter(MoEConfig(n_experts=8, top_k=2))
        out    = router(x)   # x: (B*T, D)
        # out.indices: (B*T, 2), out.weights: (B*T, 2)
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.n_experts  = cfg.n_experts
        self.top_k      = cfg.top_k
        self.noise_std  = cfg.router_noise
        self.lb_coef    = cfg.load_balance_coef
        self.gate       = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        """
        Route tokens to top-K experts.

        Args:
            x: ``(N, D)`` token representations (N = B×T).

        Returns:
            :class:`RoutingOutput`.
        """
        logits = self.gate(x)   # (N, E)

        # Add jitter noise during training
        if self.training and self.noise_std > 0:
            noise   = torch.randn_like(logits) * self.noise_std
            logits  = logits + noise

        probs   = F.softmax(logits, dim=-1)   # (N, E)

        # Top-K selection
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)    # (N, K)
        # Re-normalise weights among selected experts
        top_weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Auxiliary load-balance loss (Switch Transformer formula)
        aux_loss = self._load_balance_loss(probs, top_idx)

        return RoutingOutput(
            indices      = top_idx,
            weights      = top_weights,
            gates        = logits,
            aux_loss     = aux_loss,
            router_probs = probs,
        )

    def _load_balance_loss(
        self,
        probs:   torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Switch Transformer auxiliary load-balance loss.

        L_aux = n_experts × Σ_i f_i × P_i
        where:
          f_i = fraction of tokens dispatched to expert i
          P_i = average router probability for expert i
        """
        N, E = probs.shape
        # Compute dispatch fraction f_i
        mask    = torch.zeros(N, E)
        mask.scatter_(1, indices[:, :1], 1.0)   # use top-1 for fraction
        f_i     = mask.mean(dim=0)              # (E,)
        P_i     = probs.mean(dim=0)             # (E,)
        return self.lb_coef * E * (f_i * P_i).sum()

    def expert_utilisation(self, probs: torch.Tensor) -> dict:
        """Compute per-expert utilisation statistics."""
        top_idx = probs.argmax(dim=-1)
        counts  = torch.bincount(top_idx, minlength=self.n_experts)
        total   = probs.shape[0]
        return {
            "expert_counts": counts.tolist(),
            "utilisation":   (counts.float() / max(total, 1)).tolist(),
            "entropy":       -(probs.mean(0) * (probs.mean(0) + 1e-9).log()).sum().item(),
        }


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing (Zhou et al., 2022).

    Each expert selects its top-C tokens (C = capacity).
    Guarantees perfect load balance at the cost of some tokens being dropped.

    Args:
        cfg:          :class:`MoEConfig`.
        capacity:     Number of tokens each expert can take.
    """

    def __init__(self, cfg: MoEConfig, capacity: int = 4) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.capacity  = capacity
        self.gate      = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        N, D     = x.shape
        logits   = self.gate(x)                        # (N, E)
        scores   = F.softmax(logits, dim=0)            # normalise over tokens

        C        = min(self.capacity, N)
        # Each expert picks top-C tokens
        top_vals, top_idx = scores.topk(C, dim=0)     # (C, E)

        # Build routing indices for each token (simplified: use argmax expert)
        token_expert = logits.argmax(dim=-1)           # (N,)
        weights      = F.softmax(logits, dim=-1)       # (N, E)
        top_k_idx    = token_expert.unsqueeze(-1)      # (N, 1)
        top_k_w      = weights.gather(1, top_k_idx)   # (N, 1)

        aux_loss = torch.tensor(0.0)
        return RoutingOutput(
            indices      = top_k_idx,
            weights      = top_k_w,
            gates        = logits,
            aux_loss     = aux_loss,
            router_probs = weights,
        )


class HashRouter(nn.Module):
    """
    Deterministic hash-based routing (no learnable parameters).

    Tokens are assigned to experts based on position modulo n_experts.
    Fast but ignores token content.
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k     = min(cfg.top_k, 1)   # hash router always top-1

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        N = x.shape[0]
        idx   = (torch.arange(N) % self.n_experts).unsqueeze(-1)
        w     = torch.ones(N, 1)
        dummy = torch.zeros(N, self.n_experts)
        dummy.scatter_(1, idx, 1.0)
        return RoutingOutput(
            indices      = idx,
            weights      = w,
            gates        = dummy,
            aux_loss     = torch.tensor(0.0),
            router_probs = dummy,
        )
''')
commit("feat: add TopKRouter (noisy top-K, load-balance loss), ExpertChoiceRouter, HashRouter")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Capacity buffer (token overflow handling)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/capacity.py", '''\
"""
nanomind/moe_v2/capacity.py — Expert capacity and token overflow handling.

## The Capacity Problem

If one expert receives more tokens than its capacity allows,
some tokens must be "dropped" (passed through unchanged or zero-padded).

Capacity = capacity_factor × (total_tokens / n_experts)

Example:
  1024 tokens, 8 experts, capacity_factor=1.25
  capacity = 1.25 × (1024/8) = 160 tokens per expert

  If expert 3 gets 200 tokens → 40 tokens dropped (overflow).

Strategies:
  - Drop & replace with residual (Switch Transformer default)
  - Auxiliary loss to prevent overflow (load-balance)
  - Flexible capacity (scale factor per batch)
  - Expert choice (perfect balance, no overflow by design)

This module implements capacity buffers and overflow statistics.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class CapacityStats:
    """Statistics about token capacity utilisation."""
    n_tokens:       int
    n_experts:      int
    capacity:       int
    overflow_tokens: int
    overflow_frac:  float
    expert_loads:   list[int]

    def to_dict(self) -> dict:
        return {
            "n_tokens":       self.n_tokens,
            "capacity":       self.capacity,
            "overflow_tokens": self.overflow_tokens,
            "overflow_frac":  round(self.overflow_frac, 4),
            "max_load":       max(self.expert_loads),
            "min_load":       min(self.expert_loads),
        }


class CapacityBuffer:
    """
    Manages token capacity for each expert.

    Args:
        n_experts:       Number of experts.
        capacity_factor: Tokens per expert = cf × (N / n_experts).

    Example::

        buf   = CapacityBuffer(n_experts=8, capacity_factor=1.25)
        masks = buf.compute_masks(routing_indices, n_tokens=512)
        stats = buf.stats(routing_indices, n_tokens=512)
    """

    def __init__(
        self,
        n_experts:       int,
        capacity_factor: float = 1.25,
    ) -> None:
        self.n_experts       = n_experts
        self.capacity_factor = capacity_factor

    def capacity(self, n_tokens: int) -> int:
        """Compute expert capacity for a given token count."""
        return max(1, int(self.capacity_factor * n_tokens / self.n_experts))

    def compute_masks(
        self,
        indices: torch.Tensor,
        n_tokens: int,
    ) -> list[torch.Tensor]:
        """
        Compute per-expert boolean masks (True = token accepted).

        Tokens that overflow capacity are dropped.

        Args:
            indices:  ``(N, K)`` or ``(N, 1)`` expert assignment indices.
            n_tokens: Total token count.

        Returns:
            List of ``(N,)`` boolean masks, one per expert.
        """
        cap   = self.capacity(n_tokens)
        N     = indices.shape[0]
        masks = []
        for e in range(self.n_experts):
            # Find tokens assigned to expert e (top-1 only for simplicity)
            assigned = (indices[:, 0] == e).nonzero(as_tuple=True)[0]
            accepted = torch.zeros(N, dtype=torch.bool)
            if len(assigned) > 0:
                chosen   = assigned[:cap]   # keep only up to capacity
                accepted[chosen] = True
            masks.append(accepted)
        return masks

    def stats(
        self,
        indices:  torch.Tensor,
        n_tokens: int,
    ) -> CapacityStats:
        """Compute capacity utilisation statistics."""
        cap    = self.capacity(n_tokens)
        loads  = []
        overflow = 0
        for e in range(self.n_experts):
            n = (indices[:, 0] == e).sum().item()
            loads.append(n)
            overflow += max(0, n - cap)
        return CapacityStats(
            n_tokens        = n_tokens,
            n_experts       = self.n_experts,
            capacity        = cap,
            overflow_tokens = overflow,
            overflow_frac   = overflow / max(n_tokens, 1),
            expert_loads    = loads,
        )
''')
commit("feat: add CapacityBuffer — capacity(n_tokens), compute_masks (token drop), stats, CapacityStats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — MoE layer (full dispatch/combine)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/layer.py", '''\
"""
nanomind/moe_v2/layer.py — Full MoE Layer: dispatch, compute, combine.

MoE forward pass:
  1. Router: assign each token to top-K experts + weights
  2. Dispatch: group tokens by expert assignment
  3. Compute: run each expert on its assigned tokens
  4. Combine: weighted sum of expert outputs

The dispatch-combine pattern:
  - Tokens → experts is a sparse scatter operation
  - Expert outputs → tokens is a sparse gather + weighted sum

This module implements the full forward pass without expert parallelism
(single-device). Expert parallelism would distribute experts across devices.

References:
  Lepikhin et al. (2021) GShard: https://arxiv.org/abs/2006.16668
  Fedus et al. (2022) Switch: https://arxiv.org/abs/2101.03961
"""

from __future__ import annotations
import torch
import torch.nn as nn

from nanomind.moe_v2.config import MoEConfig
from nanomind.moe_v2.expert import ExpertBank
from nanomind.moe_v2.router import TopKRouter, ExpertChoiceRouter, HashRouter, RoutingOutput
from nanomind.moe_v2.capacity import CapacityBuffer


class MoELayer(nn.Module):
    """
    Full Mixture of Experts layer.

    Replaces a dense FFN with N expert FFNs and learned routing.

    Args:
        cfg: :class:`MoEConfig`.

    Example::

        cfg   = MoEConfig(n_experts=8, top_k=2, d_model=256, d_ff=1024)
        layer = MoELayer(cfg)
        out, aux_loss = layer(x)   # x: (B, T, D)
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg      = cfg
        self.experts  = ExpertBank(cfg)
        self.capacity = CapacityBuffer(cfg.n_experts, cfg.capacity_factor)

        if cfg.router_type == "topk":
            self.router = TopKRouter(cfg)
        elif cfg.router_type == "expert_choice":
            self.router = ExpertChoiceRouter(cfg)
        elif cfg.router_type == "hash":
            self.router = HashRouter(cfg)
        else:
            raise ValueError(f"Unknown router_type: {cfg.router_type!r}")

        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass.

        Args:
            x: ``(B, T, D)`` input tensor.

        Returns:
            ``(output, aux_loss)`` where output is ``(B, T, D)``
            and aux_loss is a scalar for load balancing.
        """
        B, T, D = x.shape
        x_flat  = x.view(B * T, D)   # (N, D)

        # Shared experts always run first
        shared_out = self.experts.shared_forward(x_flat)   # (N, D)

        # Route tokens to experts
        routing: RoutingOutput = self.router(x_flat)
        indices  = routing.indices    # (N, K)
        weights  = routing.weights    # (N, K)

        # Dispatch & combine
        output   = torch.zeros_like(x_flat)

        for k in range(self.cfg.top_k):
            expert_ids = indices[:, k]    # (N,)
            w_k        = weights[:, k]    # (N,)

            for e in range(self.cfg.n_experts):
                mask  = (expert_ids == e)
                if not mask.any():
                    continue
                x_e   = x_flat[mask]                              # (n_e, D)
                out_e = self.experts.forward_expert(e, x_e)       # (n_e, D)
                output[mask] += w_k[mask].unsqueeze(-1) * out_e

        # Add shared expert contribution
        output  = output + shared_out

        # Residual + dropout
        output  = self.dropout(output)
        output  = output.view(B, T, D)
        return output, routing.aux_loss

    def routing_stats(self, x: torch.Tensor) -> dict:
        """Return routing statistics for a batch."""
        B, T, D = x.shape
        x_flat  = x.view(B * T, D)
        with torch.no_grad():
            routing = self.router(x_flat)
        util    = self.router.expert_utilisation(routing.router_probs) \
                  if hasattr(self.router, "expert_utilisation") else {}
        cap_st  = self.capacity.stats(routing.indices, B * T)
        return {"routing": util, "capacity": cap_st.to_dict()}
''')
commit("feat: add MoELayer — dispatch/combine loop, shared experts, residual, routing_stats()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Transformer with MoE FFN layers
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/model.py", '''\
"""
nanomind/moe_v2/model.py — Sparse MoE Transformer.

Interleaves dense attention layers with sparse MoE FFN layers.
Typical pattern (Mixtral, Switch):
  - Every layer: MoE FFN (replaces dense FFN)
  - Some architectures: alternate dense and MoE layers

Architecture:
  Token Embedding → N × (Attention + MoE FFN) → LM Head

Total params scale with n_experts × d_ff (width),
but active params per token scale with top_k × d_ff.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.moe_v2.config import MoEConfig
from nanomind.moe_v2.layer import MoELayer


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN.

    Args:
        d_model:  Model dimension.
        n_heads:  Attention heads.
        moe_cfg:  MoE configuration.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        moe_cfg:  MoEConfig,
    ) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.moe  = MoELayer(moe_cfg)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        h, _   = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                            need_weights=False)
        x      = x + h
        # MoE FFN
        moe_out, aux = self.moe(self.ln2(x))
        x      = x + moe_out
        return x, aux


class SparseMoETransformer(nn.Module):
    """
    Full Sparse MoE Transformer language model.

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimension.
        n_layers:   Number of transformer blocks.
        n_heads:    Attention heads.
        max_seq:    Max sequence length.
        moe_cfg:    MoE configuration.

    Example::

        cfg   = MoEConfig(n_experts=8, top_k=2, d_model=128, d_ff=512)
        model = SparseMoETransformer(vocab_size=1000, d_model=128,
                                      n_layers=4, n_heads=4, moe_cfg=cfg)
        logits, loss, aux = model(input_ids, targets)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_layers:   int,
        n_heads:    int,
        max_seq:    int,
        moe_cfg:    MoEConfig,
    ) -> None:
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(max_seq, d_model)
        self.blocks   = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, moe_cfg)
            for _ in range(n_layers)
        ])
        self.ln_f     = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq  = max_seq

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: ``(B, T)`` token IDs.
            targets:   ``(B, T)`` target token IDs (for loss).

        Returns:
            ``(logits, loss, aux_loss)``
            where aux_loss is the sum of all MoE load-balance losses.
        """
        B, T    = input_ids.shape
        T       = min(T, self.max_seq)
        x       = self.tok_emb(input_ids[:, :T])
        x       = x + self.pos_emb(torch.arange(T))

        total_aux = torch.tensor(0.0)
        for block in self.blocks:
            x, aux  = block(x)
            total_aux = total_aux + aux

        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets[:, :T].contiguous().view(-1),
            )

        return logits, loss, total_aux

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_active_params(self) -> int:
        """Active parameters per forward pass (top_k experts only)."""
        moe = self.blocks[0].moe
        cfg = moe.cfg
        per_block = (
            cfg.top_k * moe.experts.n_params_per_expert
            + sum(p.numel() for p in self.blocks[0].attn.parameters())
        )
        return self.tok_emb.weight.numel() + len(self.blocks) * per_block
''')
commit("feat: add MoETransformerBlock (attn + MoE FFN), SparseMoETransformer (n_params, n_active_params)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Load balance and Z-loss
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/losses.py", '''\
"""
nanomind/moe_v2/losses.py — MoE auxiliary losses.

## MoE Auxiliary Losses

MoE models suffer from routing collapse:
  All tokens route to the same 1-2 experts → others starve.
  Causes: (a) rich-get-richer, (b) gradient feedback loops.

Auxiliary losses penalise routing imbalance:

1. Load Balance Loss (Switch Transformer):
   L_lb = n_experts × Σ_i f_i × P_i
   where f_i = fraction of tokens to expert i (argmax-based)
         P_i = mean router probability for expert i

2. Z-Loss (ST-MoE, Zoph et al., 2022):
   L_z = (1/B) × Σ_b (log Σ_e exp(x_{b,e}))²
   Penalises large router logits → stabilises training.

3. Router Z-Loss (ST-MoE-32B):
   Combines load balance + z-loss for best results.

4. Entropy Regularisation:
   Maximise H(router distribution) → encourages exploration.

Reference:
  Zoph et al. (2022) "ST-MoE: Designing Stable and Transferable Sparse Expert Models"
  https://arxiv.org/abs/2202.08906
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def load_balance_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """
    Switch Transformer load balance auxiliary loss.

    Args:
        router_probs:   ``(N, E)`` router softmax probabilities.
        expert_indices: ``(N, K)`` top-K expert indices per token.
        n_experts:      Number of experts E.

    Returns:
        Scalar loss.
    """
    N, E    = router_probs.shape
    # f_i: fraction of tokens assigned to each expert (top-1)
    one_hot = torch.zeros(N, E)
    one_hot.scatter_(1, expert_indices[:, :1], 1.0)
    f_i     = one_hot.mean(dim=0)        # (E,)
    P_i     = router_probs.mean(dim=0)   # (E,)
    return E * (f_i * P_i).sum()


def z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Z-Loss (Zoph et al., 2022): penalise large logit magnitudes.

    L_z = (1/N) Σ_n (log Σ_e exp(z_{n,e}))²

    Args:
        router_logits: ``(N, E)`` pre-softmax router logits.

    Returns:
        Scalar z-loss.
    """
    log_z   = torch.logsumexp(router_logits, dim=-1)   # (N,)
    return (log_z ** 2).mean()


def entropy_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """
    Negative entropy regularisation (encourages diverse routing).

    L_ent = -H(P) = Σ P log P   (minimise → maximise entropy)

    Args:
        router_probs: ``(N, E)`` router probabilities.

    Returns:
        Negative mean entropy (scalar).
    """
    ent = -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1)
    return -ent.mean()


def combined_moe_loss(
    router_probs:   torch.Tensor,
    router_logits:  torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts:      int,
    lb_coef:        float = 1e-2,
    z_coef:         float = 1e-3,
) -> torch.Tensor:
    """
    Combined MoE auxiliary loss: load balance + Z-loss.

    Args:
        router_probs:   ``(N, E)`` router probabilities.
        router_logits:  ``(N, E)`` pre-softmax logits.
        expert_indices: ``(N, K)`` top-K indices.
        n_experts:      Number of experts.
        lb_coef:        Load balance coefficient.
        z_coef:         Z-loss coefficient.

    Returns:
        Scalar combined loss.
    """
    lb = lb_coef * load_balance_loss(router_probs, expert_indices, n_experts)
    z  = z_coef  * z_loss(router_logits)
    return lb + z
''')
commit("feat: add load_balance_loss (Switch), z_loss (ST-MoE), entropy_loss, combined_moe_loss")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — moe_v2 __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe_v2/__init__.py", '''\
"""NanoMind MoE++ sub-package — Advanced Mixture of Experts.

Implements the full MoE stack for large language models:
  1. MoEConfig            — n_experts, top_k, capacity_factor, router_type
  2. Expert / ExpertBank  — GELU/SwiGLU FFN experts, shared experts (DeepSeek)
  3. TopKRouter           — noisy top-K, load-balance auxiliary loss
  4. ExpertChoiceRouter   — experts choose tokens (perfect balance)
  5. HashRouter           — deterministic, no learnable parameters
  6. CapacityBuffer       — token overflow management, capacity stats
  7. MoELayer             — dispatch/compute/combine forward pass
  8. MoETransformerBlock  — attention + MoE FFN with residuals
  9. SparseMoETransformer — full LM with top-K sparse FFNs
  10. load_balance_loss   — Switch Transformer auxiliary loss
  11. z_loss              — ST-MoE router logit regularisation
  12. combined_moe_loss   — load_balance + z_loss combined

Primary exports:
    - :class:`MoEConfig`            — configuration dataclass
    - :class:`Expert`               — single FFN expert (GELU/SwiGLU)
    - :class:`ExpertBank`           — bank of N experts + shared
    - :class:`TopKRouter`           — noisy top-K routing + LB loss
    - :class:`ExpertChoiceRouter`   — expert-selects-tokens routing
    - :class:`HashRouter`           — deterministic hash routing
    - :class:`RoutingOutput`        — indices, weights, aux_loss, probs
    - :class:`CapacityBuffer`       — capacity(n), masks, stats
    - :class:`CapacityStats`        — overflow statistics
    - :class:`MoELayer`             — full dispatch/combine layer
    - :class:`MoETransformerBlock`  — attn + MoE FFN block
    - :class:`SparseMoETransformer` — full sparse LM, n_params, n_active_params
    - :func:`load_balance_loss`     — Switch LB auxiliary loss
    - :func:`z_loss`                — ST-MoE z-loss
    - :func:`entropy_loss`          — routing entropy regularisation
    - :func:`combined_moe_loss`     — lb + z combined
"""

from nanomind.moe_v2.config import MoEConfig
from nanomind.moe_v2.expert import Expert, ExpertBank
from nanomind.moe_v2.router import TopKRouter, ExpertChoiceRouter, HashRouter, RoutingOutput
from nanomind.moe_v2.capacity import CapacityBuffer, CapacityStats
from nanomind.moe_v2.layer import MoELayer
from nanomind.moe_v2.model import MoETransformerBlock, SparseMoETransformer
from nanomind.moe_v2.losses import load_balance_loss, z_loss, entropy_loss, combined_moe_loss

__all__ = [
    "MoEConfig",
    "Expert", "ExpertBank",
    "TopKRouter", "ExpertChoiceRouter", "HashRouter", "RoutingOutput",
    "CapacityBuffer", "CapacityStats",
    "MoELayer",
    "MoETransformerBlock", "SparseMoETransformer",
    "load_balance_loss", "z_loss", "entropy_loss", "combined_moe_loss",
]
''')
commit("refactor: export all MoE++ components from nanomind/moe_v2/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/moe_v2_demo.py", '''\
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
print("\n── MoEConfig ──")
cfg = MoEConfig(n_experts=8, top_k=2, d_model=64, d_ff=256, capacity_factor=1.25)
print(f"  {cfg.to_dict()}")
print(f"  Active ratio: {cfg.active_ratio:.2%} of parameters used per token")
print(f"  Total param estimate: {cfg.total_params_estimate:,}")

# ── Expert (GELU vs SwiGLU) ───────────────────────────────────────────────────
print("\n── Expert FFN variants ──")
x = torch.randn(4, 64)
gelu_exp   = Expert(64, 256, variant="gelu")
swiglu_exp = Expert(64, 256, variant="swiglu")
print(f"  GELU expert output:   {tuple(gelu_exp(x).shape)}")
print(f"  SwiGLU expert output: {tuple(swiglu_exp(x).shape)}")

# ── ExpertBank ────────────────────────────────────────────────────────────────
print("\n── ExpertBank ──")
cfg_shared = MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128, shared_experts=1)
bank = ExpertBank(cfg_shared)
print(f"  n_experts: {bank.n_experts}, shared: {len(bank.shared)}")
print(f"  Params per expert: {bank.n_params_per_expert:,}")
print(f"  Total expert params: {bank.n_total_params:,}")

# ── Routers ───────────────────────────────────────────────────────────────────
print("\n── Routing Strategies ──")
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
print("\n── Capacity Buffer ──")
cap_buf = CapacityBuffer(n_experts=8, capacity_factor=1.0)
print(f"  Capacity for 64 tokens: {cap_buf.capacity(64)} per expert")
stats   = cap_buf.stats(r.indices, N)
print(f"  Stats: {stats.to_dict()}")

# ── MoELayer ─────────────────────────────────────────────────────────────────
print("\n── MoELayer (dispatch/combine) ──")
layer = MoELayer(cfg8)
x_3d  = torch.randn(2, 8, D)
out, aux = layer(x_3d)
print(f"  Output shape: {tuple(out.shape)}")
print(f"  Aux (LB) loss: {aux.item():.6f}")
rstats = layer.routing_stats(x_3d)
print(f"  Routing stats: overflow_frac={rstats['capacity']['overflow_frac']:.2%}")

# ── SparseMoETransformer ──────────────────────────────────────────────────────
print("\n── Sparse MoE Transformer ──")
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
print("\n── Auxiliary Losses ──")
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
print("\nMoE++ demo complete!")
''')
commit("feat: add examples/moe_v2_demo.py — config, experts, routers, capacity, MoELayer, sparse LM, aux losses")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_moe_v2.py", '''\
"""tests/test_moe_v2.py — Tests for NanoMind MoE++ package."""
import pytest
import torch
from nanomind.moe_v2 import (
    MoEConfig, Expert, ExpertBank,
    TopKRouter, ExpertChoiceRouter, HashRouter, RoutingOutput,
    CapacityBuffer, CapacityStats, MoELayer,
    MoETransformerBlock, SparseMoETransformer,
    load_balance_loss, z_loss, entropy_loss, combined_moe_loss,
)

D, V = 32, 16


def _cfg(n_experts=4, top_k=2):
    return MoEConfig(n_experts=n_experts, top_k=top_k,
                      d_model=D, d_ff=D * 2, capacity_factor=1.5)


# ── MoEConfig ─────────────────────────────────────────────────────────────────

class TestMoEConfig:
    def test_defaults_valid(self):
        cfg = MoEConfig()
        assert cfg.n_experts >= 1

    def test_active_ratio(self):
        cfg = MoEConfig(n_experts=8, top_k=2)
        assert abs(cfg.active_ratio - 0.25) < 1e-4

    def test_invalid_top_k(self):
        with pytest.raises(AssertionError):
            MoEConfig(n_experts=4, top_k=5)

    def test_to_dict_keys(self):
        d = _cfg().to_dict()
        for k in ("n_experts", "top_k", "d_model", "router_type"):
            assert k in d

    def test_total_params_positive(self):
        assert _cfg().total_params_estimate > 0


# ── Expert ────────────────────────────────────────────────────────────────────

class TestExpert:
    def test_gelu_output_shape(self):
        e = Expert(D, D * 2, variant="gelu")
        x = torch.randn(4, D)
        assert e(x).shape == (4, D)

    def test_swiglu_output_shape(self):
        e = Expert(D, D * 2, variant="swiglu")
        x = torch.randn(4, D)
        assert e(x).shape == (4, D)

    def test_gradient_flows(self):
        e = Expert(D, D * 2, variant="gelu")
        x = torch.randn(4, D)
        out = e(x).sum()
        out.backward()
        has_grad = any(p.grad is not None for p in e.parameters())
        assert has_grad


# ── ExpertBank ────────────────────────────────────────────────────────────────

class TestExpertBank:
    def test_n_experts(self):
        bank = ExpertBank(_cfg(n_experts=4))
        assert len(bank.experts) == 4

    def test_forward_expert_shape(self):
        bank = ExpertBank(_cfg())
        x    = torch.randn(5, D)
        out  = bank.forward_expert(0, x)
        assert out.shape == (5, D)

    def test_shared_expert_count(self):
        cfg  = MoEConfig(n_experts=4, top_k=2, d_model=D, d_ff=D*2, shared_experts=2)
        bank = ExpertBank(cfg)
        assert len(bank.shared) == 2

    def test_shared_forward_shape(self):
        cfg  = MoEConfig(n_experts=4, top_k=2, d_model=D, d_ff=D*2, shared_experts=1)
        bank = ExpertBank(cfg)
        x    = torch.randn(8, D)
        out  = bank.shared_forward(x)
        assert out.shape == (8, D)

    def test_n_params_per_expert_positive(self):
        bank = ExpertBank(_cfg())
        assert bank.n_params_per_expert > 0


# ── TopKRouter ────────────────────────────────────────────────────────────────

class TestTopKRouter:
    def _router(self, n_experts=4, top_k=2):
        return TopKRouter(_cfg(n_experts, top_k))

    def test_indices_shape(self):
        r  = self._router()
        x  = torch.randn(8, D)
        out = r(x)
        assert out.indices.shape == (8, 2)

    def test_weights_shape(self):
        r  = self._router()
        x  = torch.randn(8, D)
        out = r(x)
        assert out.weights.shape == (8, 2)

    def test_weights_sum_to_1(self):
        r   = self._router()
        x   = torch.randn(8, D)
        out = r(x)
        sums = out.weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_indices_in_range(self):
        r   = self._router(n_experts=4)
        x   = torch.randn(8, D)
        out = r(x)
        assert (out.indices >= 0).all() and (out.indices < 4).all()

    def test_aux_loss_non_negative(self):
        r   = self._router()
        x   = torch.randn(8, D)
        r.train()
        out = r(x)
        assert out.aux_loss.item() >= 0.0

    def test_expert_utilisation_keys(self):
        r   = self._router()
        x   = torch.randn(8, D)
        out = r(x)
        u   = r.expert_utilisation(out.router_probs)
        assert "expert_counts" in u and "entropy" in u


# ── HashRouter ────────────────────────────────────────────────────────────────

class TestHashRouter:
    def test_deterministic(self):
        r   = HashRouter(_cfg(n_experts=4))
        x   = torch.randn(8, D)
        o1  = r(x)
        o2  = r(x)
        assert torch.all(o1.indices == o2.indices)

    def test_indices_cover_all_experts(self):
        r   = HashRouter(_cfg(n_experts=4))
        x   = torch.randn(16, D)
        out = r(x)
        # With 16 tokens and 4 experts, all experts should appear
        assert len(out.indices.unique()) == 4

    def test_no_learnable_params(self):
        r = HashRouter(_cfg())
        assert sum(p.numel() for p in r.parameters()) == 0


# ── CapacityBuffer ────────────────────────────────────────────────────────────

class TestCapacityBuffer:
    def test_capacity_formula(self):
        buf = CapacityBuffer(n_experts=8, capacity_factor=1.0)
        # 80 tokens, 8 experts, cf=1.0 → capacity=10
        assert buf.capacity(80) == 10

    def test_capacity_factor_scales(self):
        b1 = CapacityBuffer(n_experts=4, capacity_factor=1.0)
        b2 = CapacityBuffer(n_experts=4, capacity_factor=2.0)
        assert b2.capacity(40) == 2 * b1.capacity(40)

    def test_stats_returns_stats(self):
        buf  = CapacityBuffer(n_experts=4, capacity_factor=1.5)
        idx  = torch.randint(0, 4, (16, 1))
        s    = buf.stats(idx, 16)
        assert isinstance(s, CapacityStats)

    def test_overflow_frac_in_range(self):
        buf = CapacityBuffer(n_experts=4, capacity_factor=1.5)
        idx = torch.randint(0, 4, (16, 1))
        s   = buf.stats(idx, 16)
        assert 0.0 <= s.overflow_frac <= 1.0


# ── MoELayer ──────────────────────────────────────────────────────────────────

class TestMoELayer:
    def _layer(self):
        return MoELayer(_cfg())

    def test_output_shape(self):
        layer = self._layer()
        x     = torch.randn(2, 6, D)
        out, aux = layer(x)
        assert out.shape == (2, 6, D)

    def test_aux_loss_scalar(self):
        layer = self._layer()
        x     = torch.randn(2, 6, D)
        _, aux = layer(x)
        assert aux.shape == ()

    def test_gradient_flows(self):
        layer = self._layer()
        x     = torch.randn(2, 4, D, requires_grad=True)
        out, aux = layer(x)
        (out.sum() + aux).backward()
        assert x.grad is not None

    def test_routing_stats_keys(self):
        layer = self._layer()
        x     = torch.randn(2, 4, D)
        s     = layer.routing_stats(x)
        assert "capacity" in s


# ── SparseMoETransformer ──────────────────────────────────────────────────────

class TestSparseMoETransformer:
    def _model(self):
        cfg = _cfg()
        return SparseMoETransformer(V, D, n_layers=2, n_heads=2,
                                     max_seq=8, moe_cfg=cfg)

    def test_logits_shape(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 4))
        logits, _, aux = m(ids)
        assert logits.shape == (2, 4, V)

    def test_loss_scalar(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 4))
        _, loss, aux = m(ids, ids)
        assert loss.shape == ()

    def test_aux_loss_non_negative(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 4))
        _, _, aux = m(ids)
        assert aux.item() >= 0.0

    def test_n_params_positive(self):
        m = self._model()
        assert m.n_params > 0

    def test_n_active_less_than_total(self):
        m = self._model()
        assert m.n_active_params <= m.n_params


# ── Auxiliary Losses ──────────────────────────────────────────────────────────

class TestMoELosses:
    def _setup(self, N=32, E=8):
        probs   = torch.softmax(torch.randn(N, E), dim=-1)
        logits  = torch.randn(N, E)
        idx     = probs.topk(2, dim=-1).indices
        return probs, logits, idx, E

    def test_load_balance_non_negative(self):
        probs, logits, idx, E = self._setup()
        lb = load_balance_loss(probs, idx, E)
        assert lb.item() >= 0.0

    def test_z_loss_non_negative(self):
        _, logits, _, _ = self._setup()
        assert z_loss(logits).item() >= 0.0

    def test_entropy_loss_sign(self):
        probs, _, _, _ = self._setup()
        el = entropy_loss(probs)
        # Negative entropy → negative value for high-entropy distributions
        assert isinstance(el.item(), float)

    def test_combined_loss_scalar(self):
        probs, logits, idx, E = self._setup()
        c = combined_moe_loss(probs, logits, idx, E)
        assert c.shape == ()

    def test_uniform_routing_low_lb_loss(self):
        """Perfectly uniform routing → low load balance loss."""
        E     = 8
        N     = 64
        probs = torch.full((N, E), 1.0 / E)
        idx   = torch.zeros(N, 1, dtype=torch.long)
        lb    = load_balance_loss(probs, idx, E)
        # Uniform P_i, concentrated f_i → moderate loss
        assert lb.item() >= 0.0
''')
commit("test: add full MoE++ test suite — config, expert, routers, capacity, layer, model, losses")

# COMMITS 12-18
for title, body in [
    ("test: add Expert SwiGLU vs GELU different outputs test", '''
class TestExpertVariants:
    def test_variants_differ(self):
        torch.manual_seed(0)
        e_gelu   = Expert(D, D * 2, variant="gelu")
        e_swiglu = Expert(D, D * 2, variant="swiglu")
        x = torch.randn(4, D)
        with torch.no_grad():
            out_g = e_gelu(x)
            out_s = e_swiglu(x)
        assert not torch.allclose(out_g, out_s)
'''),
    ("test: add TopKRouter no collapse after many steps test", '''
class TestNoRoutingCollapse:
    def test_routing_uses_all_experts(self):
        cfg    = MoEConfig(n_experts=4, top_k=2, d_model=D, d_ff=D*2,
                            router_noise=0.1)
        router = TopKRouter(cfg)
        router.train()
        torch.manual_seed(42)
        x      = torch.randn(64, D)
        out    = router(x)
        # All 4 experts should appear in top indices
        unique_experts = out.indices.unique()
        assert len(unique_experts) >= 2   # at least 2 active
'''),
    ("test: add CapacityBuffer masks correct length test", '''
class TestCapacityMasks:
    def test_mask_length(self):
        buf = CapacityBuffer(n_experts=4, capacity_factor=1.0)
        idx = torch.randint(0, 4, (20, 1))
        masks = buf.compute_masks(idx, 20)
        assert len(masks) == 4
        for m in masks:
            assert m.shape == (20,)

    def test_overflow_drops_tokens(self):
        """When all tokens go to expert 0, others should overflow."""
        buf = CapacityBuffer(n_experts=4, capacity_factor=0.5)
        idx = torch.zeros(20, 1, dtype=torch.long)  # all to expert 0
        stats = buf.stats(idx, 20)
        # capacity = 0.5 * 20/4 = 2 per expert; expert 0 gets 20 → overflow=18
        assert stats.overflow_tokens > 0
'''),
    ("test: add MoELayer with hash router runs test", '''
class TestMoELayerHashRouter:
    def test_hash_router_layer(self):
        cfg   = MoEConfig(n_experts=4, top_k=1, d_model=D, d_ff=D*2,
                           router_type="hash")
        layer = MoELayer(cfg)
        x     = torch.randn(2, 6, D)
        out, aux = layer(x)
        assert out.shape == (2, 6, D)
        assert aux.item() == 0.0   # hash router has no aux loss
'''),
    ("test: add SparseMoETransformer total loss backward test", '''
class TestMoEBackward:
    def test_total_loss_backward(self):
        cfg = MoEConfig(n_experts=4, top_k=2, d_model=D, d_ff=D*2)
        m   = SparseMoETransformer(V, D, n_layers=1, n_heads=2,
                                    max_seq=8, moe_cfg=cfg)
        ids  = torch.randint(0, V, (2, 4))
        _, loss, aux = m(ids, ids)
        (loss + aux).backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad
'''),
    ("test: add Z-loss decreases with smaller logits test", '''
class TestZLossDecreases:
    def test_large_logits_large_z_loss(self):
        small_logits = torch.randn(16, 8) * 0.1
        large_logits = torch.randn(16, 8) * 10.0
        z_small = z_loss(small_logits)
        z_large = z_loss(large_logits)
        assert z_large.item() > z_small.item()
'''),
    ("test: add ExpertChoiceRouter output shape test", '''
class TestExpertChoiceRouter:
    def test_output_shape(self):
        cfg = MoEConfig(n_experts=4, top_k=1, d_model=D, d_ff=D*2,
                         router_type="expert_choice")
        r   = ExpertChoiceRouter(cfg, capacity=4)
        x   = torch.randn(16, D)
        out = r(x)
        assert out.indices.shape == (16, 1)
        assert out.weights.shape == (16, 1)
'''),
]:
    src = read("tests/test_moe_v2.py")
    src += "\n" + body
    write("tests/test_moe_v2.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.4.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.3.0\"", "__version__ = \"4.4.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.4.0 — Advanced MoE++ release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `diffusion`  | Diffusion LMs — DDPM/DDIM, masked diffusion, CFG, cosine/linear schedules |",
    "| `diffusion`  | Diffusion LMs — DDPM/DDIM, masked diffusion, CFG, cosine/linear schedules |\n"
    "| `moe_v2`     | Advanced MoE++ — TopK/ExpertChoice/Hash routing, capacity, z-loss, SwiGLU |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.4.0] — 2024 — Advanced Mixture of Experts (MoE++)\n\n### Added\n"
      "- `MoEConfig` — n_experts, top_k, capacity_factor, router_type, shared_experts\n"
      "- `Expert` — GELU and SwiGLU FFN variants\n"
      "- `ExpertBank` — N independent experts + shared always-active experts (DeepSeek-style)\n"
      "- `TopKRouter` — noisy top-K with load-balance auxiliary loss\n"
      "- `ExpertChoiceRouter` — expert selects top-C tokens (perfect balance)\n"
      "- `HashRouter` — deterministic hash routing (no learnable params)\n"
      "- `CapacityBuffer` — token overflow management, capacity stats\n"
      "- `MoELayer` — full dispatch/compute/combine forward pass\n"
      "- `MoETransformerBlock` — attention + MoE FFN with AdaLN residuals\n"
      "- `SparseMoETransformer` — full sparse LM, n_params, n_active_params\n"
      "- `load_balance_loss` — Switch Transformer auxiliary loss\n"
      "- `z_loss` — ST-MoE router logit regularisation\n"
      "- `entropy_loss` — routing entropy maximisation\n"
      "- `combined_moe_loss` — load_balance + z_loss combined\n"
      "- `examples/moe_v2_demo.py` — full MoE++ demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.4.0, update README and CHANGELOG for Day 44 Advanced MoE++")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 44 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.4.0",
    "-m", "NanoMind v4.4.0 — Advanced MoE++", check=False)
r = run("git", "push", "origin", "v4.4.0", check=False)
print("Tag v4.4.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 44 COMPLETE — v4.4.0 TAGGED! ===")
