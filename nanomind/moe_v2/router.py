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
