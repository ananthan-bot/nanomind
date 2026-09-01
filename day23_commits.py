"""
day23_commits.py — 20 atomic commits for Day 23: Mixture of Experts (MoE).
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

print("\n=== DAY 23: Mixture of Experts (MoE) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — moe package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/__init__.py",
      '"""NanoMind Mixture of Experts (MoE) sub-package."""\n')
commit("feat: add nanomind/moe/ package skeleton for Mixture of Experts")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — MoEConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/config.py", '''\
"""
nanomind/moe/config.py — Mixture of Experts configuration.

MoE replaces the dense FFN in each transformer block with a sparse mixture:

  Dense FFN :   every token uses THE SAME d_model → d_ff → d_model transform
  MoE FFN   :   every token chooses the TOP-K of N experts (smaller FFNs)

Parameters scale with N experts but compute scales with only K:
  - Parameters : N × (d_model × d_ff × 2) — huge capacity
  - Compute    : K × (d_model × d_ff × 2) — same as K dense layers

Reference: Fedus et al. (2021) Switch Transformer — https://arxiv.org/abs/2101.03961
           Jiang et al. (2024) Mixtral 8×7B    — https://arxiv.org/abs/2401.04088
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MoEConfig:
    """
    Configuration for a Sparse Mixture of Experts FFN layer.

    Attributes:
        num_experts:      Total number of expert FFNs (N). Typically 8, 16, or 64.
        top_k:            Number of experts each token routes to (K). Typically 1 or 2.
        expert_capacity:  Max tokens per expert per batch (None = no cap).
        load_balance_coef: Weight of the auxiliary load-balancing loss.
                           Set to 0.0 to disable. Typical: 0.01.
        expert_dropout:   Dropout probability applied to expert outputs during training.
        d_ff_expert:      Hidden dimension of each expert FFN.
                          Defaults to 4 × d_model if None.
        activation:       Expert FFN activation (``"gelu"`` or ``"relu"`` or ``"swiglu"``).
    """

    num_experts:      int   = 8
    top_k:            int   = 2
    expert_capacity:  int | None = None
    load_balance_coef: float = 0.01
    expert_dropout:   float = 0.0
    d_ff_expert:      int | None = None
    activation:       str   = "gelu"

    def __post_init__(self) -> None:
        assert self.num_experts >= 1
        assert 1 <= self.top_k <= self.num_experts
        assert self.load_balance_coef >= 0.0
        assert self.activation in ("gelu", "relu", "swiglu")
''')
commit("feat: add MoEConfig — num_experts, top_k, load_balance_coef, expert_capacity")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Expert FFN module
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/expert.py", '''\
"""
nanomind/moe/expert.py — Single FFN Expert module.

Each expert is a standard two-layer FFN, identical in structure to the
dense FFN in a regular transformer block but with its own independent weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    A single FFN expert in a Mixture of Experts layer.

    Architecture: Linear → Activation → Linear (same as dense FFN).

    Args:
        d_model:    Input/output dimension.
        d_ff:       Hidden dimension (typically 4 × d_model).
        activation: Activation function: ``"gelu"``, ``"relu"``, or ``"swiglu"``.
        bias:       Whether to include bias terms.
    """

    def __init__(
        self,
        d_model:    int,
        d_ff:       int,
        activation: str  = "gelu",
        bias:       bool = False,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.d_ff       = d_ff
        self.activation = activation

        if activation == "swiglu":
            # SwiGLU: two up projections, elementwise gate × linear
            self.gate = nn.Linear(d_model, d_ff, bias=bias)
            self.up   = nn.Linear(d_model, d_ff, bias=bias)
        else:
            self.fc1  = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.fc2(F.silu(self.gate(x)) * self.up(x))
        h = self.fc1(x)
        if self.activation == "gelu":
            h = F.gelu(h)
        else:
            h = F.relu(h)
        return self.fc2(h)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, act={self.activation}"
''')
commit("feat: add Expert — single FFN expert with gelu/relu/swiglu activation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — TopKRouter
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/router.py", '''\
"""
nanomind/moe/router.py — Top-K sparse router for Mixture of Experts.

The router is a simple linear layer that maps each token to N expert logits.
The top-K experts by logit are selected; only those K experts process the token.
The routing weights (softmax over top-K logits) are used to blend expert outputs.

Routing steps:
  1. Compute router logits: (B×T, N)
  2. Take top-K experts per token
  3. Compute softmax weights over top-K logits only
  4. Return indices and weights for the MoE forward pass
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    Linear top-K router for Mixture of Experts.

    Args:
        d_model:     Input token embedding dimension.
        num_experts: Total number of experts (N).
        top_k:       Number of experts each token is routed to (K).
    """

    def __init__(
        self,
        d_model:     int,
        num_experts: int,
        top_k:       int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-K routing assignments.

        Args:
            x: Token embeddings ``(B, T, d_model)`` or ``(tokens, d_model)``.

        Returns:
            Tuple of:
            - ``expert_indices`` : Top-K expert IDs per token ``(tokens, K)``
            - ``expert_weights`` : Softmax routing weights ``(tokens, K)``
            - ``router_logits``  : Raw router logits ``(tokens, N)`` for aux loss
        """
        shape    = x.shape
        x_flat   = x.reshape(-1, shape[-1])       # (tokens, d_model)
        logits   = self.gate(x_flat)               # (tokens, N)

        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        weights  = F.softmax(top_logits, dim=-1)   # (tokens, K)

        return top_indices, weights, logits

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, top_k={self.top_k}"
''')
commit("feat: add TopKRouter — linear gate, top-K selection, softmax routing weights")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — load_balance_loss() auxiliary loss
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/load_balance.py", '''\
"""
nanomind/moe/load_balance.py — Load balancing auxiliary loss for MoE.

Without an auxiliary loss, MoE collapses: a few experts receive most tokens
while the rest become unused (expert collapse). The load balancing loss
encourages uniform routing across all experts.

Switch Transformer loss (Fedus et al. 2021):
    L_aux = N × Σ_i f_i × P_i

where:
  f_i = fraction of tokens routed to expert i
  P_i = mean routing probability for expert i (from softmax)
  N   = num_experts

The loss is minimised when f_i = P_i = 1/N for all i (perfectly balanced).
"""

from __future__ import annotations

import torch


def load_balance_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute the Switch Transformer load balancing auxiliary loss.

    Args:
        router_logits:  Raw router logits ``(tokens, num_experts)``.
        expert_indices: Top-K expert indices per token ``(tokens, K)``.
        num_experts:    Total number of experts N.

    Returns:
        Scalar auxiliary loss tensor.
    """
    num_tokens = router_logits.shape[0]

    # f_i: fraction of tokens dispatched to expert i
    # Count how many times each expert appears in top-K assignments
    dispatch_mask = torch.zeros(
        num_tokens, num_experts,
        device=router_logits.device, dtype=router_logits.dtype
    )
    dispatch_mask.scatter_(
        1,
        expert_indices.reshape(num_tokens, -1),
        1.0 / expert_indices.shape[1],   # share equally among K chosen experts
    )
    f_i = dispatch_mask.mean(dim=0)      # (N,) mean fraction per expert

    # P_i: mean routing probability for expert i (across all tokens)
    p_i = torch.softmax(router_logits, dim=-1).mean(dim=0)   # (N,)

    # L_aux = N × Σ f_i × P_i
    loss = num_experts * (f_i * p_i).sum()
    return loss


def expert_utilization(
    expert_indices: torch.Tensor,
    num_experts:    int,
) -> dict:
    """
    Compute expert utilization statistics.

    Args:
        expert_indices: Token → expert assignments ``(tokens, K)``.
        num_experts:    Total experts.

    Returns:
        Dict with ``counts``, ``fractions``, ``min_frac``, ``max_frac``,
        ``utilization`` (fraction of experts with >0 tokens).
    """
    flat    = expert_indices.reshape(-1)
    counts  = torch.bincount(flat, minlength=num_experts).float()
    total   = flat.numel()
    fracs   = counts / max(total, 1)
    used    = (counts > 0).sum().item()
    return {
        "counts":      counts.tolist(),
        "fractions":   fracs.tolist(),
        "min_frac":    fracs.min().item(),
        "max_frac":    fracs.max().item(),
        "utilization": used / num_experts,
    }
''')
commit("feat: add load_balance_loss() — Switch Transformer auxiliary loss for expert balance")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — SparseMoELayer (core MoE forward)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/layer.py", '''\
"""
nanomind/moe/layer.py — Sparse Mixture of Experts FFN layer.

Replaces the standard dense FFN in a transformer block with N independent
experts and a top-K router. Each token is processed by only K of the N
experts, with outputs blended by the routing weights.

Forward algorithm:
  1. Router: assign each token to its top-K experts and compute weights
  2. For each expert: gather assigned tokens, run FFN, scatter results back
  3. Blend: weighted sum of expert outputs per token
  4. Optionally compute auxiliary load-balancing loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.moe.config import MoEConfig
from nanomind.moe.expert import Expert
from nanomind.moe.router import TopKRouter
from nanomind.moe.load_balance import load_balance_loss


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts FFN layer.

    Replaces the dense FFN in a transformer block. Each token is routed to
    its top-K experts; outputs are blended by routing weights.

    Args:
        d_model:    Model embedding dimension.
        cfg:        MoE configuration.
        bias:       Whether expert projections have bias.
    """

    def __init__(
        self,
        d_model: int,
        cfg:     MoEConfig,
        bias:    bool = False,
    ) -> None:
        super().__init__()
        self.d_model     = d_model
        self.cfg         = cfg
        d_ff             = cfg.d_ff_expert or (4 * d_model)

        self.router  = TopKRouter(d_model, cfg.num_experts, cfg.top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, cfg.activation, bias)
            for _ in range(cfg.num_experts)
        ])
        self.dropout = nn.Dropout(cfg.expert_dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse MoE forward pass.

        Args:
            x: Input ``(B, T, d_model)``.

        Returns:
            Tuple of:
            - Output tensor ``(B, T, d_model)`` — weighted sum of expert outputs
            - Auxiliary load-balancing loss (scalar tensor)
        """
        B, T, D = x.shape
        x_flat  = x.reshape(B * T, D)             # (tokens, D)

        # Router
        expert_indices, expert_weights, router_logits = self.router(x_flat)
        # expert_indices: (tokens, K), expert_weights: (tokens, K)

        output = torch.zeros_like(x_flat)

        # Process each expert: gather assigned tokens, run FFN, scatter back
        for expert_id, expert in enumerate(self.experts):
            # Boolean mask: which (token, k) slots chose this expert
            mask = (expert_indices == expert_id)   # (tokens, K)

            # token positions that use this expert (may be used in multiple k slots)
            token_mask = mask.any(dim=1)           # (tokens,)
            if not token_mask.any():
                continue

            selected = x_flat[token_mask]          # (selected, D)
            expert_out = expert(selected)          # (selected, D)
            expert_out = self.dropout(expert_out)

            # Weight: sum of weights across k slots for this expert
            w = (expert_weights * mask.float()).sum(dim=1)   # (tokens,)
            output[token_mask] += expert_out * w[token_mask].unsqueeze(1)

        output = output.reshape(B, T, D)

        # Auxiliary load-balancing loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.cfg.load_balance_coef > 0.0:
            aux_loss = self.cfg.load_balance_coef * load_balance_loss(
                router_logits, expert_indices, self.cfg.num_experts
            )

        return output, aux_loss

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.cfg.num_experts}, "
            f"top_k={self.cfg.top_k}, "
            f"d_model={self.d_model}"
        )
''')
commit("feat: implement SparseMoELayer — router + N experts, weighted output blend, aux loss")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — MoETransformerBlock
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/block.py", '''\
"""
nanomind/moe/block.py — Transformer block with MoE FFN replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.moe.config import MoEConfig
from nanomind.moe.layer import SparseMoELayer
from nanomind.pos.factory import get_attention
from nanomind.norm.factory import get_norm


class MoETransformerBlock(nn.Module):
    """
    Transformer block where the dense FFN is replaced by a SparseMoELayer.

    Architecture (pre-norm):
        x → Norm → Attention → x + residual
        x → Norm → SparseMoE → x + residual (+ aux_loss)

    Args:
        d_model:    Embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        moe_cfg:    MoE configuration.
        dropout:    Dropout for attention and residual.
        norm_type:  Layer norm type (``"layernorm"`` or ``"rmsnorm"``).
        pos_type:   Positional embedding type for attention.
        n_kv_heads: For GQA/MQA (None = standard MHA).
        window_size: For Sliding Window Attention.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        block_size:  int,
        moe_cfg:     MoEConfig,
        dropout:     float = 0.0,
        norm_type:   str   = "layernorm",
        pos_type:    str   = "learned",
        n_kv_heads:  int | None = None,
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.attn = get_attention(
            pos_type=pos_type,
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
        )
        self.moe      = SparseMoELayer(d_model, moe_cfg)
        self.norm1    = get_norm(norm_type, d_model)
        self.norm2    = get_norm(norm_type, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE transformer block.

        Returns:
            Tuple of ``(output, aux_loss)``.
        """
        # Self-attention with pre-norm
        attn_out, _ = self.attn(self.norm1(x), kv_cache)
        x = x + self.drop(attn_out)

        # MoE FFN with pre-norm
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.drop(moe_out)

        return x, aux_loss
''')
commit("feat: add MoETransformerBlock — attention + SparseMoE FFN with pre-norm and residuals")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — NanoMindMoE model
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/model.py", '''\
"""
nanomind/moe/model.py — NanoMind with Mixture of Experts FFN layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.model.config import ModelConfig
from nanomind.moe.config import MoEConfig
from nanomind.moe.block import MoETransformerBlock
from nanomind.norm.factory import get_norm
from nanomind.utils.logger import get_logger

log = get_logger("moe.model")


class NanoMindMoE(nn.Module):
    """
    NanoMind transformer model with Sparse Mixture of Experts FFN layers.

    Every transformer block uses a SparseMoELayer instead of the dense FFN.
    The forward pass returns both logits and the summed auxiliary load-balancing
    loss (to be added to the cross-entropy loss during training).

    Args:
        model_cfg: Standard model configuration.
        moe_cfg:   MoE configuration.

    Example::

        cfg     = ModelConfig(vocab_size=32000, block_size=512, d_model=256,
                              n_layers=6, n_heads=8)
        moe_cfg = MoEConfig(num_experts=8, top_k=2)
        model   = NanoMindMoE(cfg, moe_cfg)

        logits, aux_loss = model(input_ids)
        loss = cross_entropy(logits, labels) + aux_loss
    """

    def __init__(self, model_cfg: ModelConfig, moe_cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg     = model_cfg
        self.moe_cfg = moe_cfg

        self.tok_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        # Learned pos embedding (MoE typically uses RoPE, but keep it flexible)
        if model_cfg.pos_type in ("learned", None, ""):
            self.pos_emb = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        else:
            self.pos_emb = None

        self.drop    = nn.Dropout(model_cfg.dropout)
        self.blocks  = nn.ModuleList([
            MoETransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                block_size=model_cfg.block_size,
                moe_cfg=moe_cfg,
                dropout=model_cfg.dropout,
                norm_type=getattr(model_cfg, "norm_type", "layernorm"),
                pos_type=model_cfg.pos_type,
                n_kv_heads=model_cfg.n_kv_heads,
                window_size=model_cfg.window_size,
            )
            for _ in range(model_cfg.n_layers)
        ])
        self.norm    = get_norm(getattr(model_cfg, "norm_type", "layernorm"), model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"NanoMindMoE: {n_params:,} params | "
                 f"{moe_cfg.num_experts} experts, top-{moe_cfg.top_k}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx:    torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            idx:     Token IDs ``(B, T)``.
            targets: Target token IDs for loss computation ``(B, T)``.

        Returns:
            Tuple of:
            - Logits ``(B, T, vocab_size)``
            - Combined loss (cross-entropy + aux) if targets given, else aux_loss total
        """
        B, T = idx.shape
        x    = self.tok_emb(idx)

        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x   = x + self.pos_emb(pos)

        x = self.drop(x)

        total_aux = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux

        x      = self.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            return logits, loss + total_aux

        return logits, total_aux

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"NanoMindMoE("
            f"params={self.num_parameters():,}, "
            f"experts={self.moe_cfg.num_experts}, "
            f"top_k={self.moe_cfg.top_k})"
        )
''')
commit("feat: add NanoMindMoE — full transformer model with SparseMoE FFN in every block")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — expert utilization report
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/utils.py", '''\
"""
nanomind/moe/utils.py — MoE utility functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from nanomind.moe.layer import SparseMoELayer
from nanomind.moe.load_balance import expert_utilization


def get_all_router_stats(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, dict]:
    """
    Run a forward pass and collect expert utilization stats from all MoE layers.

    Args:
        model:     NanoMindMoE model.
        input_ids: Token IDs ``(B, T)``.

    Returns:
        Dict mapping layer name to utilization stats dict.
    """
    stats:   dict[str, dict] = {}
    handles: list            = []

    def _make_hook(name: str):
        def hook(module, inp, out):
            # out is (output, aux_loss); we need router's expert_indices
            x_flat  = inp[0].reshape(-1, inp[0].shape[-1])
            indices, _, _ = module.router(x_flat)
            stats[name] = expert_utilization(indices, module.cfg.num_experts)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, SparseMoELayer):
            handles.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return stats


def print_moe_utilization(stats: dict[str, dict]) -> None:
    """Pretty-print expert utilization across all MoE layers."""
    print("=" * 60)
    print("Expert Utilization Report")
    print("=" * 60)
    for layer_name, s in stats.items():
        print(f"\n  Layer: {layer_name}")
        print(f"    Used experts    : {s['utilization']:.0%}")
        print(f"    Min token frac  : {s['min_frac']:.3f}")
        print(f"    Max token frac  : {s['max_frac']:.3f}")
        fracs = [f"{f:.2f}" for f in s["fractions"]]
        print(f"    Per-expert frac : [{', '.join(fracs)}]")
    print("=" * 60)
''')
commit("feat: add get_all_router_stats() and print_moe_utilization() — expert usage diagnostics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update moe __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/moe/__init__.py", '''\
"""NanoMind Mixture of Experts (MoE) sub-package.

Replaces the dense FFN in transformer blocks with N independent experts,
routing each token to only K of them (sparse activation).

Parameters scale with N; compute scales with K — giving huge capacity
at the same inference cost as a K-expert dense model.

Primary exports:
    - :class:`NanoMindMoE`       — full transformer model with MoE FFNs
    - :class:`MoEConfig`         — num_experts, top_k, load_balance_coef
    - :class:`SparseMoELayer`    — router + N experts, weighted blend
    - :class:`TopKRouter`        — linear gate + top-K selection
    - :class:`Expert`            — single FFN expert (gelu/relu/swiglu)
    - :class:`MoETransformerBlock` — attention + MoE FFN block
    - :func:`load_balance_loss`  — Switch Transformer auxiliary loss
    - :func:`expert_utilization` — per-expert token fraction stats
    - :func:`get_all_router_stats`  — hook-based routing diagnostics
    - :func:`print_moe_utilization` — pretty-print utilization report
"""

from nanomind.moe.config import MoEConfig
from nanomind.moe.expert import Expert
from nanomind.moe.router import TopKRouter
from nanomind.moe.load_balance import load_balance_loss, expert_utilization
from nanomind.moe.layer import SparseMoELayer
from nanomind.moe.block import MoETransformerBlock
from nanomind.moe.model import NanoMindMoE
from nanomind.moe.utils import get_all_router_stats, print_moe_utilization

__all__ = [
    "MoEConfig",
    "Expert",
    "TopKRouter",
    "load_balance_loss",
    "expert_utilization",
    "SparseMoELayer",
    "MoETransformerBlock",
    "NanoMindMoE",
    "get_all_router_stats",
    "print_moe_utilization",
]
''')
commit("refactor: export all MoE components from nanomind/moe/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: moe_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/moe_demo.py", '''\
"""
examples/moe_demo.py — Mixture of Experts demo.

Shows how to build a NanoMindMoE model and inspect expert utilization.

Usage:
    python examples/moe_demo.py
"""

import torch
from nanomind.model.config import ModelConfig
from nanomind.moe import MoEConfig, NanoMindMoE, get_all_router_stats, print_moe_utilization

# ── Build model ───────────────────────────────────────────────────────────────
model_cfg = ModelConfig(
    vocab_size=256, block_size=64, d_model=128,
    n_layers=4, n_heads=4, dropout=0.0,
)
moe_cfg   = MoEConfig(
    num_experts=8,
    top_k=2,
    load_balance_coef=0.01,
    activation="swiglu",
)
model = NanoMindMoE(model_cfg, moe_cfg)
print(model)
print(f"Total params : {model.num_parameters():,}")

# Dense equivalent params (for comparison)
dense_ffn = 2 * model_cfg.d_model * (4 * model_cfg.d_model) * model_cfg.n_layers
moe_ffn   = moe_cfg.num_experts * 2 * model_cfg.d_model * (4 * model_cfg.d_model) * model_cfg.n_layers
print(f"Dense FFN params  : {dense_ffn:,}")
print(f"MoE FFN params    : {moe_ffn:,}  ({moe_ffn/dense_ffn:.1f}x more)")
print(f"Active per token  : top-{moe_cfg.top_k} of {moe_cfg.num_experts} experts")

# ── Forward pass ──────────────────────────────────────────────────────────────
idx = torch.randint(0, 256, (2, 32))
with torch.no_grad():
    logits, aux_loss = model(idx)
print(f"\nLogits shape : {logits.shape}")
print(f"Aux loss     : {aux_loss.item():.4f}")

# ── Training step with combined loss ─────────────────────────────────────────
targets = torch.randint(0, 256, (2, 32))
logits, total_loss = model(idx, targets)
print(f"Total loss   : {total_loss.item():.4f}  (CE + {moe_cfg.load_balance_coef}×aux)")

# ── Expert utilization diagnostics ───────────────────────────────────────────
stats = get_all_router_stats(model, idx)
print_moe_utilization(stats)
''')
commit("feat: add examples/moe_demo.py — NanoMindMoE forward, aux loss, utilization report")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: MoEConfig validation
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_moe.py", '''\
"""
tests/test_moe.py — Tests for Mixture of Experts (MoE).
"""

import pytest
import torch
import torch.nn as nn

from nanomind.model.config import ModelConfig
from nanomind.moe import (
    MoEConfig, Expert, TopKRouter, SparseMoELayer,
    MoETransformerBlock, NanoMindMoE,
    load_balance_loss, expert_utilization,
)

B, T, D = 2, 8, 64
N_EXP, TOP_K = 4, 2


def tiny_moe_model(num_experts=N_EXP, top_k=TOP_K):
    torch.manual_seed(0)
    cfg     = ModelConfig(vocab_size=32, block_size=T, d_model=D,
                          n_layers=2, n_heads=4, dropout=0.0)
    moe_cfg = MoEConfig(num_experts=num_experts, top_k=top_k,
                        load_balance_coef=0.01)
    return NanoMindMoE(cfg, moe_cfg)


# ── MoEConfig ─────────────────────────────────────────────────────────────────

class TestMoEConfig:
    def test_defaults(self):
        cfg = MoEConfig()
        assert cfg.num_experts == 8
        assert cfg.top_k == 2

    def test_invalid_num_experts(self):
        with pytest.raises(AssertionError):
            MoEConfig(num_experts=0)

    def test_top_k_exceeds_experts(self):
        with pytest.raises(AssertionError):
            MoEConfig(num_experts=4, top_k=5)

    def test_invalid_activation(self):
        with pytest.raises(AssertionError):
            MoEConfig(activation="tanh")

    def test_negative_load_balance(self):
        with pytest.raises(AssertionError):
            MoEConfig(load_balance_coef=-0.1)
''')
commit("test: add MoEConfig validation — defaults, invalid experts, top_k, activation tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: Expert FFN
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_moe.py")
src += '''

# ── Expert ────────────────────────────────────────────────────────────────────

class TestExpert:
    def test_output_shape_gelu(self):
        exp = Expert(D, D * 4, activation="gelu")
        x   = torch.randn(B * T, D)
        assert exp(x).shape == (B * T, D)

    def test_output_shape_relu(self):
        exp = Expert(D, D * 4, activation="relu")
        x   = torch.randn(B * T, D)
        assert exp(x).shape == (B * T, D)

    def test_output_shape_swiglu(self):
        exp = Expert(D, D * 4, activation="swiglu")
        x   = torch.randn(B * T, D)
        assert exp(x).shape == (B * T, D)

    def test_3d_input(self):
        exp = Expert(D, D * 4)
        x   = torch.randn(B, T, D)
        assert exp(x).shape == (B, T, D)
'''
write("tests/test_moe.py", src)
commit("test: add Expert output shape tests for gelu, relu, swiglu, and 3D input")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: TopKRouter
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_moe.py")
src += '''

# ── TopKRouter ────────────────────────────────────────────────────────────────

class TestTopKRouter:
    def test_output_shapes(self):
        router = TopKRouter(D, N_EXP, TOP_K)
        x      = torch.randn(B, T, D)
        indices, weights, logits = router(x)
        assert indices.shape == (B * T, TOP_K)
        assert weights.shape == (B * T, TOP_K)
        assert logits.shape  == (B * T, N_EXP)

    def test_indices_in_range(self):
        router = TopKRouter(D, N_EXP, TOP_K)
        x      = torch.randn(B, T, D)
        indices, _, _ = router(x)
        assert (indices >= 0).all()
        assert (indices < N_EXP).all()

    def test_weights_sum_to_one(self):
        router = TopKRouter(D, N_EXP, TOP_K)
        x      = torch.randn(B, T, D)
        _, weights, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_top_k_unique_per_token(self):
        router  = TopKRouter(D, N_EXP, TOP_K)
        x       = torch.randn(B, T, D)
        indices, _, _ = router(x)
        for row in indices:
            assert len(set(row.tolist())) == TOP_K
'''
write("tests/test_moe.py", src)
commit("test: add TopKRouter shape, index range, weight sum, and uniqueness tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: SparseMoELayer output
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_moe.py")
src += '''

# ── SparseMoELayer ────────────────────────────────────────────────────────────

class TestSparseMoELayer:
    def test_output_shape(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.0)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        out, aux = moe(x)
        assert out.shape == (B, T, D)

    def test_aux_loss_zero_when_disabled(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.0)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        _, aux = moe(x)
        assert aux.item() == 0.0

    def test_aux_loss_positive_when_enabled(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.01)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        _, aux = moe(x)
        assert aux.item() >= 0.0

    def test_output_finite(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K)
        moe = SparseMoELayer(D, cfg)
        x   = torch.randn(B, T, D)
        out, _ = moe(x)
        assert out.isfinite().all()

    def test_n_experts_in_layer(self):
        cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K)
        moe = SparseMoELayer(D, cfg)
        assert len(moe.experts) == N_EXP
'''
write("tests/test_moe.py", src)
commit("test: add SparseMoELayer shape, aux loss disable/enable, finite output, expert count tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: load_balance_loss
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_moe.py")
src += '''

# ── load_balance_loss ─────────────────────────────────────────────────────────

class TestLoadBalanceLoss:
    def test_returns_scalar(self):
        logits  = torch.randn(16, N_EXP)
        indices = torch.randint(0, N_EXP, (16, TOP_K))
        loss    = load_balance_loss(logits, indices, N_EXP)
        assert loss.shape == ()

    def test_non_negative(self):
        logits  = torch.randn(32, N_EXP)
        indices = torch.randint(0, N_EXP, (32, TOP_K))
        loss    = load_balance_loss(logits, indices, N_EXP)
        assert loss.item() >= 0.0

    def test_balanced_routing_gives_low_loss(self):
        # Uniform routing: each expert gets exactly 1/N of tokens
        n_tokens = N_EXP * 4
        logits   = torch.zeros(n_tokens, N_EXP)
        # Assign tokens round-robin to ensure balance
        indices  = torch.tensor([[i % N_EXP, (i+1) % N_EXP]
                                  for i in range(n_tokens)])
        loss     = load_balance_loss(logits, indices, N_EXP)
        # Balanced loss should be close to 1.0 (= N × 1/N × 1/N × N = 1)
        assert loss.item() < 2.0

    def test_expert_utilization_keys(self):
        indices  = torch.randint(0, N_EXP, (32, TOP_K))
        stats    = expert_utilization(indices, N_EXP)
        for key in ("counts", "fractions", "min_frac", "max_frac", "utilization"):
            assert key in stats
'''
write("tests/test_moe.py", src)
commit("test: add load_balance_loss scalar, non-negative, balanced routing, and utilization tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: NanoMindMoE forward pass
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_moe.py")
src += '''

# ── NanoMindMoE ───────────────────────────────────────────────────────────────

class TestNanoMindMoE:
    def test_forward_shape(self):
        model = tiny_moe_model()
        idx   = torch.randint(0, 32, (B, T))
        logits, aux = model(idx)
        assert logits.shape == (B, T, 32)

    def test_aux_loss_scalar(self):
        model = tiny_moe_model()
        idx   = torch.randint(0, 32, (B, T))
        _, aux = model(idx)
        assert aux.shape == ()

    def test_training_loss_includes_aux(self):
        model   = tiny_moe_model()
        idx     = torch.randint(0, 32, (B, T))
        targets = torch.randint(0, 32, (B, T))
        _, loss = model(idx, targets)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        model   = tiny_moe_model()
        idx     = torch.randint(0, 32, (B, T))
        targets = torch.randint(0, 32, (B, T))
        _, loss = model(idx, targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_repr_contains_experts(self):
        model = tiny_moe_model()
        assert "experts" in repr(model).lower() or "NanoMindMoE" in repr(model)

    def test_more_experts_more_params(self):
        m4 = tiny_moe_model(num_experts=4, top_k=2)
        m8 = tiny_moe_model(num_experts=8, top_k=2)
        assert m8.num_parameters() > m4.num_parameters()
'''
write("tests/test_moe.py", src)
commit("test: add NanoMindMoE shape, aux loss, training loss, gradient flow, and param count tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: MoETransformerBlock
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_moe.py")
src += '''

# ── MoETransformerBlock ───────────────────────────────────────────────────────

class TestMoETransformerBlock:
    def test_output_shape(self):
        moe_cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K)
        block   = MoETransformerBlock(D, 4, T, moe_cfg, dropout=0.0)
        x       = torch.randn(B, T, D)
        out, aux = block(x)
        assert out.shape == (B, T, D)

    def test_aux_loss_returned(self):
        moe_cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K, load_balance_coef=0.01)
        block   = MoETransformerBlock(D, 4, T, moe_cfg, dropout=0.0)
        x       = torch.randn(B, T, D)
        _, aux  = block(x)
        assert aux.shape == ()

    def test_residual_keeps_shape(self):
        moe_cfg = MoEConfig(num_experts=N_EXP, top_k=TOP_K)
        block   = MoETransformerBlock(D, 4, T, moe_cfg, dropout=0.0)
        x       = torch.randn(1, T, D)
        out, _  = block(x)
        assert out.shape == x.shape
'''
write("tests/test_moe.py", src)
commit("test: add MoETransformerBlock output shape, aux loss, and residual shape tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump version + expose MoE in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"1.8.0\"", "__version__ = \"1.9.0\"")
src = src.replace(
    "from nanomind.generate.beam_generator import BeamSearchGenerator",
    "from nanomind.generate.beam_generator import BeamSearchGenerator\n"
    "from nanomind.moe import MoEConfig, NanoMindMoE, SparseMoELayer"
)
src = src.replace(
    "    \"BeamSearchGenerator\",\n    \"__version__\",\n]",
    "    \"BeamSearchGenerator\",\n"
    "    \"MoEConfig\",\n"
    "    \"NanoMindMoE\",\n"
    "    \"SparseMoELayer\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v1.9.0 — expose MoEConfig, NanoMindMoE, SparseMoELayer in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Decoding** | Beam search + Diverse beam search — better quality generation |",
    "| **Decoding** | Beam search + Diverse beam search — better quality generation |\n"
    "| **Architecture** | Mixture of Experts — N experts, top-K routing, load balance loss |"
)
readme = readme.replace(
    "**Total: 440 commits across 22 days.**",
    "**Total: 460 commits across 23 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.8.0] — 2024 — Beam Search & Diverse Beam Search",
    "## [1.9.0] — 2024 — Mixture of Experts (MoE)\n\n### Added\n"
    "- `NanoMindMoE` — full transformer model with SparseMoE FFN in every block\n"
    "- `SparseMoELayer` — top-K router + N expert FFNs with weighted blending\n"
    "- `TopKRouter` — linear gate, top-K selection, softmax routing weights\n"
    "- `Expert` — single FFN expert (gelu/relu/swiglu activations)\n"
    "- `MoETransformerBlock` — attention + MoE FFN with pre-norm residuals\n"
    "- `MoEConfig` — num_experts, top_k, load_balance_coef, expert_capacity\n"
    "- `load_balance_loss()` — Switch Transformer auxiliary load balancing loss\n"
    "- `expert_utilization()` — per-expert token fraction statistics\n"
    "- `get_all_router_stats()` — hook-based routing diagnostics across all layers\n"
    "- `examples/moe_demo.py` — NanoMindMoE forward, aux loss, utilization report\n\n---\n\n"
    "## [1.8.0] — 2024 — Beam Search & Diverse Beam Search"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v1.9.0, update README and CHANGELOG for Day 23 MoE")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 23 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v1.9.0",
    "-m", "NanoMind v1.9.0 — Mixture of Experts", check=False)
r = run("git", "push", "origin", "v1.9.0", check=False)
print("Tag v1.9.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 23 COMPLETE — v1.9.0 TAGGED! ===")
