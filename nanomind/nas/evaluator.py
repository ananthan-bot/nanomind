"""
nanomind/nas/evaluator.py — Architecture proxy evaluators.

Evaluating each candidate architecture by training to convergence is
expensive (GPU-hours per config × thousands of configs = months).

Proxy metrics approximate final performance cheaply:
  1. Parameter count:     larger models often perform better (but overfit)
  2. Synaptic flow:        sum of absolute gradient-parameter products at init
                           correlates with trainability (Tanaka et al., 2020)
  3. Training loss proxy:  train for N_PROXY steps, record loss
  4. Zero-cost proxies:   gradient norms, activation diversity at init

References:
  Synaptic flow: Tanaka et al. (2020) https://arxiv.org/abs/2006.05467
  Zero-cost NAS: Mellor et al. (2021) https://arxiv.org/abs/2102.08099
  Training-free: Chen et al. (2021) https://arxiv.org/abs/2108.11014
"""

from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.nas.search_space import ArchConfig


def _build_model(cfg: ArchConfig, vocab_size: int = 32) -> nn.Module:
    """Build a tiny transformer from an ArchConfig (for proxy evaluation)."""

    class Block(nn.Module):
        def __init__(self, D, H, ratio, drop):
            super().__init__()
            self.ln1 = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, H, dropout=drop, batch_first=True)
            self.ln2  = nn.LayerNorm(D)
            d_ff      = D * ratio
            self.ffn  = nn.Sequential(
                nn.Linear(D, d_ff), nn.GELU(), nn.Dropout(drop), nn.Linear(d_ff, D)
            )
        def forward(self, x):
            h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
            x = x + h
            return x + self.ffn(self.ln2(x))

    class TinyTF(nn.Module):
        def __init__(self):
            super().__init__()
            D = cfg.d_model
            T = min(cfg.max_seq, 32)
            self.T   = T
            self.tok = nn.Embedding(vocab_size, D)
            self.pos = nn.Embedding(T, D)
            self.blocks = nn.ModuleList([
                Block(D, cfg.n_heads, cfg.ffn_ratio, cfg.dropout)
                for _ in range(cfg.n_layers)
            ])
            self.ln  = nn.LayerNorm(D)
            self.lm  = nn.Linear(D, vocab_size, bias=False)
        def forward(self, x, t=None):
            B, S = x.shape
            h = self.tok(x) + self.pos(torch.arange(S))
            for block in self.blocks:
                h = block(h)
            h = self.ln(h)
            logits = self.lm(h)
            loss = F.cross_entropy(logits.view(-1, vocab_size), t.view(-1)) if t is not None else None
            return logits, loss
    return TinyTF()


class ProxyEvaluator:
    """
    Multi-proxy architecture evaluator.

    Scores architectures using a weighted combination of proxies:
      - ``param_score``:   normalised parameter efficiency
      - ``synflow_score``:  synaptic flow (trainability signal)
      - ``loss_score``:    training loss after N_PROXY steps

    Args:
        vocab_size:   Vocabulary size for proxy model.
        proxy_steps:  Training steps for loss proxy (0 = skip).
        proxy_lr:     Learning rate for loss proxy.

    Example::

        ev  = ProxyEvaluator(vocab_size=32, proxy_steps=5)
        score = ev.evaluate(ArchConfig(d_model=64, n_layers=2, n_heads=4))
    """

    def __init__(
        self,
        vocab_size:   int   = 32,
        proxy_steps:  int   = 5,
        proxy_lr:     float = 1e-2,
    ) -> None:
        self.vocab_size  = vocab_size
        self.proxy_steps = proxy_steps
        self.proxy_lr    = proxy_lr

    def param_score(self, cfg: ArchConfig) -> float:
        """Lower param count → higher score (efficiency-focused)."""
        n = cfg.n_params_estimate
        return 1.0 / math.log(max(n, 2))

    def synflow_score(self, cfg: ArchConfig) -> float:
        """
        Synaptic flow score: sum of |grad × param| at initialisation.

        Higher → more trainable architecture.
        """
        model = _build_model(cfg, self.vocab_size)
        model.train()
        # Forward with all-ones input
        x = torch.ones(1, min(cfg.max_seq, 8), dtype=torch.long)
        logits, _ = model(x)
        # Synaptic flow: use sum of logits as pseudo-loss
        loss = logits.sum()
        loss.backward()
        score = sum(
            (p.grad * p.data).abs().sum().item()
            for p in model.parameters() if p.grad is not None
        )
        return score

    def loss_proxy(self, cfg: ArchConfig, seed: int = 42) -> float:
        """
        Train for proxy_steps steps and return final loss.

        Lower loss → better architecture.
        """
        if self.proxy_steps == 0:
            return 0.0
        torch.manual_seed(seed)
        model = _build_model(cfg, self.vocab_size)
        model.train()
        opt   = torch.optim.Adam(model.parameters(), lr=self.proxy_lr)
        T     = min(cfg.max_seq, 8)
        for _ in range(self.proxy_steps):
            x = torch.randint(0, self.vocab_size, (2, T))
            y = torch.randint(0, self.vocab_size, (2, T))
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            opt.step()
        return loss.item()

    def evaluate(self, cfg: ArchConfig) -> dict:
        """
        Evaluate an architecture and return a composite score.

        Returns:
            Dict with ``param_score``, ``synflow``, ``loss_proxy``,
            ``composite`` (higher is better), ``elapsed_s``.
        """
        t0           = time.perf_counter()
        p_score      = self.param_score(cfg)
        sf_score     = self.synflow_score(cfg)
        loss         = self.loss_proxy(cfg)
        elapsed      = time.perf_counter() - t0

        # Composite: high synflow, low loss, low param count
        # Normalise synflow (log scale) and loss (negate)
        composite    = (
            0.3 * p_score
            + 0.4 * math.log(max(sf_score, 1e-9))
            - 0.3 * loss
        )
        return {
            "param_score":  round(p_score, 6),
            "synflow":      round(sf_score, 4),
            "loss_proxy":   round(loss, 4),
            "composite":    round(composite, 4),
            "elapsed_s":    round(elapsed, 3),
        }
