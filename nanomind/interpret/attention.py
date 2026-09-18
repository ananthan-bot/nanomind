"""
nanomind/interpret/attention.py — Attention weight extraction and analysis.

## Why Attention Visualization?

The attention mechanism computes a weighted sum over tokens:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The weight matrix softmax(QK^T / sqrt(d_k)) ∈ R^(T × T) tells us:
  "How much does token i attend to token j?"

Visualizing these weights reveals:
  - Which tokens the model focuses on for each prediction
  - Induction heads: patterns where token n attends to token n-S
  - Duplicate token heads: attend to identical earlier tokens
  - Positional heads: attend to nearby tokens (local context)

Important caveat: attention ≠ importance!
  Jain & Wallace (2019) showed attention is not explanation.
  Wiegreffe & Pinter (2019) showed it can be explanation.
  Gradient × attention (GradCAM-style) is more reliable.

References:
  Bahdanau et al. (2015) "Neural Machine Translation by Jointly..."
  Vig (2019) "A Multiscale Visualization of Attention in NLP"
  Jain & Wallace (2019) "Attention is not Explanation"
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class AttentionMap:
    """
    Captured attention weights for one layer.

    Attributes:
        layer:    Layer index.
        weights:  ``(B, H, T, T)`` attention weight tensor.
        tokens:   Optional list of token strings.
    """
    layer:   int
    weights: torch.Tensor          # (B, H, T, T)
    tokens:  list[str] = field(default_factory=list)

    @property
    def n_heads(self) -> int:
        return self.weights.shape[1]

    @property
    def seq_len(self) -> int:
        return self.weights.shape[2]

    def head(self, h: int) -> torch.Tensor:
        """Return attention matrix for head h: ``(T, T)``."""
        return self.weights[0, h]

    def mean_head(self) -> torch.Tensor:
        """Average over heads: ``(T, T)``."""
        return self.weights[0].mean(0)

    def rollout(self) -> torch.Tensor:
        """
        Attention rollout (Abnar & Zuidema, 2020):
        recursively multiply attention matrices across layers
        to get information flow from input to output tokens.

        Returns:
            ``(T, T)`` rollout matrix.
        """
        A = self.mean_head()
        # Add residual (identity) connection
        eye  = torch.eye(A.shape[0])
        A    = 0.5 * A + 0.5 * eye
        A    = A / A.sum(dim=-1, keepdim=True)
        return A

    def entropy(self) -> torch.Tensor:
        """
        Attention entropy per head (higher = more diffuse attention).

        Returns:
            ``(H,)`` tensor of per-head entropy values.
        """
        w   = self.weights[0].clamp(min=1e-9)   # (H, T, T)
        ent = -(w * w.log()).sum(dim=-1).mean(dim=-1)   # (H,)
        return ent

    def to_dict(self) -> dict:
        return {
            "layer":    self.layer,
            "n_heads":  self.n_heads,
            "seq_len":  self.seq_len,
            "mean":     self.mean_head().tolist(),
            "entropy":  self.entropy().tolist(),
        }


class AttentionExtractor:
    """
    Extract attention weights from a NanoMind model via forward hooks.

    Args:
        model: Language model with ``nn.MultiheadAttention`` layers.

    Example::

        extractor = AttentionExtractor(model)
        maps      = extractor.extract(input_ids)
        # maps[0].weights → (1, n_heads, T, T) for layer 0
    """

    def __init__(self, model: nn.Module) -> None:
        self.model  = model
        self._hooks: list = []
        self._maps:  list[AttentionMap] = []

    def _hook_fn(self, layer_idx: int):
        def fn(module, inp, out):
            # MultiheadAttention returns (output, weights)
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                self._maps.append(AttentionMap(
                    layer   = layer_idx,
                    weights = out[1].detach().cpu(),
                ))
        return fn

    def register_hooks(self) -> None:
        """Register forward hooks on all MultiheadAttention layers."""
        self._hooks.clear()
        idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(self._hook_fn(idx))
                self._hooks.append(h)
                idx += 1

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(
        self,
        input_ids: torch.Tensor,
        tokens:    list[str] = None,
    ) -> list[AttentionMap]:
        """
        Run a forward pass and capture all attention weights.

        Args:
            input_ids: ``(B, T)`` token ID tensor.
            tokens:    Optional token strings for labelling.

        Returns:
            List of :class:`AttentionMap` per layer.
        """
        self._maps.clear()
        self.register_hooks()
        try:
            self.model(input_ids)
        finally:
            self.remove_hooks()
        if tokens:
            for m in self._maps:
                m.tokens = tokens
        return list(self._maps)
