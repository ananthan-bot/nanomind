"""
nanomind/interpret/logit_lens.py — Logit Lens for mechanistic interpretability.

## Logit Lens (nostalgebraist, 2020)

The Logit Lens projects intermediate hidden states directly through
the language model head to see "what the model is predicting" at
each layer — before the final layer processes the representations.

Algorithm:
  For each layer l, hidden state h_l:
    logits_l = LayerNorm(h_l) @ W_lm
    probs_l  = softmax(logits_l)
    top_token_l = argmax(probs_l)

This reveals:
  - When does the model "commit" to the correct token?
  - Which layers refine vs maintain predictions?
  - Superposition: does the model route through wrong predictions?

Used for: GPT-2, GPT-J, LLaMA mechanistic analysis.

Reference:
  nostalgebraist (2020) "interpreting GPT: the logit lens"
  https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru
  Belrose et al. (2023) "Eliciting Latent Predictions from Transformers"
  https://arxiv.org/abs/2303.08112
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class LogitLensResult:
    """
    Logit Lens result: predictions at each layer.

    Attributes:
        layer_probs:  List of ``(T, V)`` probability tensors per layer.
        layer_tokens: List of ``(T,)`` top-1 token IDs per layer.
        n_layers:     Total number of layers analysed.
        vocab_size:   Vocabulary size.
    """
    layer_probs:  list[torch.Tensor] = field(default_factory=list)
    layer_tokens: list[torch.Tensor] = field(default_factory=list)
    n_layers:     int = 0
    vocab_size:   int = 0

    def get_layer_top_tokens(self, layer: int, k: int = 5) -> list[int]:
        """Return top-K predicted token IDs at a given layer."""
        if layer >= len(self.layer_probs):
            return []
        return torch.topk(self.layer_probs[layer][-1], k).indices.tolist()

    def prediction_change(self) -> list[bool]:
        """
        Return which layers changed the top-1 prediction vs previous layer.
        """
        changes = []
        for i in range(1, len(self.layer_tokens)):
            prev = self.layer_tokens[i - 1]
            curr = self.layer_tokens[i]
            changed = not torch.all(prev == curr).item()
            changes.append(changed)
        return changes

    def rank_of_correct(self, correct_id: int, pos: int = -1) -> list[int]:
        """
        Rank of the correct token at each layer (lower = better).

        Args:
            correct_id: The correct next-token ID.
            pos:        Token position to analyse.

        Returns:
            List of ranks (1-indexed) per layer.
        """
        ranks = []
        for probs in self.layer_probs:
            p      = probs[pos]
            sorted_ids = torch.argsort(p, descending=True)
            rank   = (sorted_ids == correct_id).nonzero(as_tuple=True)
            r      = rank[0].item() + 1 if len(rank[0]) > 0 else len(p)
            ranks.append(r)
        return ranks


class LogitLens:
    """
    Logit Lens: apply LM head to intermediate hidden states.

    Args:
        model:    Language model with ``lm_head`` or ``lm`` output projection.
        ln_final: Final LayerNorm (applied before LM head).

    Example::

        lens    = LogitLens(model)
        result  = lens.analyse(input_ids)
        # result.layer_tokens[0] → predictions after layer 0
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._hidden_states: list[torch.Tensor] = []

    def _get_lm_head(self) -> nn.Linear:
        for attr in ("lm_head", "lm", "output_proj"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        raise AttributeError("Cannot find LM head")

    def _get_final_ln(self) -> nn.LayerNorm | None:
        for attr in ("ln_f", "ln", "norm"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        return None

    @torch.no_grad()
    def analyse(
        self,
        input_ids: torch.Tensor,
        k_layers:  int | None = None,
    ) -> LogitLensResult:
        """
        Run Logit Lens analysis.

        Args:
            input_ids: ``(1, T)`` token ID tensor.
            k_layers:  Analyse only the first k layers (None = all).

        Returns:
            :class:`LogitLensResult`.
        """
        self._hidden_states.clear()

        def hook_fn(m, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                self._hidden_states.append(out.detach().clone())

        hooks = []
        for m in self.model.modules():
            if isinstance(m, nn.LayerNorm):
                hooks.append(m.register_forward_hook(hook_fn))

        self.model(input_ids)
        for h in hooks:
            h.remove()

        lm_head = self._get_lm_head()
        ln      = self._get_final_ln()

        result  = LogitLensResult(vocab_size=lm_head.out_features)
        layers  = self._hidden_states[:k_layers] if k_layers else self._hidden_states

        for h in layers:
            if h.shape[-1] != lm_head.in_features:
                continue
            h_n     = ln(h[0]) if ln is not None else h[0]
            logits  = lm_head(h_n)
            probs   = F.softmax(logits, dim=-1)
            top_tok = logits.argmax(dim=-1)
            result.layer_probs.append(probs.cpu())
            result.layer_tokens.append(top_tok.cpu())

        result.n_layers = len(result.layer_probs)
        return result
