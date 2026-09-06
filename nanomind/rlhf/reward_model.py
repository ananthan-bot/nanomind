"""
nanomind/rlhf/reward_model.py — Reward Model for RLHF.

The Reward Model (RM) takes a (prompt, completion) token sequence and
outputs a scalar score representing how preferred the completion is.

Architecture:
  RM = SFT backbone (frozen or fine-tuned) + linear scalar head

Training: Bradley-Terry pairwise ranking loss over (chosen, rejected) pairs.
Inference: score any completion; higher = more preferred.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.model.config import ModelConfig
from nanomind.rlhf.config import RewardModelConfig
from nanomind.utils.logger import get_logger

log = get_logger("rlhf.reward_model")


class RewardModel(nn.Module):
    """
    Scalar reward model for RLHF preference learning.

    Wraps a transformer backbone with a linear reward head that maps
    the final hidden state to a scalar reward value.

    Args:
        backbone:   Transformer model (e.g., NanoMind) whose hidden states
                    are used as features.
        model_cfg:  Model configuration (for d_model, vocab_size).
        rm_cfg:     Reward model configuration.

    Example::

        rm  = RewardModel(backbone, model_cfg, RewardModelConfig())
        r_w = rm(chosen_ids)     # scalar reward for preferred completion
        r_l = rm(rejected_ids)   # scalar reward for rejected completion
        loss = preference_loss(r_w, r_l)
    """

    def __init__(
        self,
        backbone:  nn.Module,
        model_cfg: ModelConfig,
        rm_cfg:    RewardModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rm_cfg   = rm_cfg or RewardModelConfig()
        self.drop     = nn.Dropout(self.rm_cfg.dropout)
        self.reward_head = nn.Linear(model_cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        log.info(f"RewardModel: d_model={model_cfg.d_model}, pooling={self.rm_cfg.pooling}")

    def _get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone hidden states for the input.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            Hidden states ``(B, T, d_model)``.
        """
        # Get logits from backbone — we need hidden states
        # Hook into the norm layer output before lm_head
        hooks, hidden = [], [None]

        def _hook(_, __, output):
            hidden[0] = output

        # Register hook on the final norm layer
        handle = None
        for name, module in self.backbone.named_modules():
            if name in ("norm", "ln_f", "final_norm"):
                handle = module.register_forward_hook(_hook)
                break

        self.backbone(input_ids)

        if handle is not None:
            handle.remove()

        if hidden[0] is not None:
            return hidden[0]

        # Fallback: use embedding + positional encoding directly
        tok = self.backbone.tok_emb(input_ids)
        if hasattr(self.backbone, "pos_emb"):
            pos = torch.arange(input_ids.size(1), device=input_ids.device)
            tok = tok + self.backbone.pos_emb(pos)
        return tok

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar reward for each sequence in the batch.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            Scalar rewards ``(B,)`` — one per sequence.
        """
        hidden = self._get_hidden(input_ids)   # (B, T, d_model)

        if self.rm_cfg.pooling == "last":
            features = hidden[:, -1, :]        # (B, d_model)
        else:
            features = hidden.mean(dim=1)      # (B, d_model)

        features = self.drop(features)
        reward   = self.reward_head(features).squeeze(-1)   # (B,)
        return reward
