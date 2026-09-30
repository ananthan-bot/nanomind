"""
nanomind/alignment/dpo.py — Direct Preference Optimization (DPO).

## DPO: The Key Insight (Rafailov et al., 2023)

Classic RLHF objective:
  max_π E[r(x, y)] - β KL[π || π_ref]

This can be solved analytically! The optimal policy is:
  π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)

Rewriting: r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)

Substituting into the Bradley-Terry preference model:
  p(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))

We get the DPO loss (no reward model needed!):

  L_DPO = -E[log σ(β × (log π(y_w|x) - log π_ref(y_w|x))
                  - β × (log π(y_l|x) - log π_ref(y_l|x)))]

This is just a binary cross-entropy on log-ratio differences!

Training:
  - Keep a frozen reference model π_ref (SFT model)
  - Optimise the policy π using gradient descent
  - No RL, no reward model, no sampling loop needed!

Reference:
  Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model"
  https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta:          float = 0.1    # KL divergence weight
    label_smoothing: float = 0.0  # label smoothing for BCE
    reference_free:  bool  = False  # skip ref model (SLiC-HF style)
    loss_type:       str   = "dpo"  # "dpo" | "ipo" | "kto_pair"

    def __post_init__(self):
        assert self.loss_type in ("dpo", "ipo", "kto_pair")


def compute_log_probs(
    model:     nn.Module,
    input_ids: torch.Tensor,
    labels:    torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for the labelled tokens.

    Args:
        model:     LM model returning ``(B, T, V)`` logits.
        input_ids: ``(B, T)`` input token IDs.
        labels:    ``(B, T)`` target token IDs.

    Returns:
        ``(B,)`` sum of log probs over non-padding positions.
    """
    with torch.no_grad() if False else torch.enable_grad():
        out    = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out   # (B, T, V)
    # Shift: predict token at position t+1 from position t
    shift_logits = logits[:, :-1, :]     # (B, T-1, V)
    shift_labels = labels[:, 1:]         # (B, T-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log probs for the actual tokens
    tok_log_p = log_probs.gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)                          # (B, T-1)
    # Sum over non-padding (label != -100)
    mask = (shift_labels != -100).float()
    return (tok_log_p * mask).sum(-1)     # (B,)


def dpo_loss(
    policy_logp_chosen:    torch.Tensor,
    policy_logp_rejected:  torch.Tensor,
    ref_logp_chosen:       torch.Tensor,
    ref_logp_rejected:     torch.Tensor,
    cfg:                   DPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Compute DPO / IPO loss from log probabilities.

    Args:
        policy_logp_chosen:   ``(B,)`` policy log probs for chosen.
        policy_logp_rejected: ``(B,)`` policy log probs for rejected.
        ref_logp_chosen:      ``(B,)`` reference log probs for chosen.
        ref_logp_rejected:    ``(B,)`` reference log probs for rejected.
        cfg:                  :class:`DPOConfig`.

    Returns:
        ``(loss, metrics_dict)``
    """
    β = cfg.beta

    # Log-ratio differences
    if cfg.reference_free:
        logits = β * (policy_logp_chosen - policy_logp_rejected)
    else:
        chosen_ratio   = policy_logp_chosen   - ref_logp_chosen
        rejected_ratio = policy_logp_rejected - ref_logp_rejected
        logits = β * (chosen_ratio - rejected_ratio)

    if cfg.loss_type == "dpo":
        # Standard DPO: binary cross-entropy
        loss = -F.logsigmoid(logits)
        if cfg.label_smoothing > 0:
            loss = (1 - cfg.label_smoothing) * loss -                    cfg.label_smoothing * F.logsigmoid(-logits)

    elif cfg.loss_type == "ipo":
        # IPO: squared hinge loss (avoids over-fitting)
        loss = (logits - 1 / (2 * β)) ** 2

    elif cfg.loss_type == "kto_pair":
        # KTO pair loss: separate desirable/undesirable
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)) +
            F.binary_cross_entropy_with_logits(-logits, torch.ones_like(logits))
        )

    loss = loss.mean()

    # Metrics
    chosen_rewards   = (β * (policy_logp_chosen   - ref_logp_chosen)).detach()
    rejected_rewards = (β * (policy_logp_rejected - ref_logp_rejected)).detach()
    accuracy         = (chosen_rewards > rejected_rewards).float().mean()

    metrics = {
        "loss":             loss.item(),
        "accuracy":         accuracy.item(),
        "reward_chosen":    chosen_rewards.mean().item(),
        "reward_rejected":  rejected_rewards.mean().item(),
        "reward_margin":    (chosen_rewards - rejected_rewards).mean().item(),
        "logits_mean":      logits.mean().item(),
    }
    return loss, metrics


class DPOTrainer:
    """
    DPO training loop.

    Wraps policy + reference model + optimizer for DPO training.

    Args:
        policy_model:    The model to train (initialised from SFT).
        ref_model:       Frozen reference model (SFT model, not updated).
        optimizer:       PyTorch optimizer for policy.
        cfg:             :class:`DPOConfig`.

    Example::

        trainer = DPOTrainer(policy, ref, optimizer, DPOConfig(beta=0.1))
        for chosen_ids, rejected_ids in dataloader:
            metrics = trainer.step(prompt_ids, chosen_ids, rejected_ids)
            print(f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.2%}")
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model:    nn.Module,
        optimizer:    torch.optim.Optimizer,
        cfg:          DPOConfig | None = None,
    ) -> None:
        self.policy = policy_model
        self.ref    = ref_model
        self.opt    = optimizer
        self.cfg    = cfg or DPOConfig()
        # Freeze reference model
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def step(
        self,
        chosen_ids:   torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict:
        """
        One DPO training step.

        Args:
            chosen_ids:   ``(B, T)`` chosen response token IDs.
            rejected_ids: ``(B, T)`` rejected response token IDs.

        Returns:
            Metrics dict.
        """
        self.opt.zero_grad()

        # Policy log probs
        policy_logp_chosen   = compute_log_probs(self.policy, chosen_ids,   chosen_ids)
        policy_logp_rejected = compute_log_probs(self.policy, rejected_ids, rejected_ids)

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_logp_chosen   = compute_log_probs(self.ref, chosen_ids,   chosen_ids)
            ref_logp_rejected = compute_log_probs(self.ref, rejected_ids, rejected_ids)

        loss, metrics = dpo_loss(
            policy_logp_chosen, policy_logp_rejected,
            ref_logp_chosen,    ref_logp_rejected,
            self.cfg,
        )
        loss.backward()
        self.opt.step()
        return metrics
