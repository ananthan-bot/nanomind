"""
nanomind/rlhf/rollout.py — PPO rollout buffer for RLHF.

Stores generated trajectories (tokens + rewards + values + log-probs)
and computes GAE advantages for the PPO update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch

from nanomind.rlhf.value_head import compute_gae


@dataclass
class RolloutBatch:
    """A single (prompt + response) rollout with all PPO signals."""
    input_ids:         torch.Tensor   # (T,)  prompt + generated tokens
    log_probs:         torch.Tensor   # (T_gen,) log-probs of generated tokens
    ref_log_probs:     torch.Tensor   # (T_gen,) reference model log-probs
    values:            torch.Tensor   # (T_gen,) value estimates
    rewards:           torch.Tensor   # (T_gen,) per-token rewards (RM + KL)
    advantages:        torch.Tensor   # (T_gen,) GAE advantages
    returns:           torch.Tensor   # (T_gen,) value targets


class PPORolloutBuffer:
    """
    Buffer that collects and processes PPO rollouts.

    Collects raw trajectories (tokens, log-probs, rewards, values),
    computes GAE advantages, and normalises advantages before the update.

    Args:
        gamma: Discount factor.
        lam:   GAE lambda.
        kl_coef: KL penalty coefficient.
    """

    def __init__(
        self,
        gamma:   float = 1.0,
        lam:     float = 0.95,
        kl_coef: float = 0.1,
    ) -> None:
        self.gamma   = gamma
        self.lam     = lam
        self.kl_coef = kl_coef
        self._rollouts: list[RolloutBatch] = []

    def add(
        self,
        input_ids:     torch.Tensor,
        log_probs:     torch.Tensor,
        ref_log_probs: torch.Tensor,
        values:        torch.Tensor,
        reward_score:  float,
    ) -> None:
        """
        Add a single rollout to the buffer.

        Args:
            input_ids:     Full token sequence (prompt + response).
            log_probs:     Log-probs of generated tokens ``(T_gen,)``.
            ref_log_probs: Reference model log-probs ``(T_gen,)``.
            values:        Value estimates ``(T_gen,)``.
            reward_score:  Scalar reward from the Reward Model (last token).
        """
        T_gen = log_probs.shape[0]

        # Per-token KL penalty
        kl      = log_probs - ref_log_probs
        rewards = -self.kl_coef * kl

        # Add terminal reward to last token
        rewards[-1] = rewards[-1] + reward_score

        advantages, returns = compute_gae(
            rewards, values, self.gamma, self.lam
        )
        self._rollouts.append(RolloutBatch(
            input_ids=input_ids,
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            values=values,
            rewards=rewards,
            advantages=advantages,
            returns=returns,
        ))

    def finalize(self) -> list[RolloutBatch]:
        """
        Normalise advantages across all rollouts and return them.

        Advantage normalisation (zero mean, unit variance) stabilises PPO.
        """
        all_adv = torch.cat([r.advantages for r in self._rollouts])
        mean, std = all_adv.mean(), all_adv.std() + 1e-8
        for r in self._rollouts:
            r.advantages = (r.advantages - mean) / std
        result = self._rollouts.copy()
        self._rollouts.clear()
        return result

    def __len__(self) -> int:
        return len(self._rollouts)
