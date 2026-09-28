"""
nanomind/specd/sampler.py — Speculative sampling: accept/reject with target model.

The core algorithm verifies K draft tokens using the target model in ONE pass,
accepting each based on probability ratios. Mathematically identical to
sampling from the target model directly — no accuracy loss!

Acceptance rule:
  For token x at position i:
    If p_target(x) / p_draft(x) >= 1: accept (target agrees or is more confident)
    Else: accept with prob = p_target(x) / p_draft(x) (rejection sampling)

  On rejection at position i:
    Sample correction token from: max(0, p_target - p_draft) (normalised)
    Discard all tokens after i
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SpeculativeResult:
    """Result of one speculative decoding step."""
    accepted_ids:   torch.Tensor    # (B, n_accepted) accepted token IDs
    n_accepted:     int             # mean accepted tokens this step
    acceptance_rate: float          # fraction of draft tokens accepted
    target_calls:   int = 1        # target model forward passes used

    def efficiency_gain(self, k: int) -> float:
        """Tokens generated per target model call."""
        return self.n_accepted / max(self.target_calls, 1)


class SpeculativeSampler:
    """
    Speculative decoding accept/reject sampler.

    Takes draft logits and target logits, applies the rejection sampling
    algorithm to produce tokens with the target model distribution.

    Args:
        temperature: Sampling temperature (1.0 = standard, 0 = greedy).
        top_p:       Nucleus sampling probability cutoff.

    Example::

        sampler    = SpeculativeSampler(temperature=1.0)
        result     = sampler.verify(
            draft_ids    = ...,   # (B, K)
            draft_logits = ...,   # (B, K, V)
            target_logits = ...,  # (B, K+1, V)  — target sees all K+1 positions
        )
        new_ids    = result.accepted_ids
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_p:       float = 1.0,
    ) -> None:
        self.temperature = max(temperature, 1e-8)
        self.top_p       = top_p

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def _apply_top_p(self, probs: torch.Tensor) -> torch.Tensor:
        """Nucleus sampling: zero out low-probability tokens."""
        if self.top_p >= 1.0:
            return probs
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_p.cumsum(dim=-1)
        remove = cumsum - sorted_p > self.top_p
        sorted_p[remove] = 0.0
        # Scatter back
        out = torch.zeros_like(probs)
        out.scatter_(-1, sorted_idx, sorted_p)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-8)

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample a token from probability distribution."""
        probs = self._apply_top_p(probs)
        return torch.multinomial(probs + 1e-10, num_samples=1).squeeze(-1)

    def verify(
        self,
        draft_ids:     torch.Tensor,
        draft_logits:  torch.Tensor,
        target_logits: torch.Tensor,
    ) -> SpeculativeResult:
        """
        Apply speculative acceptance/rejection.

        Args:
            draft_ids:     ``(B, K)`` draft token IDs.
            draft_logits:  ``(B, K, V)`` draft model logits.
            target_logits: ``(B, K+1, V)`` target model logits.
                           Position K+1 is for the token AFTER all K draft tokens.

        Returns:
            :class:`SpeculativeResult` with accepted token IDs.
        """
        B, K   = draft_ids.shape
        V      = target_logits.shape[-1]

        # Convert logits to probabilities
        draft_probs  = F.softmax(self._apply_temperature(draft_logits),  dim=-1)   # (B,K,V)
        target_probs = F.softmax(self._apply_temperature(target_logits), dim=-1)   # (B,K+1,V)

        accepted      = torch.zeros(B, K + 1, dtype=torch.long)
        n_accepted    = torch.zeros(B, dtype=torch.long)
        all_rejected  = torch.zeros(B, dtype=torch.bool)

        for b in range(B):
            if all_rejected[b]:
                continue
            n_acc = 0
            for k in range(K):
                tok       = draft_ids[b, k].item()
                p_draft   = draft_probs[b, k, tok].item()
                p_target  = target_probs[b, k, tok].item()

                # Rejection sampling accept criterion
                if p_draft <= 0:
                    ratio = 1.0
                else:
                    ratio = min(1.0, p_target / (p_draft + 1e-10))

                u = torch.rand(1).item()
                if u <= ratio:
                    # Accept: keep draft token
                    accepted[b, n_acc] = tok
                    n_acc             += 1
                else:
                    # Reject: sample correction from target - draft residual
                    residual = (target_probs[b, k] - draft_probs[b, k]).clamp(min=0)
                    if residual.sum() < 1e-8:
                        residual = target_probs[b, k]
                    residual = residual / (residual.sum() + 1e-8)
                    correction = self._sample(residual)
                    accepted[b, n_acc] = correction
                    n_acc             += 1
                    break   # Stop at rejection

            # Always add at least one token from target if no rejections
            if n_acc == K:
                next_tok = self._sample(target_probs[b, K])
                accepted[b, n_acc] = next_tok
                n_acc             += 1

            n_accepted[b] = n_acc

        # Trim to actual accepted length
        max_acc = n_accepted.max().item()
        accepted_trimmed = accepted[:, :max_acc]

        acc_rate = (n_accepted.float().mean() / K).item()
        return SpeculativeResult(
            accepted_ids    = accepted_trimmed,
            n_accepted      = int(n_accepted.float().mean().item()),
            acceptance_rate = acc_rate,
        )

    def greedy_verify(
        self,
        draft_ids:     torch.Tensor,
        target_logits: torch.Tensor,
    ) -> SpeculativeResult:
        """
        Greedy verification: accept if target's argmax matches draft.

        Faster but not exactly equivalent to target sampling.
        Used for temperature=0 (greedy decoding) scenarios.
        """
        B, K  = draft_ids.shape
        # Target greedy predictions for each position
        target_tokens = target_logits[:, :K].argmax(dim=-1)   # (B, K)

        # Find first mismatch
        matches       = (draft_ids == target_tokens)           # (B, K)
        accepted_list = []
        n_acc_list    = []

        for b in range(B):
            first_reject = K
            for k in range(K):
                if not matches[b, k]:
                    first_reject = k
                    break
            # Accept tokens up to first mismatch + correction
            acc_tokens = draft_ids[b, :first_reject].tolist()
            # Add target's correction at first_reject position
            correction = target_logits[b, first_reject].argmax().item()
            acc_tokens.append(correction)
            accepted_list.append(acc_tokens)
            n_acc_list.append(len(acc_tokens))

        max_len   = max(len(x) for x in accepted_list)
        accepted  = torch.zeros(B, max_len, dtype=torch.long)
        for b, tokens in enumerate(accepted_list):
            accepted[b, :len(tokens)] = torch.tensor(tokens)

        mean_acc  = sum(n_acc_list) / B
        return SpeculativeResult(
            accepted_ids    = accepted,
            n_accepted      = int(mean_acc),
            acceptance_rate = mean_acc / K,
        )
