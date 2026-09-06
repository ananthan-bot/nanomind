"""
day28_commits.py — 20 atomic commits for Day 28: RLHF (Reward Model + PPO).
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

print("\n=== DAY 28: RLHF — Reward Model + PPO — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — rlhf package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/__init__.py",
      '"""NanoMind RLHF sub-package — Reward Model and PPO fine-tuning."""\n')
commit("feat: add nanomind/rlhf/ package skeleton for RLHF (Reward Model + PPO)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — RLHFConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/config.py", '''\
"""
nanomind/rlhf/config.py — RLHF configuration (Reward Model + PPO).

## RLHF Pipeline (Ouyang et al. 2022 — InstructGPT)

Step 1 — Supervised Fine-Tuning (SFT):
  Fine-tune the base LM on high-quality human demonstrations.
  (This uses the standard NanoMind Trainer — already implemented)

Step 2 — Reward Model (RM) Training:
  Given pairs of completions (chosen, rejected) for the same prompt,
  train a scalar-output model to score completions by preference.
  Loss: Bradley-Terry pairwise ranking loss
    L = -log σ(r_θ(x, y_w) - r_θ(x, y_l))
  where y_w = preferred (won), y_l = rejected (lost).

Step 3 — PPO Fine-Tuning:
  Use the Reward Model as an environment reward signal.
  Train the SFT model (policy) with PPO to maximise expected reward
  while staying close to the SFT reference (KL penalty):
    objective = E[r_θ(x,y)] - β · KL(π_θ || π_ref)
  PPO clip: prevents large policy updates that destabilise training.

References:
  InstructGPT: Ouyang et al. (2022) https://arxiv.org/abs/2203.02155
  PPO:         Schulman et al. (2017) https://arxiv.org/abs/1707.06347
  RL4LMs:      Ramamurthy et al. (2022) https://arxiv.org/abs/2210.01241
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RewardModelConfig:
    """
    Configuration for the Reward Model.

    Attributes:
        dropout:      Dropout on reward head.
        pooling:      How to pool sequence → scalar
                      (``"last"`` = last token, ``"mean"`` = average).
    """
    dropout: float = 0.1
    pooling: str   = "last"

    def __post_init__(self) -> None:
        assert self.pooling in ("last", "mean")
        assert 0.0 <= self.dropout <= 1.0


@dataclass
class PPOConfig:
    """
    Configuration for PPO fine-tuning.

    Attributes:
        clip_ratio:     PPO clip epsilon (0.2 is standard).
        value_coef:     Weight of value function loss.
        entropy_coef:   Weight of entropy bonus (exploration).
        kl_coef:        KL penalty coefficient β (distance from reference).
        kl_target:      Target KL divergence (adaptive controller).
        gamma:          Discount factor.
        lam:            GAE lambda for advantage estimation.
        ppo_epochs:     Number of PPO update epochs per rollout.
        rollout_len:    Token length of generated rollouts.
        mini_batch_size: Mini-batch size for PPO updates.
    """
    clip_ratio:      float = 0.2
    value_coef:      float = 0.1
    entropy_coef:    float = 0.01
    kl_coef:         float = 0.1
    kl_target:       float = 6.0
    gamma:           float = 1.0
    lam:             float = 0.95
    ppo_epochs:      int   = 4
    rollout_len:     int   = 64
    mini_batch_size: int   = 4

    def __post_init__(self) -> None:
        assert 0.0 < self.clip_ratio < 1.0
        assert 0.0 <= self.kl_coef
        assert 0.0 < self.gamma <= 1.0
        assert 0.0 <= self.lam  <= 1.0
        assert self.ppo_epochs >= 1
''')
commit("feat: add RewardModelConfig and PPOConfig — RLHF hyperparameters with full docstring")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — RewardModel
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/reward_model.py", '''\
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
''')
commit("feat: add RewardModel — transformer backbone + scalar head, last/mean pooling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — preference_loss (Bradley-Terry)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/preference_loss.py", '''\
"""
nanomind/rlhf/preference_loss.py — Pairwise preference losses for reward model training.

## Bradley-Terry Model
The Bradley-Terry model (1952) gives the probability that item A is preferred over B:
  P(A ≻ B) = σ(r(A) - r(B)) = exp(r(A)) / (exp(r(A)) + exp(r(B)))

Maximising the log-likelihood gives the pairwise ranking loss:
  L = -log P(y_w ≻ y_l) = -log σ(r_θ(y_w) - r_θ(y_l))

This is equivalent to binary cross-entropy on the score difference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def preference_loss(
    reward_chosen:   torch.Tensor,
    reward_rejected: torch.Tensor,
    margin:          float = 0.0,
) -> torch.Tensor:
    """
    Bradley-Terry pairwise preference loss.

    Trains the reward model to assign higher scores to chosen completions.

    Args:
        reward_chosen:   Scalar rewards for preferred completions ``(B,)``.
        reward_rejected: Scalar rewards for rejected completions ``(B,)``.
        margin:          Optional margin: require r_w - r_l > margin.

    Returns:
        Scalar loss (mean over batch).

    Example::

        r_w  = reward_model(chosen_ids)
        r_l  = reward_model(rejected_ids)
        loss = preference_loss(r_w, r_l)
        loss.backward()
    """
    diff = reward_chosen - reward_rejected - margin
    return -F.logsigmoid(diff).mean()


def preference_accuracy(
    reward_chosen:   torch.Tensor,
    reward_rejected: torch.Tensor,
) -> float:
    """
    Compute the fraction of pairs where chosen reward > rejected reward.

    Args:
        reward_chosen:   ``(B,)`` rewards for preferred completions.
        reward_rejected: ``(B,)`` rewards for rejected completions.

    Returns:
        Accuracy in [0, 1] — 1.0 means perfect ranking.
    """
    return (reward_chosen > reward_rejected).float().mean().item()


def reward_stats(
    reward_chosen:   torch.Tensor,
    reward_rejected: torch.Tensor,
) -> dict:
    """
    Compute summary statistics for reward model monitoring.

    Args:
        reward_chosen:   ``(B,)`` rewards for preferred completions.
        reward_rejected: ``(B,)`` rewards for rejected completions.

    Returns:
        Dict with ``mean_chosen``, ``mean_rejected``, ``mean_margin``, ``accuracy``.
    """
    return {
        "mean_chosen":   reward_chosen.mean().item(),
        "mean_rejected": reward_rejected.mean().item(),
        "mean_margin":   (reward_chosen - reward_rejected).mean().item(),
        "accuracy":      preference_accuracy(reward_chosen, reward_rejected),
    }
''')
commit("feat: add preference_loss() (Bradley-Terry), preference_accuracy(), reward_stats()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — PreferenceDataset
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/preference_dataset.py", '''\
"""
nanomind/rlhf/preference_dataset.py — Dataset of (chosen, rejected) completion pairs.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from nanomind.tokenizer.base import BaseTokenizer


class PreferenceDataset(Dataset):
    """
    Dataset of (prompt, chosen, rejected) preference pairs for reward model training.

    Each item yields a pair of token sequences:
      - ``chosen``:   the preferred completion (higher human rating)
      - ``rejected``: the rejected completion (lower human rating)

    Args:
        pairs:      List of ``(prompt, chosen_text, rejected_text)`` tuples.
        tokenizer:  Tokenizer for encoding.
        max_length: Maximum sequence length (prompt + completion).

    Example::

        pairs = [
            ("Q: What is 2+2?", "A: 4", "A: 5"),
            ("Q: Sky color?",   "A: Blue", "A: Green"),
        ]
        ds = PreferenceDataset(pairs, tokenizer, max_length=64)
        chosen, rejected = ds[0]
    """

    def __init__(
        self,
        pairs:      list[tuple[str, str, str]],
        tokenizer:  BaseTokenizer,
        max_length: int = 128,
    ) -> None:
        self.chosen_ids:   list[torch.Tensor] = []
        self.rejected_ids: list[torch.Tensor] = []

        for prompt, chosen, rejected in pairs:
            enc_c = tokenizer.encode(prompt + chosen)[:max_length]
            enc_r = tokenizer.encode(prompt + rejected)[:max_length]
            self.chosen_ids.append(torch.tensor(enc_c, dtype=torch.long))
            self.rejected_ids.append(torch.tensor(enc_r, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chosen_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.chosen_ids[idx], self.rejected_ids[idx]

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad and stack chosen/rejected tensors into batches."""
        chosen   = [c for c, _ in batch]
        rejected = [r for _, r in batch]
        chosen_padded   = torch.nn.utils.rnn.pad_sequence(chosen,   batch_first=True)
        rejected_padded = torch.nn.utils.rnn.pad_sequence(rejected, batch_first=True)
        return chosen_padded, rejected_padded
''')
commit("feat: add PreferenceDataset — (prompt, chosen, rejected) pairs for reward model training")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — ValueHead
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/value_head.py", '''\
"""
nanomind/rlhf/value_head.py — Value function head for PPO actor-critic.

In PPO, the policy (actor) and value function (critic) often share a backbone.
The ValueHead adds a linear layer on top of the backbone hidden states to
predict V(s) — the expected cumulative reward from state s.

V(s) is used to compute advantages:
  A(s, a) = Q(s, a) - V(s)   (advantage of taking action a in state s)

Generalised Advantage Estimation (GAE, Schulman 2016):
  δ_t     = r_t + γ · V(s_{t+1}) - V(s_t)
  A_t     = Σ_{l≥0} (γλ)^l · δ_{t+l}

Advantages normalised to zero mean and unit variance before the PPO update.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Scalar value function head added on top of a transformer backbone.

    Maps hidden states ``(B, T, d_model)`` → value estimates ``(B, T)``.

    Args:
        d_model: Backbone hidden dimension.
        dropout: Dropout before the linear layer.

    Example::

        value_head = ValueHead(d_model=256)
        values     = value_head(hidden_states)   # (B, T)
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token value estimates.

        Args:
            hidden: ``(B, T, d_model)`` hidden states from backbone.

        Returns:
            Value estimates ``(B, T)``.
        """
        return self.proj(self.drop(hidden)).squeeze(-1)


def compute_gae(
    rewards:  torch.Tensor,
    values:   torch.Tensor,
    gamma:    float = 1.0,
    lam:      float = 0.95,
    last_val: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalised Advantage Estimation (GAE).

    Args:
        rewards: ``(T,)`` per-step rewards.
        values:  ``(T,)`` value estimates per step.
        gamma:   Discount factor.
        lam:     GAE lambda.
        last_val: Bootstrap value for the step after the rollout.

    Returns:
        Tuple of ``(advantages, returns)`` each ``(T,)``
        where returns = advantages + values (used as value targets).
    """
    T           = rewards.shape[0]
    advantages  = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae         = 0.0

    for t in reversed(range(T)):
        next_val = last_val if t == T - 1 else values[t + 1].item()
        delta    = rewards[t].item() + gamma * next_val - values[t].item()
        gae      = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
''')
commit("feat: add ValueHead — per-token value estimates + compute_gae() for PPO advantages")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — KL divergence penalty
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/kl_penalty.py", '''\
"""
nanomind/rlhf/kl_penalty.py — KL divergence penalty for RLHF PPO.

A critical component of RLHF is keeping the policy model close to the
SFT reference model to prevent reward hacking and maintain language quality.

The KL penalty adds a per-token penalty to the reward:
  r_total(t) = r_RM(t) - β · KL(π_θ(t) || π_ref(t))

where β is the KL coefficient and:
  KL(π_θ || π_ref) = Σ_v π_θ(v|x) · log(π_θ(v|x) / π_ref(v|x))

Approximation used in practice (for efficiency):
  KL ≈ log π_θ(a_t|s_t) - log π_ref(a_t|s_t)
  (sample-based estimate using the generated action a_t)

Adaptive KL controller (Ziegler et al. 2019):
  If KL > target:   increase β  (tighten constraint)
  If KL < target:   decrease β  (relax constraint)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_kl_divergence(
    logits_policy:    torch.Tensor,
    logits_reference: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token KL divergence: KL(policy || reference).

    Args:
        logits_policy:    Policy logits ``(B, T, V)``.
        logits_reference: Reference model logits ``(B, T, V)``.

    Returns:
        Per-token KL ``(B, T)`` — non-negative.
    """
    log_p   = F.log_softmax(logits_policy,    dim=-1)
    log_q   = F.log_softmax(logits_reference, dim=-1)
    p       = log_p.exp()
    # KL(p || q) = Σ p * (log p - log q)
    kl      = (p * (log_p - log_q)).sum(dim=-1)
    return kl.clamp(min=0.0)   # numerical safety


def approx_token_kl(
    log_probs_policy:    torch.Tensor,
    log_probs_reference: torch.Tensor,
) -> torch.Tensor:
    """
    Sample-based approximate KL: log π_θ(a_t) - log π_ref(a_t).

    This is the cheap approximation used in TRL / InstructGPT:
    instead of summing over the full vocabulary, just use the
    log-prob of the actually sampled token.

    Args:
        log_probs_policy:    ``(B, T)`` log-probs of sampled tokens under policy.
        log_probs_reference: ``(B, T)`` log-probs of sampled tokens under reference.

    Returns:
        Approximate KL ``(B, T)``.
    """
    return log_probs_policy - log_probs_reference


class AdaptiveKLController:
    """
    Adaptive KL coefficient controller (Ziegler et al. 2019).

    Adjusts the KL penalty coefficient β based on the observed KL divergence
    relative to a target:
      β_new = β * (1 + 0.2 * (KL - target) / target)

    Args:
        init_kl_coef: Initial β value.
        target_kl:    Target KL divergence.
        horizon:      Adaptation horizon (number of steps).
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl:    float = 6.0,
        horizon:      int   = 10_000,
    ) -> None:
        self.kl_coef  = init_kl_coef
        self.target   = target_kl
        self.horizon  = horizon

    def update(self, current_kl: float) -> float:
        """
        Update β based on observed KL and return the new value.

        Args:
            current_kl: Observed mean KL divergence in the last batch.

        Returns:
            Updated β.
        """
        proportional_error = (current_kl - self.target) / self.target
        multiplier         = 1 + 0.2 * proportional_error
        self.kl_coef       = self.kl_coef * multiplier
        self.kl_coef       = max(0.01, min(self.kl_coef, 10.0))   # clamp
        return self.kl_coef
''')
commit("feat: add token_kl_divergence(), approx_token_kl(), AdaptiveKLController for PPO penalty")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — PPORolloutBuffer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/rollout.py", '''\
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
''')
commit("feat: add PPORolloutBuffer — collect rollouts, KL+RM rewards, GAE advantages, normalise")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — PPOLoss
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/ppo_loss.py", '''\
"""
nanomind/rlhf/ppo_loss.py — PPO clipped objective and value function loss.

PPO-Clip (Schulman et al. 2017) objective per token:
  ratio = π_θ(a_t|s_t) / π_old(a_t|s_t)  = exp(log_π - log_π_old)
  L_clip = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)

The clipping prevents too-large policy updates:
  - If ratio > 1+ε: policy moved too far → cap the gradient
  - If ratio < 1-ε: policy moved back too much → cap the gradient

Value function loss (Huber loss for stability):
  L_V = 0.5 · (V_θ(s_t) - returns_t)²
  (sometimes also clipped, but we use simple MSE here)

Entropy bonus (optional, encourages exploration):
  L_H = H(π_θ(·|s_t)) = -Σ π_θ(v) log π_θ(v)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ppo_policy_loss(
    log_probs:     torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    clip_ratio:    float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    PPO-Clip policy loss.

    Args:
        log_probs:     Current policy log-probs ``(N,)``.
        old_log_probs: Old policy log-probs ``(N,)`` (from rollout).
        advantages:    GAE advantages ``(N,)`` (pre-normalised).
        clip_ratio:    PPO clip epsilon ε.

    Returns:
        Tuple of ``(loss, info)`` where ``info`` contains clip fraction and ratio stats.
    """
    ratio      = torch.exp(log_probs - old_log_probs)
    surr1      = ratio * advantages
    surr2      = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    loss       = -torch.min(surr1, surr2).mean()
    clipped    = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()

    return loss, {
        "policy_loss":    loss.item(),
        "clip_fraction":  clipped,
        "ratio_mean":     ratio.mean().item(),
        "ratio_max":      ratio.max().item(),
    }


def ppo_value_loss(
    values:  torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Value function MSE loss.

    Args:
        values:  Current value estimates ``(N,)``.
        returns: GAE returns (value targets) ``(N,)``.

    Returns:
        Scalar value loss.
    """
    return F.mse_loss(values, returns)


def ppo_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of the policy distribution (encourages exploration).

    Args:
        logits: Policy logits ``(N, vocab_size)`` or ``(vocab_size,)``.

    Returns:
        Scalar entropy (positive — we maximise entropy).
    """
    probs  = F.softmax(logits, dim=-1)
    log_p  = F.log_softmax(logits, dim=-1)
    return -(probs * log_p).sum(dim=-1).mean()


def ppo_total_loss(
    log_probs:     torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    values:        torch.Tensor,
    returns:       torch.Tensor,
    logits:        torch.Tensor | None = None,
    clip_ratio:    float = 0.2,
    value_coef:    float = 0.1,
    entropy_coef:  float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    Combined PPO loss: policy + value + entropy.

    Returns:
        Tuple of ``(total_loss, info_dict)``.
    """
    policy_loss, policy_info = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_ratio)
    val_loss                 = ppo_value_loss(values, returns)
    entropy                  = ppo_entropy_bonus(logits) if logits is not None else torch.tensor(0.0)

    total = policy_loss + value_coef * val_loss - entropy_coef * entropy
    return total, {
        **policy_info,
        "value_loss":   val_loss.item(),
        "entropy":      entropy.item(),
        "total_loss":   total.item(),
    }
''')
commit("feat: add ppo_policy_loss(), ppo_value_loss(), ppo_entropy_bonus(), ppo_total_loss()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — RewardModelTrainer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/rm_trainer.py", '''\
"""
nanomind/rlhf/rm_trainer.py — Reward Model training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.rlhf.reward_model import RewardModel
from nanomind.rlhf.preference_loss import preference_loss, reward_stats
from nanomind.utils.logger import get_logger

log = get_logger("rlhf.rm_trainer")


class RewardModelTrainer:
    """
    Training loop for the Reward Model on preference data.

    Args:
        model:     RewardModel to train.
        optimizer: PyTorch optimizer.
        device:    Training device.

    Example::

        trainer = RewardModelTrainer(rm, optimizer, device)
        metrics = trainer.train_epoch(preference_loader)
    """

    def __init__(
        self,
        model:     RewardModel,
        optimizer: torch.optim.Optimizer,
        device:    torch.device | str = "cpu",
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.device    = torch.device(device)

    def train_step(
        self,
        chosen:   torch.Tensor,
        rejected: torch.Tensor,
    ) -> dict:
        """
        One gradient step on a (chosen, rejected) preference batch.

        Args:
            chosen:   ``(B, T)`` chosen completion token IDs.
            rejected: ``(B, T)`` rejected completion token IDs.

        Returns:
            Dict with ``loss`` and ``accuracy``.
        """
        chosen, rejected = chosen.to(self.device), rejected.to(self.device)

        r_w  = self.model(chosen)
        r_l  = self.model(rejected)
        loss = preference_loss(r_w, r_l)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {**reward_stats(r_w.detach(), r_l.detach()), "loss": loss.item()}

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one full training epoch over the preference dataset."""
        self.model.train()
        total_loss, total_acc, steps = 0.0, 0.0, 0
        for chosen, rejected in loader:
            m = self.train_step(chosen, rejected)
            total_loss += m["loss"]
            total_acc  += m["accuracy"]
            steps      += 1
        return {
            "loss":     total_loss / max(steps, 1),
            "accuracy": total_acc  / max(steps, 1),
            "steps":    steps,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on a preference dataset without gradient updates."""
        self.model.eval()
        total_loss, total_acc, steps = 0.0, 0.0, 0
        for chosen, rejected in loader:
            chosen, rejected = chosen.to(self.device), rejected.to(self.device)
            r_w  = self.model(chosen)
            r_l  = self.model(rejected)
            loss = preference_loss(r_w, r_l)
            total_loss += loss.item()
            total_acc  += (r_w > r_l).float().mean().item()
            steps      += 1
        return {
            "val_loss":     total_loss / max(steps, 1),
            "val_accuracy": total_acc  / max(steps, 1),
        }
''')
commit("feat: add RewardModelTrainer — train_step(), train_epoch(), evaluate() on preference data")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — update rlhf __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rlhf/__init__.py", '''\
"""NanoMind RLHF sub-package — Reward Model + PPO Fine-Tuning.

Implements the InstructGPT 3-step RLHF pipeline:
  Step 1: SFT      — supervised fine-tuning (use NanoMind Trainer)
  Step 2: RM       — reward model training on preference pairs
  Step 3: PPO      — policy optimization with RM reward + KL penalty

Primary exports:
    - :class:`RewardModel`           — backbone + scalar head for preference scoring
    - :class:`RewardModelTrainer`    — train_step() + train_epoch() on preference data
    - :class:`PreferenceDataset`     — (prompt, chosen, rejected) dataset
    - :class:`RewardModelConfig`     — pooling, dropout config
    - :class:`PPOConfig`             — clip_ratio, kl_coef, GAE lambda, ppo_epochs
    - :class:`ValueHead`             — per-token value estimates for actor-critic
    - :class:`PPORolloutBuffer`      — collect rollouts, compute GAE advantages
    - :class:`AdaptiveKLController`  — adaptive β for KL penalty
    - :func:`preference_loss`        — Bradley-Terry pairwise ranking loss
    - :func:`preference_accuracy`    — fraction of pairs ranked correctly
    - :func:`reward_stats`           — mean chosen/rejected/margin/accuracy
    - :func:`token_kl_divergence`    — per-token KL(policy || reference)
    - :func:`approx_token_kl`        — sample-based approximate KL
    - :func:`compute_gae`            — Generalised Advantage Estimation
    - :func:`ppo_total_loss`         — policy + value + entropy PPO loss
"""

from nanomind.rlhf.config import RewardModelConfig, PPOConfig
from nanomind.rlhf.reward_model import RewardModel
from nanomind.rlhf.preference_loss import preference_loss, preference_accuracy, reward_stats
from nanomind.rlhf.preference_dataset import PreferenceDataset
from nanomind.rlhf.value_head import ValueHead, compute_gae
from nanomind.rlhf.kl_penalty import (
    token_kl_divergence, approx_token_kl, AdaptiveKLController
)
from nanomind.rlhf.rollout import PPORolloutBuffer, RolloutBatch
from nanomind.rlhf.ppo_loss import (
    ppo_policy_loss, ppo_value_loss, ppo_entropy_bonus, ppo_total_loss
)
from nanomind.rlhf.rm_trainer import RewardModelTrainer

__all__ = [
    "RewardModelConfig", "PPOConfig",
    "RewardModel", "RewardModelTrainer",
    "PreferenceDataset",
    "ValueHead", "compute_gae",
    "token_kl_divergence", "approx_token_kl", "AdaptiveKLController",
    "PPORolloutBuffer", "RolloutBatch",
    "preference_loss", "preference_accuracy", "reward_stats",
    "ppo_policy_loss", "ppo_value_loss", "ppo_entropy_bonus", "ppo_total_loss",
]
''')
commit("refactor: export all RLHF components from nanomind/rlhf/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — example: rlhf_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/rlhf_demo.py", '''\
"""
examples/rlhf_demo.py — RLHF demo: Reward Model training + PPO signals.

Demonstrates:
  1. Building a RewardModel from a NanoMind backbone
  2. Training on synthetic preference pairs
  3. Computing PPO signals (KL penalty + advantages)

Usage:
    python examples/rlhf_demo.py
"""

import torch
from torch.utils.data import DataLoader

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.rlhf import (
    RewardModelConfig, PPOConfig,
    RewardModel, RewardModelTrainer,
    PreferenceDataset, preference_accuracy, reward_stats,
    ValueHead, compute_gae,
    token_kl_divergence, AdaptiveKLController,
    ppo_total_loss,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog. " * 20
tokenizer = CharTokenizer().build(CORPUS)
V         = tokenizer.vocab_size

model_cfg = ModelConfig(vocab_size=V, block_size=32, d_model=64,
                        n_layers=2, n_heads=4, dropout=0.0)
backbone  = NanoMind(model_cfg)

# ── Step 2: Reward Model ──────────────────────────────────────────────────────
rm_cfg = RewardModelConfig(pooling="last", dropout=0.1)
rm     = RewardModel(backbone, model_cfg, rm_cfg)

# Synthetic preference pairs
pairs = [
    ("the quick", " brown fox", " slow turtle"),
    ("jumps over", " the lazy", " a tree"),
] * 8

ds      = PreferenceDataset(pairs, tokenizer, max_length=32)
loader  = DataLoader(ds, batch_size=4, collate_fn=PreferenceDataset.collate_fn)
opt_rm  = torch.optim.Adam(rm.parameters(), lr=1e-3)
trainer = RewardModelTrainer(rm, opt_rm, device="cpu")

print("Training Reward Model...")
for epoch in range(3):
    m = trainer.train_epoch(loader)
    print(f"  Epoch {epoch+1}: loss={m['loss']:.4f}, acc={m['accuracy']:.2%}")

# ── Step 3: PPO Signals ───────────────────────────────────────────────────────
value_head = ValueHead(d_model=64)
kl_ctrl    = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0)

# Simulate a rollout
torch.manual_seed(42)
T_gen = 10
old_log_probs = torch.randn(T_gen).log_softmax(dim=0)
cur_log_probs = old_log_probs + 0.1 * torch.randn(T_gen)
advantages    = torch.randn(T_gen)
values        = torch.randn(T_gen)
returns       = advantages + values
logits        = torch.randn(T_gen, V)

total_loss, info = ppo_total_loss(
    cur_log_probs, old_log_probs, advantages,
    values, returns, logits,
    clip_ratio=0.2, value_coef=0.1, entropy_coef=0.01
)
print(f"\nPPO loss    : {info['total_loss']:.4f}")
print(f"Clip fraction: {info['clip_fraction']:.2%}")
print(f"Entropy      : {info['entropy']:.4f}")

new_kl_coef = kl_ctrl.update(current_kl=7.0)
print(f"\nAdaptive KL coef: {new_kl_coef:.4f} (target KL=6.0, observed KL=7.0)")
''')
commit("feat: add examples/rlhf_demo.py — RM training, preference loss, PPO signals demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: RewardModelConfig + RewardModel
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_rlhf.py", '''\
"""
tests/test_rlhf.py — Tests for RLHF components.
"""

import pytest
import torch
import torch.nn as nn

from nanomind import NanoMind, ModelConfig
from nanomind.rlhf import (
    RewardModelConfig, PPOConfig,
    RewardModel, PreferenceDataset,
    preference_loss, preference_accuracy, reward_stats,
    ValueHead, compute_gae,
    token_kl_divergence, approx_token_kl, AdaptiveKLController,
    PPORolloutBuffer, ppo_total_loss,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = "abcdefghij " * 5
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg), cfg

def tiny_rm():
    m, cfg = tiny_model()
    return RewardModel(m, cfg, RewardModelConfig()), cfg


# ── RewardModelConfig ─────────────────────────────────────────────────────────

class TestRewardModelConfig:
    def test_defaults(self):
        cfg = RewardModelConfig()
        assert cfg.pooling == "last"

    def test_invalid_pooling(self):
        with pytest.raises(AssertionError):
            RewardModelConfig(pooling="max")

    def test_invalid_dropout(self):
        with pytest.raises(AssertionError):
            RewardModelConfig(dropout=1.5)


# ── RewardModel ───────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_output_shape(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (B, T))
        r     = rm(ids)
        assert r.shape == (B,)

    def test_output_scalar_per_sequence(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (1, T))
        r     = rm(ids)
        assert r.ndim == 1

    def test_mean_pooling(self):
        m, cfg = tiny_model()
        rm     = RewardModel(m, cfg, RewardModelConfig(pooling="mean"))
        ids    = torch.randint(0, VOCAB, (B, T))
        r      = rm(ids)
        assert r.shape == (B,)

    def test_output_finite(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (B, T))
        assert rm(ids).isfinite().all()
''')
commit("test: add RewardModelConfig defaults, invalid pooling, and RewardModel shape/finite tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: preference_loss + dataset
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rlhf.py")
src += '''

# ── preference_loss ───────────────────────────────────────────────────────────

class TestPreferenceLoss:
    def test_loss_positive(self):
        r_w  = torch.tensor([1.0, 2.0])
        r_l  = torch.tensor([0.0, 1.5])
        loss = preference_loss(r_w, r_l)
        assert loss.item() > 0.0

    def test_perfect_ranking_low_loss(self):
        r_w  = torch.tensor([5.0] * 8)
        r_l  = torch.tensor([0.0] * 8)
        loss = preference_loss(r_w, r_l)
        assert loss.item() < 0.01

    def test_accuracy_all_correct(self):
        r_w = torch.tensor([1.0, 2.0, 3.0])
        r_l = torch.tensor([0.0, 1.0, 2.0])
        assert preference_accuracy(r_w, r_l) == 1.0

    def test_accuracy_all_wrong(self):
        r_w = torch.tensor([0.0, 0.0])
        r_l = torch.tensor([1.0, 1.0])
        assert preference_accuracy(r_w, r_l) == 0.0

    def test_reward_stats_keys(self):
        r_w = torch.randn(4)
        r_l = torch.randn(4)
        s   = reward_stats(r_w, r_l)
        assert all(k in s for k in ("mean_chosen","mean_rejected","mean_margin","accuracy"))


class TestPreferenceDataset:
    def test_len(self):
        pairs = [("a", "b", "c")] * 5
        ds    = PreferenceDataset(pairs, TOK, max_length=T)
        assert len(ds) == 5

    def test_item_shapes(self):
        pairs = [("ab", "cd", "ef")]
        ds    = PreferenceDataset(pairs, TOK, max_length=T)
        c, r  = ds[0]
        assert c.dtype == torch.long
        assert r.dtype == torch.long
'''
write("tests/test_rlhf.py", src)
commit("test: add preference_loss, accuracy, reward_stats, and PreferenceDataset tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: ValueHead + compute_gae
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rlhf.py")
src += '''

# ── ValueHead + compute_gae ───────────────────────────────────────────────────

class TestValueHead:
    def test_output_shape(self):
        vh     = ValueHead(D)
        hidden = torch.randn(B, T, D)
        v      = vh(hidden)
        assert v.shape == (B, T)

    def test_gradient_flows(self):
        vh     = ValueHead(D)
        hidden = torch.randn(B, T, D, requires_grad=True)
        loss   = vh(hidden).sum()
        loss.backward()
        assert hidden.grad is not None


class TestComputeGAE:
    def test_output_shapes(self):
        T_     = 8
        r      = torch.zeros(T_)
        v      = torch.ones(T_)
        adv, ret = compute_gae(r, v, gamma=1.0, lam=0.95)
        assert adv.shape == (T_,)
        assert ret.shape == (T_,)

    def test_returns_equals_advantages_plus_values(self):
        T_     = 6
        r      = torch.rand(T_)
        v      = torch.rand(T_)
        adv, ret = compute_gae(r, v)
        assert torch.allclose(ret, adv + v, atol=1e-5)

    def test_zero_reward_negative_advantage(self):
        """With positive values and zero rewards, advantages should be negative."""
        T_     = 4
        r      = torch.zeros(T_)
        v      = torch.full((T_,), 2.0)
        adv, _ = compute_gae(r, v, gamma=1.0, lam=1.0, last_val=0.0)
        assert adv[-1].item() < 0.0
'''
write("tests/test_rlhf.py", src)
commit("test: add ValueHead shape/gradient, compute_gae shape/returns/advantage tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: KL divergence + AdaptiveKLController
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rlhf.py")
src += '''

# ── KL divergence ─────────────────────────────────────────────────────────────

class TestKLDivergence:
    def test_same_policy_zero_kl(self):
        logits = torch.randn(B, T, VOCAB)
        kl     = token_kl_divergence(logits, logits)
        assert kl.abs().max().item() < 1e-5

    def test_kl_non_negative(self):
        logits_p = torch.randn(B, T, VOCAB)
        logits_q = torch.randn(B, T, VOCAB)
        kl       = token_kl_divergence(logits_p, logits_q)
        assert (kl >= 0).all()

    def test_approx_kl_shape(self):
        lp = torch.randn(B, T)
        lr = torch.randn(B, T)
        kl = approx_token_kl(lp, lr)
        assert kl.shape == (B, T)

    def test_adaptive_kl_increases_on_high_kl(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0)
        new  = ctrl.update(current_kl=10.0)
        assert new > 0.1

    def test_adaptive_kl_decreases_on_low_kl(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.5, target_kl=6.0)
        new  = ctrl.update(current_kl=2.0)
        assert new < 0.5
'''
write("tests/test_rlhf.py", src)
commit("test: add token_kl_divergence, approx_kl shape, AdaptiveKLController update tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: PPO losses
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rlhf.py")
src += '''

# ── PPO losses ────────────────────────────────────────────────────────────────

class TestPPOLoss:
    def _make_tensors(self, n=16):
        lp  = torch.randn(n)
        olp = lp + 0.1 * torch.randn(n)
        adv = torch.randn(n)
        v   = torch.randn(n)
        ret = adv + v
        log = torch.randn(n, VOCAB)
        return lp, olp, adv, v, ret, log

    def test_total_loss_returns_dict(self):
        lp, olp, adv, v, ret, log = self._make_tensors()
        _, info = ppo_total_loss(lp, olp, adv, v, ret, log)
        for k in ("policy_loss","value_loss","entropy","total_loss","clip_fraction"):
            assert k in info

    def test_loss_scalar(self):
        lp, olp, adv, v, ret, log = self._make_tensors()
        total, _ = ppo_total_loss(lp, olp, adv, v, ret, log)
        assert total.ndim == 0

    def test_identical_policy_zero_clip(self):
        """If log_probs == old_log_probs, ratio=1, no clipping occurs."""
        lp  = torch.zeros(8)
        adv = torch.ones(8)
        v   = torch.zeros(8)
        ret = v
        _, info = ppo_total_loss(lp, lp, adv, v, ret, clip_ratio=0.2)
        assert info["clip_fraction"] == 0.0
'''
write("tests/test_rlhf.py", src)
commit("test: add ppo_total_loss dict, scalar, and zero clip fraction tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: PPORolloutBuffer
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rlhf.py")
src += '''

# ── PPORolloutBuffer ──────────────────────────────────────────────────────────

class TestPPORolloutBuffer:
    def _make_rollout(self, T_gen=8):
        return (
            torch.randint(0, VOCAB, (T + T_gen,)),  # input_ids
            torch.randn(T_gen),                      # log_probs
            torch.randn(T_gen),                      # ref_log_probs
            torch.randn(T_gen),                      # values
            1.5,                                     # reward_score
        )

    def test_add_and_len(self):
        buf = PPORolloutBuffer()
        buf.add(*self._make_rollout())
        assert len(buf) == 1

    def test_finalize_returns_rollouts(self):
        buf = PPORolloutBuffer()
        for _ in range(3):
            buf.add(*self._make_rollout())
        rollouts = buf.finalize()
        assert len(rollouts) == 3

    def test_advantages_normalised(self):
        buf = PPORolloutBuffer()
        for _ in range(4):
            buf.add(*self._make_rollout())
        rollouts = buf.finalize()
        all_adv  = torch.cat([r.advantages for r in rollouts])
        assert all_adv.mean().abs().item() < 0.5   # close to zero mean
'''
write("tests/test_rlhf.py", src)
commit("test: add PPORolloutBuffer add/len, finalize, and advantage normalisation tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v2.4.0 + expose RLHF in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"2.3.0\"", "__version__ = \"2.4.0\"")
src = src.replace(
    "from nanomind.amp import AMPConfig, AMPTrainer, GradAccumulator, mixed_precision_context",
    "from nanomind.amp import AMPConfig, AMPTrainer, GradAccumulator, mixed_precision_context\n"
    "from nanomind.rlhf import RewardModel, PPOConfig, RewardModelConfig, preference_loss"
)
src = src.replace(
    "    \"mixed_precision_context\",\n    \"__version__\",\n]",
    "    \"mixed_precision_context\",\n"
    "    \"RewardModel\",\n"
    "    \"PPOConfig\",\n"
    "    \"RewardModelConfig\",\n"
    "    \"preference_loss\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v2.4.0 — expose RewardModel, PPOConfig, preference_loss in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Training** | AMP + Grad Checkpointing — bfloat16, grad accum, loss scaling |",
    "| **Training** | AMP + Grad Checkpointing — bfloat16, grad accum, loss scaling |\n"
    "| **Alignment** | RLHF — Bradley-Terry reward model, PPO with KL penalty |"
)
readme = readme.replace(
    "**Total: 545 commits across 27 days.**",
    "**Total: 565 commits across 28 days.**"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [2.3.0] — 2024 — Mixed Precision Training & Gradient Checkpointing",
    "## [2.4.0] — 2024 — RLHF: Reward Model + PPO\n\n### Added\n"
    "- `RewardModel` — transformer backbone + scalar head for preference scoring\n"
    "- `RewardModelTrainer` — train_step() + evaluate() on preference pairs\n"
    "- `PreferenceDataset` — (prompt, chosen, rejected) dataset with collate_fn\n"
    "- `RewardModelConfig` / `PPOConfig` — RLHF hyperparameter dataclasses\n"
    "- `preference_loss()` — Bradley-Terry pairwise ranking loss\n"
    "- `preference_accuracy()` / `reward_stats()` — RM evaluation metrics\n"
    "- `ValueHead` — per-token value estimates for actor-critic PPO\n"
    "- `compute_gae()` — Generalised Advantage Estimation (λ-returns)\n"
    "- `token_kl_divergence()` / `approx_token_kl()` — KL penalty computation\n"
    "- `AdaptiveKLController` — dynamic β controller (Ziegler et al. 2019)\n"
    "- `PPORolloutBuffer` — rollout collection with GAE + advantage normalisation\n"
    "- `ppo_total_loss()` — clip + value + entropy combined PPO objective\n"
    "- `examples/rlhf_demo.py` — reward model training + PPO signals demo\n\n---\n\n"
    "## [2.3.0] — 2024 — Mixed Precision Training & Gradient Checkpointing"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v2.4.0, update README and CHANGELOG for Day 28 RLHF")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 28 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v2.4.0",
    "-m", "NanoMind v2.4.0 — RLHF: Reward Model + PPO", check=False)
r = run("git", "push", "origin", "v2.4.0", check=False)
print("Tag v2.4.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 28 COMPLETE — v2.4.0 TAGGED! ===")
