"""
day50_commits.py — 20 atomic commits for Day 50: Constitutional AI, DPO & Modern Alignment.
GOLDEN JUBILEE — v5.0.0 MAJOR RELEASE!
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

print("\n=== DAY 50 🥇 GOLDEN JUBILEE: Constitutional AI, DPO & Modern Alignment — v5.0.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — alignment package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/__init__.py",
      '"""NanoMind Alignment sub-package — Constitutional AI, DPO & Modern Alignment."""\n')
commit("feat: add nanomind/alignment/ package skeleton — Constitutional AI, DPO, modern alignment")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Preference dataset and data structures
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/preference.py", '''\
"""
nanomind/alignment/preference.py — Preference data structures.

## From RLHF to Preference Learning

Classic RLHF (Day 11) uses a reward model + PPO loop.
Modern alignment methods use preference data directly:
  - DPO:  learns directly from (chosen, rejected) pairs — no RM!
  - IPO:  identity preference optimisation (fixes DPO over-fitting)
  - KTO:  learns from (prompt, response, good/bad) labels
  - RLAIF: AI feedback instead of human feedback (Constitutional AI)

## Preference Triplet

Standard preference dataset:
  (prompt, chosen, rejected)
  chosen:   the better response (human-preferred)
  rejected: the worse response (human-rejected)

Used in: Anthropic HH-RLHF, OpenAssistant, UltraFeedback.

## Constitutional AI (Bai et al., 2022)

Instead of human labellers, use the model itself to generate preferences:
  1. Model generates response to a harmful prompt
  2. Model critiques its own response using a CONSTITUTION
     (a list of principles: "be helpful, harmless, honest")
  3. Model revises its response based on the critique
  4. Use (original, revised) as preference pair → train with RL

This allows scalable alignment without human labellers.
Used by: Claude (Anthropic).

Reference:
  Bai et al. (2022) "Constitutional AI: Harmlessness from AI Feedback"
  https://arxiv.org/abs/2212.06950

  Rafailov et al. (2023) DPO:
  https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PreferencePair:
    """A single preference pair: (prompt, chosen, rejected)."""
    prompt:   str
    chosen:   str
    rejected: str
    score_chosen:   float | None = None   # optional reward score
    score_rejected: float | None = None
    source:   str = "human"               # human | ai | synthetic

    @property
    def margin(self) -> float | None:
        if self.score_chosen and self.score_rejected:
            return self.score_chosen - self.score_rejected
        return None

    def to_dict(self) -> dict:
        return {
            "prompt":   self.prompt,
            "chosen":   self.chosen,
            "rejected": self.rejected,
            "source":   self.source,
        }

    def flip(self) -> "PreferencePair":
        """Flip chosen/rejected (for data augmentation / sanity check)."""
        return PreferencePair(
            prompt        = self.prompt,
            chosen        = self.rejected,
            rejected      = self.chosen,
            score_chosen  = self.score_rejected,
            score_rejected= self.score_chosen,
            source        = self.source,
        )


@dataclass
class BinaryFeedback:
    """Single response with binary good/bad label (for KTO)."""
    prompt:   str
    response: str
    is_good:  bool        # True = desirable, False = undesirable
    source:   str = "human"


class PreferenceDataset:
    """
    Dataset of preference pairs for alignment training.

    Args:
        pairs: Initial list of :class:`PreferencePair`.

    Example::

        dataset = PreferenceDataset()
        dataset.add(PreferencePair("What is 2+2?", "4", "fish"))
        print(f"Size: {len(dataset)}")
        for pair in dataset:
            ...
    """

    def __init__(self, pairs: list[PreferencePair] | None = None) -> None:
        self._pairs: list[PreferencePair] = list(pairs or [])

    def add(self, pair: PreferencePair) -> None:
        self._pairs.append(pair)

    def filter_by_margin(self, min_margin: float) -> "PreferenceDataset":
        """Keep only pairs where chosen is clearly better."""
        filtered = [p for p in self._pairs
                    if p.margin is not None and p.margin >= min_margin]
        return PreferenceDataset(filtered)

    def filter_by_source(self, source: str) -> "PreferenceDataset":
        return PreferenceDataset([p for p in self._pairs if p.source == source])

    def stats(self) -> dict:
        margins = [p.margin for p in self._pairs if p.margin is not None]
        return {
            "n_pairs":     len(self._pairs),
            "n_human":     sum(1 for p in self._pairs if p.source == "human"),
            "n_ai":        sum(1 for p in self._pairs if p.source == "ai"),
            "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
        }

    def to_binary(self) -> list[BinaryFeedback]:
        """Convert to binary feedback list (for KTO)."""
        out = []
        for p in self._pairs:
            out.append(BinaryFeedback(p.prompt, p.chosen,   is_good=True))
            out.append(BinaryFeedback(p.prompt, p.rejected, is_good=False))
        return out

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs)

    def __getitem__(self, idx):
        return self._pairs[idx]
''')
commit("feat: add PreferencePair, BinaryFeedback, PreferenceDataset — preference data, stats, filter_by_margin")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — DPO (Direct Preference Optimization)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/dpo.py", '''\
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
            loss = (1 - cfg.label_smoothing) * loss - \
                   cfg.label_smoothing * F.logsigmoid(-logits)

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
''')
commit("feat: add DPOConfig, dpo_loss (DPO/IPO/KTO), compute_log_probs, DPOTrainer — reward margin, accuracy")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Constitutional AI
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/constitutional.py", '''\
"""
nanomind/alignment/constitutional.py — Constitutional AI (CAI).

## Constitutional AI (Bai et al., 2022)

Key idea: use the model itself to generate alignment signal.

PHASE 1 — Supervised Learning from AI Feedback (SL-CAI):
  1. Prompt model to generate a potentially harmful response
  2. Ask model to identify principles violated ("is this response safe?")
  3. Ask model to revise the response to be more helpful and harmless
  4. Finetune on (prompt, revised_response) pairs

PHASE 2 — RL from AI Feedback (RLAIF):
  1. Generate pairs of (response_A, response_B) for same prompt
  2. Ask model: "Which response is more helpful and harmless?"
     using constitutional principles
  3. Use AI preferences to train reward model
  4. PPO/DPO with this RM

This scales to unlimited preference data without human labellers!

The CONSTITUTION is a set of natural language principles:
  - "Choose the response that is least harmful"
  - "Choose the response that most avoids discrimination"
  - "Choose the response that is most truthful"

NanoMind implements the CAI critique-revision loop and AI feedback collection.

Reference:
  Bai et al. (2022) https://arxiv.org/abs/2212.06950
  Lee et al. (2023) RLAIF https://arxiv.org/abs/2309.00267
"""

from __future__ import annotations
from dataclasses import dataclass, field
from nanomind.alignment.preference import PreferencePair, PreferenceDataset


# Simplified Anthropic-style constitution
DEFAULT_CONSTITUTION = [
    "Choose the response that is least likely to cause harm.",
    "Choose the response that is most helpful to the user.",
    "Choose the response that is most honest and avoids deception.",
    "Choose the response that is most respectful of human dignity.",
    "Choose the response that avoids discrimination and bias.",
]


@dataclass
class CritiqueRevision:
    """A single critique-revision cycle."""
    original_response: str
    critique:          str
    revised_response:  str
    principle:         str


@dataclass
class ConstitutionalConfig:
    """Configuration for Constitutional AI training."""
    constitution:       list[str] = field(default_factory=lambda: DEFAULT_CONSTITUTION)
    n_critique_rounds:  int       = 1
    critique_prompt_tmpl: str     = (
        "Here is a response: '{response}'\n\n"
        "Principle: {principle}\n\n"
        "Identify any ways in which this response violates the principle."
    )
    revision_prompt_tmpl: str = (
        "Rewrite the following response to better follow this principle:\n"
        "Principle: {principle}\n\n"
        "Original: {response}\n\n"
        "Critique: {critique}\n\n"
        "Revised response:"
    )
    feedback_prompt_tmpl: str = (
        "Consider these two responses to: '{prompt}'\n\n"
        "Response A: {response_a}\n"
        "Response B: {response_b}\n\n"
        "Principle: {principle}\n\n"
        "Which response better follows the principle? Reply A or B."
    )


class ConstitutionalAI:
    """
    Constitutional AI critique-revision and AI feedback pipeline.

    Uses a model_fn callable to simulate LLM calls for:
      1. Generating critiques
      2. Revising responses
      3. Comparing response pairs (RLAIF)

    Args:
        cfg:       :class:`ConstitutionalConfig`.
        model_fn:  Callable (prompt: str) → str. Can be real LLM or mock.

    Example::

        cai = ConstitutionalAI(cfg, model_fn=my_llm)
        cr  = cai.critique_and_revise(harmful_response, principle)
        pair = cai.generate_preference_pair(prompt, resp_a, resp_b)
    """

    def __init__(
        self,
        cfg:      ConstitutionalConfig | None = None,
        model_fn: object = None,
    ) -> None:
        self.cfg      = cfg or ConstitutionalConfig()
        self.model_fn = model_fn or self._mock_model

    def _mock_model(self, prompt: str) -> str:
        """Mock LLM response for testing."""
        if "Identify" in prompt:
            return "This response may be harmful because it lacks care."
        if "Rewrite" in prompt:
            return "Here is a helpful and harmless revised response."
        if "Which response" in prompt:
            return "A"
        return "A helpful response."

    def critique(self, response: str, principle: str) -> str:
        """Generate a critique of a response against a principle."""
        prompt = self.cfg.critique_prompt_tmpl.format(
            response=response, principle=principle
        )
        return self.model_fn(prompt)

    def revise(self, response: str, critique: str, principle: str) -> str:
        """Revise a response based on a critique."""
        prompt = self.cfg.revision_prompt_tmpl.format(
            response=response, critique=critique, principle=principle
        )
        return self.model_fn(prompt)

    def critique_and_revise(
        self,
        response:  str,
        principle: str | None = None,
    ) -> CritiqueRevision:
        """
        Full critique-revision cycle for one response.

        Args:
            response:  Original model response.
            principle: Constitutional principle (random if None).

        Returns:
            :class:`CritiqueRevision`.
        """
        import random
        principle = principle or random.choice(self.cfg.constitution)
        critique  = self.critique(response, principle)
        revised   = self.revise(response, critique, principle)
        return CritiqueRevision(
            original_response = response,
            critique          = critique,
            revised_response  = revised,
            principle         = principle,
        )

    def ai_feedback(
        self,
        prompt:     str,
        response_a: str,
        response_b: str,
        principle:  str | None = None,
    ) -> str:
        """
        Get AI preference between two responses.

        Returns:
            ``"A"`` or ``"B"`` (which response is preferred).
        """
        import random
        principle = principle or random.choice(self.cfg.constitution)
        feedback_prompt = self.cfg.feedback_prompt_tmpl.format(
            prompt=prompt, response_a=response_a,
            response_b=response_b, principle=principle,
        )
        response = self.model_fn(feedback_prompt).strip()
        # Normalise to A or B
        if "B" in response and "A" not in response:
            return "B"
        return "A"

    def build_preference_dataset(
        self,
        prompts:     list[str],
        responses_a: list[str],
        responses_b: list[str],
    ) -> PreferenceDataset:
        """
        Build preference dataset using AI feedback (RLAIF).

        For each (prompt, resp_a, resp_b), the AI labels which is preferred.
        """
        dataset = PreferenceDataset()
        for prompt, a, b in zip(prompts, responses_a, responses_b):
            preferred = self.ai_feedback(prompt, a, b)
            chosen, rejected = (a, b) if preferred == "A" else (b, a)
            dataset.add(PreferencePair(
                prompt=prompt, chosen=chosen, rejected=rejected, source="ai"
            ))
        return dataset

    def generate_sft_data(
        self,
        prompts:   list[str],
        responses: list[str],
    ) -> list[dict]:
        """
        Generate SFT training data via critique-revision.

        Returns list of {prompt, response} dicts for SFT fine-tuning.
        """
        sft_data = []
        for prompt, response in zip(prompts, responses):
            cr = self.critique_and_revise(response)
            sft_data.append({
                "prompt":    prompt,
                "response":  cr.revised_response,
                "original":  cr.original_response,
                "critique":  cr.critique,
                "principle": cr.principle,
            })
        return sft_data
''')
commit("feat: add ConstitutionalAI — critique_and_revise, ai_feedback (RLAIF), build_preference_dataset")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — KTO (Kahneman-Tversky Optimization)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/kto.py", '''\
"""
nanomind/alignment/kto.py — KTO: Kahneman-Tversky Optimization.

## KTO (Ethayarajh et al., 2024)

DPO needs paired (chosen, rejected) data for the SAME prompt.
This is often unavailable — real human feedback is:
  "This response was good" / "This response was bad"
without a paired comparison.

KTO uses prospect theory (Kahneman & Tversky, 1979):
  Humans are loss-averse: losing $100 feels worse than gaining $100!

KTO loss:
  For desirable responses (y_w):
    L_w = 1 - σ(r(x, y_w) - z₀)    where z₀ = KL[π || π_ref]

  For undesirable responses (y_l):
    L_l = 1 - σ(z₀ - r(x, y_l))

The z₀ reference point is the expected reward under the reference model.

Benefits:
  ✓ Works with binary feedback (good/bad), not pairs
  ✓ More natural data collection (rate responses, don't compare)
  ✓ Better data efficiency (each sample is useful independently)

Reference:
  Ethayarajh et al. (2024) "KTO: Model Alignment as Prospect Theoretic Optimization"
  https://arxiv.org/abs/2402.01306
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanomind.alignment.dpo import compute_log_probs


@dataclass
class KTOConfig:
    """Configuration for KTO training."""
    beta:        float = 0.1    # KL weight
    desirable_weight:   float = 1.0   # weight for desirable samples
    undesirable_weight: float = 1.0   # weight for undesirable samples


def kto_loss(
    policy_logp:  torch.Tensor,
    ref_logp:     torch.Tensor,
    is_desirable: torch.Tensor,   # (B,) bool
    kl_estimate:  torch.Tensor,   # scalar KL estimate
    cfg:          KTOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Compute KTO loss from log probabilities and binary labels.

    Args:
        policy_logp:  ``(B,)`` policy log probs for each response.
        ref_logp:     ``(B,)`` reference log probs.
        is_desirable: ``(B,)`` bool — True = desirable, False = undesirable.
        kl_estimate:  Scalar KL divergence estimate (z₀ reference point).
        cfg:          :class:`KTOConfig`.

    Returns:
        ``(loss, metrics_dict)``
    """
    β  = cfg.beta
    z0 = kl_estimate.detach()

    rewards = β * (policy_logp - ref_logp)   # (B,)

    # Desirable loss: 1 - σ(r - z₀)
    loss_desirable   = cfg.desirable_weight * (
        1 - F.sigmoid(rewards[is_desirable] - z0)
    )
    # Undesirable loss: 1 - σ(z₀ - r)
    loss_undesirable = cfg.undesirable_weight * (
        1 - F.sigmoid(z0 - rewards[~is_desirable])
    )

    # Combine (mean over non-empty subsets)
    total = torch.zeros(1)
    if loss_desirable.numel() > 0:
        total = total + loss_desirable.mean()
    if loss_undesirable.numel() > 0:
        total = total + loss_undesirable.mean()

    metrics = {
        "loss":             total.item(),
        "reward_desirable":   rewards[is_desirable].mean().item() if is_desirable.any() else 0.0,
        "reward_undesirable": rewards[~is_desirable].mean().item() if (~is_desirable).any() else 0.0,
        "kl_estimate":        z0.item(),
    }
    return total, metrics


class KTOTrainer:
    """
    KTO training loop.

    Args:
        policy_model:  LM to train.
        ref_model:     Frozen reference LM.
        optimizer:     Policy optimizer.
        cfg:           :class:`KTOConfig`.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model:    nn.Module,
        optimizer:    torch.optim.Optimizer,
        cfg:          KTOConfig | None = None,
    ) -> None:
        self.policy = policy_model
        self.ref    = ref_model
        self.opt    = optimizer
        self.cfg    = cfg or KTOConfig()
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def estimate_kl(self, ids: torch.Tensor) -> torch.Tensor:
        """Estimate E[log π/π_ref] on a batch for the z₀ reference."""
        with torch.no_grad():
            pol = compute_log_probs(self.policy, ids, ids)
            ref = compute_log_probs(self.ref,    ids, ids)
        return (pol - ref).mean()

    def step(
        self,
        ids:          torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> dict:
        """
        One KTO training step.

        Args:
            ids:          ``(B, T)`` response token IDs.
            is_desirable: ``(B,)`` bool labels.
        """
        self.opt.zero_grad()
        kl_est = self.estimate_kl(ids)

        policy_logp = compute_log_probs(self.policy, ids, ids)
        with torch.no_grad():
            ref_logp = compute_log_probs(self.ref, ids, ids)

        loss, metrics = kto_loss(policy_logp, ref_logp, is_desirable, kl_est, self.cfg)
        loss.backward()
        self.opt.step()
        return metrics
''')
commit("feat: add KTOConfig, kto_loss (prospect theory), KTOTrainer — binary feedback, desirable/undesirable")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Reward model (RM) for RLHF/RLAIF
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/reward.py", '''\
"""
nanomind/alignment/reward.py — Reward model for alignment training.

The reward model (RM) learns to score responses based on preference data.
Used as a signal source for PPO and as a judge for RLAIF.

Training objective (Bradley-Terry):
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
  L_RM = -log σ(r(x, y_w) - r(x, y_l))

After training, RM can score any response in [roughly] (-3, +3) range:
  r > 0: response is better than average
  r < 0: response is worse than average
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class RewardModel(nn.Module):
    """
    Reward model: LM backbone + scalar regression head.

    Takes (prompt + response) token IDs and outputs a scalar reward.

    Args:
        backbone:   LM model (used as feature extractor).
        d_model:    Backbone output dimension.
        dropout:    Dropout on reward head.

    Example::

        rm      = RewardModel(backbone, d_model=128)
        rewards = rm(chosen_ids)     # (B,) scalar rewards
        loss, m = rm.preference_loss(chosen_ids, rejected_ids)
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model:  int,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar reward for each sequence.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(B,)`` reward scores.
        """
        out = self.backbone(input_ids)
        # Use last token hidden state as sequence representation
        if isinstance(out, tuple):
            hidden = out[0][:, -1, :]   # (B, D)
        else:
            hidden = out[:, -1, :]
        return self.reward_head(hidden).squeeze(-1)   # (B,)

    def preference_loss(
        self,
        chosen_ids:   torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Bradley-Terry preference loss.

        Args:
            chosen_ids:   ``(B, T)`` chosen responses.
            rejected_ids: ``(B, T)`` rejected responses.

        Returns:
            ``(loss, metrics)``
        """
        r_w = self(chosen_ids)     # (B,)
        r_l = self(rejected_ids)   # (B,)
        loss     = -F.logsigmoid(r_w - r_l).mean()
        accuracy = (r_w > r_l).float().mean()
        return loss, {
            "loss":        loss.item(),
            "accuracy":    accuracy.item(),
            "reward_chosen":   r_w.mean().item(),
            "reward_rejected": r_l.mean().item(),
            "reward_margin":   (r_w - r_l).mean().item(),
        }


@dataclass
class RMTrainingConfig:
    """Configuration for reward model training."""
    lr:         float = 1e-4
    beta:       float = 0.0   # margin bonus (r_w - r_l > beta)
    center_reward: bool = True  # subtract mean reward for stability
''')
commit("feat: add RewardModel — LM backbone + scalar head, preference_loss (Bradley-Terry), RMTrainingConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Alignment evaluator (win rate, reward hacking detection)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/eval.py", '''\
"""
nanomind/alignment/eval.py — Alignment evaluation metrics.

Key alignment metrics:
  - Win rate: fraction of time aligned model beats baseline
  - Reward score: mean RM score on test prompts
  - KL divergence: how much policy drifted from reference
  - Reward hacking detection: high reward but low quality

## Goodhart's Law and Reward Hacking

"When a measure becomes a target, it ceases to be a good measure."

In RLHF: models learn to exploit reward model weaknesses:
  - Generating very long responses (RM rewards verbosity)
  - Sycophantic responses ("Great question!")
  - Repetitive token patterns that confuse RM

Mitigation:
  - KL penalty (keeps policy close to reference)
  - Diverse RM ensembles (harder to hack all)
  - Human evaluation checkpoints
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class AlignmentMetrics:
    """Collected alignment evaluation metrics."""
    win_rate:    float
    mean_reward: float
    kl_div:      float
    reward_std:  float

    def to_dict(self) -> dict:
        return {
            "win_rate":    round(self.win_rate,    3),
            "mean_reward": round(self.mean_reward, 4),
            "kl_div":      round(self.kl_div,      4),
            "reward_std":  round(self.reward_std,  4),
        }

    def is_reward_hacking(self, threshold: float = 2.0) -> bool:
        """Heuristic: high reward + high KL = suspicious."""
        return self.mean_reward > threshold and self.kl_div > threshold


class AlignmentEvaluator:
    """
    Evaluate alignment quality of a trained model.

    Args:
        reward_model:  Trained reward model for scoring.
        ref_model:     Reference (SFT) model for KL estimation.

    Example::

        eval    = AlignmentEvaluator(reward_model, sft_model)
        metrics = eval.evaluate(policy_model, test_ids)
        print(metrics.to_dict())
    """

    def __init__(
        self,
        reward_model: nn.Module,
        ref_model:    nn.Module,
    ) -> None:
        self.rm  = reward_model
        self.ref = ref_model

    @torch.no_grad()
    def compute_rewards(self, model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        """Score responses with reward model."""
        return self.rm(ids)   # (B,)

    @torch.no_grad()
    def compute_kl(
        self,
        policy:    nn.Module,
        ids:       torch.Tensor,
    ) -> float:
        """Estimate KL(policy || ref) on a batch."""
        def log_probs(m, x):
            out    = m(x)
            logits = out[0] if isinstance(out, tuple) else out
            return F.log_softmax(logits[:, :-1], dim=-1)

        policy_lp = log_probs(policy, ids)
        ref_lp    = log_probs(self.ref, ids)
        kl = (torch.exp(policy_lp) * (policy_lp - ref_lp)).sum(-1).mean()
        return kl.item()

    @torch.no_grad()
    def win_rate(
        self,
        policy:    nn.Module,
        baseline:  nn.Module,
        ids:       torch.Tensor,
    ) -> float:
        """Fraction of samples where policy gets higher reward than baseline."""
        r_policy   = self.compute_rewards(policy,   ids)
        r_baseline = self.compute_rewards(baseline, ids)
        return (r_policy > r_baseline).float().mean().item()

    @torch.no_grad()
    def evaluate(
        self,
        policy:   nn.Module,
        baseline: nn.Module,
        test_ids: torch.Tensor,
    ) -> AlignmentMetrics:
        """
        Full alignment evaluation.

        Args:
            policy:   Aligned model.
            baseline: Reference/SFT model.
            test_ids: ``(B, T)`` test prompt+response token IDs.

        Returns:
            :class:`AlignmentMetrics`.
        """
        rewards  = self.compute_rewards(policy, test_ids)
        kl       = self.compute_kl(policy, test_ids)
        wr       = self.win_rate(policy, baseline, test_ids)

        return AlignmentMetrics(
            win_rate    = wr,
            mean_reward = rewards.mean().item(),
            kl_div      = kl,
            reward_std  = rewards.std().item(),
        )
''')
commit("feat: add AlignmentEvaluator — win_rate, compute_kl, evaluate(), AlignmentMetrics, reward_hacking detect")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — alignment __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/alignment/__init__.py", '''\
"""NanoMind Alignment sub-package — Constitutional AI, DPO & Modern Alignment.

Implements the modern post-training alignment stack:
  1. PreferencePair / BinaryFeedback — data structures
  2. PreferenceDataset               — preference pairs, filter, stats
  3. DPOConfig / dpo_loss            — DPO/IPO/KTO-pair loss
  4. compute_log_probs               — per-token log probs
  5. DPOTrainer                      — policy + ref model training loop
  6. ConstitutionalAI                — critique/revise, RLAIF feedback
  7. CritiqueRevision                — single critique-revision record
  8. ConstitutionalConfig            — constitution, prompts
  9. KTOConfig / kto_loss            — Kahneman-Tversky prospect loss
  10. KTOTrainer                     — binary feedback training
  11. RewardModel                    — backbone + scalar head
  12. RMTrainingConfig               — RM training settings
  13. AlignmentEvaluator             — win_rate, kl_div, reward_hacking
  14. AlignmentMetrics               — win_rate, mean_reward, kl_div

Primary exports:
    - :class:`PreferencePair`        — prompt, chosen, rejected
    - :class:`BinaryFeedback`        — prompt, response, is_good
    - :class:`PreferenceDataset`     — add, filter_by_margin, stats
    - :class:`DPOConfig`             — beta, loss_type
    - :func:`dpo_loss`               — DPO/IPO/KTO_pair loss
    - :func:`compute_log_probs`      — sequence log probs
    - :class:`DPOTrainer`            — step, policy + ref training
    - :class:`ConstitutionalAI`      — critique_and_revise, ai_feedback
    - :class:`ConstitutionalConfig`  — constitution, prompt templates
    - :class:`CritiqueRevision`      — original, critique, revised
    - :class:`KTOConfig`             — beta, desirable/undesirable weight
    - :func:`kto_loss`               — prospect-theoretic loss
    - :class:`KTOTrainer`            — step, estimate_kl
    - :class:`RewardModel`           — forward, preference_loss
    - :class:`AlignmentEvaluator`    — evaluate, win_rate, kl
    - :class:`AlignmentMetrics`      — to_dict, is_reward_hacking
"""

from nanomind.alignment.preference import PreferencePair, BinaryFeedback, PreferenceDataset
from nanomind.alignment.dpo import (
    DPOConfig, dpo_loss, compute_log_probs, DPOTrainer,
)
from nanomind.alignment.constitutional import (
    ConstitutionalAI, ConstitutionalConfig, CritiqueRevision,
    DEFAULT_CONSTITUTION,
)
from nanomind.alignment.kto import KTOConfig, kto_loss, KTOTrainer
from nanomind.alignment.reward import RewardModel, RMTrainingConfig
from nanomind.alignment.eval import AlignmentEvaluator, AlignmentMetrics

__all__ = [
    "PreferencePair", "BinaryFeedback", "PreferenceDataset",
    "DPOConfig", "dpo_loss", "compute_log_probs", "DPOTrainer",
    "ConstitutionalAI", "ConstitutionalConfig", "CritiqueRevision", "DEFAULT_CONSTITUTION",
    "KTOConfig", "kto_loss", "KTOTrainer",
    "RewardModel", "RMTrainingConfig",
    "AlignmentEvaluator", "AlignmentMetrics",
]
''')
commit("refactor: export all alignment components from nanomind/alignment/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/alignment_demo.py", '''\
"""
examples/alignment_demo.py — NanoMind Constitutional AI, DPO & Alignment demo.

Demonstrates:
  1. Preference dataset: pairs, filtering, stats
  2. DPO loss: DPO / IPO / KTO-pair variants
  3. DPO trainer: policy + frozen reference
  4. Constitutional AI: critique-revision, RLAIF
  5. KTO: binary feedback training
  6. Reward model: Bradley-Terry preference loss
  7. Alignment evaluation: win rate, KL divergence

Usage:
    python examples/alignment_demo.py
"""
import torch
import torch.nn as nn
from nanomind.alignment import (
    PreferencePair, BinaryFeedback, PreferenceDataset,
    DPOConfig, dpo_loss, compute_log_probs, DPOTrainer,
    ConstitutionalAI, ConstitutionalConfig, CritiqueRevision,
    KTOConfig, kto_loss, KTOTrainer,
    RewardModel, RMTrainingConfig,
    AlignmentEvaluator, AlignmentMetrics,
)

V = 32   # small vocab for demo

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind Constitutional AI, DPO & Alignment Demo")
print("=" * 60)

# ── Preference Dataset ────────────────────────────────────────────────────────
print("\n── Preference Dataset ──")
dataset = PreferenceDataset([
    PreferencePair("What is 2+2?", "4", "fish",
                    score_chosen=0.9, score_rejected=0.1, source="human"),
    PreferencePair("Tell me a joke", "Why did the chicken cross the road?",
                    "I refuse to answer", score_chosen=0.8, score_rejected=0.2,
                    source="human"),
    PreferencePair("What is AI?", "Artificial Intelligence is...", "IDK",
                    score_chosen=0.7, score_rejected=0.3, source="ai"),
])
print(f"  Dataset size: {len(dataset)}")
print(f"  Stats: {dataset.stats()}")
filtered = dataset.filter_by_margin(min_margin=0.5)
print(f"  After filter(margin≥0.5): {len(filtered)} pairs")
binary = dataset.to_binary()
print(f"  Binary feedback samples: {len(binary)}")

# ── DPO Loss ──────────────────────────────────────────────────────────────────
print("\n── DPO Loss (Direct Preference Optimization) ──")
B = 4
policy_logp_w = torch.tensor([-2.1, -1.8, -2.5, -1.9])
policy_logp_l = torch.tensor([-3.2, -3.5, -3.8, -2.8])
ref_logp_w    = torch.tensor([-2.0, -2.0, -2.4, -2.0])
ref_logp_l    = torch.tensor([-3.0, -3.2, -3.5, -2.7])

for loss_type in ["dpo", "ipo", "kto_pair"]:
    cfg  = DPOConfig(beta=0.1, loss_type=loss_type)
    loss, m = dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, cfg)
    print(f"  [{loss_type:8}] loss={m['loss']:.4f} acc={m['accuracy']:.2%} "
          f"margin={m['reward_margin']:.4f}")

# ── DPO Trainer ───────────────────────────────────────────────────────────────
print("\n── DPO Trainer ──")
policy = TinyLM(); ref = TinyLM()
opt    = torch.optim.AdamW(policy.parameters(), lr=1e-4)
trainer = DPOTrainer(policy, ref, opt, DPOConfig(beta=0.1))
chosen_ids   = torch.randint(0, V, (2, 8))
rejected_ids = torch.randint(0, V, (2, 8))
metrics = trainer.step(chosen_ids, rejected_ids)
print(f"  DPO step: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2%}")
print(f"  Chosen reward: {metrics['reward_chosen']:.4f}")
print(f"  Rejected reward: {metrics['reward_rejected']:.4f}")

# ── Constitutional AI ──────────────────────────────────────────────────────────
print("\n── Constitutional AI ──")
cai = ConstitutionalAI()
cr  = cai.critique_and_revise(
    response  = "This question is stupid and I won't answer it.",
    principle = "Choose the response that is most respectful of human dignity.",
)
print(f"  Original: {cr.original_response[:50]}...")
print(f"  Critique: {cr.critique[:50]}...")
print(f"  Revised:  {cr.revised_response[:50]}...")

# RLAIF: AI feedback on response pairs
print("\n  RLAIF — AI Feedback:")
preferred = cai.ai_feedback(
    "What is the capital of France?",
    "Paris is the capital of France.",
    "I don't know.",
)
print(f"  Preferred response: {preferred}")

# Build preference dataset via RLAIF
rlaif_dataset = cai.build_preference_dataset(
    prompts     = ["What is 2+2?", "Explain gravity"],
    responses_a = ["The answer is 4.", "Gravity is a force..."],
    responses_b = ["I don't know", "It's complicated."],
)
print(f"  RLAIF dataset size: {len(rlaif_dataset)}")
print(f"  RLAIF stats: {rlaif_dataset.stats()}")

# SFT data generation
sft_data = cai.generate_sft_data(
    ["How are you?"], ["I am functioning."]
)
print(f"  SFT data generated: {sft_data[0].keys()}")

# ── KTO ───────────────────────────────────────────────────────────────────────
print("\n── KTO (Kahneman-Tversky Optimization) ──")
policy2 = TinyLM(); ref2 = TinyLM()
opt2    = torch.optim.AdamW(policy2.parameters(), lr=1e-4)
kto_trainer = KTOTrainer(policy2, ref2, opt2, KTOConfig(beta=0.1))
ids        = torch.randint(0, V, (4, 8))
is_good    = torch.tensor([True, False, True, False])
kto_m      = kto_trainer.step(ids, is_good)
print(f"  KTO step: loss={kto_m['loss']:.4f}")
print(f"  Reward desirable: {kto_m['reward_desirable']:.4f}")
print(f"  Reward undesirable: {kto_m['reward_undesirable']:.4f}")

# Direct kto_loss
policy_lp = torch.tensor([-2.0, -3.0, -2.5, -3.5])
ref_lp    = torch.tensor([-2.1, -2.9, -2.4, -3.3])
kl_est    = torch.tensor(0.1)
cfg_kto   = KTOConfig(beta=0.1)
l, m      = kto_loss(policy_lp, ref_lp, is_good, kl_est, cfg_kto)
print(f"  KTO loss: {l.item():.4f}")

# ── Reward Model ──────────────────────────────────────────────────────────────
print("\n── Reward Model (Bradley-Terry) ──")
rm     = RewardModel(TinyLM(), d_model=V)
chosen = torch.randint(0, V, (4, 8))
reject = torch.randint(0, V, (4, 8))
rewards = rm(chosen)
print(f"  Reward scores shape: {tuple(rewards.shape)}")
print(f"  Mean reward: {rewards.mean().item():.4f}")
rm_loss, rm_m = rm.preference_loss(chosen, reject)
print(f"  RM loss: {rm_m['loss']:.4f}, acc: {rm_m['accuracy']:.2%}")
print(f"  Reward margin: {rm_m['reward_margin']:.4f}")

# ── Alignment Evaluator ────────────────────────────────────────────────────────
print("\n── Alignment Evaluation ──")
evaluator = AlignmentEvaluator(rm, TinyLM())
test_ids  = torch.randint(0, V, (8, 10))
metrics   = evaluator.evaluate(policy, ref, test_ids)
print(f"  Win rate:     {metrics.win_rate:.2%}")
print(f"  Mean reward:  {metrics.mean_reward:.4f}")
print(f"  KL divergence:{metrics.kl_div:.4f}")
print(f"  Reward hacking: {metrics.is_reward_hacking()}")
print(f"  Metrics: {metrics.to_dict()}")

print("\nAlignment demo complete!")
''')
commit("feat: add examples/alignment_demo.py — DPO, constitutional AI, KTO, reward model, evaluation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_alignment.py", '''\
"""tests/test_alignment.py — Tests for NanoMind alignment package."""
import pytest
import torch
import torch.nn as nn
from nanomind.alignment import (
    PreferencePair, BinaryFeedback, PreferenceDataset,
    DPOConfig, dpo_loss, compute_log_probs, DPOTrainer,
    ConstitutionalAI, ConstitutionalConfig,
    KTOConfig, kto_loss, KTOTrainer,
    RewardModel,
    AlignmentEvaluator, AlignmentMetrics,
)

V = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── PreferenceDataset ─────────────────────────────────────────────────────────

class TestPreferenceDataset:
    def test_add_and_len(self):
        d = PreferenceDataset()
        d.add(PreferencePair("p", "c", "r"))
        assert len(d) == 1

    def test_filter_by_margin(self):
        d = PreferenceDataset([
            PreferencePair("p", "c", "r", score_chosen=0.9, score_rejected=0.1),
            PreferencePair("p", "c", "r", score_chosen=0.6, score_rejected=0.5),
        ])
        f = d.filter_by_margin(0.5)
        assert len(f) == 1

    def test_filter_by_source(self):
        d = PreferenceDataset([
            PreferencePair("p", "c", "r", source="human"),
            PreferencePair("p", "c", "r", source="ai"),
        ])
        assert len(d.filter_by_source("human")) == 1

    def test_stats_keys(self):
        d = PreferenceDataset([PreferencePair("p", "c", "r")])
        s = d.stats()
        assert "n_pairs" in s

    def test_to_binary(self):
        d = PreferenceDataset([PreferencePair("p", "c", "r")])
        b = d.to_binary()
        assert len(b) == 2
        assert any(x.is_good for x in b)
        assert any(not x.is_good for x in b)

    def test_flip(self):
        p = PreferencePair("q", "chosen", "rejected")
        f = p.flip()
        assert f.chosen == "rejected"
        assert f.rejected == "chosen"


# ── DPO ───────────────────────────────────────────────────────────────────────

class TestDPO:
    def _logps(self, B=4):
        pw = torch.tensor([-2.0] * B)
        pl = torch.tensor([-3.0] * B)
        rw = torch.tensor([-2.1] * B)
        rl = torch.tensor([-2.9] * B)
        return pw, pl, rw, rl

    def test_dpo_loss_scalar(self):
        pw, pl, rw, rl = self._logps()
        l, m = dpo_loss(pw, pl, rw, rl, DPOConfig(loss_type="dpo"))
        assert l.shape == ()

    def test_ipo_loss_scalar(self):
        pw, pl, rw, rl = self._logps()
        l, m = dpo_loss(pw, pl, rw, rl, DPOConfig(loss_type="ipo"))
        assert l.shape == ()

    def test_kto_pair_loss(self):
        pw, pl, rw, rl = self._logps()
        l, m = dpo_loss(pw, pl, rw, rl, DPOConfig(loss_type="kto_pair"))
        assert l.item() >= 0

    def test_metrics_keys(self):
        pw, pl, rw, rl = self._logps()
        _, m = dpo_loss(pw, pl, rw, rl, DPOConfig())
        for k in ("loss", "accuracy", "reward_chosen", "reward_rejected", "reward_margin"):
            assert k in m

    def test_accuracy_in_range(self):
        pw, pl, rw, rl = self._logps()
        _, m = dpo_loss(pw, pl, rw, rl, DPOConfig())
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_dpo_trainer_step(self):
        p   = TinyLM(); r = TinyLM()
        opt = torch.optim.AdamW(p.parameters())
        t   = DPOTrainer(p, r, opt, DPOConfig())
        c   = torch.randint(0, V, (2, 4))
        rej = torch.randint(0, V, (2, 4))
        m   = t.step(c, rej)
        assert "loss" in m

    def test_ref_model_frozen(self):
        p   = TinyLM(); r = TinyLM()
        opt = torch.optim.AdamW(p.parameters())
        DPOTrainer(p, r, opt)
        for param in r.parameters():
            assert not param.requires_grad


# ── Constitutional AI ──────────────────────────────────────────────────────────

class TestConstitutionalAI:
    def test_critique_returns_str(self):
        cai = ConstitutionalAI()
        c   = cai.critique("a response", "be helpful")
        assert isinstance(c, str) and len(c) > 0

    def test_revise_returns_str(self):
        cai = ConstitutionalAI()
        r   = cai.revise("bad response", "it is bad", "be helpful")
        assert isinstance(r, str)

    def test_critique_and_revise(self):
        cai = ConstitutionalAI()
        cr  = cai.critique_and_revise("some response")
        assert cr.original_response == "some response"
        assert cr.revised_response != ""
        assert cr.principle != ""

    def test_ai_feedback_returns_AB(self):
        cai = ConstitutionalAI()
        fb  = cai.ai_feedback("prompt", "response A", "response B")
        assert fb in ("A", "B")

    def test_build_preference_dataset(self):
        cai = ConstitutionalAI()
        ds  = cai.build_preference_dataset(["p"], ["a"], ["b"])
        assert len(ds) == 1
        assert ds[0].source == "ai"

    def test_generate_sft_data(self):
        cai  = ConstitutionalAI()
        data = cai.generate_sft_data(["q"], ["resp"])
        assert len(data) == 1
        assert "prompt" in data[0] and "response" in data[0]


# ── KTO ───────────────────────────────────────────────────────────────────────

class TestKTO:
    def test_kto_loss_scalar(self):
        lp = torch.tensor([-2.0, -3.0, -2.5, -3.5])
        rl = torch.tensor([-2.1, -2.9, -2.4, -3.3])
        kl = torch.tensor(0.1)
        is_d = torch.tensor([True, False, True, False])
        l, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        assert l.shape == ()

    def test_kto_metrics_keys(self):
        lp = torch.tensor([-2.0, -3.0])
        rl = torch.tensor([-2.1, -2.9])
        kl = torch.tensor(0.05)
        is_d = torch.tensor([True, False])
        _, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        for k in ("loss", "reward_desirable", "reward_undesirable", "kl_estimate"):
            assert k in m

    def test_kto_trainer_step(self):
        p   = TinyLM(); r = TinyLM()
        opt = torch.optim.AdamW(p.parameters())
        t   = KTOTrainer(p, r, opt, KTOConfig())
        ids = torch.randint(0, V, (4, 6))
        is_d = torch.tensor([True, False, True, False])
        m = t.step(ids, is_d)
        assert "loss" in m


# ── RewardModel ───────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_forward_shape(self):
        rm = RewardModel(TinyLM(), d_model=V)
        ids = torch.randint(0, V, (4, 8))
        r   = rm(ids)
        assert r.shape == (4,)

    def test_preference_loss_keys(self):
        rm = RewardModel(TinyLM(), d_model=V)
        c  = torch.randint(0, V, (2, 6))
        r  = torch.randint(0, V, (2, 6))
        l, m = rm.preference_loss(c, r)
        for k in ("loss", "accuracy", "reward_margin"):
            assert k in m

    def test_accuracy_in_range(self):
        rm = RewardModel(TinyLM(), d_model=V)
        c  = torch.randint(0, V, (4, 6))
        r  = torch.randint(0, V, (4, 6))
        _, m = rm.preference_loss(c, r)
        assert 0.0 <= m["accuracy"] <= 1.0


# ── AlignmentEvaluator ────────────────────────────────────────────────────────

class TestAlignmentEvaluator:
    def test_evaluate_returns_metrics(self):
        rm   = RewardModel(TinyLM(), V)
        ev   = AlignmentEvaluator(rm, TinyLM())
        ids  = torch.randint(0, V, (4, 6))
        m    = ev.evaluate(TinyLM(), TinyLM(), ids)
        assert isinstance(m, AlignmentMetrics)

    def test_win_rate_in_range(self):
        rm   = RewardModel(TinyLM(), V)
        ev   = AlignmentEvaluator(rm, TinyLM())
        ids  = torch.randint(0, V, (4, 6))
        wr   = ev.win_rate(TinyLM(), TinyLM(), ids)
        assert 0.0 <= wr <= 1.0

    def test_reward_hacking_detection(self):
        m = AlignmentMetrics(win_rate=0.9, mean_reward=3.0, kl_div=3.0, reward_std=0.5)
        assert m.is_reward_hacking()

    def test_no_reward_hacking_normal(self):
        m = AlignmentMetrics(win_rate=0.6, mean_reward=0.5, kl_div=0.1, reward_std=0.2)
        assert not m.is_reward_hacking()

    def test_to_dict_keys(self):
        m = AlignmentMetrics(win_rate=0.7, mean_reward=1.0, kl_div=0.3, reward_std=0.2)
        d = m.to_dict()
        for k in ("win_rate", "mean_reward", "kl_div", "reward_std"):
            assert k in d
''')
commit("test: add full alignment test suite — preference, DPO, constitutional AI, KTO, reward model, evaluator")

# COMMITS 12-18
for title, body in [
    ("test: add DPOConfig invalid loss_type raises test", '''
class TestDPOConfigValidation:
    def test_invalid_loss_type(self):
        import pytest
        with pytest.raises(AssertionError):
            DPOConfig(loss_type="bad")
'''),
    ("test: add DPO chosen reward higher than rejected test", '''
class TestDPORewardOrdering:
    def test_good_pairs_have_positive_margin(self):
        # If chosen clearly better, margin should be positive
        pw = torch.tensor([-1.0] * 4)   # high chosen log-p
        pl = torch.tensor([-5.0] * 4)   # low rejected log-p
        rw = torch.tensor([-2.0] * 4)
        rl = torch.tensor([-4.0] * 4)
        _, m = dpo_loss(pw, pl, rw, rl, DPOConfig())
        assert m["reward_margin"] > 0
'''),
    ("test: add DPO reference-free mode test", '''
class TestDPOReferenceFree:
    def test_reference_free_loss(self):
        pw = torch.tensor([-2.0, -2.5])
        pl = torch.tensor([-3.0, -3.5])
        rw = torch.zeros(2)
        rl = torch.zeros(2)
        cfg = DPOConfig(reference_free=True)
        l, m = dpo_loss(pw, pl, rw, rl, cfg)
        assert l.item() >= 0
'''),
    ("test: add ConstitutionalAI custom constitution test", '''
class TestCustomConstitution:
    def test_custom_constitution(self):
        cfg = ConstitutionalConfig(constitution=["Be brief.", "Be kind."])
        cai = ConstitutionalAI(cfg)
        cr  = cai.critique_and_revise("response")
        assert cr.principle in ["Be brief.", "Be kind."]
'''),
    ("test: add KTO all desirable no error test", '''
class TestKTOAllDesirable:
    def test_all_desirable(self):
        lp = torch.tensor([-2.0, -2.5, -2.3])
        rl = torch.tensor([-2.1, -2.4, -2.2])
        kl = torch.tensor(0.05)
        is_d = torch.tensor([True, True, True])
        l, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        assert l.item() >= 0

    def test_all_undesirable(self):
        lp = torch.tensor([-3.0, -3.5])
        rl = torch.tensor([-2.9, -3.3])
        kl = torch.tensor(0.05)
        is_d = torch.tensor([False, False])
        l, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        assert l.item() >= 0
'''),
    ("test: add RewardModel gradient flows test", '''
class TestRMGradient:
    def test_gradients_flow(self):
        rm = RewardModel(TinyLM(), d_model=V)
        c  = torch.randint(0, V, (2, 4))
        r  = torch.randint(0, V, (2, 4))
        l, _ = rm.preference_loss(c, r)
        l.backward()
        has_grad = any(p.grad is not None for p in rm.parameters())
        assert has_grad
'''),
    ("test: add AlignmentMetrics compute_kl shape test", '''
class TestKLComputation:
    def test_compute_kl_returns_float(self):
        rm = RewardModel(TinyLM(), V)
        ev = AlignmentEvaluator(rm, TinyLM())
        ids = torch.randint(0, V, (2, 6))
        kl  = ev.compute_kl(TinyLM(), ids)
        assert isinstance(kl, float)
'''),
]:
    src = read("tests/test_alignment.py")
    src += "\n" + body
    write("tests/test_alignment.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.0.0 (MAJOR!)
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.9.0\"", "__version__ = \"5.0.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.0.0 — GOLDEN JUBILEE MAJOR RELEASE: 50 days, 1000+ commits, 20 subpackages")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `serving`    | Continuous Batching — PagedAttention KV cache, scheduler, prefix caching, LLM engine |",
    "| `serving`    | Continuous Batching — PagedAttention KV cache, scheduler, prefix caching, LLM engine |\n"
    "| `alignment`  | Modern Alignment — DPO/IPO, Constitutional AI, KTO, RLAIF, reward model, eval |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.0.0] 🥇 GOLDEN JUBILEE — Day 50: Constitutional AI, DPO & Modern Alignment\n\n"
      "### 🏆 Milestone: 1000+ commits, 50 days, 20 subpackages\n\n"
      "### Added\n"
      "- `PreferencePair` / `BinaryFeedback` / `PreferenceDataset` — preference data\n"
      "- `DPOConfig` / `dpo_loss` — Direct Preference Optimization (DPO/IPO/KTO-pair)\n"
      "- `compute_log_probs` — per-token log probabilities\n"
      "- `DPOTrainer` — policy + frozen reference, step(), reward margin\n"
      "- `ConstitutionalAI` — critique/revise, RLAIF ai_feedback, build_preference_dataset\n"
      "- `ConstitutionalConfig` — constitution, prompt templates\n"
      "- `KTOConfig` / `kto_loss` — Kahneman-Tversky prospect-theoretic loss\n"
      "- `KTOTrainer` — binary feedback training, estimate_kl\n"
      "- `RewardModel` — LM backbone + scalar head, preference_loss (Bradley-Terry)\n"
      "- `AlignmentEvaluator` — win_rate, compute_kl, evaluate, is_reward_hacking\n"
      "- `AlignmentMetrics` — win_rate, mean_reward, kl_div, reward_hacking\n"
      "- `examples/alignment_demo.py` — full alignment demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.0.0 GOLDEN JUBILEE — README and CHANGELOG for Day 50 Alignment milestone")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 50 🥇 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v5.0.0",
    "-m", "NanoMind v5.0.0 — Golden Jubilee: 50 Days, 1000+ Commits, 20 Subpackages", check=False)
r = run("git", "push", "origin", "v5.0.0", check=False)
print("Tag v5.0.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
total_n = int(total.stdout.strip())
print(f"\n{'🥇'*5} DAY 50 GOLDEN JUBILEE COMPLETE {'🥇'*5}")
print(f"🏆 TOTAL COMMITS: {total_n}")
print(f"📦 SUBPACKAGES:   20")
print(f"🏷️  VERSION:       v5.0.0 MAJOR RELEASE")
print("=== DAY 50 COMPLETE — v5.0.0 TAGGED! ===")
