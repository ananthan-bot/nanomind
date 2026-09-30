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
        "Here is a response: '{response}'

"
        "Principle: {principle}

"
        "Identify any ways in which this response violates the principle."
    )
    revision_prompt_tmpl: str = (
        "Rewrite the following response to better follow this principle:
"
        "Principle: {principle}

"
        "Original: {response}

"
        "Critique: {critique}

"
        "Revised response:"
    )
    feedback_prompt_tmpl: str = (
        "Consider these two responses to: '{prompt}'

"
        "Response A: {response_a}
"
        "Response B: {response_b}

"
        "Principle: {principle}

"
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
