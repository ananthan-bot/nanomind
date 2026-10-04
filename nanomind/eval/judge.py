"""
nanomind/eval/judge.py — LLM-as-a-Judge evaluation.

## LLM-as-a-Judge (Zheng et al., 2023)

Use a powerful LLM (GPT-4, Claude) to evaluate model responses.
Addresses limitations of automated metrics:
  - ROUGE/BLEU miss semantic equivalence
  - Exact match too strict for free-form answers
  - LLM judges capture nuance, helpfulness, harmlessness

## MT-Bench Protocol

80 challenging multi-turn questions across 8 categories:
  Writing, Roleplay, Reasoning, Math, Coding,
  Extraction, STEM, Humanities

Judge rates responses 1-10 on:
  - Correctness
  - Helpfulness
  - Safety
  - Format adherence

## Pairwise vs Pointwise Judging

Pointwise:  Judge rates ONE response → "Rate this response 1-10"
Pairwise:   Judge compares TWO responses → "Which is better, A or B?"
            → More reliable, avoids position bias
            → Used in: LMSYS Chatbot Arena

## Position Bias

LLM judges prefer responses in position A over B.
Mitigation: swap and average (A vs B, then B vs A).

Reference:
  Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
  https://arxiv.org/abs/2306.05685
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class JudgeResult:
    """Result from LLM-as-a-Judge evaluation."""
    question:   str
    response:   str
    score:      float    # 1-10
    reasoning:  str
    judge_model: str = "mock"
    category:   str = "general"

    @property
    def normalized_score(self) -> float:
        """Score normalized to [0, 1]."""
        return (self.score - 1) / 9.0

    def to_dict(self) -> dict:
        return {
            "score":    self.score,
            "reasoning": self.reasoning[:100],
            "category": self.category,
        }


@dataclass
class PairwiseResult:
    """Result from pairwise LLM comparison."""
    question:    str
    response_a:  str
    response_b:  str
    winner:      str     # "A" | "B" | "tie"
    score_a:     float
    score_b:     float
    reasoning:   str
    judge_model: str = "mock"

    @property
    def model_a_wins(self) -> bool:
        return self.winner == "A"

    def to_dict(self) -> dict:
        return {
            "winner":   self.winner,
            "score_a":  self.score_a,
            "score_b":  self.score_b,
            "reasoning": self.reasoning[:100],
        }


class LLMJudge:
    """
    LLM-as-a-Judge evaluator.

    Args:
        judge_fn:  Callable (prompt: str) → str. The judge LLM.
        model_name: Name of the judge model.
        swap_debiasing: Run both orderings for pairwise to reduce position bias.

    Example::

        judge   = LLMJudge(judge_fn=gpt4_fn, swap_debiasing=True)
        result  = judge.score_response(question, response)
        print(f"Score: {result.score}/10")

        pair = judge.compare(question, response_a, response_b)
        print(f"Winner: {pair.winner}")
    """

    SCORE_PROMPT = (
        "You are an expert evaluator. Rate the following response on a scale of 1-10.

"
        "Question: {question}

"
        "Response: {response}

"
        "Criteria: helpfulness, accuracy, safety, and clarity.
"
        "Provide a score (1-10) and brief reasoning.
"
        "Score:"
    )

    PAIRWISE_PROMPT = (
        "Compare the following two responses and determine which is better.

"
        "Question: {question}

"
        "Response A: {response_a}

"
        "Response B: {response_b}

"
        "Which response is better? Answer A, B, or tie, with a brief reason.
"
        "Answer:"
    )

    def __init__(
        self,
        judge_fn:       object = None,
        model_name:     str    = "mock",
        swap_debiasing: bool   = True,
    ) -> None:
        self.judge_fn       = judge_fn or self._mock_judge
        self.model_name     = model_name
        self.swap_debiasing = swap_debiasing

    def _mock_judge(self, prompt: str) -> str:
        """Mock judge for testing."""
        if "Score:" in prompt:
            return "8
This is a helpful and accurate response."
        if "Answer:" in prompt:
            return "A
Response A is more detailed and accurate."
        return "7
Decent response."

    def _parse_score(self, output: str) -> tuple[float, str]:
        """Extract numeric score and reasoning from judge output."""
        import re
        numbers = re.findall(r"([1-9]|10)", output)
        score   = float(numbers[0]) if numbers else 5.0
        # Everything after the number is reasoning
        lines     = output.strip().split("
", 1)
        reasoning = lines[1].strip() if len(lines) > 1 else output
        return score, reasoning

    def _parse_winner(self, output: str) -> str:
        """Extract A/B/tie winner from pairwise output."""
        out = output.strip().upper()
        if out.startswith("A"):
            return "A"
        if out.startswith("B"):
            return "B"
        return "tie"

    def score_response(
        self,
        question:  str,
        response:  str,
        category:  str = "general",
    ) -> JudgeResult:
        """
        Score a single response on a 1-10 scale.

        Args:
            question: The original question/instruction.
            response: Model's response to evaluate.
            category: Task category.

        Returns:
            :class:`JudgeResult`.
        """
        prompt = self.SCORE_PROMPT.format(question=question, response=response)
        output = self.judge_fn(prompt)
        score, reasoning = self._parse_score(output)
        return JudgeResult(
            question    = question,
            response    = response,
            score       = score,
            reasoning   = reasoning,
            judge_model = self.model_name,
            category    = category,
        )

    def compare(
        self,
        question:   str,
        response_a: str,
        response_b: str,
    ) -> PairwiseResult:
        """
        Pairwise comparison of two responses.

        If swap_debiasing=True, runs both orderings and averages.

        Args:
            question:   Original prompt.
            response_a: First response.
            response_b: Second response.

        Returns:
            :class:`PairwiseResult`.
        """
        prompt = self.PAIRWISE_PROMPT.format(
            question=question, response_a=response_a, response_b=response_b
        )
        out1    = self.judge_fn(prompt)
        winner1 = self._parse_winner(out1)

        score_a = self.score_response(question, response_a).score
        score_b = self.score_response(question, response_b).score

        if self.swap_debiasing:
            prompt2 = self.PAIRWISE_PROMPT.format(
                question=question, response_a=response_b, response_b=response_a
            )
            out2    = self.judge_fn(prompt2)
            winner2 = self._parse_winner(out2)
            # Flip winner2 (A in swap = B originally)
            winner2 = {"A": "B", "B": "A", "tie": "tie"}[winner2]
            # Aggregate
            if winner1 == winner2:
                winner = winner1
            else:
                # Disagreement → use scores to decide
                winner = "A" if score_a > score_b else ("B" if score_b > score_a else "tie")
        else:
            winner = winner1

        return PairwiseResult(
            question    = question,
            response_a  = response_a,
            response_b  = response_b,
            winner      = winner,
            score_a     = score_a,
            score_b     = score_b,
            reasoning   = out1.strip(),
            judge_model = self.model_name,
        )

    def batch_score(
        self,
        questions:  list[str],
        responses:  list[str],
    ) -> list[JudgeResult]:
        """Score a batch of (question, response) pairs."""
        return [
            self.score_response(q, r)
            for q, r in zip(questions, responses)
        ]

    def win_rate(
        self,
        questions:    list[str],
        responses_a:  list[str],
        responses_b:  list[str],
    ) -> dict:
        """Compute win rate of model A vs model B."""
        results = [
            self.compare(q, a, b)
            for q, a, b in zip(questions, responses_a, responses_b)
        ]
        n   = len(results)
        win_a = sum(1 for r in results if r.winner == "A")
        win_b = sum(1 for r in results if r.winner == "B")
        ties  = sum(1 for r in results if r.winner == "tie")
        return {
            "win_rate_a": win_a / max(n, 1),
            "win_rate_b": win_b / max(n, 1),
            "tie_rate":   ties  / max(n, 1),
            "n_samples":  n,
        }
