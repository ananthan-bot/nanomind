"""
nanomind/eval/task.py — Core evaluation task and sample types.

## LLM Evaluation Task Types

1. Multiple Choice (MMLU, ARC, HellaSwag, WinoGrande):
   - Model selects best answer from A/B/C/D
   - Measured by: accuracy, log-likelihood

2. Open-ended Generation (TruthfulQA, NaturalQuestions):
   - Model generates free-form answer
   - Measured by: ROUGE, BERTScore, exact match, LLM judge

3. Math Reasoning (GSM8K, MATH):
   - Model generates step-by-step solution
   - Measured by: final answer exact match

4. Coding (HumanEval, MBPP):
   - Model generates code that passes unit tests
   - Measured by: pass@k

5. Dialogue / Instruction Following (MT-Bench, AlpacaEval):
   - Multi-turn conversations or instructions
   - Measured by: LLM-as-a-judge scores (1-10)

## Evaluation Protocols

- Zero-shot: No examples in context (tests true capability)
- Few-shot:  K examples in context before test question
- Chain-of-thought: "Let's think step by step" before answering

References:
  Hendrycks et al. (2020) MMLU:        https://arxiv.org/abs/2009.03300
  Cobbe et al. (2021) GSM8K:           https://arxiv.org/abs/2110.14168
  Chen et al. (2021) HumanEval:        https://arxiv.org/abs/2107.03374
  Zheng et al. (2023) MT-Bench/Judge:  https://arxiv.org/abs/2306.05685
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    MULTIPLE_CHOICE    = "multiple_choice"
    OPEN_GENERATION    = "open_generation"
    MATH_REASONING     = "math_reasoning"
    CODE_GENERATION    = "code_generation"
    CLASSIFICATION     = "classification"
    DIALOGUE           = "dialogue"


class EvalProtocol(Enum):
    ZERO_SHOT   = "zero_shot"
    FEW_SHOT    = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class EvalSample:
    """
    A single evaluation sample.

    Args:
        sample_id:   Unique identifier.
        question:    The question or prompt.
        choices:     Answer choices (for multiple choice tasks).
        answer:      Correct answer (letter "A"-"D" or text).
        task_type:   Type of evaluation task.
        subject:     Topic/subject category.
        difficulty:  easy / medium / hard.
        metadata:    Additional metadata.

    Example::

        sample = EvalSample(
            sample_id = "mmlu_001",
            question  = "What is the capital of France?",
            choices   = ["London", "Berlin", "Paris", "Madrid"],
            answer    = "C",
            task_type = TaskType.MULTIPLE_CHOICE,
        )
    """
    sample_id:  str
    question:   str
    answer:     str
    task_type:  TaskType      = TaskType.MULTIPLE_CHOICE
    choices:    list[str]     = field(default_factory=list)
    subject:    str           = "general"
    difficulty: str           = "medium"
    explanation: str          = ""
    metadata:   dict          = field(default_factory=dict)

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    @property
    def answer_text(self) -> str:
        """Return the answer text (for multiple choice: the chosen option)."""
        if self.task_type == TaskType.MULTIPLE_CHOICE and self.choices:
            idx = ord(self.answer.upper()) - ord("A")
            if 0 <= idx < len(self.choices):
                return self.choices[idx]
        return self.answer

    def format_choices(self) -> str:
        """Format choices as 'A. choice1
B. choice2
...'"""
        letters = "ABCDEFGHIJ"
        return "
".join(f"{letters[i]}. {c}" for i, c in enumerate(self.choices))

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "question":  self.question,
            "answer":    self.answer,
            "task_type": self.task_type.value,
            "subject":   self.subject,
        }


@dataclass
class EvalTask:
    """
    A collection of evaluation samples forming a benchmark.

    Args:
        name:      Benchmark name (e.g., "MMLU", "GSM8K").
        samples:   List of :class:`EvalSample`.
        task_type: Task type for all samples.
        protocol:  Evaluation protocol (zero/few-shot).
        n_shots:   Number of few-shot examples.

    Example::

        task = EvalTask("MMLU", samples, TaskType.MULTIPLE_CHOICE,
                         protocol=EvalProtocol.FIVE_SHOT, n_shots=5)
    """
    name:      str
    samples:   list[EvalSample]
    task_type: TaskType      = TaskType.MULTIPLE_CHOICE
    protocol:  EvalProtocol  = EvalProtocol.ZERO_SHOT
    n_shots:   int           = 0
    description: str         = ""

    def __len__(self) -> int:
        return len(self.samples)

    def subjects(self) -> list[str]:
        return list(set(s.subject for s in self.samples))

    def by_subject(self) -> dict[str, list[EvalSample]]:
        out: dict = {}
        for s in self.samples:
            out.setdefault(s.subject, []).append(s)
        return out

    def by_difficulty(self) -> dict[str, list[EvalSample]]:
        out: dict = {}
        for s in self.samples:
            out.setdefault(s.difficulty, []).append(s)
        return out

    def to_dict(self) -> dict:
        return {
            "name":      self.name,
            "n_samples": len(self),
            "task_type": self.task_type.value,
            "protocol":  self.protocol.value,
            "subjects":  self.subjects(),
        }
