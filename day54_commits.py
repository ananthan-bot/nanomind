"""
day54_commits.py — 20 atomic commits for Day 54: Evaluation & Benchmarking Suite.
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

print("\n=== DAY 54: Evaluation & Benchmarking Suite — 20 commits, v5.4.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — eval package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/__init__.py",
      '"""NanoMind Eval sub-package — Evaluation & Benchmarking Suite."""\n')
commit("feat: add nanomind/eval/ package skeleton for evaluation and benchmarking")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Core task and metric types
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/task.py", '''\
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
        """Format choices as 'A. choice1\nB. choice2\n...'"""
        letters = "ABCDEFGHIJ"
        return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(self.choices))

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
''')
commit("feat: add EvalSample, EvalTask, TaskType, EvalProtocol — core evaluation data structures")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Benchmark generators (MMLU, GSM8K, HellaSwag style)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/benchmarks.py", '''\
"""
nanomind/eval/benchmarks.py — Synthetic benchmark generators.

Generates MMLU-style, GSM8K-style, HellaSwag-style, and
TruthfulQA-style eval samples for testing the evaluation pipeline.

In production, these would load from Hugging Face datasets:
  datasets.load_dataset("cais/mmlu", "all")
  datasets.load_dataset("gsm8k", "main")
"""

from __future__ import annotations
import random
from nanomind.eval.task import EvalSample, EvalTask, TaskType, EvalProtocol


# ── MMLU-style multiple choice ─────────────────────────────────────────────────

MMLU_SAMPLES = [
    # STEM
    ("What is the derivative of x²?",
     ["x", "2x", "x²/2", "2"],            "B", "math"),
    ("Which planet is closest to the Sun?",
     ["Venus", "Earth", "Mercury", "Mars"], "C", "science"),
    ("What does DNA stand for?",
     ["Deoxyribonucleic acid", "Dynamic Nucleotide Amplifier",
      "Digital Neuronal Activator", "Dense Nucleic Array"],
     "A", "biology"),
    ("What is the speed of light (approx)?",
     ["3×10⁸ m/s", "3×10⁶ m/s", "3×10¹⁰ m/s", "3×10⁴ m/s"],
     "A", "physics"),
    ("What is log₂(8)?",
     ["2", "3", "4", "8"],                 "B", "math"),
    # Humanities
    ("In what year did World War II end?",
     ["1943", "1944", "1945", "1946"],     "C", "history"),
    ("Who wrote 'Pride and Prejudice'?",
     ["Charlotte Brontë", "Jane Austen", "George Eliot", "Virginia Woolf"],
     "B", "literature"),
    ("Which country was Napoleon Bonaparte from?",
     ["Italy", "Spain", "France", "Austria"],  "C", "history"),
    # Social science
    ("What is GDP an acronym for?",
     ["Gross Domestic Product", "General Data Protocol",
      "Global Development Plan", "Government Debt Position"],
     "A", "economics"),
    ("Who proposed the theory of evolution?",
     ["Isaac Newton", "Gregor Mendel", "Charles Darwin", "Albert Einstein"],
     "C", "biology"),
]


def make_mmlu_task(n: int = 10, seed: int = 42) -> EvalTask:
    """Generate MMLU-style multiple choice task."""
    rng     = random.Random(seed)
    pool    = list(MMLU_SAMPLES) * ((n // len(MMLU_SAMPLES)) + 1)
    pool    = pool[:n]
    rng.shuffle(pool)
    samples = []
    for i, (q, choices, ans, subj) in enumerate(pool):
        samples.append(EvalSample(
            sample_id = f"mmlu_{i:04d}",
            question  = q,
            choices   = choices,
            answer    = ans,
            task_type = TaskType.MULTIPLE_CHOICE,
            subject   = subj,
        ))
    return EvalTask(
        name        = "MMLU",
        samples     = samples,
        task_type   = TaskType.MULTIPLE_CHOICE,
        protocol    = EvalProtocol.ZERO_SHOT,
        description = "Massive Multitask Language Understanding",
    )


# ── GSM8K-style math reasoning ─────────────────────────────────────────────────

GSM8K_SAMPLES = [
    ("Janet has 3 apples. She buys 5 more. How many does she have?",
     "8", "3 + 5 = 8"),
    ("A train travels 60 mph for 2 hours. How far does it travel?",
     "120", "60 × 2 = 120 miles"),
    ("If a pizza has 8 slices and 3 people each eat 2 slices, how many are left?",
     "2", "8 - 3×2 = 8 - 6 = 2"),
    ("A store has 100 items. 40% are on sale. How many items are on sale?",
     "40", "100 × 0.40 = 40"),
    ("Mark runs 5 miles every day for a week. How many miles total?",
     "35", "5 × 7 = 35 miles"),
    ("A rectangle is 8m long and 4m wide. What is its area?",
     "32", "8 × 4 = 32 m²"),
    ("If 12 eggs cost $3, how much do 4 eggs cost?",
     "1", "$3/12 × 4 = $1"),
    ("A class has 30 students. 6 are absent. What fraction are present?",
     "4/5", "(30-6)/30 = 24/30 = 4/5"),
]


def make_gsm8k_task(n: int = 8, seed: int = 42) -> EvalTask:
    """Generate GSM8K-style math reasoning task."""
    rng     = random.Random(seed)
    pool    = list(GSM8K_SAMPLES) * ((n // len(GSM8K_SAMPLES)) + 1)
    pool    = pool[:n]
    samples = [
        EvalSample(
            sample_id   = f"gsm8k_{i:04d}",
            question    = q,
            answer      = ans,
            explanation = sol,
            task_type   = TaskType.MATH_REASONING,
            subject     = "math",
        )
        for i, (q, ans, sol) in enumerate(pool[:n])
    ]
    return EvalTask(
        name        = "GSM8K",
        samples     = samples,
        task_type   = TaskType.MATH_REASONING,
        protocol    = EvalProtocol.CHAIN_OF_THOUGHT,
        n_shots     = 8,
        description = "Grade School Math 8K",
    )


# ── HellaSwag-style commonsense ─────────────────────────────────────────────────

HELLASWAG_SAMPLES = [
    ("A woman is cooking pasta. She boils water and adds pasta.",
     ["She then drains the pasta and adds sauce.",
      "She starts painting the wall.",
      "She goes to sleep immediately.",
      "She calls the fire department."],
     "A", "commonsense"),
    ("A man is teaching a child to ride a bike. He holds the seat.",
     ["He lets go and the child rides on their own.",
      "He throws the bike away.",
      "He starts singing loudly.",
      "He falls asleep standing."],
     "A", "commonsense"),
    ("The programmer opens their code editor and types a function.",
     ["They test the function and fix any bugs.",
      "They throw the computer out the window.",
      "They start baking a cake.",
      "They immediately retire."],
     "A", "commonsense"),
]


def make_hellaswag_task(n: int = 6) -> EvalTask:
    """Generate HellaSwag-style commonsense task."""
    pool    = list(HELLASWAG_SAMPLES) * ((n // len(HELLASWAG_SAMPLES)) + 1)
    samples = [
        EvalSample(
            sample_id = f"hella_{i:04d}",
            question  = q,
            choices   = c,
            answer    = a,
            task_type = TaskType.MULTIPLE_CHOICE,
            subject   = subj,
        )
        for i, (q, c, a, subj) in enumerate(pool[:n])
    ]
    return EvalTask(
        name      = "HellaSwag",
        samples   = samples,
        task_type = TaskType.MULTIPLE_CHOICE,
        protocol  = EvalProtocol.ZERO_SHOT,
    )


# ── TruthfulQA-style ───────────────────────────────────────────────────────────

TRUTHFUL_SAMPLES = [
    ("Is it possible to fold a piece of paper in half more than 7 times?",
     "Yes, with large enough paper or industrial equipment it is possible.",
     ["No, it is physically impossible.", "Yes, it is possible with large enough paper.",
      "Only in zero gravity.", "Only with special paper."],
     "B"),
    ("Do humans only use 10% of their brains?",
     "No, humans use virtually all of their brain.",
     ["Yes, we only use 10%.", "No, we use virtually all of it.",
      "Yes, but only during sleep.", "Only 10% is active at once."],
     "B"),
    ("Did Napoleon Bonaparte have very short stature?",
     "No, Napoleon was average height for his time, about 5ft 7in.",
     ["Yes, he was only 5ft 2in.", "No, he was actually average height.",
      "He was very tall.", "He was exactly 6ft."],
     "B"),
]


def make_truthfulqa_task(n: int = 6) -> EvalTask:
    """Generate TruthfulQA-style task."""
    pool    = list(TRUTHFUL_SAMPLES) * ((n // len(TRUTHFUL_SAMPLES)) + 1)
    samples = [
        EvalSample(
            sample_id   = f"tqa_{i:04d}",
            question    = q,
            answer      = a,
            explanation = exp,
            choices     = c,
            task_type   = TaskType.MULTIPLE_CHOICE,
            subject     = "factual",
        )
        for i, (q, exp, c, a) in enumerate(pool[:n])
    ]
    return EvalTask(
        name      = "TruthfulQA",
        samples   = samples,
        task_type = TaskType.MULTIPLE_CHOICE,
        protocol  = EvalProtocol.ZERO_SHOT,
    )
''')
commit("feat: add make_mmlu_task, make_gsm8k_task, make_hellaswag_task, make_truthfulqa_task — benchmark generators")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Metrics
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/metrics.py", '''\
"""
nanomind/eval/metrics.py — Evaluation metrics for LLM benchmarks.

## Metrics for Different Task Types

Multiple Choice:
  - Accuracy: fraction of correct answers
  - Normalized accuracy: corrected for chance (macro-averaged across subjects)

Open Generation:
  - Exact Match (EM): does prediction == ground truth (after normalization)?
  - F1: token-level overlap between prediction and reference
  - ROUGE-L: longest common subsequence F1
  - BERTScore: semantic similarity via embeddings (requires BERT)

Math Reasoning:
  - Final Answer Exact Match: extract final number, check equality

Code Generation:
  - pass@k (Day 52): probability at least 1 of k samples passes tests

Instruction Following:
  - LLM-as-a-Judge: GPT-4 / Claude scores responses 1-10

## Perplexity

Measure how well a language model predicts a text corpus.
Lower perplexity = better model.

  PPL = exp(-1/N × Σ log p(xᵢ | x₁...xᵢ₋₁))

For context:
  GPT-2:    perplexity ~29 on WikiText-103
  GPT-3:    perplexity ~20
  LLaMA-70B: perplexity ~3.0 on some benchmarks
"""

from __future__ import annotations
import re
import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    metric_name: str
    value:       float
    n_samples:   int
    details:     dict = None

    def to_dict(self) -> dict:
        return {
            "metric":    self.metric_name,
            "value":     round(self.value, 4),
            "n_samples": self.n_samples,
        }


def exact_match(prediction: str, reference: str, normalize: bool = True) -> bool:
    """Check if prediction exactly matches reference (after normalization)."""
    if normalize:
        prediction = _normalize_text(prediction)
        reference  = _normalize_text(reference)
    return prediction == reference


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 between prediction and reference.

    Used for QA tasks where exact match is too strict.
    """
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens  = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    precision = common / max(len(pred_tokens), 1)
    recall    = common / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L: F1 based on Longest Common Subsequence."""
    pred = _normalize_text(prediction).split()
    ref  = _normalize_text(reference).split()
    lcs  = _lcs_length(pred, ref)
    if not pred or not ref:
        return 0.0
    p = lcs / len(pred)
    r = lcs / len(ref)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _lcs_length(a: list, b: list) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_number(text: str) -> str | None:
    """Extract the last number from a math reasoning response."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else None


def math_exact_match(prediction: str, reference: str) -> bool:
    """Check if math answer matches (extracts final number)."""
    pred_num = extract_number(prediction)
    ref_num  = extract_number(reference)
    if pred_num is None or ref_num is None:
        return exact_match(prediction, reference)
    try:
        return abs(float(pred_num) - float(ref_num)) < 1e-6
    except ValueError:
        return pred_num == ref_num


def multiple_choice_accuracy(
    predictions: list[str],
    references:  list[str],
) -> MetricResult:
    """Compute accuracy for multiple choice predictions."""
    n       = len(predictions)
    correct = sum(1 for p, r in zip(predictions, references)
                  if p.strip().upper()[:1] == r.strip().upper()[:1])
    return MetricResult(
        metric_name = "accuracy",
        value       = correct / max(n, 1),
        n_samples   = n,
        details     = {"n_correct": correct, "n_total": n},
    )


def generation_metrics(
    predictions: list[str],
    references:  list[str],
    task_type:   str = "open",
) -> dict[str, MetricResult]:
    """Compute all generation metrics for a list of predictions."""
    n = len(predictions)
    em_scores     = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores     = [token_f1(p, r) for p, r in zip(predictions, references)]
    rouge_scores  = [rouge_l(p, r)   for p, r in zip(predictions, references)]

    return {
        "exact_match": MetricResult("exact_match", sum(em_scores)/n, n),
        "token_f1":    MetricResult("token_f1",    sum(f1_scores)/n, n),
        "rouge_l":     MetricResult("rouge_l",     sum(rouge_scores)/n, n),
    }


@torch.no_grad()
def perplexity(
    model:     nn.Module,
    input_ids: torch.Tensor,
    stride:    int = 512,
) -> float:
    """
    Compute perplexity of a language model on a token sequence.

    Uses strided evaluation to handle sequences longer than context window.

    Args:
        model:     LM model returning ``(B, T, V)`` logits.
        input_ids: ``(1, T)`` token IDs.
        stride:    Evaluation stride (overlap between windows).

    Returns:
        Perplexity (scalar).
    """
    import torch.nn.functional as F
    T        = input_ids.shape[1]
    nll_sum  = 0.0
    n_tokens = 0

    for begin in range(0, T - 1, stride):
        end       = min(begin + stride * 2, T)
        ids       = input_ids[:, begin:end]
        target    = ids[:, 1:]
        inp       = ids[:, :-1]

        out    = model(inp)
        logits = out[0] if isinstance(out, tuple) else out   # (1, t-1, V)
        log_p  = F.log_softmax(logits, dim=-1)

        # Gather target log probs
        tgt_lp = log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, t-1)
        nll_sum  += -tgt_lp.sum().item()
        n_tokens += tgt_lp.numel()

    return math.exp(nll_sum / max(n_tokens, 1))
''')
commit("feat: add exact_match, token_f1, rouge_l, math_exact_match, multiple_choice_accuracy, perplexity")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — LLM-as-a-Judge
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/judge.py", '''\
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
        "You are an expert evaluator. Rate the following response on a scale of 1-10.\n\n"
        "Question: {question}\n\n"
        "Response: {response}\n\n"
        "Criteria: helpfulness, accuracy, safety, and clarity.\n"
        "Provide a score (1-10) and brief reasoning.\n"
        "Score:"
    )

    PAIRWISE_PROMPT = (
        "Compare the following two responses and determine which is better.\n\n"
        "Question: {question}\n\n"
        "Response A: {response_a}\n\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Answer A, B, or tie, with a brief reason.\n"
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
            return "8\nThis is a helpful and accurate response."
        if "Answer:" in prompt:
            return "A\nResponse A is more detailed and accurate."
        return "7\nDecent response."

    def _parse_score(self, output: str) -> tuple[float, str]:
        """Extract numeric score and reasoning from judge output."""
        import re
        numbers = re.findall(r"\b([1-9]|10)\b", output)
        score   = float(numbers[0]) if numbers else 5.0
        # Everything after the number is reasoning
        lines     = output.strip().split("\n", 1)
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
''')
commit("feat: add LLMJudge — score_response (1-10), pairwise compare, swap_debiasing, win_rate, batch_score")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Evaluator (runs benchmark + collects results)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/evaluator.py", '''\
"""
nanomind/eval/evaluator.py — Benchmark evaluator: runs tasks, collects results.

Ties together task + metrics + optional LLM judge to produce
a comprehensive benchmark report.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from nanomind.eval.task import EvalSample, EvalTask, TaskType
from nanomind.eval.metrics import (
    multiple_choice_accuracy, generation_metrics,
    math_exact_match, MetricResult,
)
from nanomind.utils.logger import get_logger

log = get_logger("eval.evaluator")


@dataclass
class SampleResult:
    """Result for a single sample."""
    sample:     EvalSample
    prediction: str
    correct:    bool
    score:      float   # 0-1
    time_s:     float

    def to_dict(self) -> dict:
        return {
            "sample_id":  self.sample.sample_id,
            "correct":    self.correct,
            "score":      round(self.score, 4),
            "prediction": self.prediction[:80],
            "reference":  self.sample.answer[:80],
        }


@dataclass
class TaskResult:
    """Aggregated results for an entire evaluation task."""
    task_name:   str
    n_samples:   int
    accuracy:    float
    metrics:     dict[str, float] = field(default_factory=dict)
    by_subject:  dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0
    sample_results: list[SampleResult] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.sample_results if r.correct)

    def to_dict(self) -> dict:
        return {
            "task":      self.task_name,
            "accuracy":  round(self.accuracy, 4),
            "n_samples": self.n_samples,
            "metrics":   {k: round(v, 4) for k, v in self.metrics.items()},
            "by_subject": {k: round(v, 4) for k, v in self.by_subject.items()},
            "wall_time_s": round(self.wall_time_s, 2),
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report across multiple tasks."""
    model_name:  str
    tasks:       list[TaskResult]
    wall_time_s: float = 0.0

    @property
    def overall_accuracy(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.accuracy for t in self.tasks) / len(self.tasks)

    @property
    def total_samples(self) -> int:
        return sum(t.n_samples for t in self.tasks)

    def to_dict(self) -> dict:
        return {
            "model":            self.model_name,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "total_samples":    self.total_samples,
            "n_tasks":          len(self.tasks),
            "tasks":            [t.to_dict() for t in self.tasks],
            "wall_time_s":      round(self.wall_time_s, 2),
        }


class BenchmarkEvaluator:
    """
    Runs evaluation benchmarks against a model.

    Args:
        model_fn:   Callable (sample: EvalSample) → str prediction.
        model_name: Name of the model being evaluated.

    Example::

        evaluator = BenchmarkEvaluator(model_fn=my_model, model_name="NanoMind-v5")
        task      = make_mmlu_task(n=100)
        result    = evaluator.evaluate_task(task)
        print(f"Accuracy: {result.accuracy:.2%}")

        report = evaluator.evaluate_all([mmlu, gsm8k, hellaswag])
        print(f"Overall: {report.overall_accuracy:.2%}")
    """

    def __init__(
        self,
        model_fn:   object,
        model_name: str = "model",
    ) -> None:
        self.model_fn   = model_fn
        self.model_name = model_name

    def _predict(self, sample: EvalSample) -> str:
        """Get model prediction for a sample."""
        return self.model_fn(sample)

    def _score_sample(self, sample: EvalSample, pred: str) -> tuple[bool, float]:
        """Score a prediction against the ground truth."""
        if sample.task_type == TaskType.MULTIPLE_CHOICE:
            correct = pred.strip().upper()[:1] == sample.answer.strip().upper()[:1]
            return correct, float(correct)

        elif sample.task_type == TaskType.MATH_REASONING:
            correct = math_exact_match(pred, sample.answer)
            return correct, float(correct)

        elif sample.task_type == TaskType.OPEN_GENERATION:
            from nanomind.eval.metrics import token_f1
            score   = token_f1(pred, sample.answer)
            return score > 0.5, score

        else:
            from nanomind.eval.metrics import exact_match
            correct = exact_match(pred, sample.answer)
            return correct, float(correct)

    def evaluate_task(self, task: EvalTask) -> TaskResult:
        """
        Evaluate a single task.

        Args:
            task: :class:`EvalTask` with samples.

        Returns:
            :class:`TaskResult`.
        """
        t0            = time.monotonic()
        sample_results = []

        for sample in task.samples:
            ts   = time.monotonic()
            pred = self._predict(sample)
            correct, score = self._score_sample(sample, pred)
            sample_results.append(SampleResult(
                sample     = sample,
                prediction = pred,
                correct    = correct,
                score      = score,
                time_s     = time.monotonic() - ts,
            ))

        n   = len(sample_results)
        acc = sum(r.score for r in sample_results) / max(n, 1)

        # Per-subject accuracy
        by_subj: dict = {}
        for r in sample_results:
            s = r.sample.subject
            by_subj.setdefault(s, []).append(r.score)
        by_subject = {s: sum(v)/len(v) for s, v in by_subj.items()}

        return TaskResult(
            task_name      = task.name,
            n_samples      = n,
            accuracy       = acc,
            metrics        = {"accuracy": acc},
            by_subject     = by_subject,
            wall_time_s    = time.monotonic() - t0,
            sample_results = sample_results,
        )

    def evaluate_all(self, tasks: list[EvalTask]) -> BenchmarkReport:
        """Evaluate multiple tasks and produce a combined report."""
        t0      = time.monotonic()
        results = []
        for task in tasks:
            log.info(f"Evaluating {task.name} ({len(task)} samples)...")
            results.append(self.evaluate_task(task))
        return BenchmarkReport(
            model_name  = self.model_name,
            tasks       = results,
            wall_time_s = time.monotonic() - t0,
        )
''')
commit("feat: add BenchmarkEvaluator, SampleResult, TaskResult, BenchmarkReport — run tasks, collect metrics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Subject-level leaderboard
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/leaderboard.py", '''\
"""
nanomind/eval/leaderboard.py — Model leaderboard and comparison utilities.

Compare multiple models on the same benchmarks.
Generate ranking tables, Elo ratings, and radar charts.

## Elo Rating System

Used in LMSYS Chatbot Arena to rank LLMs:
  K = Elo update factor (default 32)
  Expected score: E_A = 1 / (1 + 10^((R_B - R_A)/400))
  Update: R_A += K × (S_A - E_A)  where S_A = 1 if A wins, 0.5 if tie, 0 if loss

Starting Elo: 1000 for all models
Higher Elo = better model.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelScore:
    """A model's score on a benchmark suite."""
    model_name:  str
    scores:      dict[str, float]   # task → accuracy
    elo:         float = 1000.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / max(len(self.scores), 1)

    def to_dict(self) -> dict:
        return {
            "model":      self.model_name,
            "mean_score": round(self.mean_score, 4),
            "elo":        round(self.elo, 1),
            **{k: round(v, 4) for k, v in self.scores.items()},
        }


class Leaderboard:
    """
    Model leaderboard with Elo-based ranking.

    Args:
        task_names: Names of evaluation tasks.

    Example::

        board = Leaderboard(["MMLU", "GSM8K", "HellaSwag"])
        board.add_model("GPT-4",     {"MMLU": 0.86, "GSM8K": 0.92, "HellaSwag": 0.96})
        board.add_model("LLaMA-70B", {"MMLU": 0.79, "GSM8K": 0.77, "HellaSwag": 0.87})
        print(board.ranking())
    """

    def __init__(self, task_names: list[str]) -> None:
        self.task_names = task_names
        self._models:  list[ModelScore] = []

    def add_model(self, name: str, scores: dict[str, float]) -> None:
        """Add a model's scores to the leaderboard."""
        self._models.append(ModelScore(model_name=name, scores=scores))

    def ranking(self) -> list[ModelScore]:
        """Return models sorted by mean score (descending)."""
        return sorted(self._models, key=lambda m: -m.mean_score)

    def update_elo(self, winner: str, loser: str, k: float = 32.0) -> None:
        """Update Elo ratings after a pairwise comparison."""
        def find(name):
            for m in self._models:
                if m.model_name == name:
                    return m
            return None

        w = find(winner)
        l = find(loser)
        if w is None or l is None:
            return

        ea = 1.0 / (1.0 + 10 ** ((l.elo - w.elo) / 400.0))
        eb = 1.0 - ea
        w.elo += k * (1.0 - ea)
        l.elo += k * (0.0 - eb)

    def leaderboard_table(self) -> list[dict]:
        """Return full leaderboard as list of dicts."""
        ranked = self.ranking()
        return [
            {"rank": i + 1, **m.to_dict()}
            for i, m in enumerate(ranked)
        ]

    def best_model(self) -> ModelScore | None:
        """Return the top-ranked model."""
        ranked = self.ranking()
        return ranked[0] if ranked else None

    def task_comparison(self, task: str) -> list[tuple[str, float]]:
        """Compare all models on a specific task."""
        result = [(m.model_name, m.scores.get(task, 0.0)) for m in self._models]
        return sorted(result, key=lambda x: -x[1])
''')
commit("feat: add Leaderboard, ModelScore, Elo rating — ranking, update_elo, leaderboard_table, best_model")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — eval __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/eval/__init__.py", '''\
"""NanoMind Eval sub-package — Evaluation & Benchmarking Suite.

Implements comprehensive LLM evaluation infrastructure:
  1. EvalSample / EvalTask         — core data structures
  2. TaskType / EvalProtocol       — task and protocol enums
  3. make_mmlu_task                — MMLU-style multiple choice benchmark
  4. make_gsm8k_task               — GSM8K math reasoning benchmark
  5. make_hellaswag_task           — HellaSwag commonsense benchmark
  6. make_truthfulqa_task          — TruthfulQA factuality benchmark
  7. exact_match / token_f1        — generation metrics
  8. rouge_l                       — ROUGE-L LCS F1 metric
  9. math_exact_match              — number extraction + comparison
  10. multiple_choice_accuracy     — MC accuracy metric
  11. generation_metrics           — all generation metrics at once
  12. perplexity                   — LM perplexity evaluation
  13. LLMJudge                     — LLM-as-a-Judge (1-10 scoring)
  14. JudgeResult / PairwiseResult — judge output types
  15. BenchmarkEvaluator           — run tasks, score samples
  16. SampleResult / TaskResult    — per-sample and per-task results
  17. BenchmarkReport              — full benchmark report
  18. Leaderboard / ModelScore     — model comparison and Elo rating

Primary exports:
    - :class:`EvalSample`              — question, choices, answer
    - :class:`EvalTask`                — collection of samples
    - :class:`TaskType`                — MULTIPLE_CHOICE etc.
    - :func:`make_mmlu_task`           — MMLU benchmark generator
    - :func:`make_gsm8k_task`          — GSM8K math generator
    - :func:`make_hellaswag_task`      — HellaSwag generator
    - :func:`make_truthfulqa_task`     — TruthfulQA generator
    - :func:`exact_match`              — EM metric
    - :func:`token_f1`                 — token-level F1
    - :func:`rouge_l`                  — ROUGE-L metric
    - :func:`math_exact_match`         — extract number + compare
    - :func:`multiple_choice_accuracy` — MC accuracy
    - :func:`generation_metrics`       — all generation metrics
    - :func:`perplexity`               — LM perplexity
    - :class:`LLMJudge`                — LLM-as-a-Judge evaluator
    - :class:`JudgeResult`             — score + reasoning
    - :class:`PairwiseResult`          — A vs B comparison
    - :class:`BenchmarkEvaluator`      — run, score, report
    - :class:`TaskResult`              — accuracy + per-subject
    - :class:`BenchmarkReport`         — overall + per-task
    - :class:`Leaderboard`             — ranking + Elo
    - :class:`ModelScore`              — per-model scores
"""

from nanomind.eval.task import EvalSample, EvalTask, TaskType, EvalProtocol
from nanomind.eval.benchmarks import (
    make_mmlu_task, make_gsm8k_task, make_hellaswag_task, make_truthfulqa_task,
)
from nanomind.eval.metrics import (
    exact_match, token_f1, rouge_l, math_exact_match,
    multiple_choice_accuracy, generation_metrics, perplexity,
    MetricResult, extract_number,
)
from nanomind.eval.judge import LLMJudge, JudgeResult, PairwiseResult
from nanomind.eval.evaluator import (
    BenchmarkEvaluator, SampleResult, TaskResult, BenchmarkReport,
)
from nanomind.eval.leaderboard import Leaderboard, ModelScore

__all__ = [
    "EvalSample", "EvalTask", "TaskType", "EvalProtocol",
    "make_mmlu_task", "make_gsm8k_task", "make_hellaswag_task", "make_truthfulqa_task",
    "exact_match", "token_f1", "rouge_l", "math_exact_match",
    "multiple_choice_accuracy", "generation_metrics", "perplexity",
    "MetricResult", "extract_number",
    "LLMJudge", "JudgeResult", "PairwiseResult",
    "BenchmarkEvaluator", "SampleResult", "TaskResult", "BenchmarkReport",
    "Leaderboard", "ModelScore",
]
''')
commit("refactor: export all eval components from nanomind/eval/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/eval_demo.py", '''\
"""
examples/eval_demo.py — NanoMind Evaluation & Benchmarking demo.

Usage:
    python examples/eval_demo.py
"""
import torch
import torch.nn as nn
from nanomind.eval import (
    EvalSample, EvalTask, TaskType, EvalProtocol,
    make_mmlu_task, make_gsm8k_task, make_hellaswag_task, make_truthfulqa_task,
    exact_match, token_f1, rouge_l, math_exact_match,
    multiple_choice_accuracy, generation_metrics, perplexity,
    LLMJudge, JudgeResult, PairwiseResult,
    BenchmarkEvaluator, SampleResult, TaskResult, BenchmarkReport,
    Leaderboard, ModelScore,
)

V = 32

class TinyLM(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.rnn  = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None

print("=" * 60)
print("NanoMind Evaluation & Benchmarking Demo")
print("=" * 60)

# ── Benchmarks ────────────────────────────────────────────────────────────────
print("\n── Benchmark Tasks ──")
mmlu     = make_mmlu_task(n=10)
gsm8k    = make_gsm8k_task(n=8)
hella    = make_hellaswag_task(n=6)
truthful = make_truthfulqa_task(n=6)

for task in [mmlu, gsm8k, hella, truthful]:
    print(f"  {task.name}: {len(task)} samples, "
          f"type={task.task_type.value}, "
          f"subjects={task.subjects()[:2]}")

# ── EvalSample ────────────────────────────────────────────────────────────────
print("\n── EvalSample ──")
s = mmlu.samples[0]
print(f"  ID: {s.sample_id}")
print(f"  Q:  {s.question}")
print(f"  Choices:\n{s.format_choices()}")
print(f"  Answer: {s.answer} → '{s.answer_text}'")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n── Metrics ──")
print(f"  exact_match('Paris', 'Paris'):   {exact_match('Paris', 'Paris')}")
print(f"  exact_match('Paris', 'London'):  {exact_match('Paris', 'London')}")
print(f"  token_f1('the cat sat', 'cat sat mat'): {token_f1('the cat sat', 'cat sat mat'):.3f}")
print(f"  rouge_l('the cat', 'the cat sat'):      {rouge_l('the cat', 'the cat sat'):.3f}")
print(f"  math_exact_match('The answer is 42', '42'): {math_exact_match('The answer is 42', '42')}")

preds = ["A", "B", "A", "C", "A"]
refs  = ["A", "B", "C", "C", "B"]
acc   = multiple_choice_accuracy(preds, refs)
print(f"  MC accuracy: {acc.value:.2%} ({acc.to_dict()})")

gen_m = generation_metrics(["the cat sat on the mat"],
                             ["the cat sat on the mat"])
print(f"  Generation metrics: {gen_m}")

# Perplexity
model = TinyLM()
ids   = torch.randint(0, V, (1, 32))
ppl   = perplexity(model, ids)
print(f"  Perplexity (random model): {ppl:.2f}")

# ── LLM-as-a-Judge ────────────────────────────────────────────────────────────
print("\n── LLM-as-a-Judge ──")
judge = LLMJudge(swap_debiasing=True)

result = judge.score_response(
    "What is the capital of France?",
    "The capital of France is Paris.",
)
print(f"  Score: {result.score}/10, Normalized: {result.normalized_score:.2%}")

pair = judge.compare(
    "Explain what DNA is.",
    "DNA is deoxyribonucleic acid, containing genetic information.",
    "I don't know.",
)
print(f"  Pairwise: Winner={pair.winner}, A={pair.score_a}/10, B={pair.score_b}/10")

win_rates = judge.win_rate(
    ["Q1", "Q2", "Q3"],
    ["Good A1", "Good A2", "Good A3"],
    ["Bad B1", "Bad B2", "Bad B3"],
)
print(f"  Win rates: {win_rates}")

# ── Benchmark Evaluator ───────────────────────────────────────────────────────
print("\n── Benchmark Evaluator ──")

# Mock model: always answers "A" for MC, "42" for math
def mock_model_fn(sample: EvalSample) -> str:
    if sample.task_type == TaskType.MULTIPLE_CHOICE:
        return "A"
    if sample.task_type == TaskType.MATH_REASONING:
        return sample.answer   # perfect oracle for demo
    return "Paris"

evaluator = BenchmarkEvaluator(mock_model_fn, model_name="MockModel")
for task in [mmlu, gsm8k, hella]:
    result = evaluator.evaluate_task(task)
    print(f"  {task.name}: acc={result.accuracy:.2%}, "
          f"by_subject={list(result.by_subject.items())[:2]}")

report = evaluator.evaluate_all([mmlu, gsm8k, hella])
print(f"\n  Overall report:")
print(f"    Model: {report.model_name}")
print(f"    Overall accuracy: {report.overall_accuracy:.2%}")
print(f"    Total samples: {report.total_samples}")
for t in report.tasks:
    print(f"    {t.task_name}: {t.accuracy:.2%}")

# ── Leaderboard ───────────────────────────────────────────────────────────────
print("\n── Model Leaderboard ──")
board = Leaderboard(["MMLU", "GSM8K", "HellaSwag", "TruthfulQA"])
board.add_model("GPT-4",       {"MMLU": 0.86, "GSM8K": 0.92, "HellaSwag": 0.96, "TruthfulQA": 0.71})
board.add_model("LLaMA-70B",   {"MMLU": 0.79, "GSM8K": 0.77, "HellaSwag": 0.87, "TruthfulQA": 0.52})
board.add_model("Mistral-7B",  {"MMLU": 0.63, "GSM8K": 0.52, "HellaSwag": 0.81, "TruthfulQA": 0.44})
board.add_model("NanoMind-v5", {"MMLU": 0.35, "GSM8K": 0.21, "HellaSwag": 0.52, "TruthfulQA": 0.38})

for entry in board.leaderboard_table():
    print(f"  #{entry['rank']}: {entry['model']:15s} "
          f"avg={entry['mean_score']:.2%} "
          f"Elo={entry['elo']:.0f}")

# Elo update from pairwise comparison
board.update_elo("GPT-4", "LLaMA-70B", k=32)
print(f"\n  After GPT-4 beats LLaMA-70B:")
best = board.best_model()
print(f"  Best model: {best.model_name} (Elo={best.elo:.1f})")

print(f"\n  Task comparison MMLU:")
for name, score in board.task_comparison("MMLU"):
    print(f"    {name}: {score:.2%}")

print("\nEvaluation demo complete!")
''')
commit("feat: add examples/eval_demo.py — metrics, judge, evaluator, benchmark, leaderboard")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_eval.py", '''\
"""tests/test_eval.py — Tests for NanoMind eval package."""
import pytest
import torch
import torch.nn as nn
from nanomind.eval import (
    EvalSample, EvalTask, TaskType, EvalProtocol,
    make_mmlu_task, make_gsm8k_task, make_hellaswag_task, make_truthfulqa_task,
    exact_match, token_f1, rouge_l, math_exact_match,
    multiple_choice_accuracy, generation_metrics, perplexity,
    LLMJudge, JudgeResult, PairwiseResult,
    BenchmarkEvaluator, SampleResult, TaskResult, BenchmarkReport,
    Leaderboard, ModelScore,
)

V = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None


# ── EvalSample / EvalTask ──────────────────────────────────────────────────────

class TestEvalSample:
    def test_answer_text_mc(self):
        s = EvalSample("id", "Q?", "B", TaskType.MULTIPLE_CHOICE,
                        choices=["X", "Y", "Z"])
        assert s.answer_text == "Y"

    def test_format_choices(self):
        s = EvalSample("id", "Q?", "A", choices=["opt1", "opt2"])
        fmt = s.format_choices()
        assert "A. opt1" in fmt and "B. opt2" in fmt

    def test_to_dict(self):
        s = EvalSample("id1", "Q?", "A")
        d = s.to_dict()
        assert "sample_id" in d and "answer" in d


class TestEvalTask:
    def test_len(self):
        t = make_mmlu_task(n=5)
        assert len(t) == 5

    def test_subjects(self):
        t = make_mmlu_task(n=10)
        assert len(t.subjects()) > 0

    def test_by_subject(self):
        t = make_mmlu_task(n=10)
        d = t.by_subject()
        assert all(isinstance(v, list) for v in d.values())


# ── Benchmark Generators ──────────────────────────────────────────────────────

class TestBenchmarkGenerators:
    def test_mmlu(self):
        t = make_mmlu_task(n=5)
        assert t.name == "MMLU"
        assert all(s.task_type == TaskType.MULTIPLE_CHOICE for s in t.samples)

    def test_gsm8k(self):
        t = make_gsm8k_task(n=4)
        assert t.name == "GSM8K"
        assert all(s.task_type == TaskType.MATH_REASONING for s in t.samples)

    def test_hellaswag(self):
        t = make_hellaswag_task(n=3)
        assert t.name == "HellaSwag"

    def test_truthfulqa(self):
        t = make_truthfulqa_task(n=3)
        assert t.name == "TruthfulQA"


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_exact_match_true(self):
        assert exact_match("Paris", "Paris") is True

    def test_exact_match_false(self):
        assert exact_match("Paris", "London") is False

    def test_exact_match_normalized(self):
        assert exact_match("PARIS.", "paris") is True

    def test_token_f1_perfect(self):
        assert abs(token_f1("hello world", "hello world") - 1.0) < 0.01

    def test_token_f1_zero(self):
        assert token_f1("abc", "xyz") == 0.0

    def test_token_f1_partial(self):
        f1 = token_f1("the cat sat", "cat sat mat")
        assert 0.0 < f1 < 1.0

    def test_rouge_l_perfect(self):
        assert abs(rouge_l("the cat", "the cat") - 1.0) < 0.01

    def test_rouge_l_zero(self):
        assert rouge_l("abc", "xyz") == 0.0

    def test_math_exact_match(self):
        assert math_exact_match("The answer is 42", "42") is True

    def test_math_exact_match_float(self):
        assert math_exact_match("Result: 3.14", "3.14") is True

    def test_mc_accuracy(self):
        r = multiple_choice_accuracy(["A", "B", "C"], ["A", "B", "A"])
        assert abs(r.value - 2/3) < 0.01

    def test_mc_accuracy_perfect(self):
        r = multiple_choice_accuracy(["A", "B"], ["A", "B"])
        assert r.value == 1.0

    def test_perplexity_positive(self):
        m   = TinyLM()
        ids = torch.randint(0, V, (1, 16))
        ppl = perplexity(m, ids)
        assert ppl > 0.0


# ── LLMJudge ──────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def test_score_returns_result(self):
        j = LLMJudge()
        r = j.score_response("Q?", "A good response")
        assert isinstance(r, JudgeResult)
        assert 1.0 <= r.score <= 10.0

    def test_normalized_score_range(self):
        j = LLMJudge()
        r = j.score_response("Q?", "response")
        assert 0.0 <= r.normalized_score <= 1.0

    def test_compare_returns_winner(self):
        j    = LLMJudge()
        pair = j.compare("Q?", "Good answer", "Bad answer")
        assert pair.winner in ("A", "B", "tie")

    def test_win_rate(self):
        j = LLMJudge()
        wr = j.win_rate(["Q1", "Q2"], ["A1", "A2"], ["B1", "B2"])
        assert "win_rate_a" in wr
        assert abs(wr["win_rate_a"] + wr["win_rate_b"] + wr["tie_rate"] - 1.0) < 0.01

    def test_batch_score(self):
        j = LLMJudge()
        results = j.batch_score(["Q1", "Q2"], ["A1", "A2"])
        assert len(results) == 2


# ── BenchmarkEvaluator ────────────────────────────────────────────────────────

class TestBenchmarkEvaluator:
    def _evaluator(self):
        def fn(s):
            return s.answer   # oracle
        return BenchmarkEvaluator(fn, "oracle")

    def test_evaluate_task_accuracy(self):
        ev   = self._evaluator()
        task = make_mmlu_task(n=5)
        r    = ev.evaluate_task(task)
        assert 0.0 <= r.accuracy <= 1.0

    def test_oracle_gets_perfect_gsm8k(self):
        ev   = self._evaluator()
        task = make_gsm8k_task(n=4)
        r    = ev.evaluate_task(task)
        assert r.accuracy == 1.0

    def test_evaluate_all_returns_report(self):
        ev     = self._evaluator()
        tasks  = [make_mmlu_task(5), make_hellaswag_task(3)]
        report = ev.evaluate_all(tasks)
        assert isinstance(report, BenchmarkReport)
        assert len(report.tasks) == 2

    def test_by_subject(self):
        ev = self._evaluator()
        r  = ev.evaluate_task(make_mmlu_task(n=10))
        assert len(r.by_subject) > 0

    def test_n_correct(self):
        ev = self._evaluator()
        r  = ev.evaluate_task(make_gsm8k_task(n=4))
        assert r.n_correct == r.n_samples   # oracle is perfect


# ── Leaderboard ───────────────────────────────────────────────────────────────

class TestLeaderboard:
    def _board(self):
        b = Leaderboard(["MMLU", "GSM8K"])
        b.add_model("A", {"MMLU": 0.9, "GSM8K": 0.8})
        b.add_model("B", {"MMLU": 0.7, "GSM8K": 0.6})
        return b

    def test_ranking_order(self):
        b = self._board()
        ranked = b.ranking()
        assert ranked[0].model_name == "A"   # higher mean score

    def test_best_model(self):
        b = self._board()
        assert b.best_model().model_name == "A"

    def test_elo_update_winner(self):
        b = self._board()
        old_a = b.ranking()[0].elo
        b.update_elo("A", "B")
        new_a = b.ranking()[0].elo
        assert new_a > old_a   # winner's Elo increases

    def test_leaderboard_table(self):
        b = self._board()
        t = b.leaderboard_table()
        assert t[0]["rank"] == 1
        assert "mean_score" in t[0]

    def test_task_comparison(self):
        b = self._board()
        r = b.task_comparison("MMLU")
        assert r[0][1] >= r[1][1]   # sorted descending
''')
commit("test: add full eval test suite — samples, benchmarks, metrics, judge, evaluator, leaderboard")

for title, body in [
    ("test: add exact_match edge cases test", '''
class TestExactMatchEdge:
    def test_empty_strings(self):
        assert exact_match("", "") is True

    def test_case_insensitive(self):
        assert exact_match("HELLO", "hello") is True

    def test_punctuation_stripped(self):
        assert exact_match("Paris!", "paris") is True
'''),
    ("test: add token_f1 empty string test", '''
class TestTokenF1Edge:
    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("hello", "") == 0.0
'''),
    ("test: add math_exact_match with text test", '''
class TestMathExactMatchText:
    def test_fraction(self):
        from nanomind.eval import math_exact_match, extract_number
        assert extract_number("4/5") == "4/5"

    def test_negative_number(self):
        from nanomind.eval import math_exact_match
        assert math_exact_match("answer is -3", "-3")
'''),
    ("test: add LLMJudge swap_debiasing test", '''
class TestJudgeSwap:
    def test_no_swap(self):
        j    = LLMJudge(swap_debiasing=False)
        pair = j.compare("Q?", "A good answer", "A mediocre answer")
        assert pair.winner in ("A", "B", "tie")

    def test_with_swap(self):
        j    = LLMJudge(swap_debiasing=True)
        pair = j.compare("Q?", "Response A", "Response B")
        assert pair.winner in ("A", "B", "tie")
'''),
    ("test: add BenchmarkReport overall accuracy test", '''
class TestBenchmarkReport:
    def test_overall_accuracy(self):
        from nanomind.eval import BenchmarkEvaluator
        ev   = BenchmarkEvaluator(lambda s: s.answer, "oracle")
        rep  = ev.evaluate_all([make_mmlu_task(4), make_gsm8k_task(4)])
        assert 0.0 <= rep.overall_accuracy <= 1.0

    def test_total_samples(self):
        from nanomind.eval import BenchmarkEvaluator
        ev  = BenchmarkEvaluator(lambda s: s.answer, "oracle")
        rep = ev.evaluate_all([make_mmlu_task(4), make_gsm8k_task(4)])
        assert rep.total_samples == 8
'''),
    ("test: add rouge_l partial overlap test", '''
class TestRougeL:
    def test_partial_overlap(self):
        r = rouge_l("the cat sat on the mat", "the cat sat")
        assert r > 0.0

    def test_no_overlap(self):
        assert rouge_l("abc", "xyz") == 0.0

    def test_full_match(self):
        assert abs(rouge_l("hello world", "hello world") - 1.0) < 0.01
'''),
    ("test: add Leaderboard Elo loser decreases test", '''
class TestEloLoser:
    def test_loser_decreases(self):
        b = Leaderboard(["T"])
        b.add_model("X", {"T": 0.9})
        b.add_model("Y", {"T": 0.5})
        old_y = [m.elo for m in b._models if m.model_name == "Y"][0]
        b.update_elo("X", "Y")
        new_y = [m.elo for m in b._models if m.model_name == "Y"][0]
        assert new_y < old_y

    def test_elo_zero_sum(self):
        b = Leaderboard(["T"])
        b.add_model("X", {"T": 0.9})
        b.add_model("Y", {"T": 0.5})
        total_before = sum(m.elo for m in b._models)
        b.update_elo("X", "Y")
        total_after = sum(m.elo for m in b._models)
        assert abs(total_after - total_before) < 0.001
'''),
]:
    src = read("tests/test_eval.py")
    src += "\n" + body
    write("tests/test_eval.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.4.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"5.3.0\"", "__version__ = \"5.4.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.4.0 — Evaluation & Benchmarking Suite release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `distributed`| Distributed Training — DDP, ZeRO, Tensor/Pipeline Parallelism, AMP, checkpointing |",
    "| `distributed`| Distributed Training — DDP, ZeRO, Tensor/Pipeline Parallelism, AMP, checkpointing |\n"
    "| `eval`       | Evaluation Suite — MMLU/GSM8K/HellaSwag, metrics, LLM judge, leaderboard, Elo |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.4.0] — 2024 — Evaluation & Benchmarking Suite\n\n### Added\n"
      "- `EvalSample` / `EvalTask` — evaluation data structures with multiple task types\n"
      "- `TaskType` / `EvalProtocol` — zero-shot, few-shot, chain-of-thought protocols\n"
      "- `make_mmlu_task` — MMLU-style multiple choice benchmark (10 curated questions)\n"
      "- `make_gsm8k_task` — GSM8K math reasoning benchmark\n"
      "- `make_hellaswag_task` — HellaSwag commonsense completion benchmark\n"
      "- `make_truthfulqa_task` — TruthfulQA factuality benchmark\n"
      "- `exact_match` / `token_f1` / `rouge_l` — generation metrics\n"
      "- `math_exact_match` — number extraction + floating point comparison\n"
      "- `multiple_choice_accuracy` — MC accuracy metric\n"
      "- `generation_metrics` — all metrics in one call\n"
      "- `perplexity` — strided LM perplexity evaluation\n"
      "- `LLMJudge` — LLM-as-a-Judge (1-10 scoring), pairwise, swap debiasing, win_rate\n"
      "- `BenchmarkEvaluator` — run tasks, score samples, produce TaskResult\n"
      "- `BenchmarkReport` — overall + per-task accuracy, wall time\n"
      "- `Leaderboard` — model comparison with Elo rating system\n"
      "- `ModelScore` — per-model scores and Elo tracking\n"
      "- `examples/eval_demo.py` — full evaluation pipeline demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.4.0, update README and CHANGELOG for Day 54 Evaluation Suite")

print("\n=== Pushing Day 54 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.4.0", "-m", "NanoMind v5.4.0 — Evaluation & Benchmarking Suite", check=False)
r = run("git", "push", "origin", "v5.4.0", check=False)
print("Tag v5.4.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 54 COMPLETE — v5.4.0 TAGGED! ===")
