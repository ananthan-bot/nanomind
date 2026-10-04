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
