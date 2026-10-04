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
