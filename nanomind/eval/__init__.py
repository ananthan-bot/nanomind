"""NanoMind evaluation and metrics sub-package.

Primary exports:
    - :class:`Evaluator`             — full evaluation runner
    - :class:`EvalConfig`            — evaluation configuration
    - :class:`EvalResult`            — structured results container

Metrics:
    - :func:`perplexity`             — exp(cross-entropy loss)
    - :func:`bits_per_character`     — loss in bits
    - :func:`token_accuracy`         — top-1 prediction accuracy
    - :func:`top_k_accuracy`         — top-K prediction accuracy
    - :func:`cross_entropy_on_batch` — per-batch loss helper

Benchmarking:
    - :func:`benchmark_inference`    — time and memory profiling

Comparison:
    - :func:`compare_models`         — evaluate multiple models side-by-side
    - :func:`print_comparison`       — pretty-print comparison table

Generation quality:
    - :func:`type_token_ratio`       — lexical diversity
    - :func:`distinct_n`             — N-gram diversity
    - :func:`repetition_fraction`    — N-gram repetition
    - :func:`generation_report`      — full quality report dict
"""

from nanomind.eval.config import EvalConfig
from nanomind.eval.result import EvalResult
from nanomind.eval.evaluator import Evaluator
from nanomind.eval.metrics import (
    perplexity,
    bits_per_character,
    token_accuracy,
    top_k_accuracy,
    cross_entropy_on_batch,
)
from nanomind.eval.benchmark import benchmark_inference
from nanomind.eval.compare import compare_models, print_comparison
from nanomind.eval.generation_quality import (
    type_token_ratio,
    distinct_n,
    repetition_fraction,
    generation_report,
)

__all__ = [
    "EvalConfig",
    "EvalResult",
    "Evaluator",
    "perplexity",
    "bits_per_character",
    "token_accuracy",
    "top_k_accuracy",
    "cross_entropy_on_batch",
    "benchmark_inference",
    "compare_models",
    "print_comparison",
    "type_token_ratio",
    "distinct_n",
    "repetition_fraction",
    "generation_report",
]
