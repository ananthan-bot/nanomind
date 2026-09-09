"""
nanomind/eval/runner.py — Unified evaluation runner for NanoMind models.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.eval.config import BenchmarkConfig
from nanomind.eval.perplexity import compute_perplexity
from nanomind.eval.throughput import benchmark_prefill, benchmark_memory
from nanomind.eval.accuracy import evaluate_accuracy
from nanomind.utils.logger import get_logger

log = get_logger("eval.runner")


class EvalRunner:
    """
    Unified evaluation runner: perplexity, accuracy, throughput, memory.

    Runs all benchmark dimensions in one call and returns a structured report.

    Args:
        model:  Language model to evaluate.
        cfg:    Benchmark configuration.
        name:   Model name for reports.

    Example::

        runner  = EvalRunner(model, BenchmarkConfig(), name="NanoMind-128d")
        results = runner.run(val_loader)
        print(runner.format_report(results))
    """

    def __init__(
        self,
        model: nn.Module,
        cfg:   BenchmarkConfig | None = None,
        name:  str = "Model",
    ) -> None:
        self.model  = model
        self.cfg    = cfg or BenchmarkConfig()
        self.name   = name
        self.device = torch.device(self.cfg.device)

    def run(
        self,
        loader:      DataLoader | None = None,
        max_batches: int | None = 50,
    ) -> dict:
        """
        Run the full evaluation suite.

        Args:
            loader:      DataLoader for perplexity + accuracy evaluation.
            max_batches: Max batches for perplexity/accuracy to keep it fast.

        Returns:
            Dict with all metrics from all benchmark dimensions.
        """
        results: dict = {"model": self.name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        # Memory
        log.info(f"Evaluating memory for {self.name}...")
        mem = benchmark_memory(self.model)
        results.update(mem)

        # Throughput
        log.info(f"Evaluating throughput for {self.name}...")
        speed = benchmark_prefill(self.model, self.cfg)
        results.update(speed)

        # Perplexity + accuracy (if loader provided)
        if loader is not None:
            log.info(f"Evaluating perplexity for {self.name}...")
            ppl = compute_perplexity(self.model, loader, self.device, max_batches)
            results.update(ppl)

            log.info(f"Evaluating accuracy for {self.name}...")
            acc = evaluate_accuracy(self.model, loader, self.device,
                                    self.cfg.top_k_values, max_batches)
            results.update(acc)

        return results

    def format_report(self, results: dict) -> str:
        """Format an evaluation results dict as a human-readable report."""
        lines = [
            f"Evaluation Report: {results.get('model', 'Unknown')}",
            f"  Timestamp : {results.get('timestamp', '')}",
            "─" * 52,
        ]
        if "n_params" in results:
            lines.append(f"  Parameters: {results['n_params']:>14,}")
            lines.append(f"  Memory    : {results.get('total_mb', 0):>12.2f} MB")
        if "tokens_per_sec" in results:
            lines.append(f"  Throughput: {results['tokens_per_sec']:>12,.0f} tok/s")
            lines.append(f"  Latency   : {results.get('ms_per_batch', 0):>12.2f} ms/batch")
        if "perplexity" in results:
            lines.append(f"  Perplexity: {results['perplexity']:>12.4f}")
            lines.append(f"  Bits/char : {results.get('bits_per_char', 0):>12.4f}")
        for k in self.cfg.top_k_values:
            key = f"top_{k}"
            if key in results:
                lines.append(f"  Top-{k:<6}: {results[key]:>11.2%}")
        return "
".join(lines)

    def compare(self, other_results: list[dict]) -> str:
        """Format a comparison table for multiple model evaluation results."""
        cols = ["model", "n_params", "total_mb", "tokens_per_sec", "perplexity", "top_1"]
        header = f"{'Model':<20} {'Params':>10} {'MB':>8} {'Tok/s':>10} {'PPL':>10} {'Top-1':>8}"
        sep    = "─" * len(header)
        rows   = [header, sep]
        for r in other_results:
            row = (
                f"{r.get('model','?'):<20} "
                f"{r.get('n_params',0):>10,} "
                f"{r.get('total_mb',0):>8.1f} "
                f"{r.get('tokens_per_sec',0):>10,.0f} "
                f"{r.get('perplexity',float('inf')):>10.2f} "
                f"{r.get('top_1',0):>8.2%}"
            )
            rows.append(row)
        return "
".join(rows)
