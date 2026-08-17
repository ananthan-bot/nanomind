"""
nanomind/trainer/estimate.py — Training time and compute estimation utilities.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn


def estimate_training_time(
    model: nn.Module,
    train_loader,
    n_warmup: int = 3,
    n_measure: int = 10,
    device: torch.device | None = None,
) -> dict:
    """
    Estimate training throughput and total training time.

    Runs a few warm-up steps and then times ``n_measure`` steps to
    estimate tokens/second and total training duration.

    Args:
        model:       The model to benchmark.
        train_loader: The training DataLoader.
        n_warmup:    Number of warm-up steps (not timed).
        n_measure:   Number of steps to time.
        device:      Device to run on.

    Returns:
        Dict with keys:
        - ``tokens_per_second``: float
        - ``seconds_per_iter``:  float
        - ``estimated_total_s``: float (for 1000 iters)
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    data_it = iter(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # Warm-up
    for _ in range(n_warmup):
        try:
            x, y = next(data_it)
        except StopIteration:
            data_it = iter(train_loader)
            x, y = next(data_it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_tokens = 0
    for _ in range(n_measure):
        try:
            x, y = next(data_it)
        except StopIteration:
            data_it = iter(train_loader)
            x, y = next(data_it)
        x, y = x.to(device), y.to(device)
        total_tokens += x.numel()
        _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps     = total_tokens / elapsed
    spi     = elapsed / n_measure
    return {
        "tokens_per_second":  tps,
        "seconds_per_iter":   spi,
        "estimated_total_s":  spi * 1000,
    }
