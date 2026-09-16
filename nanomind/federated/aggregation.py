"""
nanomind/federated/aggregation.py — Federated gradient aggregation strategies.

FedAvg (McMahan et al., 2017):
  Weighted average of client gradient deltas by dataset size:
    global_model += sum_i(n_i / N × delta_i)
  Simple, widely used, converges for i.i.d. data.
  Struggles with non-i.i.d. (heterogeneous) data.

FedMedian (Yin et al., 2018):
  Take coordinate-wise median instead of mean.
  Robust to Byzantine (malicious) clients:
    If < n/2 clients are malicious, median is unaffected.
  Slower convergence than FedAvg but more secure.

References:
  FedAvg: McMahan et al. (2017) https://arxiv.org/abs/1602.05629
  FedMedian: Yin et al. (2018) https://arxiv.org/abs/1803.01498
  Byzantine robustness survey: https://arxiv.org/abs/1906.10907
"""

from __future__ import annotations
import torch


def fedavg(
    client_updates: list[dict],
    global_state:   dict,
) -> dict:
    """
    Federated Averaging (FedAvg) aggregation.

    Weighted average of gradient deltas by number of samples.

    Args:
        client_updates: List of dicts from ``FederatedClient.train_round()``.
                        Each must have ``gradient_deltas`` and ``n_samples``.
        global_state:   Current global model state_dict.

    Returns:
        Updated global model state_dict.
    """
    total_n   = sum(u["n_samples"] for u in client_updates)
    new_state = {k: v.clone().float() for k, v in global_state.items()}

    for update in client_updates:
        weight = update["n_samples"] / max(total_n, 1)
        for k, delta in update["gradient_deltas"].items():
            if k in new_state:
                new_state[k] += weight * delta.float()

    return new_state


def fedmedian(
    client_updates: list[dict],
    global_state:   dict,
) -> dict:
    """
    Coordinate-wise Median aggregation (Byzantine-robust).

    Args:
        client_updates: List of dicts from ``FederatedClient.train_round()``.
        global_state:   Current global model state_dict.

    Returns:
        Updated global model state_dict with median-aggregated deltas.
    """
    new_state = {k: v.clone().float() for k, v in global_state.items()}

    # Collect all deltas per key
    for k in new_state:
        if k not in client_updates[0]["gradient_deltas"]:
            continue
        stacked = torch.stack([
            u["gradient_deltas"][k].float() for u in client_updates
        ], dim=0)                                  # (n_clients, *shape)
        median_delta = stacked.median(dim=0).values
        new_state[k] = new_state[k] + median_delta

    return new_state


def aggregate(
    client_updates: list[dict],
    global_state:   dict,
    strategy:       str = "fedavg",
) -> dict:
    """
    Dispatch to the requested aggregation strategy.

    Args:
        client_updates: List of client round results.
        global_state:   Current global state_dict.
        strategy:       ``"fedavg"`` or ``"fedmedian"``.

    Returns:
        New global state_dict.
    """
    if strategy == "fedavg":
        return fedavg(client_updates, global_state)
    elif strategy == "fedmedian":
        return fedmedian(client_updates, global_state)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")
