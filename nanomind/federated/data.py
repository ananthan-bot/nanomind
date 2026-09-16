"""
nanomind/federated/data.py — Federated data partition utilities.

Federated learning assumes data is NOT i.i.d. across clients:
  - Different users have different writing styles, languages, topics
  - Non-i.i.d. data hurts FedAvg convergence (client drift)

Partition strategies:
  IID:       Shuffle and split equally → each client sees full distribution
  Dirichlet: Sample proportions from Dir(alpha) → controls heterogeneity
             alpha → ∞: i.i.d.
             alpha → 0: each client has only 1 class

  Pathological: Each client gets exactly K classes (standard benchmark)

Reference:
  Li et al. (2020) "Federated Learning on Non-IID Data Silos: An Experimental Study"
  https://arxiv.org/abs/2102.02079
"""

from __future__ import annotations
import random
import torch
from dataclasses import dataclass


@dataclass
class FederatedDataset:
    """A client's local dataset."""
    client_id: str
    batches:   list[tuple]   # list of (x, y) pairs

    @property
    def n_samples(self) -> int:
        return len(self.batches)


def iid_partition(
    data:       list[tuple],
    n_clients:  int,
    batch_size: int = 4,
) -> list[FederatedDataset]:
    """
    Partition data IID across clients.

    Each client receives an equal-sized random subset.

    Args:
        data:       List of ``(x, y)`` samples.
        n_clients:  Number of clients.
        batch_size: Batch size for each client dataset.

    Returns:
        List of :class:`FederatedDataset`.
    """
    shuffled  = data[:]
    random.shuffle(shuffled)
    size      = len(shuffled) // n_clients
    datasets  = []
    for i in range(n_clients):
        shard   = shuffled[i * size:(i + 1) * size]
        # Group into batches
        batches = []
        for j in range(0, len(shard), batch_size):
            sub = shard[j:j + batch_size]
            if not sub:
                continue
            xs = torch.stack([s[0] for s in sub])
            ys = torch.stack([s[1] for s in sub])
            batches.append((xs, ys))
        datasets.append(FederatedDataset(f"client-{i:03d}", batches))
    return datasets


def dirichlet_partition(
    data:       list[tuple],
    n_clients:  int,
    alpha:      float = 0.5,
    batch_size: int   = 4,
    n_classes:  int   = 2,
) -> list[FederatedDataset]:
    """
    Partition data via Dirichlet distribution (non-IID).

    Args:
        data:       List of ``(x, y)`` samples.
        n_clients:  Number of clients.
        alpha:      Dirichlet concentration (lower → more heterogeneous).
        batch_size: Batch size per client.
        n_classes:  Number of label classes.

    Returns:
        List of :class:`FederatedDataset` with heterogeneous distributions.
    """
    # Group by class
    class_data: dict[int, list] = {c: [] for c in range(n_classes)}
    for x, y in data:
        label = int(y.item()) % n_classes
        class_data[label].append((x, y))

    # Sample proportions from Dirichlet
    client_data: list[list] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        props = torch.distributions.Dirichlet(
            torch.ones(n_clients) * alpha
        ).sample()
        props = (props / props.sum()).tolist()
        items = class_data[c]
        random.shuffle(items)
        idx   = 0
        for i, p in enumerate(props):
            count = int(p * len(items))
            client_data[i].extend(items[idx:idx + count])
            idx   += count

    datasets = []
    for i, shard in enumerate(client_data):
        batches = []
        for j in range(0, len(shard), batch_size):
            sub = shard[j:j + batch_size]
            if not sub:
                continue
            xs = torch.stack([s[0] for s in sub])
            ys = torch.stack([s[1] for s in sub])
            batches.append((xs, ys))
        datasets.append(FederatedDataset(f"client-{i:03d}", batches))
    return datasets
