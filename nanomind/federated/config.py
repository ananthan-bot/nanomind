"""
nanomind/federated/config.py — Federated learning configuration.

## What is Federated Learning?

Traditional training: all data goes to a central server.
  Problem: privacy — user data never leaves the device.

Federated Learning (McMahan et al., 2017):
  1. Server sends model to N clients
  2. Each client trains locally on its OWN data
  3. Clients send GRADIENTS (not data) back to server
  4. Server aggregates gradients (FedAvg) → updated model
  5. Repeat for R rounds

Key insight: raw data never leaves the device.
Used by: Google Gboard (keyboard predictions), Apple Siri, medical AI.

## Differential Privacy

Problem: gradients can leak private training data.
Solution: add calibrated Gaussian noise to gradients before upload.

  DP guarantee: for any two neighbouring datasets D, D' differing
  in one example, the gradient distribution is (ε, δ)-DP:
    P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ

  Smaller ε → stronger privacy (but worse accuracy).
  δ < 1/n (dataset size) → meaningful guarantee.

References:
  FedAvg: McMahan et al. (2017) "Communication-Efficient Learning of
  Deep Networks from Decentralized Data" https://arxiv.org/abs/1602.05629

  DP-SGD: Abadi et al. (2016) "Deep Learning with Differential Privacy"
  https://arxiv.org/abs/1607.00133
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FederatedConfig:
    """
    Configuration for a federated learning experiment.

    Attributes:
        n_clients:        Total number of clients.
        clients_per_round: How many clients participate per round.
        n_rounds:         Total number of federated rounds.
        local_epochs:     Epochs each client trains locally per round.
        local_lr:         Client-side learning rate.
        aggregation:      Aggregation strategy: ``"fedavg"`` or ``"fedmedian"``.
        dp_epsilon:       Differential privacy epsilon (0 = disabled).
        dp_delta:         Differential privacy delta.
        dp_max_grad_norm: Gradient clipping norm for DP-SGD.
        compress_gradients: Enable top-K gradient compression.
        compress_ratio:   Fraction of gradients to keep (0.01 = top 1%).
    """
    n_clients:          int   = 10
    clients_per_round:  int   = 5
    n_rounds:           int   = 10
    local_epochs:       int   = 3
    local_lr:           float = 1e-3
    aggregation:        str   = "fedavg"
    dp_epsilon:         float = 0.0       # 0 = no DP
    dp_delta:           float = 1e-5
    dp_max_grad_norm:   float = 1.0
    compress_gradients: bool  = False
    compress_ratio:     float = 0.1

    def __post_init__(self) -> None:
        assert self.n_clients          >= 1
        assert 1 <= self.clients_per_round <= self.n_clients
        assert self.n_rounds           >= 1
        assert self.local_epochs       >= 1
        assert self.local_lr           >  0
        assert self.aggregation        in ("fedavg", "fedmedian")
        assert self.dp_epsilon         >= 0.0
        assert 0 < self.compress_ratio <= 1.0

    @property
    def dp_enabled(self) -> bool:
        """True if differential privacy is enabled."""
        return self.dp_epsilon > 0.0

    @property
    def participation_rate(self) -> float:
        """Fraction of clients participating per round."""
        return self.clients_per_round / self.n_clients
