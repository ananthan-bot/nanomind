"""NanoMind Federated sub-package — privacy-preserving distributed learning.

Implements a complete federated learning system:
  1. FederatedClient   — local training with DP + gradient compression
  2. FederatedServer   — orchestration, client selection, aggregation
  3. FedAvg / FedMedian — aggregation strategies
  4. DifferentialPrivacyEngine — DP-SGD: clip + Gaussian noise
  5. TopKCompressor    — top-K sparsification with error feedback
  6. SecureAggregator  — mask/unmask for gradient privacy
  7. Data partitioning — IID and Dirichlet non-IID splits

Primary exports:
    - :class:`FederatedConfig`           — n_clients, rounds, DP, compression
    - :class:`FederatedClient`           — train_round, load_global_weights
    - :class:`FederatedServer`           — add_client, train_round, full train()
    - :class:`DifferentialPrivacyEngine` — clip_and_noise, privacy_spent
    - :class:`TopKCompressor`            — compress, decompress, error feedback
    - :class:`QuantisedCompressor`       — float32→int8 gradient quantisation
    - :class:`SecureAggregator`          — mask, unmask
    - :func:`fedavg`                     — weighted FedAvg aggregation
    - :func:`fedmedian`                  — Byzantine-robust median
    - :func:`aggregate`                  — dispatch to strategy
    - :class:`FederatedDataset`          — client local dataset
    - :func:`iid_partition`              — IID data split
    - :func:`dirichlet_partition`        — non-IID Dirichlet split
"""

from nanomind.federated.config import FederatedConfig
from nanomind.federated.privacy import DifferentialPrivacyEngine
from nanomind.federated.compression import TopKCompressor, QuantisedCompressor
from nanomind.federated.aggregation import fedavg, fedmedian, aggregate
from nanomind.federated.client import FederatedClient
from nanomind.federated.server import FederatedServer
from nanomind.federated.secure_agg import SecureAggregator
from nanomind.federated.data import FederatedDataset, iid_partition, dirichlet_partition

__all__ = [
    "FederatedConfig",
    "DifferentialPrivacyEngine",
    "TopKCompressor", "QuantisedCompressor",
    "fedavg", "fedmedian", "aggregate",
    "FederatedClient", "FederatedServer",
    "SecureAggregator",
    "FederatedDataset", "iid_partition", "dirichlet_partition",
]
