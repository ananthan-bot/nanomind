"""
nanomind/federated/server.py — Federated learning server / coordinator.

The server orchestrates the federated training loop:
  1. Maintain the global model
  2. Select clients for each round
  3. Broadcast global weights
  4. Collect and aggregate gradient updates
  5. Log round statistics
"""

from __future__ import annotations
import copy
import random
import torch.nn as nn
from nanomind.federated.config import FederatedConfig
from nanomind.federated.aggregation import aggregate
from nanomind.federated.client import FederatedClient
from nanomind.utils.logger import get_logger

log = get_logger("federated.server")


class FederatedServer:
    """
    Central coordinator for federated training.

    Args:
        global_model: The global model to train.
        cfg:          Federated configuration.

    Example::

        server = FederatedServer(model, cfg)
        server.add_client(FederatedClient("c0", model_copy, dataset, cfg))
        history = server.train()
    """

    def __init__(self, global_model: nn.Module, cfg: FederatedConfig) -> None:
        self.global_model = global_model
        self.cfg          = cfg
        self._clients:    list[FederatedClient] = []
        self._round_logs: list[dict]            = []

    def add_client(self, client: FederatedClient) -> None:
        """Register a client with the server."""
        self._clients.append(client)

    def add_clients(self, clients: list[FederatedClient]) -> None:
        """Register multiple clients."""
        for c in clients:
            self.add_client(c)

    def _select_clients(self) -> list[FederatedClient]:
        """Randomly select clients_per_round clients."""
        k = min(self.cfg.clients_per_round, len(self._clients))
        return random.sample(self._clients, k)

    def _global_state(self) -> dict:
        return copy.deepcopy(self.global_model.state_dict())

    def train_round(self, round_idx: int) -> dict:
        """
        Run a single federated round.

        Args:
            round_idx: Current round index (0-indexed).

        Returns:
            Dict with ``round``, ``avg_loss``, ``n_clients``.
        """
        selected = self._select_clients()
        global_s = self._global_state()
        updates  = []

        for client in selected:
            update = client.train_round(global_s)
            updates.append(update)

        # Aggregate updates
        new_state = aggregate(updates, global_s, self.cfg.aggregation)
        self.global_model.load_state_dict(new_state)

        avg_loss = sum(u["loss"] for u in updates) / len(updates)
        log_entry = {
            "round":     round_idx + 1,
            "avg_loss":  round(avg_loss, 6),
            "n_clients": len(selected),
        }
        self._round_logs.append(log_entry)
        log.info(f"Round {round_idx + 1}/{self.cfg.n_rounds}  "
                 f"loss={avg_loss:.4f}  clients={len(selected)}")
        return log_entry

    def train(self) -> list[dict]:
        """
        Run the full federated training loop.

        Returns:
            List of round log dicts.
        """
        log.info(f"Starting federated training: "
                 f"{self.cfg.n_rounds} rounds, {len(self._clients)} clients")
        for r in range(self.cfg.n_rounds):
            self.train_round(r)
        return self._round_logs

    @property
    def round_logs(self) -> list[dict]:
        return list(self._round_logs)

    @property
    def n_clients(self) -> int:
        return len(self._clients)
