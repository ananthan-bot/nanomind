"""
nanomind/federated/client.py — Federated learning client.

Each client:
  1. Receives global model weights from server
  2. Trains locally for E epochs on its own dataset
  3. Optionally applies DP noise to gradients
  4. Optionally compresses gradients
  5. Returns gradient deltas to server
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
from nanomind.federated.config import FederatedConfig
from nanomind.federated.privacy import DifferentialPrivacyEngine
from nanomind.federated.compression import TopKCompressor
from nanomind.utils.logger import get_logger

log = get_logger("federated.client")


class FederatedClient:
    """
    Federated learning client — local training + gradient upload.

    Args:
        client_id:  Unique client identifier.
        model:      Local model (deep copy of global model).
        dataset:    List of ``(x, y)`` training batches.
        cfg:        Federated config.
        dp_engine:  Optional DP engine for gradient privatisation.
        compressor: Optional gradient compressor.

    Example::

        client = FederatedClient("client-0", model, dataset, cfg)
        deltas = client.train_round(global_weights)
    """

    def __init__(
        self,
        client_id:  str,
        model:      nn.Module,
        dataset:    list,
        cfg:        FederatedConfig,
        dp_engine:  DifferentialPrivacyEngine | None = None,
        compressor: TopKCompressor | None            = None,
    ) -> None:
        self.client_id  = client_id
        self.model      = model
        self.dataset    = dataset
        self.cfg        = cfg
        self.dp_engine  = dp_engine
        self.compressor = compressor
        self._train_loss_history: list[float] = []

    def load_global_weights(self, global_state: dict) -> None:
        """Load global model weights into local model."""
        self.model.load_state_dict(copy.deepcopy(global_state))

    def train_round(self, global_state: dict) -> dict:
        """
        Run a local training round and return gradient deltas.

        Args:
            global_state: Global model state_dict from server.

        Returns:
            Dict with ``gradient_deltas``, ``n_samples``, ``loss``.
        """
        self.load_global_weights(global_state)
        initial_state = copy.deepcopy(self.model.state_dict())

        opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg.local_lr)
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for epoch in range(self.cfg.local_epochs):
            for x, y in self.dataset:
                opt.zero_grad()
                logits, loss = self.model(x, y)
                if loss is None:
                    import torch.nn.functional as F
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()

                if self.dp_engine:
                    self.dp_engine.clip_and_noise(self.model)

                opt.step()
                total_loss += loss.item()
                n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        self._train_loss_history.append(avg_loss)

        # Compute gradient delta = final - initial weights
        final_state = self.model.state_dict()
        deltas = {
            k: final_state[k].float() - initial_state[k].float()
            for k in final_state
        }

        log.info(f"Client {self.client_id}: loss={avg_loss:.4f}, "
                 f"batches={n_batches}")
        return {
            "gradient_deltas": deltas,
            "n_samples":       len(self.dataset),
            "loss":            avg_loss,
            "client_id":       self.client_id,
        }

    @property
    def last_loss(self) -> float | None:
        return self._train_loss_history[-1] if self._train_loss_history else None
