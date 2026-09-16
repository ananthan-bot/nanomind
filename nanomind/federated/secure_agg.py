"""
nanomind/federated/secure_agg.py — Secure Aggregation (SecAgg).

Standard federated learning: server sees individual client gradients.
Secure Aggregation: server sees ONLY the SUM, never individual updates.

Protocol (Bonawitz et al., 2017):
  1. Each pair of clients (i, j) agree on a random seed s_ij
  2. Client i adds:  +mask(s_ij)  for j > i
                     -mask(s_ji)  for j < i
  3. Server sums all masked updates → masks cancel out
  4. Server only sees the true sum

This NanoMind implementation uses a simplified version with shared seeds:
  - Each client generates a random mask using its client ID as seed
  - Server is given the sum of all masks for cancellation

Real SecAgg uses pairwise Diffie-Hellman key agreement.

Reference:
  Bonawitz et al. (2017) "Practical Secure Aggregation for Privacy-Preserving
  Machine Learning" https://dl.acm.org/doi/10.1145/3133956.3133982
"""

from __future__ import annotations
import torch
import hashlib


def _deterministic_mask(seed_str: str, shape: torch.Size, dtype=torch.float32) -> torch.Tensor:
    """Generate a deterministic random mask from a string seed."""
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
    gen  = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=dtype)


class SecureAggregator:
    """
    Simplified Secure Aggregation for federated gradients.

    Each client masks its gradients with a random tensor.
    The server subtracts the sum of all masks to recover the true sum.

    Args:
        round_seed: Global seed for this federated round.

    Example::

        agg = SecureAggregator(round_seed="round-1")
        masked = agg.mask(gradient_flat, client_id="client-0")
        # ... collect all masked from clients ...
        true_sum = agg.unmask(sum_of_masked, client_ids=["client-0", ...])
    """

    def __init__(self, round_seed: str = "nanomind-secagg") -> None:
        self.round_seed = round_seed

    def _mask_for_client(self, client_id: str, shape: torch.Size) -> torch.Tensor:
        """Deterministic mask for (round, client) pair."""
        seed_str = f"{self.round_seed}:{client_id}"
        return _deterministic_mask(seed_str, shape)

    def mask(self, gradient: torch.Tensor, client_id: str) -> torch.Tensor:
        """
        Mask a gradient tensor for upload.

        Args:
            gradient:  Client gradient (flat or shaped tensor).
            client_id: Client identifier.

        Returns:
            Masked gradient (gradient + mask).
        """
        m = self._mask_for_client(client_id, gradient.shape)
        return gradient + m

    def unmask(
        self,
        masked_sum: torch.Tensor,
        client_ids: list[str],
    ) -> torch.Tensor:
        """
        Remove sum of all client masks from the aggregated sum.

        Args:
            masked_sum: Sum of all clients' masked gradients.
            client_ids: List of participating client IDs.

        Returns:
            True gradient sum (unmasked).
        """
        total_mask = sum(
            self._mask_for_client(cid, masked_sum.shape)
            for cid in client_ids
        )
        return masked_sum - total_mask
