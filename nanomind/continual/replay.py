"""
nanomind/continual/replay.py — Experience Replay Buffer.

Replay buffers store a subset of past task data and mix it into the
current task's training batches to prevent forgetting.

Reservoir sampling (Vitter, 1985):
  Keep a uniform random sample of size M from a stream of N items.
  Each new item replaces a random existing item with prob M/N.
  This guarantees a uniform sample without storing all N items.

DER++ (Buzzega et al., 2020):
  Also stores logits at the time of storage as "dark experience".
  During replay, adds KL loss between current logits and stored logits.
  This soft constraint is stronger than just replaying labels.

References:
  Buzzega et al. (2020) "Dark Experience for General Continual Learning"
  https://arxiv.org/abs/2004.07211

  Rebuffi et al. (2017) "iCaRL: Incremental Classifier and Representation Learning"
  https://arxiv.org/abs/1611.07725
"""

from __future__ import annotations
import random
import torch
from dataclasses import dataclass


@dataclass
class ReplayEntry:
    """A single entry in the replay buffer."""
    x:       torch.Tensor    # input tokens (T,)
    y:       torch.Tensor    # target tokens (T,)
    task_id: int
    logits:  torch.Tensor | None = None   # stored logits for DER++


class ReplayBuffer:
    """
    Reservoir-sampled experience replay buffer.

    Maintains a fixed-size uniform random sample of training examples
    across all tasks seen so far.

    Args:
        max_size:  Maximum number of examples to store.

    Example::

        buf = ReplayBuffer(max_size=500)
        buf.add(x, y, task_id=0)
        samples = buf.sample(batch_size=16)
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._buffer: list[ReplayEntry] = []
        self._n_seen: int = 0    # total examples seen (for reservoir sampling)

    def add(
        self,
        x:       torch.Tensor,
        y:       torch.Tensor,
        task_id: int,
        logits:  torch.Tensor | None = None,
    ) -> None:
        """
        Add a single example via reservoir sampling.

        Args:
            x:       ``(T,)`` input token sequence.
            y:       ``(T,)`` target token sequence.
            task_id: Task this example belongs to.
            logits:  Optional stored logits for DER++.
        """
        entry = ReplayEntry(x.clone(), y.clone(), task_id,
                            logits.detach().clone() if logits is not None else None)
        self._n_seen += 1
        if len(self._buffer) < self.max_size:
            self._buffer.append(entry)
        else:
            idx = random.randint(0, self._n_seen - 1)
            if idx < self.max_size:
                self._buffer[idx] = entry

    def add_batch(
        self,
        xs:      torch.Tensor,
        ys:      torch.Tensor,
        task_id: int,
        logits:  torch.Tensor | None = None,
    ) -> None:
        """Add a batch of examples to the replay buffer."""
        for i in range(xs.shape[0]):
            lg = logits[i] if logits is not None else None
            self.add(xs[i], ys[i], task_id, lg)

    def sample(self, batch_size: int) -> list[ReplayEntry]:
        """
        Sample a random batch from the replay buffer.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of :class:`ReplayEntry`.
        """
        k = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, k) if k > 0 else []

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a collated ``(x_batch, y_batch)`` tensor.

        Returns:
            ``(x, y)`` tensors of shape ``(k, T)``.
        """
        entries = self.sample(batch_size)
        if not entries:
            raise RuntimeError("Replay buffer is empty")
        xs = torch.stack([e.x for e in entries])
        ys = torch.stack([e.y for e in entries])
        return xs, ys

    def task_counts(self) -> dict[int, int]:
        """Count examples per task."""
        counts: dict[int, int] = {}
        for e in self._buffer:
            counts[e.task_id] = counts.get(e.task_id, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def n_seen(self) -> int:
        return self._n_seen
