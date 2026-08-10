"""
nanomind/data/prefetch.py — DataLoader prefetching utility.

Wraps a DataLoader to overlap CPU->GPU transfer with computation,
improving throughput on CUDA devices.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator

import torch
from torch.utils.data import DataLoader


class PrefetchLoader:
    """
    Wraps a DataLoader and pre-loads the next batch to GPU asynchronously.

    On non-CUDA devices this is a transparent pass-through.

    Args:
        loader: The DataLoader to wrap.
        device: Target device for tensor transfer.

    Example::

        loader = PrefetchLoader(train_loader, device)
        for x, y in loader:
            loss = model(x, y)
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self._loader = loader
        self._device = device
        self._use_cuda = device.type == "cuda"

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if not self._use_cuda:
            # Non-CUDA: simple pass-through
            for batch in self._loader:
                yield tuple(t.to(self._device) for t in batch)  # type: ignore[misc]
            return

        stream = torch.cuda.Stream()
        it = iter(self._loader)

        try:
            next_x, next_y = next(it)
        except StopIteration:
            return

        with torch.cuda.stream(stream):
            next_x = next_x.to(self._device, non_blocking=True)
            next_y = next_y.to(self._device, non_blocking=True)

        while True:
            cur_x, cur_y = next_x, next_y
            try:
                raw_x, raw_y = next(it)
                with torch.cuda.stream(stream):
                    next_x = raw_x.to(self._device, non_blocking=True)
                    next_y = raw_y.to(self._device, non_blocking=True)
            except StopIteration:
                torch.cuda.current_stream().wait_stream(stream)
                yield cur_x, cur_y
                break

            torch.cuda.current_stream().wait_stream(stream)
            yield cur_x, cur_y

    def __len__(self) -> int:
        return len(self._loader)
