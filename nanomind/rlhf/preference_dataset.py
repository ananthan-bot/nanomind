"""
nanomind/rlhf/preference_dataset.py — Dataset of (chosen, rejected) completion pairs.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from nanomind.tokenizer.base import BaseTokenizer


class PreferenceDataset(Dataset):
    """
    Dataset of (prompt, chosen, rejected) preference pairs for reward model training.

    Each item yields a pair of token sequences:
      - ``chosen``:   the preferred completion (higher human rating)
      - ``rejected``: the rejected completion (lower human rating)

    Args:
        pairs:      List of ``(prompt, chosen_text, rejected_text)`` tuples.
        tokenizer:  Tokenizer for encoding.
        max_length: Maximum sequence length (prompt + completion).

    Example::

        pairs = [
            ("Q: What is 2+2?", "A: 4", "A: 5"),
            ("Q: Sky color?",   "A: Blue", "A: Green"),
        ]
        ds = PreferenceDataset(pairs, tokenizer, max_length=64)
        chosen, rejected = ds[0]
    """

    def __init__(
        self,
        pairs:      list[tuple[str, str, str]],
        tokenizer:  BaseTokenizer,
        max_length: int = 128,
    ) -> None:
        self.chosen_ids:   list[torch.Tensor] = []
        self.rejected_ids: list[torch.Tensor] = []

        for prompt, chosen, rejected in pairs:
            enc_c = tokenizer.encode(prompt + chosen)[:max_length]
            enc_r = tokenizer.encode(prompt + rejected)[:max_length]
            self.chosen_ids.append(torch.tensor(enc_c, dtype=torch.long))
            self.rejected_ids.append(torch.tensor(enc_r, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chosen_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.chosen_ids[idx], self.rejected_ids[idx]

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad and stack chosen/rejected tensors into batches."""
        chosen   = [c for c, _ in batch]
        rejected = [r for _, r in batch]
        chosen_padded   = torch.nn.utils.rnn.pad_sequence(chosen,   batch_first=True)
        rejected_padded = torch.nn.utils.rnn.pad_sequence(rejected, batch_first=True)
        return chosen_padded, rejected_padded
