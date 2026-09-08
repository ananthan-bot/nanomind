"""
nanomind/dpo/dataset.py — Dataset for DPO preference training.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.dpo.config import DPOConfig


class DPODataset(Dataset):
    """
    Dataset of (prompt, chosen, rejected) pairs for DPO training.

    Each item yields:
      - ``chosen_ids``:    prompt + chosen completion tokens
      - ``rejected_ids``:  prompt + rejected completion tokens
      - ``prompt_len``:    number of prompt tokens (for masking)

    The prompt tokens are masked out during loss computation — we only
    compute log-probs over the completion portion.

    Args:
        pairs:      List of ``(prompt, chosen_text, rejected_text)`` tuples.
        tokenizer:  Tokenizer for encoding.
        cfg:        DPO configuration.
    """

    def __init__(
        self,
        pairs:     list[tuple[str, str, str]],
        tokenizer: BaseTokenizer,
        cfg:       DPOConfig | None = None,
    ) -> None:
        cfg = cfg or DPOConfig()
        self.items: list[dict] = []

        for prompt, chosen, rejected in pairs:
            prompt_ids   = tokenizer.encode(prompt)[:cfg.max_prompt_length]
            chosen_ids   = tokenizer.encode(chosen)
            rejected_ids = tokenizer.encode(rejected)

            # Concatenate prompt + completion, truncate to max_length
            c_full = (prompt_ids + chosen_ids)[:cfg.max_length]
            r_full = (prompt_ids + rejected_ids)[:cfg.max_length]

            self.items.append({
                "chosen_ids":   torch.tensor(c_full,   dtype=torch.long),
                "rejected_ids": torch.tensor(r_full,   dtype=torch.long),
                "prompt_len":   len(prompt_ids),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Pad sequences and return completion masks."""
        chosen   = [b["chosen_ids"]   for b in batch]
        rejected = [b["rejected_ids"] for b in batch]
        plens    = [b["prompt_len"]    for b in batch]

        chosen_padded   = torch.nn.utils.rnn.pad_sequence(chosen,   batch_first=True)
        rejected_padded = torch.nn.utils.rnn.pad_sequence(rejected, batch_first=True)

        # Completion masks: 1 for completion tokens, 0 for prompt + padding
        def make_mask(seqs, lengths):
            masks = []
            for seq, plen in zip(seqs, lengths):
                m = torch.zeros(seq.shape[0], dtype=torch.bool)
                m[plen:] = True
                masks.append(m)
            return torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)

        return {
            "chosen_ids":       chosen_padded,
            "rejected_ids":     rejected_padded,
            "chosen_mask":      make_mask(chosen,   plens),
            "rejected_mask":    make_mask(rejected, plens),
            "prompt_lengths":   torch.tensor(plens, dtype=torch.long),
        }
