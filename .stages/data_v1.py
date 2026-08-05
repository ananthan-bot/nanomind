"""
data.py - TextDataset for NanoMind (DataLoader support coming next)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    Sliding-window character-level language modelling dataset.
    Each sample is (x, y) where y = x shifted by one token.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        self.n = len(tokens) - block_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y
