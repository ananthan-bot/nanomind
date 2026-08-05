"""
data.py — Dataset and DataLoader utilities for MiniGPT
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    Sliding-window character-level language modelling dataset.

    Each sample is a pair (x, y) where:
        x = tokens[i : i + block_size]       (input)
        y = tokens[i + 1 : i + block_size + 1]  (target, shifted by 1)

    The model learns to predict y[t] given x[0..t].
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        """
        Args:
            tokens:     1-D integer tensor of all token IDs
            block_size: context window length
        """
        self.tokens = tokens
        self.block_size = block_size
        # Number of valid starting positions
        self.n = len(tokens) - block_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def get_dataloaders(
    data_path: str,
    block_size: int,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    Load a text file, build a tokenizer, and return train/val DataLoaders.

    Args:
        data_path:     path to the .txt training corpus
        block_size:    context window length
        batch_size:    samples per batch
        val_fraction:  fraction of data to use for validation (default 10%)
        num_workers:   DataLoader worker processes

    Returns:
        train_loader, val_loader, tokenizer
    """
    text = Path(data_path).read_text(encoding="utf-8")
    print(f"[data] Loaded {len(text):,} characters from '{data_path}'")

    tokenizer = CharTokenizer().build(text)
    print(f"[data] Vocabulary: {tokenizer.vocab_size} characters")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"[data] Token count: {len(ids):,}")

    dataset = TextDataset(ids, block_size)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(
        f"[data] Train samples: {n_train:,} | "
        f"Val samples: {n_val:,} | "
        f"Batches/epoch (train): {len(train_loader):,}"
    )

    return train_loader, val_loader, tokenizer
