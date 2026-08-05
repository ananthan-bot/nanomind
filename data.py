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


from torch.utils.data import DataLoader, random_split


def get_dataloaders(
    data_path: str,
    block_size: int,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
):
    """Load text, tokenize, and return train/val DataLoaders + tokenizer."""
    from pathlib import Path as _P
    text = _P(data_path).read_text(encoding="utf-8")
    print(f"[data] Loaded {len(text):,} characters from '{data_path}'")

    tokenizer = CharTokenizer().build(text)
    print(f"[data] Vocabulary: {tokenizer.vocab_size} characters")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"[data] Token count: {len(ids):,}")

    dataset = TextDataset(ids, block_size)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    print(f"[data] Train: {n_train:,} | Val: {n_val:,} | Batches/epoch: {len(train_loader):,}")
    return train_loader, val_loader, tokenizer
