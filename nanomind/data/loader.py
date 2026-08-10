"""
nanomind/data/loader.py — DataLoader factory for NanoMind training.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from nanomind.data.split import split_dataset
from nanomind.tokenizer.base import BaseTokenizer


def get_dataloaders(
    text: str,
    tokenizer: BaseTokenizer,
    block_size: int,
    batch_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from raw text.

    Tokenizes the corpus, creates a :class:`~nanomind.data.TextDataset`,
    splits it, and returns ready-to-use DataLoaders.

    Args:
        text:         Raw training corpus string.
        tokenizer:    A fitted tokenizer instance.
        block_size:   Context window length in tokens.
        batch_size:   Number of sequences per batch.
        val_fraction: Fraction held out for validation.
        seed:         Random seed for the train/val split.
        num_workers:  DataLoader worker processes.
        pin_memory:   Pin tensors to page-locked memory (faster GPU transfer).

    Returns:
        ``(train_loader, val_loader)``
    """
    from nanomind.data.dataset import TextDataset

    dataset = TextDataset.from_string(text, tokenizer, block_size)
    train_ds, val_ds = split_dataset(dataset, val_fraction=val_fraction, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return train_loader, val_loader


def get_dataloaders_from_config(
    cfg: "DataConfig",  # type: ignore[name-defined]  # noqa: F821
) -> tuple[DataLoader, DataLoader, BaseTokenizer]:
    """
    Build DataLoaders and tokenizer from a :class:`~nanomind.data.DataConfig`.

    This is the primary entry point used by the training script.
    Reads the text file, builds/fits the tokenizer, and returns
    everything needed for training.

    Args:
        cfg: A :class:`~nanomind.data.DataConfig` instance.

    Returns:
        ``(train_loader, val_loader, tokenizer)``
    """
    from pathlib import Path
    from nanomind.data.config import DataConfig
    from nanomind.tokenizer.factory import get_tokenizer
    from nanomind.utils.logger import get_logger

    log = get_logger(__name__)
    text = Path(cfg.data_path).read_text(encoding="utf-8")
    log.info(f"Loaded {len(text):,} characters from '{cfg.data_path}'")

    TokenizerClass = get_tokenizer(cfg.tokenizer)
    tok = TokenizerClass()
    if cfg.tokenizer == "bpe":
        tok.train(text, vocab_size=cfg.bpe_vocab_size)  # type: ignore[attr-defined]
    else:
        tok.build(text)  # type: ignore[attr-defined]
    log.info(f"Tokenizer: {tok}")

    train_loader, val_loader = get_dataloaders(
        text=text,
        tokenizer=tok,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    log.info(
        f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}"
    )
    return train_loader, val_loader, tok
