"""
nanomind/data/config.py — Configuration dataclass for the data pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """All configuration needed to build dataloaders for NanoMind training.

    Attributes:
        data_path:      Path to the training text file.
        tokenizer:      Tokenizer name (``"char"`` or ``"bpe"``).
        block_size:     Context window length in tokens.
        batch_size:     Number of sequences per training batch.
        val_fraction:   Fraction of data held out for validation (0.0–1.0).
        num_workers:    DataLoader worker processes (0 = main process only).
        pin_memory:     Whether to pin DataLoader tensors to page-locked memory.
        bpe_vocab_size: Target vocabulary size when using the BPE tokenizer.
        seed:           Random seed for the train/val split.
    """

    data_path: str = "data.txt"
    tokenizer: str = "char"
    block_size: int = 128
    batch_size: int = 32
    val_fraction: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    bpe_vocab_size: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        assert 0.0 < self.val_fraction < 1.0, "val_fraction must be in (0, 1)"
        assert self.block_size > 0, "block_size must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.tokenizer in ("char", "bpe"), (
            f"Unknown tokenizer '{self.tokenizer}'. Choose 'char' or 'bpe'."
        )
