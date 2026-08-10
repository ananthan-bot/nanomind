"""
day4_commits.py — 20 atomic commits for Day 4: Data Pipeline.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 4: Data Pipeline — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — data package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/__init__.py", '"""NanoMind data pipeline sub-package."""\n')
commit("feat: add nanomind/data/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — DataConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/config.py", '''\
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
''')
commit("feat: add DataConfig dataclass with validation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — TextDataset skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/dataset.py", '''\
"""
nanomind/data/dataset.py — PyTorch Dataset for language model training.

Implements a sliding-window dataset where each sample is a pair
(x, y) of consecutive token sequences of length `block_size`.
The model learns to predict y[t] given x[0..t].
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from nanomind.tokenizer.base import BaseTokenizer


class TextDataset(Dataset):
    """
    Sliding-window character/token language modelling dataset.

    Each sample ``(x, y)`` satisfies:
        - ``x = tokens[i : i + block_size]``
        - ``y = tokens[i + 1 : i + block_size + 1]``

    so the model learns to predict the *next* token at every position.

    Args:
        tokens:     1-D integer tensor of all token IDs.
        block_size: Context window length in tokens.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        assert len(tokens) > block_size, (
            f"Dataset too small: {len(tokens)} tokens <= block_size {block_size}"
        )
        self._tokens = tokens
        self._block_size = block_size

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def block_size(self) -> int:
        """Context window length."""
        return self._block_size

    @property
    def num_tokens(self) -> int:
        """Total number of tokens in the dataset."""
        return len(self._tokens)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
''')
commit("feat: add TextDataset class skeleton with tokens, block_size, and properties")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — implement __len__
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/data/dataset.py")
src = src.replace(
    "    def __len__(self) -> int:\n        raise NotImplementedError",
    """\
    def __len__(self) -> int:
        \"\"\"Number of valid (x, y) pairs in the dataset.\"\"\"
        return len(self._tokens) - self._block_size"""
)
write("nanomind/data/dataset.py", src)
commit("feat: implement TextDataset.__len__() — number of sliding windows")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — implement __getitem__
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/data/dataset.py")
src = src.replace(
    "    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:\n        raise NotImplementedError",
    """\
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Return the (input, target) pair at position ``idx``.

        Args:
            idx: Starting position in the token stream.

        Returns:
            Tuple ``(x, y)`` where both are 1-D LongTensors of length
            ``block_size``.
        \"\"\"
        chunk = self._tokens[idx : idx + self._block_size + 1]
        x = chunk[:-1].clone()   # input  tokens
        y = chunk[1:].clone()    # target tokens (shifted by 1)
        return x, y"""
)
write("nanomind/data/dataset.py", src)
commit("feat: implement TextDataset.__getitem__() — sliding window (x, y) pairs")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — from_string() constructor
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/data/dataset.py")
src += '''
    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_string(
        cls,
        text: str,
        tokenizer: "BaseTokenizer",
        block_size: int,
    ) -> "TextDataset":
        """
        Build a :class:`TextDataset` from a raw text string.

        Args:
            text:       The training corpus as a string.
            tokenizer:  A fitted tokenizer (CharTokenizer or BPETokenizer).
            block_size: Context window length.

        Returns:
            A :class:`TextDataset` ready for use with a DataLoader.
        """
        ids = tokenizer.encode(text)
        tokens = torch.tensor(ids, dtype=torch.long)
        return cls(tokens, block_size)
'''
write("nanomind/data/dataset.py", src)
commit("feat: add TextDataset.from_string() factory constructor")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — from_file() constructor
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/data/dataset.py")
src += '''
    @classmethod
    def from_file(
        cls,
        path: str,
        tokenizer: "BaseTokenizer",
        block_size: int,
        encoding: str = "utf-8",
    ) -> "TextDataset":
        """
        Build a :class:`TextDataset` by reading a text file from disk.

        Args:
            path:       Path to the training text file.
            tokenizer:  A fitted tokenizer.
            block_size: Context window length.
            encoding:   File encoding (default: utf-8).

        Returns:
            A :class:`TextDataset` ready for use with a DataLoader.
        """
        from pathlib import Path
        text = Path(path).read_text(encoding=encoding)
        return cls.from_string(text, tokenizer, block_size)

    # ── Info ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TextDataset("
            f"num_tokens={self.num_tokens:,}, "
            f"block_size={self.block_size}, "
            f"samples={len(self):,})"
        )
'''
write("nanomind/data/dataset.py", src)
commit("feat: add TextDataset.from_file() constructor and __repr__")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — train/val split function
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/split.py", '''\
"""
nanomind/data/split.py — Train/validation split utilities.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, random_split, Subset


def split_dataset(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Split a dataset into train and validation subsets.

    Uses a reproducible random split (seeded via a dedicated Generator)
    so the split is identical across runs.

    Args:
        dataset:      The full dataset to split.
        val_fraction: Fraction of samples for validation (default: 10%).
        seed:         Random seed for reproducibility.

    Returns:
        ``(train_subset, val_subset)`` — both are :class:`torch.utils.data.Subset`.

    Example::

        train_ds, val_ds = split_dataset(dataset, val_fraction=0.1, seed=42)
    """
    n = len(dataset)  # type: ignore[arg-type]
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds
''')
commit("feat: add split_dataset() — reproducible train/val split with seeded Generator")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — get_dataloaders() factory (basic)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/loader.py", '''\
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
''')
commit("feat: add get_dataloaders() factory — tokenize, split, and wrap in DataLoaders")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — get_dataloaders_from_config()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/data/loader.py")
src += '''

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
'''
write("nanomind/data/loader.py", src)
commit("feat: add get_dataloaders_from_config() — one-call setup from DataConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — dataset statistics helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/stats.py", '''\
"""
nanomind/data/stats.py — Dataset statistics and diagnostics.
"""

from __future__ import annotations

import torch

from nanomind.tokenizer.base import BaseTokenizer


def dataset_stats(text: str, tokenizer: BaseTokenizer) -> dict:
    """
    Compute basic statistics about a tokenized corpus.

    Args:
        text:      Raw corpus string.
        tokenizer: A fitted tokenizer.

    Returns:
        Dictionary with keys:
        - ``num_chars``     : Total characters
        - ``num_tokens``    : Total tokens after encoding
        - ``vocab_size``    : Tokenizer vocabulary size
        - ``coverage``      : Fraction of unique chars in tokenizer vocab
        - ``compression``   : tokens / chars ratio (< 1 means compression)
    """
    ids = tokenizer.encode(text)
    unique_chars = set(text)
    covered = sum(1 for ch in unique_chars if ch in getattr(tokenizer, "_char_to_id", {}))

    return {
        "num_chars":   len(text),
        "num_tokens":  len(ids),
        "vocab_size":  tokenizer.vocab_size,
        "coverage":    covered / max(len(unique_chars), 1),
        "compression": len(ids) / max(len(text), 1),
    }


def print_stats(stats: dict) -> None:
    """Pretty-print dataset statistics."""
    print("Dataset Statistics")
    print("─" * 35)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v:,}")
''')
commit("feat: add dataset_stats() — corpus statistics (tokens, coverage, compression)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — IterableTextDataset for streaming large files
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/iterable.py", '''\
"""
nanomind/data/iterable.py — Streaming dataset for large text corpora.

For corpora too large to fit in RAM, IterableTextDataset streams
fixed-size chunks from disk without loading the full file.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from nanomind.tokenizer.base import BaseTokenizer


class IterableTextDataset(IterableDataset):
    """
    Streaming language model dataset for very large text files.

    Reads the file line-by-line, encodes on-the-fly, and yields
    ``(x, y)`` sliding-window pairs without holding everything in RAM.

    Args:
        path:       Path to the text file.
        tokenizer:  A fitted tokenizer.
        block_size: Context window length.
        encoding:   File text encoding.
    """

    def __init__(
        self,
        path: str,
        tokenizer: BaseTokenizer,
        block_size: int,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._encoding = encoding

    def __iter__(self):
        buffer: list[int] = []
        bs = self._block_size

        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                buffer.extend(self._tokenizer.encode(line))
                while len(buffer) >= bs + 1:
                    chunk = buffer[: bs + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield x, y
                    buffer = buffer[bs:]  # Advance by block_size (non-overlapping)

    def __repr__(self) -> str:
        return (
            f"IterableTextDataset("
            f"path='{self._path}', "
            f"block_size={self._block_size})"
        )
''')
commit("feat: add IterableTextDataset — streaming dataset for large text files")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — prefetch context manager
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/prefetch.py", '''\
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
''')
commit("feat: add PrefetchLoader — async CPU-to-GPU prefetching for DataLoader")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — update data __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/__init__.py", '''\
"""NanoMind data pipeline sub-package.

Primary entry points:
    - :func:`get_dataloaders`             — build loaders from text + tokenizer
    - :func:`get_dataloaders_from_config` — build loaders from DataConfig
    - :class:`TextDataset`                — map-style sliding window dataset
    - :class:`IterableTextDataset`        — streaming dataset for large files
    - :class:`DataConfig`                 — configuration dataclass
"""

from nanomind.data.config import DataConfig
from nanomind.data.dataset import TextDataset
from nanomind.data.iterable import IterableTextDataset
from nanomind.data.split import split_dataset
from nanomind.data.loader import get_dataloaders, get_dataloaders_from_config
from nanomind.data.stats import dataset_stats, print_stats
from nanomind.data.prefetch import PrefetchLoader

__all__ = [
    "DataConfig",
    "TextDataset",
    "IterableTextDataset",
    "split_dataset",
    "get_dataloaders",
    "get_dataloaders_from_config",
    "dataset_stats",
    "print_stats",
    "PrefetchLoader",
]
''')
commit("refactor: export all data pipeline components from nanomind/data/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: sliding window correctness
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_data.py", '''\
"""
tests/test_data.py — Tests for the NanoMind data pipeline.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from nanomind.data import (
    DataConfig,
    TextDataset,
    IterableTextDataset,
    split_dataset,
    get_dataloaders,
    dataset_stats,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = (
    "abcdefghijklmnopqrstuvwxyz " * 40
)
BLOCK_SIZE = 16
BATCH_SIZE = 4


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer().build(CORPUS)


@pytest.fixture
def dataset(tokenizer) -> TextDataset:
    return TextDataset.from_string(CORPUS, tokenizer, BLOCK_SIZE)


# ── TextDataset ───────────────────────────────────────────────────────────────

class TestTextDataset:
    def test_len(self, dataset):
        expected = dataset.num_tokens - BLOCK_SIZE
        assert len(dataset) == expected

    def test_item_shapes(self, dataset):
        x, y = dataset[0]
        assert x.shape == (BLOCK_SIZE,)
        assert y.shape == (BLOCK_SIZE,)

    def test_x_y_shifted_by_one(self, dataset):
        x, y = dataset[0]
        # y should be x shifted right by 1
        assert torch.equal(x[1:], y[:-1])

    def test_consecutive_windows_overlap(self, dataset):
        x0, _ = dataset[0]
        x1, _ = dataset[1]
        # Consecutive windows overlap by (block_size - 1) tokens
        assert torch.equal(x0[1:], x1[:-1])

    def test_dtype_is_long(self, dataset):
        x, y = dataset[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long
''')
commit("test: add TextDataset sliding window correctness and shape tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: train/val split
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── Split ─────────────────────────────────────────────────────────────────────

class TestSplitDataset:
    def test_sizes_sum_to_total(self, dataset):
        train_ds, val_ds = split_dataset(dataset, val_fraction=0.1)
        assert len(train_ds) + len(val_ds) == len(dataset)

    def test_val_fraction_respected(self, dataset):
        frac = 0.2
        _, val_ds = split_dataset(dataset, val_fraction=frac)
        ratio = len(val_ds) / len(dataset)
        assert abs(ratio - frac) < 0.01

    def test_reproducible_with_same_seed(self, dataset):
        train1, _ = split_dataset(dataset, seed=99)
        train2, _ = split_dataset(dataset, seed=99)
        # Same seed => same split => same first indices
        assert train1.indices[:5] == train2.indices[:5]

    def test_different_seeds_differ(self, dataset):
        train1, _ = split_dataset(dataset, seed=0)
        train2, _ = split_dataset(dataset, seed=1)
        assert train1.indices[:5] != train2.indices[:5]
'''
write("tests/test_data.py", src)
commit("test: add train/val split size and reproducibility tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: DataLoader batch shapes
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── DataLoader ────────────────────────────────────────────────────────────────

class TestGetDataloaders:
    def test_returns_two_loaders(self, tokenizer):
        train, val = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        assert train is not None
        assert val is not None

    def test_train_batch_shape(self, tokenizer):
        train, _ = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        x, y = next(iter(train))
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert y.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_val_batch_shape(self, tokenizer):
        _, val = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        x, y = next(iter(val))
        assert x.shape[1] == BLOCK_SIZE

    def test_train_larger_than_val(self, tokenizer):
        train, val = get_dataloaders(CORPUS, tokenizer, BLOCK_SIZE, BATCH_SIZE)
        assert len(train) >= len(val)
'''
write("tests/test_data.py", src)
commit("test: add DataLoader batch shape and size tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: from_file constructor
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── from_file ─────────────────────────────────────────────────────────────────

class TestFromFile:
    def test_from_file_matches_from_string(self, tokenizer, tmp_path):
        p = tmp_path / "corpus.txt"
        p.write_text(CORPUS, encoding="utf-8")
        ds_file   = TextDataset.from_file(str(p), tokenizer, BLOCK_SIZE)
        ds_string = TextDataset.from_string(CORPUS, tokenizer, BLOCK_SIZE)
        assert len(ds_file) == len(ds_string)
        x_file, y_file     = ds_file[0]
        x_string, y_string = ds_string[0]
        assert torch.equal(x_file, x_string)
        assert torch.equal(y_file, y_string)
'''
write("tests/test_data.py", src)
commit("test: add from_file constructor test (matches from_string output)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: dataset stats + IterableTextDataset
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── Stats ─────────────────────────────────────────────────────────────────────

class TestDatasetStats:
    def test_keys_present(self, tokenizer):
        stats = dataset_stats(CORPUS, tokenizer)
        assert "num_chars" in stats
        assert "num_tokens" in stats
        assert "vocab_size" in stats

    def test_num_chars_correct(self, tokenizer):
        stats = dataset_stats(CORPUS, tokenizer)
        assert stats["num_chars"] == len(CORPUS)

    def test_compression_positive(self, tokenizer):
        stats = dataset_stats(CORPUS, tokenizer)
        assert stats["compression"] > 0


# ── IterableTextDataset ───────────────────────────────────────────────────────

class TestIterableTextDataset:
    def test_yields_correct_shapes(self, tokenizer, tmp_path):
        p = tmp_path / "big.txt"
        p.write_text(CORPUS, encoding="utf-8")
        ds = IterableTextDataset(str(p), tokenizer, BLOCK_SIZE)
        batch = next(iter(ds))
        x, y = batch
        assert x.shape == (BLOCK_SIZE,)
        assert y.shape == (BLOCK_SIZE,)

    def test_x_y_shifted(self, tokenizer, tmp_path):
        p = tmp_path / "big.txt"
        p.write_text(CORPUS, encoding="utf-8")
        ds = IterableTextDataset(str(p), tokenizer, BLOCK_SIZE)
        x, y = next(iter(ds))
        assert torch.equal(x[1:], y[:-1])
'''
write("tests/test_data.py", src)
commit("test: add dataset_stats and IterableTextDataset tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update DataConfig in configs, README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
# Update small.yaml to include data section
old_small = read("configs/small.yaml")
write("configs/small.yaml", '''\
# NanoMind — Small model config (~1M params, fast CPU training)
data:
  tokenizer: char
  block_size: 128
  batch_size: 32
  val_fraction: 0.1

model:
  d_model: 128
  n_layers: 4
  n_heads: 4
  block_size: 128
  dropout: 0.1

train:
  max_iters: 5000
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  warmup_iters: 100
  grad_clip: 1.0
  eval_interval: 200
  device: auto
''')

readme = read("README.md")
readme = readme.replace(
    "| 4 | Data pipeline | 🔜 |",
    "| 4 | Data pipeline | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- BPE tokenizer with merge learning, encode/decode, persistence, factory (Day 3)",
    "- BPE tokenizer with merge learning, encode/decode, persistence, factory (Day 3)\n- Data pipeline: TextDataset, IterableTextDataset, DataLoaders, PrefetchLoader, stats (Day 4)"
)
write("CHANGELOG.md", cl)
commit("chore: export data pipeline from package; update configs, README, CHANGELOG for Day 4")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 4 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 4 COMPLETE ===")
