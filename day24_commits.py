"""
day24_commits.py — 20 atomic commits for Day 24: Streaming Data Pipeline.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

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

print("\n=== DAY 24: Streaming Data Pipeline — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — data package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/__init__.py",
      '"""NanoMind data pipeline sub-package — streaming tokenization and loading."""\n')
commit("feat: add nanomind/data/ package skeleton for streaming data pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — DataConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/config.py", '''\
"""
nanomind/data/config.py — Data pipeline configuration.

Efficient training on large text corpora requires:
  1. Streaming       — read data without loading everything into RAM
  2. Online tokenization — tokenize on-the-fly with caching
  3. Document packing    — pack multiple docs into one block_size chunk
                          (avoids wasting padding tokens)
  4. Data mixing         — blend multiple datasets with configurable weights
  5. Sharding            — split data into shards for multi-worker loading
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """
    Configuration for the NanoMind data pipeline.

    Attributes:
        block_size:      Context window length — all samples are exactly this long.
        batch_size:      Samples per batch.
        pack_documents:  If True, concatenate documents and chunk into block_size
                         windows instead of padding short documents. Maximises
                         GPU utilisation (no wasted padding tokens).
        eos_token_id:    End-of-document token inserted between packed documents.
        stride:          Sliding window stride when packing (block_size = no overlap).
        num_workers:     DataLoader worker processes.
        prefetch:        Number of batches to prefetch per worker.
        seed:            Random seed for shuffling.
        split_ratio:     (train, val) split ratio if single file provided.
        sources:         List of (path_or_name, weight) tuples for data mixing.
    """

    block_size:      int              = 512
    batch_size:      int              = 32
    pack_documents:  bool             = True
    eos_token_id:    int              = 0
    stride:          int | None       = None    # None = block_size (no overlap)
    num_workers:     int              = 2
    prefetch:        int              = 2
    seed:            int              = 42
    split_ratio:     tuple[float, float] = (0.9, 0.1)
    sources:         list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.block_size > 0
        assert self.batch_size > 0
        assert abs(sum(self.split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1"
        if self.stride is None:
            self.stride = self.block_size
        if self.sources:
            weights = [w for _, w in self.sources]
            assert all(w > 0 for w in weights), "source weights must be positive"
''')
commit("feat: add DataConfig — block_size, packing, mixing sources, stride, split_ratio")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — document packing utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/packing.py", '''\
"""
nanomind/data/packing.py — Document packing for efficient training.

Instead of padding short documents to block_size (wasting GPU compute on
padding tokens), document packing concatenates multiple documents end-to-end
and chunks the resulting stream into fixed-length block_size windows.

Example (block_size=8, EOS=0):
  Doc A: [1, 2, 3]
  Doc B: [4, 5, 6, 7, 8, 9]
  Doc C: [10, 11]

  Stream: [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 0, 10, 11, 0]
  Chunks: [1,2,3,0,4,5,6,7], [8,9,0,10,11,0,<pad>,<pad>]  ← last may be dropped

No wasted padding tokens! Used in: GPT-2, LLaMA, PaLM training.
"""

from __future__ import annotations

import torch


def pack_documents(
    token_lists:  list[list[int]],
    block_size:   int,
    eos_token_id: int  = 0,
    stride:       int | None = None,
    drop_last:    bool = True,
) -> list[list[int]]:
    """
    Pack a list of tokenised documents into fixed-length chunks.

    Concatenates all documents (separated by EOS) into a flat stream,
    then slices into chunks of exactly ``block_size`` tokens.

    Args:
        token_lists:   List of token ID lists, one per document.
        block_size:    Desired chunk length.
        eos_token_id:  Token inserted between documents.
        stride:        Sliding window stride (None = block_size, no overlap).
        drop_last:     Drop the final incomplete chunk.

    Returns:
        List of token chunks, each of length ``block_size``.
    """
    stride = stride or block_size

    # Flatten: [doc1_tokens, EOS, doc2_tokens, EOS, ...]
    flat: list[int] = []
    for tokens in token_lists:
        flat.extend(tokens)
        flat.append(eos_token_id)

    chunks: list[list[int]] = []
    i = 0
    while i + block_size <= len(flat):
        chunks.append(flat[i:i + block_size])
        i += stride

    if not drop_last and i < len(flat):
        last = flat[i:]
        # Pad to block_size
        last += [eos_token_id] * (block_size - len(last))
        chunks.append(last)

    return chunks


def make_input_target_pairs(
    chunks: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert packed chunks into (input, target) pairs for language modelling.

    Each chunk ``[t0, t1, ..., t_{n}]`` becomes:
      input  = ``[t0, t1, ..., t_{n-1}]``
      target = ``[t1, t2, ..., t_{n}]``

    Args:
        chunks: List of token ID lists of equal length.

    Returns:
        Tuple of ``(inputs, targets)`` tensors each ``(N, block_size - 1)``.
        Returns empty tensors if chunks is empty.
    """
    if not chunks:
        return torch.zeros(0, 1, dtype=torch.long), torch.zeros(0, 1, dtype=torch.long)

    data    = torch.tensor(chunks, dtype=torch.long)   # (N, block_size)
    inputs  = data[:, :-1]
    targets = data[:, 1:]
    return inputs, targets
''')
commit("feat: add pack_documents() and make_input_target_pairs() — zero-waste document packing")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — TextFileDataset
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/text_dataset.py", '''\
"""
nanomind/data/text_dataset.py — Text file dataset with online tokenization.
"""

from __future__ import annotations

import torch
from pathlib import Path
from torch.utils.data import Dataset

from nanomind.data.config import DataConfig
from nanomind.data.packing import pack_documents, make_input_target_pairs
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger

log = get_logger("data.text_dataset")


class TextFileDataset(Dataset):
    """
    Dataset that reads a plain text file, tokenizes it, and packs into blocks.

    Each item is a (input_ids, target_ids) tuple of length ``block_size - 1``.

    Tokenization is performed once at construction time (suitable for files
    that fit in memory). For very large files, use StreamingTextDataset instead.

    Args:
        path:      Path to the text file.
        tokenizer: Tokenizer for encoding the text.
        cfg:       Data pipeline configuration.
        split:     ``"train"`` or ``"val"`` — determines which portion to use.

    Example::

        ds = TextFileDataset("data/shakespeare.txt", tokenizer, DataConfig())
        loader = DataLoader(ds, batch_size=32)
    """

    def __init__(
        self,
        path:      str | Path,
        tokenizer: BaseTokenizer,
        cfg:       DataConfig | None = None,
        split:     str = "train",
    ) -> None:
        cfg  = cfg or DataConfig()
        path = Path(path)
        assert path.exists(), f"File not found: {path}"

        text = path.read_text(encoding="utf-8")
        log.info(f"Loaded {path.name}: {len(text):,} chars")

        # Tokenize
        all_ids = tokenizer.encode(text)
        log.info(f"Tokenized: {len(all_ids):,} tokens")

        # Split
        split_at = int(len(all_ids) * cfg.split_ratio[0])
        if split == "train":
            ids = all_ids[:split_at]
        else:
            ids = all_ids[split_at:]

        # Pack into block_size chunks
        if cfg.pack_documents:
            # Treat the entire split as one "document"
            chunks = pack_documents(
                [ids], cfg.block_size + 1,
                eos_token_id=cfg.eos_token_id,
                stride=cfg.stride,
            )
        else:
            # Naive chunking
            chunks = [ids[i:i + cfg.block_size + 1]
                      for i in range(0, len(ids) - cfg.block_size, cfg.block_size)]

        # Build (input, target) pairs
        valid = [c for c in chunks if len(c) == cfg.block_size + 1]
        self.inputs, self.targets = make_input_target_pairs(valid)
        log.info(f"TextFileDataset [{split}]: {len(self)} samples × {cfg.block_size} tokens")

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
''')
commit("feat: add TextFileDataset — read text file, tokenize, pack, split into train/val")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — InMemoryTokenDataset (fast packed dataset from raw token list)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/token_dataset.py", '''\
"""
nanomind/data/token_dataset.py — Fast in-memory dataset from a pre-tokenized token array.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class InMemoryTokenDataset(Dataset):
    """
    Efficient dataset from a pre-tokenized integer array.

    Stores a flat token array and creates sliding-window (input, target)
    pairs on the fly. Zero memory overhead beyond the token array itself.

    Args:
        tokens:     Flat token ID tensor or list.
        block_size: Context window length.
        stride:     Step between windows (None = block_size, no overlap).

    Example::

        tokens = tokenizer.encode(large_text)
        ds     = InMemoryTokenDataset(tokens, block_size=512)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        tokens:     list[int] | torch.Tensor,
        block_size: int,
        stride:     int | None = None,
    ) -> None:
        self.data       = (
            tokens if isinstance(tokens, torch.Tensor)
            else torch.tensor(tokens, dtype=torch.long)
        )
        self.block_size = block_size
        self.stride     = stride or block_size

        # Precompute start indices
        self.starts = list(range(0, len(self.data) - block_size, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s   = self.starts[idx]
        tok = self.data[s:s + self.block_size + 1]
        return tok[:-1], tok[1:]

    def split(self, ratio: float = 0.9) -> tuple["InMemoryTokenDataset", "InMemoryTokenDataset"]:
        """Split into train and val datasets at the given ratio."""
        split_at = int(len(self.data) * ratio)
        train    = InMemoryTokenDataset(self.data[:split_at], self.block_size, self.stride)
        val      = InMemoryTokenDataset(self.data[split_at:], self.block_size, self.stride)
        return train, val
''')
commit("feat: add InMemoryTokenDataset — zero-overhead sliding window dataset from token array")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — MixedDataset (blend multiple datasets by weight)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/mixed_dataset.py", '''\
"""
nanomind/data/mixed_dataset.py — Blend multiple datasets by sampling weight.

Data mixing allows training on heterogeneous corpora (e.g., code + web + books)
with custom per-source ratios. Each __getitem__ samples a dataset proportionally
to its weight and draws a random item from it.

Used in: LLaMA (code+books+web), Gemini, PaLM training recipes.
"""

from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    """
    Blend multiple datasets with configurable sampling weights.

    Args:
        datasets: List of ``(dataset, weight)`` pairs. Weights need not sum to 1.
        seed:     Random seed for reproducibility.
        length:   Virtual dataset length (number of samples to expose).

    Example::

        mixed = MixedDataset([
            (code_ds, 0.35),
            (web_ds,  0.45),
            (book_ds, 0.20),
        ], length=100_000)
        loader = DataLoader(mixed, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        datasets: list[tuple[Dataset, float]],
        seed:     int = 42,
        length:   int | None = None,
    ) -> None:
        assert len(datasets) > 0, "Need at least one dataset"
        assert all(w > 0 for _, w in datasets), "All weights must be positive"

        self.datasets = [ds for ds, _ in datasets]
        raw_weights   = [w for _, w in datasets]
        total         = sum(raw_weights)
        self.weights  = [w / total for w in raw_weights]
        self.rng      = random.Random(seed)
        self._length  = length or sum(len(ds) for ds in self.datasets)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample a dataset proportional to weight
        ds     = self.rng.choices(self.datasets, weights=self.weights, k=1)[0]
        inner  = self.rng.randint(0, len(ds) - 1)
        return ds[inner]

    def source_stats(self) -> dict:
        """Return per-source name, size, and weight."""
        return {
            f"source_{i}": {
                "size":   len(ds),
                "weight": self.weights[i],
            }
            for i, ds in enumerate(self.datasets)
        }
''')
commit("feat: add MixedDataset — blend multiple datasets with configurable sampling weights")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — ShardedFileLoader — load from multiple shard files
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/sharded.py", '''\
"""
nanomind/data/sharded.py — Sharded dataset: load from multiple text files.

Large training corpora are stored as multiple shard files (e.g., shard_000.txt,
shard_001.txt, ...). ShardedDataset loads shards lazily, one at a time,
to avoid exceeding RAM.
"""

from __future__ import annotations

import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset

from nanomind.data.config import DataConfig
from nanomind.data.packing import pack_documents, make_input_target_pairs
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger

log = get_logger("data.sharded")


class ShardedDataset(Dataset):
    """
    Dataset that concatenates multiple shard files into one logical dataset.

    Tokenizes and packs each shard file independently, then concatenates
    all samples across shards.

    Args:
        shard_paths: List of text file paths (one shard per file).
        tokenizer:   Tokenizer for encoding.
        cfg:         Data pipeline configuration.

    Note:
        All shards are loaded into memory at construction. For truly streaming
        (out-of-core) loading, use a custom IterableDataset instead.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        tokenizer:   BaseTokenizer,
        cfg:         DataConfig | None = None,
    ) -> None:
        cfg  = cfg or DataConfig()
        all_inputs:  list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for path in shard_paths:
            path = Path(path)
            text = path.read_text(encoding="utf-8")
            ids  = tokenizer.encode(text)
            chunks = pack_documents(
                [ids], cfg.block_size + 1,
                eos_token_id=cfg.eos_token_id,
                stride=cfg.stride,
            )
            valid  = [c for c in chunks if len(c) == cfg.block_size + 1]
            if valid:
                x, y = make_input_target_pairs(valid)
                all_inputs.append(x)
                all_targets.append(y)
            log.debug(f"Loaded shard {path.name}: {len(valid)} samples")

        if all_inputs:
            self.inputs  = torch.cat(all_inputs,  dim=0)
            self.targets = torch.cat(all_targets, dim=0)
        else:
            self.inputs  = torch.zeros(0, cfg.block_size, dtype=torch.long)
            self.targets = torch.zeros(0, cfg.block_size, dtype=torch.long)

        log.info(f"ShardedDataset: {len(shard_paths)} shards → {len(self):,} samples")

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
''')
commit("feat: add ShardedDataset — load and pack multiple shard files into one dataset")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — DataPipeline high-level builder
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/pipeline.py", '''\
"""
nanomind/data/pipeline.py — High-level data pipeline builder.

DataPipeline provides a single entry point for constructing training and
validation DataLoaders from text files, with optional document packing
and data mixing.
"""

from __future__ import annotations

import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from nanomind.data.config import DataConfig
from nanomind.data.text_dataset import TextFileDataset
from nanomind.data.token_dataset import InMemoryTokenDataset
from nanomind.data.mixed_dataset import MixedDataset
from nanomind.tokenizer.base import BaseTokenizer
from nanomind.utils.logger import get_logger

log = get_logger("data.pipeline")


class DataPipeline:
    """
    High-level data pipeline for NanoMind training.

    Supports:
    - Single text file with train/val split
    - Multiple text files with mixing weights
    - Document packing for efficient GPU utilisation

    Args:
        tokenizer: Tokenizer for encoding.
        cfg:       Data configuration.

    Example::

        cfg      = DataConfig(block_size=512, batch_size=32, pack_documents=True)
        pipeline = DataPipeline(tokenizer, cfg)
        pipeline.add_source("data/books.txt", weight=0.6)
        pipeline.add_source("data/web.txt",   weight=0.4)

        train_loader, val_loader = pipeline.build()
        for x, y in train_loader:
            ...
    """

    def __init__(self, tokenizer: BaseTokenizer, cfg: DataConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.cfg       = cfg or DataConfig()
        self._sources: list[tuple[Path, float]] = []

    def add_source(self, path: str | Path, weight: float = 1.0) -> "DataPipeline":
        """Add a text file source with a mixing weight."""
        self._sources.append((Path(path), weight))
        return self   # enable chaining

    def build(self) -> tuple[DataLoader, DataLoader]:
        """
        Build and return ``(train_loader, val_loader)`` DataLoaders.

        Returns:
            Tuple of train and validation DataLoaders.
        """
        if not self._sources:
            raise ValueError("No data sources added. Call add_source() first.")

        train_datasets, val_datasets = [], []

        for path, weight in self._sources:
            train_ds = TextFileDataset(path, self.tokenizer, self.cfg, split="train")
            val_ds   = TextFileDataset(path, self.tokenizer, self.cfg, split="val")
            train_datasets.append((train_ds, weight))
            val_datasets.append((val_ds, weight))

        # Mix datasets
        if len(train_datasets) == 1:
            train_ds_final = train_datasets[0][0]
            val_ds_final   = val_datasets[0][0]
        else:
            train_ds_final = MixedDataset(
                train_datasets, seed=self.cfg.seed,
                length=sum(len(d) for d, _ in train_datasets)
            )
            val_ds_final = MixedDataset(
                val_datasets, seed=self.cfg.seed,
                length=sum(len(d) for d, _ in val_datasets)
            )

        def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
            return DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=shuffle,
                num_workers=self.cfg.num_workers,
                prefetch_factor=self.cfg.prefetch if self.cfg.num_workers > 0 else None,
                pin_memory=torch.cuda.is_available(),
            )

        train_loader = _loader(train_ds_final, shuffle=True)
        val_loader   = _loader(val_ds_final,   shuffle=False)

        log.info(
            f"DataPipeline: {len(train_ds_final)} train samples, "
            f"{len(val_ds_final)} val samples, "
            f"batch={self.cfg.batch_size}"
        )
        return train_loader, val_loader
''')
commit("feat: add DataPipeline — high-level builder for train/val DataLoaders with mixing")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — data stats utility
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/stats.py", '''\
"""
nanomind/data/stats.py — Dataset statistics and inspection utilities.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


def dataset_stats(dataset: Dataset, max_samples: int = 1000) -> dict:
    """
    Compute basic statistics for a token dataset.

    Args:
        dataset:     Token dataset yielding (x, y) pairs.
        max_samples: Maximum samples to scan.

    Returns:
        Dict with ``n_samples``, ``block_size``, ``token_coverage``,
        ``mean_target``, ``unique_tokens``.
    """
    n   = min(len(dataset), max_samples)
    all_tokens: list[torch.Tensor] = []

    for i in range(n):
        x, _ = dataset[i]
        all_tokens.append(x)

    data         = torch.stack(all_tokens)       # (n, block_size)
    unique       = data.unique().numel()
    mean_val     = data.float().mean().item()

    return {
        "n_samples":      len(dataset),
        "block_size":     data.shape[1],
        "scanned":        n,
        "unique_tokens":  unique,
        "mean_token_id":  mean_val,
        "min_token_id":   data.min().item(),
        "max_token_id":   data.max().item(),
    }


def estimate_tokens_per_second(
    loader:     DataLoader,
    n_batches:  int = 10,
) -> float:
    """
    Estimate throughput: tokens per second through the data pipeline.

    Args:
        loader:    DataLoader to benchmark.
        n_batches: Number of batches to time.

    Returns:
        Estimated tokens per second.
    """
    import time
    total_tokens = 0
    t0 = time.perf_counter()
    for i, (x, _) in enumerate(loader):
        total_tokens += x.numel()
        if i + 1 >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    return total_tokens / max(elapsed, 1e-9)


def print_dataset_report(dataset: Dataset) -> None:
    """Pretty-print a dataset summary."""
    stats = dataset_stats(dataset)
    print("=" * 50)
    print("Dataset Report")
    print("=" * 50)
    print(f"  Samples         : {stats['n_samples']:,}")
    print(f"  Block size      : {stats['block_size']}")
    print(f"  Unique tokens   : {stats['unique_tokens']:,}")
    print(f"  Token ID range  : [{stats['min_token_id']}, {stats['max_token_id']}]")
    print(f"  Mean token ID   : {stats['mean_token_id']:.1f}")
    print("=" * 50)
''')
commit("feat: add dataset_stats(), estimate_tokens_per_second(), print_dataset_report()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — update data __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/data/__init__.py", '''\
"""NanoMind data pipeline sub-package.

Efficient streaming tokenization, document packing, sharded loading,
and multi-source data mixing for large-scale language model training.

Key features:
  - Document packing: zero padding waste — pack multiple docs into block_size
  - Data mixing:      blend multiple corpora with configurable weights
  - Shard loading:    distribute large datasets across multiple files
  - Token throughput: benchmark tokens/sec through your pipeline

Primary exports:
    - :class:`DataPipeline`         — high-level train/val DataLoader builder
    - :class:`DataConfig`           — block_size, packing, mixing, stride config
    - :class:`TextFileDataset`      — text file → tokenize → pack → Dataset
    - :class:`InMemoryTokenDataset` — fast sliding-window dataset from token array
    - :class:`MixedDataset`         — blend multiple datasets by weight
    - :class:`ShardedDataset`       — load and concatenate multiple shard files
    - :func:`pack_documents`        — concatenate and chunk documents
    - :func:`make_input_target_pairs` — (input, target) pairs from chunks
    - :func:`dataset_stats`         — summary statistics for a dataset
    - :func:`print_dataset_report`  — pretty-print dataset summary
"""

from nanomind.data.config import DataConfig
from nanomind.data.packing import pack_documents, make_input_target_pairs
from nanomind.data.text_dataset import TextFileDataset
from nanomind.data.token_dataset import InMemoryTokenDataset
from nanomind.data.mixed_dataset import MixedDataset
from nanomind.data.sharded import ShardedDataset
from nanomind.data.pipeline import DataPipeline
from nanomind.data.stats import dataset_stats, print_dataset_report, estimate_tokens_per_second

__all__ = [
    "DataConfig",
    "DataPipeline",
    "TextFileDataset",
    "InMemoryTokenDataset",
    "MixedDataset",
    "ShardedDataset",
    "pack_documents",
    "make_input_target_pairs",
    "dataset_stats",
    "print_dataset_report",
    "estimate_tokens_per_second",
]
''')
commit("refactor: export all data pipeline components from nanomind/data/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: data_pipeline_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/data_pipeline_demo.py", '''\
"""
examples/data_pipeline_demo.py — Data pipeline demo.

Shows document packing, data mixing, and token throughput benchmarking.

Usage:
    python examples/data_pipeline_demo.py
"""

import tempfile, torch
from pathlib import Path

from nanomind.tokenizer.char import CharTokenizer
from nanomind.data import (
    DataConfig, DataPipeline, InMemoryTokenDataset,
    MixedDataset, pack_documents, print_dataset_report,
    estimate_tokens_per_second,
)
from torch.utils.data import DataLoader

# ── Tokenizer ─────────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog. " * 100
tokenizer = CharTokenizer().build(CORPUS)

# ── 1. Document packing demo ──────────────────────────────────────────────────
docs   = [tokenizer.encode(s) for s in CORPUS.split(". ") if s.strip()]
chunks = pack_documents(docs, block_size=33, eos_token_id=0)
print(f"Documents : {len(docs)}")
print(f"Chunks    : {len(chunks)} × 33 tokens (no padding waste!)")

# ── 2. InMemoryTokenDataset ───────────────────────────────────────────────────
tokens = torch.tensor(tokenizer.encode(CORPUS))
ds     = InMemoryTokenDataset(tokens, block_size=32)
train_ds, val_ds = ds.split(0.9)
print(f"\nInMemory train: {len(train_ds)} samples, val: {len(val_ds)} samples")

# ── 3. DataPipeline with temp files ──────────────────────────────────────────
cfg = DataConfig(block_size=32, batch_size=8, pack_documents=True, num_workers=0)
with tempfile.TemporaryDirectory() as tmp:
    # Write two "source" files
    f1 = Path(tmp) / "source_a.txt"
    f2 = Path(tmp) / "source_b.txt"
    f1.write_text(CORPUS[:len(CORPUS)//2], encoding="utf-8")
    f2.write_text(CORPUS[len(CORPUS)//2:], encoding="utf-8")

    pipeline = DataPipeline(tokenizer, cfg)
    pipeline.add_source(f1, weight=0.7).add_source(f2, weight=0.3)
    train_loader, val_loader = pipeline.build()

    print(f"\nDataPipeline: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val samples")
    x, y = next(iter(train_loader))
    print(f"Batch shape: x={x.shape}, y={y.shape}")

# ── 4. Token throughput ───────────────────────────────────────────────────────
simple_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
tps = estimate_tokens_per_second(simple_loader, n_batches=5)
print(f"\nThroughput: {tps:,.0f} tokens/sec")

print_dataset_report(train_ds)
''')
commit("feat: add examples/data_pipeline_demo.py — packing, mixing, pipeline, throughput demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: DataConfig
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_data.py", '''\
"""
tests/test_data.py — Tests for the NanoMind data pipeline.
"""

import pytest
import tempfile
import torch
from pathlib import Path

from nanomind.tokenizer.char import CharTokenizer
from nanomind.data import (
    DataConfig, InMemoryTokenDataset, MixedDataset,
    pack_documents, make_input_target_pairs, dataset_stats,
)
from nanomind.data.text_dataset import TextFileDataset

CORPUS    = "the quick brown fox jumps over the lazy dog. " * 20
TOKENIZER = CharTokenizer().build(CORPUS)
TOKENS    = torch.tensor(TOKENIZER.encode(CORPUS))
BLOCK     = 16


# ── DataConfig ────────────────────────────────────────────────────────────────

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.block_size == 512
        assert cfg.pack_documents is True
        assert cfg.stride == 512   # defaults to block_size

    def test_split_ratio_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            DataConfig(split_ratio=(0.7, 0.4))

    def test_invalid_block_size(self):
        with pytest.raises(AssertionError):
            DataConfig(block_size=0)

    def test_stride_defaults_to_block_size(self):
        cfg = DataConfig(block_size=64)
        assert cfg.stride == 64

    def test_custom_stride(self):
        cfg = DataConfig(block_size=64, stride=32)
        assert cfg.stride == 32
''')
commit("test: add DataConfig validation — defaults, split_ratio, block_size, stride tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: pack_documents
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── pack_documents ────────────────────────────────────────────────────────────

class TestPackDocuments:
    def test_output_chunk_length(self):
        docs   = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        chunks = pack_documents(docs, block_size=4)
        assert all(len(c) == 4 for c in chunks)

    def test_empty_docs(self):
        chunks = pack_documents([], block_size=8)
        assert chunks == []

    def test_eos_inserted(self):
        docs   = [[1, 2], [3, 4]]
        flat   = []
        for d in docs:
            flat.extend(d)
            flat.append(0)
        chunks = pack_documents(docs, block_size=len(flat), drop_last=False)
        assert 0 in chunks[0]

    def test_stride_overlap(self):
        docs       = [list(range(20))]
        no_overlap = pack_documents(docs, block_size=5, stride=5)
        overlap    = pack_documents(docs, block_size=5, stride=2)
        assert len(overlap) > len(no_overlap)

    def test_make_input_target_pairs_shape(self):
        chunks      = [[i for i in range(9)] for _ in range(4)]
        inputs, tgts = make_input_target_pairs(chunks)
        assert inputs.shape == (4, 8)
        assert tgts.shape   == (4, 8)
        assert torch.equal(inputs[:, 1:], tgts[:, :-1])

    def test_make_pairs_empty(self):
        x, y = make_input_target_pairs([])
        assert x.shape[0] == 0
'''
write("tests/test_data.py", src)
commit("test: add pack_documents chunk length, EOS, stride, and make_input_target_pairs tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: InMemoryTokenDataset
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── InMemoryTokenDataset ──────────────────────────────────────────────────────

class TestInMemoryTokenDataset:
    def test_len(self):
        ds = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        assert len(ds) > 0

    def test_item_shapes(self):
        ds   = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        x, y = ds[0]
        assert x.shape == (BLOCK,)
        assert y.shape == (BLOCK,)

    def test_target_is_shifted(self):
        ds   = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        x, y = ds[0]
        assert torch.equal(x[1:], y[:-1])

    def test_split(self):
        ds          = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        train, val  = ds.split(0.8)
        assert len(train) > len(val)
        assert len(train) + len(val) <= len(ds) + 2   # stride rounding

    def test_stride_fewer_samples(self):
        ds_full   = InMemoryTokenDataset(TOKENS, block_size=BLOCK, stride=BLOCK)
        ds_stride = InMemoryTokenDataset(TOKENS, block_size=BLOCK, stride=BLOCK // 2)
        assert len(ds_stride) > len(ds_full)
'''
write("tests/test_data.py", src)
commit("test: add InMemoryTokenDataset len, shape, shifted target, split, and stride tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: MixedDataset
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── MixedDataset ──────────────────────────────────────────────────────────────

class TestMixedDataset:
    def _make_ds(self, n=50):
        return InMemoryTokenDataset(TOKENS[:100], block_size=BLOCK)

    def test_len(self):
        ds1   = self._make_ds()
        ds2   = self._make_ds()
        mixed = MixedDataset([(ds1, 0.6), (ds2, 0.4)], length=100)
        assert len(mixed) == 100

    def test_item_shapes(self):
        ds1   = self._make_ds()
        ds2   = self._make_ds()
        mixed = MixedDataset([(ds1, 0.5), (ds2, 0.5)], length=50)
        x, y  = mixed[0]
        assert x.shape == (BLOCK,)

    def test_weights_normalised(self):
        ds1   = self._make_ds()
        mixed = MixedDataset([(ds1, 3.0)], length=10)
        assert abs(mixed.weights[0] - 1.0) < 1e-6

    def test_source_stats_keys(self):
        ds1   = self._make_ds()
        ds2   = self._make_ds()
        mixed = MixedDataset([(ds1, 0.7), (ds2, 0.3)])
        stats = mixed.source_stats()
        assert "source_0" in stats
        assert "weight" in stats["source_0"]

    def test_no_datasets_raises(self):
        with pytest.raises(AssertionError):
            MixedDataset([])
'''
write("tests/test_data.py", src)
commit("test: add MixedDataset len, item shape, weight normalisation, and source stats tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: TextFileDataset
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── TextFileDataset ───────────────────────────────────────────────────────────

class TestTextFileDataset:
    def test_basic_load(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(CORPUS, encoding="utf-8")
        cfg = DataConfig(block_size=BLOCK, num_workers=0)
        ds  = TextFileDataset(f, TOKENIZER, cfg, split="train")
        assert len(ds) > 0

    def test_item_shapes(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(CORPUS, encoding="utf-8")
        cfg = DataConfig(block_size=BLOCK, num_workers=0)
        ds  = TextFileDataset(f, TOKENIZER, cfg)
        x, y = ds[0]
        assert x.shape == (BLOCK,)
        assert y.shape == (BLOCK,)

    def test_train_larger_than_val(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(CORPUS * 5, encoding="utf-8")
        cfg  = DataConfig(block_size=BLOCK, num_workers=0, split_ratio=(0.9, 0.1))
        tr   = TextFileDataset(f, TOKENIZER, cfg, split="train")
        val  = TextFileDataset(f, TOKENIZER, cfg, split="val")
        assert len(tr) > len(val)

    def test_missing_file_raises(self):
        with pytest.raises(AssertionError):
            TextFileDataset("/nonexistent/file.txt", TOKENIZER)
'''
write("tests/test_data.py", src)
commit("test: add TextFileDataset load, shape, train>val, and missing file tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: dataset_stats
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── dataset_stats ─────────────────────────────────────────────────────────────

class TestDatasetStats:
    def test_keys_present(self):
        ds    = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        stats = dataset_stats(ds, max_samples=10)
        for key in ("n_samples", "block_size", "unique_tokens", "mean_token_id"):
            assert key in stats

    def test_n_samples_correct(self):
        ds    = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        stats = dataset_stats(ds)
        assert stats["n_samples"] == len(ds)

    def test_block_size_correct(self):
        ds    = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        stats = dataset_stats(ds, max_samples=5)
        assert stats["block_size"] == BLOCK

    def test_token_id_range(self):
        ds    = InMemoryTokenDataset(TOKENS, block_size=BLOCK)
        stats = dataset_stats(ds, max_samples=10)
        assert stats["min_token_id"] >= 0
        assert stats["max_token_id"] < TOKENIZER.vocab_size
'''
write("tests/test_data.py", src)
commit("test: add dataset_stats keys, n_samples, block_size, and token range tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: DataPipeline integration
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_data.py")
src += '''

# ── DataPipeline ──────────────────────────────────────────────────────────────

class TestDataPipeline:
    def test_no_sources_raises(self):
        from nanomind.data.pipeline import DataPipeline
        p = DataPipeline(TOKENIZER)
        with pytest.raises(ValueError):
            p.build()

    def test_single_source(self, tmp_path):
        from nanomind.data.pipeline import DataPipeline
        from torch.utils.data import DataLoader
        f   = tmp_path / "data.txt"
        f.write_text(CORPUS * 3, encoding="utf-8")
        cfg = DataConfig(block_size=BLOCK, batch_size=4, num_workers=0)
        p   = DataPipeline(TOKENIZER, cfg)
        p.add_source(f)
        train_dl, val_dl = p.build()
        assert isinstance(train_dl, DataLoader)
        x, y = next(iter(train_dl))
        assert x.shape == (4, BLOCK)

    def test_chaining(self, tmp_path):
        from nanomind.data.pipeline import DataPipeline
        f1  = tmp_path / "a.txt"
        f2  = tmp_path / "b.txt"
        f1.write_text(CORPUS * 2, encoding="utf-8")
        f2.write_text(CORPUS * 2, encoding="utf-8")
        cfg = DataConfig(block_size=BLOCK, batch_size=4, num_workers=0)
        p   = DataPipeline(TOKENIZER, cfg).add_source(f1, 0.6).add_source(f2, 0.4)
        train_dl, _ = p.build()
        assert len(train_dl.dataset) > 0
'''
write("tests/test_data.py", src)
commit("test: add DataPipeline no-sources, single-source, and chaining integration tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v2.0.0 + expose data package
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"1.9.0\"", "__version__ = \"2.0.0\"")
src = src.replace(
    "from nanomind.moe import MoEConfig, NanoMindMoE, SparseMoELayer",
    "from nanomind.moe import MoEConfig, NanoMindMoE, SparseMoELayer\n"
    "from nanomind.data import DataConfig, DataPipeline, InMemoryTokenDataset"
)
src = src.replace(
    "    \"SparseMoELayer\",\n    \"__version__\",\n]",
    "    \"SparseMoELayer\",\n"
    "    \"DataConfig\",\n"
    "    \"DataPipeline\",\n"
    "    \"InMemoryTokenDataset\",\n"
    "    \"__version__\",\n]"
)
write("nanomind/__init__.py", src)
commit("feat: bump to v2.0.0 — major milestone, expose DataConfig and DataPipeline in public API")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| **Architecture** | Mixture of Experts — N experts, top-K routing, load balance loss |",
    "| **Architecture** | Mixture of Experts — N experts, top-K routing, load balance loss |\n"
    "| **Data** | Streaming pipeline — document packing, multi-source mixing, sharding |"
)
readme = readme.replace(
    "**Total: 465 commits across 23 days.**",
    "**Total: 485 commits across 24 days.**"
)
if "**Total:" not in readme:
    readme = readme.replace(
        "**Total: 460 commits across 23 days.**",
        "**Total: 485 commits across 24 days.**"
    )
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "## [1.9.0] — 2024 — Mixture of Experts (MoE)",
    "## [2.0.0] — 2024 — Streaming Data Pipeline\n\n### Added\n"
    "- `DataPipeline` — high-level train/val DataLoader builder with source mixing\n"
    "- `DataConfig` — block_size, packing, mixing sources, stride, split_ratio\n"
    "- `TextFileDataset` — text file → tokenize → pack → Dataset\n"
    "- `InMemoryTokenDataset` — zero-overhead sliding-window dataset from token array\n"
    "- `MixedDataset` — blend multiple datasets with configurable sampling weights\n"
    "- `ShardedDataset` — load and concatenate multiple shard files into one dataset\n"
    "- `pack_documents()` — concatenate docs with EOS and chunk into blocks\n"
    "- `make_input_target_pairs()` — (input, target) pairs from chunks\n"
    "- `dataset_stats()` / `print_dataset_report()` — dataset inspection\n"
    "- `estimate_tokens_per_second()` — data pipeline throughput benchmark\n"
    "- `examples/data_pipeline_demo.py` — packing, mixing, and throughput demo\n\n---\n\n"
    "## [1.9.0] — 2024 — Mixture of Experts (MoE)"
)
write("CHANGELOG.md", cl)
commit("chore: bump to v2.0.0, update README and CHANGELOG for Day 24 Data Pipeline")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 24 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v2.0.0",
    "-m", "NanoMind v2.0.0 — Streaming Data Pipeline", check=False)
r = run("git", "push", "origin", "v2.0.0", check=False)
print("Tag v2.0.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 24 COMPLETE — v2.0.0 TAGGED! ===")
