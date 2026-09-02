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
print(f"
InMemory train: {len(train_ds)} samples, val: {len(val_ds)} samples")

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

    print(f"
DataPipeline: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val samples")
    x, y = next(iter(train_loader))
    print(f"Batch shape: x={x.shape}, y={y.shape}")

# ── 4. Token throughput ───────────────────────────────────────────────────────
simple_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
tps = estimate_tokens_per_second(simple_loader, n_batches=5)
print(f"
Throughput: {tps:,.0f} tokens/sec")

print_dataset_report(train_ds)
