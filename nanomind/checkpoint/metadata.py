"""
nanomind/checkpoint/metadata.py — Checkpoint metadata utilities.

Every saved checkpoint carries a metadata dict that records the full
training state at save time, making it easy to inspect checkpoints
without loading the full model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def make_metadata(
    step: int,
    train_loss: float,
    val_loss: float,
    model_config: dict,
    extra: dict | None = None,
) -> dict:
    """
    Build a checkpoint metadata dictionary.

    Args:
        step:         Training step at save time.
        train_loss:   Latest training loss.
        val_loss:     Latest validation loss.
        model_config: Serialized ModelConfig dict.
        extra:        Any additional fields to include.

    Returns:
        Metadata dict (JSON-serializable).
    """
    meta = {
        "step":         step,
        "train_loss":   float(train_loss),
        "val_loss":     float(val_loss),
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": model_config,
    }
    if extra:
        meta.update(extra)
    return meta


def save_metadata(meta: dict, path: str | Path) -> None:
    """Write metadata to a companion JSON file."""
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> dict:
    """Load metadata from a companion JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
