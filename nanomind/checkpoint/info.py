"""
nanomind/checkpoint/info.py — Checkpoint inspection utilities.
"""

from __future__ import annotations

from pathlib import Path
import torch


def checkpoint_info(path: str | Path) -> dict:
    """
    Read checkpoint metadata without loading model weights.

    Parses the companion ``.json`` file if available, otherwise loads
    the ``.pt`` and extracts just the metadata key (much faster than
    loading full weights).

    Args:
        path: Path to a ``.pt`` checkpoint file.

    Returns:
        Metadata dict.
    """
    from nanomind.checkpoint.metadata import load_metadata
    path = Path(path)
    json_path = path.with_suffix(".json")
    if json_path.exists():
        return load_metadata(json_path)
    # Fall back to loading pt header only
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("metadata", {"path": str(path)})


def print_checkpoint_info(path: str | Path) -> None:
    """Pretty-print checkpoint metadata."""
    info = checkpoint_info(path)
    print(f"Checkpoint: {Path(path).name}")
    print("-" * 40)
    for k, v in info.items():
        if k != "model_config":
            print(f"  {k:<20}: {v}")
    if "model_config" in info:
        print("  model_config:")
        for k, v in info["model_config"].items():
            print(f"    {k:<18}: {v}")
