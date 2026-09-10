"""
nanomind/export/safetensors_export.py — SafeTensors weight export.

SafeTensors (HuggingFace, 2022) is a simple, safe, zero-copy format for
storing and loading neural network weights:

  Problems with pickle (.pt):
    - Arbitrary code execution (security risk on untrusted models)
    - Slow loading (Python overhead, no memory mapping)
    - Not cross-language (Python only)

  SafeTensors advantages:
    - Zero-code execution — just raw tensors
    - Memory-mapped loading — instant access without copying
    - Cross-language: Python, Rust, C++, JavaScript
    - Used by every HuggingFace model (Llama, Mistral, Falcon, etc.)

File format:
  [8 bytes: header_size][JSON header][tensor data (contiguous)]

NanoMind implements a pure-Python SafeTensors writer (no safetensors
library dependency) for pedagogical completeness.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.utils.logger import get_logger

log = get_logger("export.safetensors")


def _dtype_str(t: torch.Tensor) -> str:
    return {
        torch.float32:  "F32",
        torch.float16:  "F16",
        torch.bfloat16: "BF16",
        torch.int32:    "I32",
        torch.int64:    "I64",
        torch.int8:     "I8",
    }.get(t.dtype, "F32")


def save_safetensors(
    tensors: dict[str, torch.Tensor],
    path:    str | Path,
) -> None:
    """
    Save tensors in SafeTensors format (pure Python implementation).

    Args:
        tensors: Dict mapping parameter names to tensors.
        path:    Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build header: {name: {dtype, shape, data_offsets}}
    offset = 0
    header: dict = {}
    raw_parts: list[bytes] = []

    for name, tensor in tensors.items():
        t_contiguous = tensor.detach().cpu().contiguous()
        raw          = t_contiguous.numpy().tobytes()
        header[name] = {
            "dtype":        _dtype_str(t_contiguous),
            "shape":        list(t_contiguous.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        raw_parts.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header).encode("utf-8")
    header_size  = struct.pack("<Q", len(header_bytes))

    with path.open("wb") as f:
        f.write(header_size)
        f.write(header_bytes)
        for part in raw_parts:
            f.write(part)

    size_mb = path.stat().st_size / (1024 ** 2)
    log.info(f"SafeTensors saved to {path} ({size_mb:.2f} MB, {len(tensors)} tensors)")


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load tensors from a SafeTensors file (pure Python).

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        Dict mapping parameter names to tensors.
    """
    import numpy as np
    path = Path(path)
    dtype_map = {
        "F32": (np.float32, torch.float32),
        "F16": (np.float16, torch.float16),
        "BF16": (np.uint16,  torch.bfloat16),
        "I32": (np.int32,   torch.int32),
        "I64": (np.int64,   torch.int64),
        "I8":  (np.int8,    torch.int8),
    }
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header      = json.loads(f.read(header_size))
        data_start  = 8 + header_size
        result      = {}
        for name, meta in header.items():
            np_dtype, t_dtype = dtype_map[meta["dtype"]]
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            arr    = np.frombuffer(f.read(end - start), dtype=np_dtype).copy()
            result[name] = torch.from_numpy(arr.reshape(meta["shape"])).to(t_dtype)
    return result


def export_safetensors(
    model: nn.Module,
    cfg:   ExportConfig,
) -> Path:
    """
    Export model weights to SafeTensors format.

    Args:
        model: PyTorch model.
        cfg:   Export configuration.

    Returns:
        Path to the saved ``.safetensors`` file.
    """
    tensors  = {k: v for k, v in model.state_dict().items()}
    out_path = cfg.filename("safetensors")
    save_safetensors(tensors, out_path)
    return out_path
