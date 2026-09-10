"""
day32_commits.py — 20 atomic commits for Day 32: Model Export (TorchScript, ONNX, SafeTensors).
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

print("\n=== DAY 32: Model Export — TorchScript, ONNX, SafeTensors — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — export package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/__init__.py",
      '"""NanoMind Export sub-package — model export utilities."""\n')
commit("feat: add nanomind/export/ package skeleton for model export utilities")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — ExportConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/config.py", '''\
"""
nanomind/export/config.py — Model export configuration.

## Why Export?

Training happens in PyTorch (research-friendly, dynamic graphs, autograd).
Deployment happens in optimised runtimes:

  TorchScript:    PyTorch\'s own compiled IR — deploy without Python.
                  Used by: mobile apps (iOS/Android via LibTorch),
                           C++ inference servers, edge devices.

  ONNX:           Open Neural Network Exchange — portable format.
                  Deploy to: ONNX Runtime, TensorRT, OpenVINO, CoreML,
                             ONNX.js, cloud ML services (Azure, AWS SageMaker).

  SafeTensors:    HuggingFace\'s safe, zero-copy weight format.
                  Replaces pickle-based .pt files — immune to code injection.
                  Used by all modern HuggingFace models.

Export pipeline:
  NanoMind (PyTorch) → export() → Optimised Runtime
                                       ↓
                               fast, safe, portable inference

Reference:
  ONNX: https://onnx.ai/
  SafeTensors: https://github.com/huggingface/safetensors
  TorchScript: https://pytorch.org/docs/stable/jit.html
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportConfig:
    """
    Configuration for model export.

    Attributes:
        format:         Export format: ``"torchscript"``, ``"onnx"``, or
                        ``"safetensors"``.
        output_path:    Directory to write exported files.
        opset_version:  ONNX opset version (default: 17).
        dynamic_batch:  If True, export with dynamic batch dimension.
        dynamic_seq:    If True, export with dynamic sequence length.
        validate:       If True, validate exported model against original.
        rtol:           Relative tolerance for output validation.
        atol:           Absolute tolerance for output validation.
        model_name:     Name used in exported filenames.
    """

    format:         str   = "torchscript"
    output_path:    str   = "exported_models"
    opset_version:  int   = 17
    dynamic_batch:  bool  = True
    dynamic_seq:    bool  = True
    validate:       bool  = True
    rtol:           float = 1e-3
    atol:           float = 1e-5
    model_name:     str   = "nanomind"

    def __post_init__(self) -> None:
        assert self.format in ("torchscript", "onnx", "safetensors")
        assert self.opset_version >= 11
        assert self.rtol > 0
        assert self.atol > 0

    @property
    def output_dir(self) -> Path:
        return Path(self.output_path)

    def filename(self, suffix: str) -> Path:
        ext = {"torchscript": ".pt", "onnx": ".onnx", "safetensors": ".safetensors"}
        return self.output_dir / f"{self.model_name}{ext.get(suffix, suffix)}"
''')
commit("feat: add ExportConfig — format, opset, dynamic_batch/seq, validate, output_path")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — TorchScript export
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/torchscript.py", '''\
"""
nanomind/export/torchscript.py — TorchScript model export.

TorchScript compiles a PyTorch model to a static IR (Intermediate Representation)
that can run without Python — used for mobile, C++ inference, and edge deployment.

Two modes:
  torch.jit.script:  Analyses Python source code and compiles to IR.
                     Handles control flow (if/for) but requires type annotations.

  torch.jit.trace:   Runs the model with example inputs and records operations.
                     Simpler, but doesn\'t handle dynamic control flow.

NanoMind uses tracing (trace mode) since transformers have static control flow.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.utils.logger import get_logger

log = get_logger("export.torchscript")


def export_torchscript(
    model:      nn.Module,
    cfg:        ExportConfig,
    example_input: torch.Tensor | None = None,
) -> Path:
    """
    Export a NanoMind model to TorchScript via tracing.

    Args:
        model:         PyTorch model to export.
        cfg:           Export configuration.
        example_input: Example input tensor ``(B, T)``.
                       Generated automatically if not provided.

    Returns:
        Path to the saved ``.pt`` TorchScript file.
    """
    model.eval()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if example_input is None:
        # Get vocab size from embedding layer
        emb = next(m for m in model.modules() if isinstance(m, nn.Embedding))
        V   = emb.num_embeddings
        example_input = torch.randint(0, V, (1, 32))

    log.info(f"Tracing model with input shape {tuple(example_input.shape)}...")
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_input,))

    out_path = cfg.filename("torchscript")
    traced.save(str(out_path))
    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"TorchScript saved to {out_path} ({size_mb:.2f} MB)")
    return out_path


def load_torchscript(path: str | Path) -> torch.jit.ScriptModule:
    """
    Load a saved TorchScript model.

    Args:
        path: Path to the ``.pt`` TorchScript file.

    Returns:
        Loaded :class:`torch.jit.ScriptModule`.
    """
    return torch.jit.load(str(path))
''')
commit("feat: add export_torchscript() — jit.trace export with auto example input, load helper")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — ONNX export
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/onnx_export.py", '''\
"""
nanomind/export/onnx_export.py — ONNX model export.

ONNX (Open Neural Network Exchange) is the industry-standard portable format
for deploying neural networks across frameworks and hardware:

  PyTorch → ONNX → ONNX Runtime (CPU/GPU, 2-10× faster than PyTorch)
                 → TensorRT     (NVIDIA GPU, 5-30× faster)
                 → OpenVINO     (Intel CPU/VPU)
                 → CoreML       (Apple Silicon)
                 → ONNX.js      (browser inference)
                 → AWS, Azure ML serving

Dynamic axes allow the exported model to handle variable batch sizes
and sequence lengths at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.utils.logger import get_logger

log = get_logger("export.onnx")


def export_onnx(
    model:         nn.Module,
    cfg:           ExportConfig,
    example_input: torch.Tensor | None = None,
) -> Path:
    """
    Export a NanoMind model to ONNX format.

    Args:
        model:         PyTorch model to export.
        cfg:           Export configuration.
        example_input: Example input tensor ``(B, T)``.

    Returns:
        Path to the saved ``.onnx`` file.
    """
    model.eval()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if example_input is None:
        emb = next(m for m in model.modules() if isinstance(m, nn.Embedding))
        V   = emb.num_embeddings
        example_input = torch.randint(0, V, (1, 32))

    # Build dynamic axes mapping
    dynamic_axes: dict = {"input_ids": {}}
    if cfg.dynamic_batch:
        dynamic_axes["input_ids"][0] = "batch_size"
        dynamic_axes["logits"] = {0: "batch_size"}
    if cfg.dynamic_seq:
        dynamic_axes["input_ids"][1] = "seq_len"
        if "logits" not in dynamic_axes:
            dynamic_axes["logits"] = {}
        dynamic_axes["logits"][1] = "seq_len"

    out_path = cfg.filename("onnx")
    log.info(f"Exporting to ONNX (opset={cfg.opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (example_input,),
            str(out_path),
            opset_version=cfg.opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes if (cfg.dynamic_batch or cfg.dynamic_seq) else None,
            do_constant_folding=True,
            export_params=True,
        )

    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"ONNX saved to {out_path} ({size_mb:.2f} MB)")
    return out_path
''')
commit("feat: add export_onnx() — torch.onnx.export with dynamic axes, opset config, do_constant_folding")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — SafeTensors export
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/safetensors_export.py", '''\
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
''')
commit("feat: add save_safetensors(), load_safetensors(), export_safetensors() — pure Python impl")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — export validation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/validate.py", '''\
"""
nanomind/export/validate.py — Export validation: compare original vs exported outputs.

After exporting, we verify that the exported model produces the same output
as the original PyTorch model (within floating-point tolerance).

This guards against:
  - Numerical precision changes from fp32 → fp16 conversion
  - Operator differences between PyTorch and ONNX Runtime
  - TorchScript compilation bugs (rare but possible)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.utils.logger import get_logger

log = get_logger("export.validate")


def validate_torchscript(
    original:  nn.Module,
    ts_path:   str | Path,
    n_inputs:  int   = 4,
    seq_len:   int   = 16,
    rtol:      float = 1e-3,
    atol:      float = 1e-4,
) -> dict:
    """
    Validate TorchScript export vs original PyTorch model.

    Args:
        original:  Original PyTorch model.
        ts_path:   Path to the TorchScript ``.pt`` file.
        n_inputs:  Number of random inputs to test.
        seq_len:   Sequence length for test inputs.
        rtol:      Relative tolerance for output comparison.
        atol:      Absolute tolerance for output comparison.

    Returns:
        Dict with ``passed``, ``max_abs_diff``, ``max_rel_diff``.
    """
    original.eval()
    ts_model = torch.jit.load(str(ts_path))

    emb  = next(m for m in original.modules() if isinstance(m, nn.Embedding))
    V    = emb.num_embeddings
    diffs: list[float] = []

    for _ in range(n_inputs):
        x         = torch.randint(0, V, (1, seq_len))
        with torch.no_grad():
            out_orig, _ = original(x)
            out_ts      = ts_model(x)
            if isinstance(out_ts, tuple):
                out_ts = out_ts[0]
        diffs.append((out_orig - out_ts).abs().max().item())

    max_diff = max(diffs)
    passed   = max_diff <= atol + rtol * out_orig.abs().max().item()
    log.info(f"TorchScript validation: passed={passed}, max_diff={max_diff:.2e}")
    return {"passed": passed, "max_abs_diff": max_diff, "n_inputs": n_inputs}


def model_size_report(model: nn.Module, name: str = "Model") -> dict:
    """
    Compute model size statistics.

    Args:
        model: PyTorch model.
        name:  Model display name.

    Returns:
        Dict with ``n_params``, ``trainable_params``, ``size_mb``, ``name``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = sum(p.nbytes for p in model.parameters()) / (1024 ** 2)
    return {
        "name":             name,
        "n_params":         total,
        "trainable_params": trainable,
        "frozen_params":    total - trainable,
        "size_mb":          size_mb,
        "size_mb_fp16":     size_mb / 2,
        "size_mb_int8":     size_mb / 4,
    }


def format_size_report(report: dict) -> str:
    """Format a model size report as a readable string."""
    return (
        f"Model: {report['name']}\n"
        f"  Parameters : {report['n_params']:>12,}\n"
        f"  Trainable  : {report['trainable_params']:>12,}\n"
        f"  Size (fp32): {report['size_mb']:>10.2f} MB\n"
        f"  Size (fp16): {report['size_mb_fp16']:>10.2f} MB\n"
        f"  Size (int8): {report['size_mb_int8']:>10.2f} MB\n"
    )
''')
commit("feat: add validate_torchscript(), model_size_report(), format_size_report()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — unified export function
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/exporter.py", '''\
"""
nanomind/export/exporter.py — Unified model exporter.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.export.torchscript import export_torchscript
from nanomind.export.onnx_export import export_onnx
from nanomind.export.safetensors_export import export_safetensors
from nanomind.export.validate import validate_torchscript, model_size_report
from nanomind.utils.logger import get_logger

log = get_logger("export.exporter")


class ModelExporter:
    """
    Unified model export interface for NanoMind.

    Supports TorchScript, ONNX, and SafeTensors formats with
    optional post-export validation.

    Args:
        model: PyTorch model to export.
        cfg:   Export configuration.

    Example::

        exporter = ModelExporter(model, ExportConfig(format="torchscript"))
        path     = exporter.export()
        report   = exporter.size_report()
        print(exporter.format_summary())
    """

    def __init__(
        self,
        model: nn.Module,
        cfg:   ExportConfig | None = None,
    ) -> None:
        self.model = model.eval()
        self.cfg   = cfg or ExportConfig()

    def export(
        self,
        example_input: torch.Tensor | None = None,
    ) -> Path:
        """
        Export the model in the configured format.

        Args:
            example_input: Optional example input tensor for tracing.

        Returns:
            Path to the exported file.
        """
        fmt = self.cfg.format
        log.info(f"Exporting model as {fmt}...")

        if fmt == "torchscript":
            path = export_torchscript(self.model, self.cfg, example_input)
            if self.cfg.validate:
                result = validate_torchscript(
                    self.model, path,
                    rtol=self.cfg.rtol, atol=self.cfg.atol,
                )
                if result["passed"]:
                    log.info("Validation passed ✓")
                else:
                    log.warning(f"Validation failed! max_diff={result['max_abs_diff']:.2e}")
        elif fmt == "onnx":
            path = export_onnx(self.model, self.cfg, example_input)
        elif fmt == "safetensors":
            path = export_safetensors(self.model, self.cfg)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        log.info(f"Export complete: {path}")
        return path

    def export_all(self, example_input: torch.Tensor | None = None) -> dict[str, Path]:
        """Export to all three formats."""
        results = {}
        for fmt in ("torchscript", "onnx", "safetensors"):
            self.cfg.format = fmt
            try:
                results[fmt] = self.export(example_input)
            except Exception as e:
                log.warning(f"{fmt} export failed: {e}")
                results[fmt] = None
        return results

    def size_report(self) -> dict:
        """Return model size statistics."""
        return model_size_report(self.model, self.cfg.model_name)

    def format_summary(self) -> str:
        """Return a formatted export summary string."""
        rep = self.size_report()
        lines = [
            f"Export Summary: {rep['name']}",
            f"  Format     : {self.cfg.format}",
            f"  Output dir : {self.cfg.output_dir}",
            f"  Parameters : {rep['n_params']:,}",
            f"  Size fp32  : {rep['size_mb']:.2f} MB",
            f"  Size fp16  : {rep['size_mb_fp16']:.2f} MB",
        ]
        return "\n".join(lines)
''')
commit("feat: add ModelExporter — export(), export_all(), size_report(), format_summary()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — quantized export helper
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/quantize_export.py", '''\
"""
nanomind/export/quantize_export.py — Post-training quantisation for export.

Reduces model size 4× with minimal quality loss by converting float32
weights to int8 using PyTorch\'s dynamic quantisation.

Dynamic quantisation:
  - Quantises weights to int8 at load time (offline)
  - Activations quantised per-batch at runtime
  - No calibration dataset needed (unlike static quantisation)
  - Ideal for NLP models (Linear layers dominate computation)

Size comparison (NanoMind 128d, 4L):
  float32 : 4 bytes/param → ~20 MB
  float16 : 2 bytes/param → ~10 MB
  int8    : 1 byte/param  →  ~5 MB
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nanomind.export.config import ExportConfig
from nanomind.export.torchscript import export_torchscript
from nanomind.utils.logger import get_logger

log = get_logger("export.quantize")


def quantize_dynamic(
    model:      nn.Module,
    dtype:      torch.dtype = torch.qint8,
    layer_types: set | None = None,
) -> nn.Module:
    """
    Apply dynamic INT8 quantisation to a model.

    Args:
        model:       PyTorch model to quantise.
        dtype:       Quantisation dtype (default: qint8).
        layer_types: Layer types to quantise. Defaults to ``{nn.Linear}``.

    Returns:
        Quantised model (in-place and returned).
    """
    if layer_types is None:
        layer_types = {nn.Linear}
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(), layer_types, dtype=dtype
    )
    n_orig = sum(p.numel() for p in model.parameters())
    log.info(f"Dynamic INT8 quantisation applied to {n_orig:,} params")
    return quantized


def export_quantized_torchscript(
    model:         nn.Module,
    cfg:           ExportConfig,
    example_input: torch.Tensor | None = None,
) -> Path:
    """
    Quantise a model and export as TorchScript.

    Equivalent to:
      quantize_dynamic(model) → export_torchscript(...)

    Args:
        model:         Original float32 model.
        cfg:           Export configuration.
        example_input: Example input for tracing.

    Returns:
        Path to quantised TorchScript file.
    """
    quantized = quantize_dynamic(model)
    orig_cfg  = cfg.format
    cfg.format = "torchscript"
    # Save with "_int8" suffix
    orig_name  = cfg.model_name
    cfg.model_name = orig_name + "_int8"
    path = export_torchscript(quantized, cfg, example_input)
    cfg.model_name = orig_name
    cfg.format     = orig_cfg
    return path
''')
commit("feat: add quantize_dynamic(), export_quantized_torchscript() — INT8 dynamic quantisation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — CLI: nanomind export
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/cli.py", '''\
"""
nanomind/export/cli.py — CLI entry point: ``nanomind export``.

Usage::

    python -m nanomind.export.cli --format torchscript --output exported/
    python -m nanomind.export.cli --format onnx --opset 17
    python -m nanomind.export.cli --format safetensors
    python -m nanomind.export.cli --all   # export to all formats
"""

from __future__ import annotations

import argparse
import torch
from nanomind.export.config import ExportConfig
from nanomind.export.exporter import ModelExporter
from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nanomind export",
        description="Export a NanoMind model to TorchScript, ONNX, or SafeTensors",
    )
    p.add_argument("--format",    default="torchscript",
                   choices=["torchscript", "onnx", "safetensors"],
                   help="Export format")
    p.add_argument("--all",       action="store_true", help="Export all formats")
    p.add_argument("--output",    default="exported_models", help="Output directory")
    p.add_argument("--opset",     type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-validate", action="store_true", help="Skip validation")
    p.add_argument("--model-name", default="nanomind", help="Model filename prefix")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    CORPUS    = "hello world " * 20
    tokenizer = CharTokenizer().build(CORPUS)
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32,
                            d_model=64, n_layers=2, n_heads=4, dropout=0.0)
    model = NanoMind(model_cfg)

    cfg = ExportConfig(
        format=args.format,
        output_path=args.output,
        opset_version=args.opset,
        validate=not args.no_validate,
        model_name=args.model_name,
    )
    exporter = ModelExporter(model, cfg)
    print(exporter.format_summary())

    if args.all:
        results = exporter.export_all()
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")
    else:
        path = exporter.export()
        print(f"  Saved to: {path}")


if __name__ == "__main__":
    main()
''')
commit("feat: add nanomind/export/cli.py — CLI for --format, --all, --opset, --output flags")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — export __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/export/__init__.py", '''\
"""NanoMind Export sub-package — model export to TorchScript, ONNX, SafeTensors.

Supported formats:
  torchscript — PyTorch JIT compiled model, runs without Python
  onnx        — Open Neural Network Exchange, deploy to any runtime
  safetensors — HuggingFace safe weight format, zero-code, memory-mapped

Primary exports:
    - :class:`ModelExporter`          — unified export() + export_all() + size_report()
    - :class:`ExportConfig`           — format, output_path, opset, validate
    - :func:`export_torchscript`      — JIT trace export
    - :func:`load_torchscript`        — load a TorchScript .pt file
    - :func:`export_onnx`             — ONNX export with dynamic axes
    - :func:`export_safetensors`      — SafeTensors weight export
    - :func:`save_safetensors`        — raw dict→safetensors writer
    - :func:`load_safetensors`        — raw safetensors→dict reader
    - :func:`validate_torchscript`    — output correctness check
    - :func:`model_size_report`       — param count + MB statistics
    - :func:`quantize_dynamic`        — INT8 dynamic quantisation
    - :func:`export_quantized_torchscript` — quantise + export
"""

from nanomind.export.config import ExportConfig
from nanomind.export.exporter import ModelExporter
from nanomind.export.torchscript import export_torchscript, load_torchscript
from nanomind.export.onnx_export import export_onnx
from nanomind.export.safetensors_export import (
    export_safetensors, save_safetensors, load_safetensors
)
from nanomind.export.validate import validate_torchscript, model_size_report, format_size_report
from nanomind.export.quantize_export import quantize_dynamic, export_quantized_torchscript

__all__ = [
    "ExportConfig", "ModelExporter",
    "export_torchscript", "load_torchscript",
    "export_onnx",
    "export_safetensors", "save_safetensors", "load_safetensors",
    "validate_torchscript", "model_size_report", "format_size_report",
    "quantize_dynamic", "export_quantized_torchscript",
]
''')
commit("refactor: export all model export components from nanomind/export/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: export_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/export_demo.py", '''\
"""
examples/export_demo.py — NanoMind model export demo.

Demonstrates exporting a trained model to all three formats:
  1. TorchScript (.pt)   — deploy without Python
  2. ONNX (.onnx)       — deploy to any runtime
  3. SafeTensors (.safetensors) — safe weight storage

Usage:
    python examples/export_demo.py
"""
import torch
from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.export import (
    ExportConfig, ModelExporter,
    model_size_report, format_size_report,
    save_safetensors, load_safetensors,
    quantize_dynamic,
)

# ── Build model ───────────────────────────────────────────────────────────────
CORPUS    = "the quick brown fox jumps over the lazy dog " * 10
tokenizer = CharTokenizer().build(CORPUS)
V         = tokenizer.vocab_size

torch.manual_seed(0)
cfg   = ModelConfig(vocab_size=V, block_size=32, d_model=64,
                    n_layers=2, n_heads=4, dropout=0.0)
model = NanoMind(cfg)

print("=" * 55)
print("NanoMind Model Export Demo")
print("=" * 55)

# ── Size report ───────────────────────────────────────────────────────────────
rep = model_size_report(model, "NanoMind-64d-2L")
print(f"\n{format_size_report(rep)}")

# ── Export to all formats ─────────────────────────────────────────────────────
export_cfg = ExportConfig(
    output_path="exported_models",
    model_name="nanomind_demo",
    validate=True,
    dynamic_batch=True,
    dynamic_seq=True,
)
exporter = ModelExporter(model, export_cfg)
print(exporter.format_summary())
print()

results = exporter.export_all()
for fmt, path in results.items():
    if path and path.exists():
        size = path.stat().st_size / (1024 ** 2)
        print(f"  ✅ {fmt:<15} → {path.name}  ({size:.2f} MB)")
    else:
        print(f"  ⚠️  {fmt:<15} → skipped")

# ── SafeTensors roundtrip ─────────────────────────────────────────────────────
print("\n── SafeTensors roundtrip ──")
st_path = "exported_models/roundtrip_test.safetensors"
save_safetensors(dict(model.state_dict()), st_path)
loaded  = load_safetensors(st_path)
first   = list(loaded.keys())[0]
match   = torch.allclose(model.state_dict()[first], loaded[first], atol=1e-6)
print(f"  Saved {len(loaded)} tensors, roundtrip match: {match}")

# ── Quantised model ───────────────────────────────────────────────────────────
print("\n── INT8 Quantisation ──")
q_model = quantize_dynamic(model)
q_rep   = model_size_report(q_model, "NanoMind-INT8")
print(f"  Original : {rep['size_mb']:.2f} MB")
print(f"  INT8     : {q_rep['size_mb']:.2f} MB  ({q_rep['size_mb']/rep['size_mb']:.1%} of original)")
print("\nExport demo complete!")
''')
commit("feat: add examples/export_demo.py — TorchScript + ONNX + SafeTensors + INT8 demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — test: ExportConfig
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_export.py", '''\
"""
tests/test_export.py — Tests for NanoMind model export.
"""
import json, struct, pytest, tempfile
from pathlib import Path

import torch

from nanomind.model.config import ModelConfig
from nanomind.model.nanomind import NanoMind
from nanomind.tokenizer.char import CharTokenizer
from nanomind.export import (
    ExportConfig, ModelExporter,
    export_torchscript, load_torchscript,
    export_onnx,
    export_safetensors, save_safetensors, load_safetensors,
    validate_torchscript, model_size_report, format_size_report,
    quantize_dynamic,
)

CORPUS = "abcde " * 10
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
T      = 16

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=32,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── ExportConfig ──────────────────────────────────────────────────────────────

class TestExportConfig:
    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.format == "torchscript"
        assert cfg.opset_version == 17

    def test_invalid_format(self):
        with pytest.raises(AssertionError):
            ExportConfig(format="pickle")

    def test_invalid_opset(self):
        with pytest.raises(AssertionError):
            ExportConfig(opset_version=9)

    def test_filename_torchscript(self):
        cfg = ExportConfig(model_name="test")
        assert cfg.filename("torchscript").suffix == ".pt"

    def test_filename_onnx(self):
        cfg = ExportConfig(model_name="test")
        assert cfg.filename("onnx").suffix == ".onnx"

    def test_filename_safetensors(self):
        cfg = ExportConfig(model_name="test")
        assert cfg.filename("safetensors").suffix == ".safetensors"
''')
commit("test: add ExportConfig defaults, invalid format/opset, filename suffix tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — test: TorchScript export
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_export.py")
src += '''

# ── TorchScript ───────────────────────────────────────────────────────────────

class TestTorchScript:
    def test_export_creates_file(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        assert path.exists()
        assert path.suffix == ".pt"

    def test_export_file_nonzero(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        assert path.stat().st_size > 0

    def test_load_torchscript(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        ts    = load_torchscript(path)
        x     = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            out = ts(x)
        assert isinstance(out, tuple) or isinstance(out, torch.Tensor)

    def test_output_matches_original(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="test")
        path  = export_torchscript(model, cfg)
        ts    = load_torchscript(path)
        x     = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            orig, _ = model(x)
            ts_out  = ts(x)
            if isinstance(ts_out, tuple): ts_out = ts_out[0]
        assert torch.allclose(orig, ts_out, atol=1e-4)
'''
write("tests/test_export.py", src)
commit("test: add TorchScript export file creation, nonzero, load, output match tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — test: SafeTensors
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_export.py")
src += '''

# ── SafeTensors ───────────────────────────────────────────────────────────────

class TestSafeTensors:
    def test_save_creates_file(self, tmp_path):
        tensors = {"a": torch.randn(4, 4), "b": torch.randn(8)}
        path    = tmp_path / "test.safetensors"
        save_safetensors(tensors, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip(self, tmp_path):
        tensors = {"w1": torch.randn(16, 8), "b1": torch.randn(16)}
        path    = tmp_path / "rt.safetensors"
        save_safetensors(tensors, path)
        loaded  = load_safetensors(path)
        assert set(loaded.keys()) == set(tensors.keys())
        for k in tensors:
            assert torch.allclose(tensors[k], loaded[k], atol=1e-6)

    def test_model_export(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(format="safetensors",
                              output_path=str(tmp_path), model_name="test")
        path  = export_safetensors(model, cfg)
        assert path.exists()
        loaded = load_safetensors(path)
        assert len(loaded) == len(list(model.state_dict()))
'''
write("tests/test_export.py", src)
commit("test: add SafeTensors save creates file, roundtrip, model export + load tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: model size report + validate
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_export.py")
src += '''

# ── model_size_report + validate ──────────────────────────────────────────────

class TestSizeReport:
    def test_keys(self):
        model = tiny_model()
        rep   = model_size_report(model, "Test")
        for k in ("n_params", "trainable_params", "size_mb", "size_mb_fp16"):
            assert k in rep

    def test_params_positive(self):
        model = tiny_model()
        rep   = model_size_report(model)
        assert rep["n_params"] > 0
        assert rep["size_mb"]  > 0.0

    def test_fp16_half_of_fp32(self):
        model = tiny_model()
        rep   = model_size_report(model)
        assert abs(rep["size_mb_fp16"] - rep["size_mb"] / 2) < 1e-6

    def test_format_is_string(self):
        model = tiny_model()
        rep   = model_size_report(model, "TestModel")
        s     = format_size_report(rep)
        assert isinstance(s, str)
        assert "TestModel" in s

    def test_validate_torchscript_passes(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(output_path=str(tmp_path), model_name="val")
        path  = export_torchscript(model, cfg)
        result = validate_torchscript(model, path, n_inputs=2, seq_len=T)
        assert result["passed"]
        assert result["max_abs_diff"] < 1e-3
'''
write("tests/test_export.py", src)
commit("test: add model_size_report keys/positive/fp16, format_size_report, validate_torchscript tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: ModelExporter
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_export.py")
src += '''

# ── ModelExporter ─────────────────────────────────────────────────────────────

class TestModelExporter:
    def _exporter(self, tmp_path, fmt="torchscript"):
        model = tiny_model()
        cfg   = ExportConfig(format=fmt, output_path=str(tmp_path),
                              model_name="test", validate=False)
        return ModelExporter(model, cfg)

    def test_export_torchscript(self, tmp_path):
        exp  = self._exporter(tmp_path, "torchscript")
        path = exp.export()
        assert path.exists()

    def test_export_safetensors(self, tmp_path):
        exp  = self._exporter(tmp_path, "safetensors")
        path = exp.export()
        assert path.exists()

    def test_size_report(self, tmp_path):
        exp = self._exporter(tmp_path)
        rep = exp.size_report()
        assert "n_params" in rep

    def test_format_summary(self, tmp_path):
        exp     = self._exporter(tmp_path)
        summary = exp.format_summary()
        assert "torchscript" in summary

    def test_export_all(self, tmp_path):
        exp     = self._exporter(tmp_path)
        results = exp.export_all()
        # At least safetensors should work (no ONNX library required)
        assert results.get("safetensors") is not None
'''
write("tests/test_export.py", src)
commit("test: add ModelExporter torchscript/safetensors export, size_report, format_summary, export_all tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: quantize_dynamic
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_export.py")
src += '''

# ── quantize_dynamic ──────────────────────────────────────────────────────────

class TestQuantizeDynamic:
    def test_returns_model(self):
        model = tiny_model()
        q     = quantize_dynamic(model)
        assert q is not None

    def test_inference_still_works(self):
        model = tiny_model()
        q     = quantize_dynamic(model)
        x     = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            logits, _ = q(x)
        assert logits.shape == (1, T, VOCAB)

    def test_output_shape_unchanged(self):
        model = tiny_model()
        q     = quantize_dynamic(model)
        x     = torch.randint(0, VOCAB, (2, T))
        with torch.no_grad():
            orig_logits, _ = model(x)
            q_logits, _    = q(x)
        assert orig_logits.shape == q_logits.shape
'''
write("tests/test_export.py", src)
commit("test: add quantize_dynamic returns model, inference works, output shape unchanged tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: ONNX export file creation
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_export.py")
src += '''

# ── ONNX ──────────────────────────────────────────────────────────────────────

class TestONNXExport:
    def test_onnx_creates_file(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(format="onnx", output_path=str(tmp_path),
                              model_name="test", dynamic_batch=False, dynamic_seq=False)
        path  = export_onnx(model, cfg)
        assert path.exists()
        assert path.suffix == ".onnx"

    def test_onnx_file_nonzero(self, tmp_path):
        model = tiny_model()
        cfg   = ExportConfig(format="onnx", output_path=str(tmp_path),
                              model_name="test", dynamic_batch=False, dynamic_seq=False)
        path  = export_onnx(model, cfg)
        assert path.stat().st_size > 0
'''
write("tests/test_export.py", src)
commit("test: add ONNX export file creation and nonzero size tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v3.2.0 + expose export in public API
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.1.0\"", "__version__ = \"3.2.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.2.0 — Model Export release (TorchScript, ONNX, SafeTensors)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `eval` | Benchmarking — perplexity, accuracy, throughput, memory |",
    "| `eval` | Benchmarking — perplexity, accuracy, throughput, memory |\n"
    "| `export` | Export — TorchScript, ONNX, SafeTensors, INT8 quantised |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.2.0] — 2024 — Model Export: TorchScript, ONNX, SafeTensors\n\n### Added\n" \
     "- `ModelExporter` — unified export() + export_all() + size_report()\n" \
     "- `ExportConfig` — format, output_path, opset, dynamic_batch/seq, validate\n" \
     "- `export_torchscript()` — JIT trace export + `load_torchscript()`\n" \
     "- `export_onnx()` — ONNX export with dynamic axes and constant folding\n" \
     "- `export_safetensors()` — SafeTensors weight export (pure Python)\n" \
     "- `save_safetensors()` / `load_safetensors()` — raw tensor I/O\n" \
     "- `validate_torchscript()` — output correctness check post-export\n" \
     "- `model_size_report()` — fp32/fp16/int8 size statistics\n" \
     "- `quantize_dynamic()` — INT8 dynamic quantisation\n" \
     "- `export_quantized_torchscript()` — quantise + export pipeline\n" \
     "- `nanomind/export/cli.py` — CLI: nanomind export --format onnx\n" \
     "- `examples/export_demo.py` — full export + roundtrip demo\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.2.0, update README and CHANGELOG for Day 32 Model Export")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 32 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.2.0",
    "-m", "NanoMind v3.2.0 — Model Export: TorchScript, ONNX, SafeTensors", check=False)
r = run("git", "push", "origin", "v3.2.0", check=False)
print("Tag v3.2.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 32 COMPLETE — v3.2.0 TAGGED! ===")
