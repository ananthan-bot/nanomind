"""NanoMind checkpoint sub-package.

Primary exports:
    - :class:`CheckpointManager`        — manages save/load/cleanup lifecycle
    - :class:`CheckpointConfig`         — checkpoint configuration dataclass
    - :func:`save_checkpoint`           — low-level atomic checkpoint save
    - :func:`load_checkpoint`           — low-level checkpoint restore
    - :func:`load_for_inference`        — inference-only model restore
    - :func:`save_for_inference`        — save weights-only checkpoint
    - :func:`load_inference_checkpoint` — load weights-only checkpoint
    - :func:`auto_resume`               — auto-detect and resume from latest
    - :func:`checkpoint_info`           — inspect checkpoint metadata
"""

from nanomind.checkpoint.config import CheckpointConfig
from nanomind.checkpoint.manager import CheckpointManager
from nanomind.checkpoint.io import save_checkpoint, load_checkpoint, load_for_inference
from nanomind.checkpoint.inference import save_for_inference, load_inference_checkpoint
from nanomind.checkpoint.resume import auto_resume
from nanomind.checkpoint.info import checkpoint_info, print_checkpoint_info

__all__ = [
    "CheckpointConfig",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "load_for_inference",
    "save_for_inference",
    "load_inference_checkpoint",
    "auto_resume",
    "checkpoint_info",
    "print_checkpoint_info",
]
