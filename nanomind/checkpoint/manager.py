"""
nanomind/checkpoint/manager.py — High-level checkpoint manager.

CheckpointManager wraps the low-level save/load functions and adds:
- Automatic naming by step number
- Retention policy (keep last N)
- Best checkpoint tracking
- Auto-resume from latest
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from nanomind.checkpoint.config import CheckpointConfig
from nanomind.checkpoint.io import save_checkpoint, load_checkpoint
from nanomind.utils.logger import get_logger


class CheckpointManager:
    """
    Manages checkpoint saving, tracking, and cleanup.

    Args:
        cfg: :class:`~nanomind.checkpoint.CheckpointConfig` instance.
    """

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg      = cfg
        self.out_dir  = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log      = get_logger("checkpoint")
        self._best_val: float = float("inf")
        self._saved:    list[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        train_loss: float = float("nan"),
        val_loss: float = float("nan"),
        model_config: dict | None = None,
    ) -> Path:
        """
        Save a checkpoint for the current training step.

        Applies retention policy and best-checkpoint tracking.

        Args:
            model:        Model to checkpoint.
            optimizer:    Optimizer (None = skip).
            step:         Current training step.
            train_loss:   Training loss at this step.
            val_loss:     Validation loss at this step.
            model_config: Model configuration dict.

        Returns:
            Path of the saved checkpoint file.
        """
        opt = optimizer if self.cfg.save_optimizer else None
        ckpt_path = self.out_dir / f"step_{step:07d}.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            model_config=model_config,
        )
        self._saved.append(ckpt_path)
        self.log.info(f"Saved checkpoint: {ckpt_path.name}")

        # Best checkpoint
        if self.cfg.save_best and val_loss < self._best_val:
            self._best_val = val_loss
            best_path = self.out_dir / "best.pt"
            import shutil
            shutil.copy2(ckpt_path, best_path)
            shutil.copy2(ckpt_path.with_suffix(".json"), best_path.with_suffix(".json"))
            self.log.info(f"  -> New best val={val_loss:.4f}, saved to best.pt")

        # Retention policy
        self._cleanup()
        return ckpt_path
