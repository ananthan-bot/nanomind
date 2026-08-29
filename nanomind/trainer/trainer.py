"""
nanomind/trainer/trainer.py — Training loop for NanoMind.

The Trainer class owns the full training lifecycle:
    - train_step()     : one forward + backward + optimizer step
    - eval_step()      : one forward without gradients
    - estimate_loss()  : average loss over N batches
    - train()          : the full training loop
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.trainer.config import TrainConfig
from nanomind.utils.logger import get_logger
from nanomind.logging import TrainingLogger, LogConfig
from nanomind.utils.format import fmt_number, fmt_time, fmt_loss, fmt_lr
from nanomind.utils.timer import Timer, tokens_per_second


class Trainer:
    """
    Handles the NanoMind training loop.

    Args:
        model:        The :class:`~nanomind.model.NanoMind` model.
        optimizer:    Configured optimizer (e.g. AdamW).
        train_loader: DataLoader for training batches.
        val_loader:   DataLoader for validation batches.
        cfg:          :class:`~nanomind.trainer.TrainConfig`.
        device:       Resolved :class:`torch.device`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.log          = get_logger("trainer")

        # State
        self.step:     int   = 0
        self.best_val: float = float("inf")
        self._timer          = Timer()

        # AMP scaler (only active when use_amp=True and CUDA available)
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if cfg.use_amp and device.type == "cuda"
            else None
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        """Yield batches endlessly, restarting the loader when exhausted."""
        while True:
            yield from loader

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform one forward + backward pass (no optimizer step).

        Supports AMP via :class:`torch.cuda.amp.GradScaler` when enabled.

        Args:
            x: Input tokens  ``(B, T)``
            y: Target tokens ``(B, T)``

        Returns:
            Loss value as a Python float.
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        if self._scaler is not None:
            with torch.cuda.amp.autocast():
                _, loss = self.model(x, y)
            self._scaler.scale(loss).backward()
        else:
            _, loss = self.model(x, y)
            loss.backward()

        return loss.item()

    def _optimizer_step(self) -> None:
        """
        Apply accumulated gradients, clip norms, and step the optimizer.

        Handles both standard and AMP (GradScaler) paths.
        Resets gradients after the update.
        """
        if self._scaler is not None:
            if self.cfg.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def eval_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single validation forward pass. Returns loss as float."""
        self.model.eval()
        x, y   = x.to(self.device), y.to(self.device)
        _, loss = self.model(x, y)
        return loss.item()

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """
        Estimate mean loss on train and val sets over ``eval_iters`` batches.

        Returns:
            Dict with keys ``"train"`` and ``"val"``.
        """
        self.model.eval()
        results: dict[str, float] = {}
        loaders = {"train": self.train_loader, "val": self.val_loader}
        for split, loader in loaders.items():
            losses: list[float] = []
            it = iter(loader)
            for _ in range(min(self.cfg.eval_iters, len(loader))):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                losses.append(self.eval_step(x, y))
            results[split] = sum(losses) / len(losses) if losses else float("nan")
        self.model.train()
        return results

    def _log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        elapsed: float,
        batch_size: int,
        block_size: int,
    ) -> None:
        """Print a compact training log line."""
        tps = tokens_per_second(
            batch_size * block_size * self.cfg.log_interval, elapsed
        )
        self.log.info(
            f"step {step:>6} | loss {fmt_loss(loss)} | "
            f"lr {fmt_lr(lr)} | {tps:,.0f} tok/s"
        )

    def _check_early_stop(self, val_loss: float) -> bool:
        """
        Check whether training should stop early.

        Tracks patience counter — incremented each eval where val loss
        does not improve. Resets to 0 on improvement.

        Args:
            val_loss: Latest validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.cfg.early_stop_patience <= 0:
            return False

        if val_loss < self.best_val:
            self._patience_counter = 0
            self.best_val = val_loss
        else:
            self._patience_counter = getattr(self, "_patience_counter", 0) + 1
            if self._patience_counter >= self.cfg.early_stop_patience:
                self.log.warning(
                    f"Early stopping: val loss did not improve for "
                    f"{self.cfg.early_stop_patience} evals."
                )
                return True
        return False

    def train(
        self,
        lr_scheduler=None,
        on_eval: "Callable | None" = None,  # noqa: F821
    ) -> dict[str, float]:
        """
        Run the full training loop.

        Args:
            lr_scheduler: Optional LR scheduler with a ``get_lr(step)`` callable.
            on_eval:      Optional callback called after each evaluation with
                          ``(step, train_loss, val_loss)``.

        Returns:
            Dict with ``"best_val"`` and ``"final_train"`` losses.
        """
        from pathlib import Path as _P
        out_dir = _P(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.log.info(
            f"Starting training | device={self.device} | "
            f"max_iters={self.cfg.max_iters:,}"
        )

        data_iter   = self._infinite_loader(self.train_loader)
        running_loss = 0.0
        self._timer.start()
        self.optimizer.zero_grad(set_to_none=True)

        for step in range(self.step, self.cfg.max_iters):
            self.step = step

            # Update LR
            if lr_scheduler is not None:
                lr = lr_scheduler(step)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            # Gradient accumulation
            for micro_step in range(self.cfg.grad_accum_steps):
                x, y       = next(data_iter)
                step_loss  = self.train_step(x, y)
                running_loss += step_loss / self.cfg.grad_accum_steps

            self._optimizer_step()

            # Logging
            if (step + 1) % self.cfg.log_interval == 0:
                elapsed = self._timer.stop()
                self._log_step(
                    step + 1, running_loss / self.cfg.log_interval,
                    lr, elapsed,
                    x.size(0), x.size(1),
                )
                running_loss = 0.0
                self._timer.start()

            # Evaluation
            if (step + 1) % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                self.log.info(
                    f"[eval] step {step+1} | "
                    f"train={fmt_loss(losses['train'])} | "
                    f"val={fmt_loss(losses['val'])}"
                )
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    self.log.info(f"  New best val: {fmt_loss(self.best_val)}")

                if on_eval:
                    on_eval(step + 1, losses["train"], losses["val"])

                if self._check_early_stop(losses["val"]):
                    break

        self.log.info(
            f"Training complete | best_val={fmt_loss(self.best_val)}"
        )
        return {"best_val": self.best_val, "final_train": running_loss}
