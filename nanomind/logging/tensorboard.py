"""
nanomind/logging/tensorboard.py — TensorBoard logger for NanoMind training.

Requires: ``pip install tensorboard``
Usage:    ``tensorboard --logdir logs/``
"""

from __future__ import annotations

from pathlib import Path
from nanomind.logging.base import BaseLogger
from nanomind.utils.logger import get_logger

log = get_logger("logging.tensorboard")


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard training logger.

    Logs scalars, optional parameter histograms, and text summaries.
    Falls back gracefully if ``tensorboard`` is not installed.

    Args:
        log_dir:     Directory for TensorBoard event files.
        run_name:    Sub-directory name for this run.
        log_params:  Log weight histograms every N steps (expensive).
    """

    def __init__(
        self,
        log_dir:    str = "logs",
        run_name:   str = "run",
        log_params: bool = False,
    ) -> None:
        self.log_dir    = Path(log_dir) / run_name
        self.log_params = log_params
        self._writer    = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer    = SummaryWriter(log_dir=str(self.log_dir))
            self._available = True
            log.info(f"TensorBoard logging to: {self.log_dir}")
        except ImportError:
            log.warning(
                "TensorBoard not installed. "
                "Install with: pip install tensorboard"
            )

    def log_config(self, config: dict) -> None:
        if not self._available:
            return
        # Log hyperparameters
        try:
            self._writer.add_hparams(
                hparam_dict={k: str(v) for k, v in config.items()},
                metric_dict={},
            )
        except Exception:
            pass  # add_hparams can fail with certain config types

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if not self._available:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self._writer.add_scalar(key, val, global_step=step)

    def log_histogram(self, name: str, values, step: int) -> None:
        if not self._available:
            return
        try:
            self._writer.add_histogram(name, values, global_step=step)
        except Exception:
            pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        if not self._available:
            return
        self._writer.add_text(tag, text, global_step=step)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            log.info("TensorBoard writer closed.")
