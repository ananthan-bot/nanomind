"""
nanomind/logging/wandb_logger.py — Weights & Biases logger for NanoMind training.

Requires: ``pip install wandb``
Usage:    ``wandb login`` then set ``backend: wandb`` in LogConfig.
"""

from __future__ import annotations

from nanomind.logging.base import BaseLogger
from nanomind.utils.logger import get_logger

log = get_logger("logging.wandb")


class WandbLogger(BaseLogger):
    """
    Weights & Biases training logger.

    Logs metrics, config, text samples, and optionally model gradients.
    Falls back gracefully if ``wandb`` is not installed or not authenticated.

    Args:
        project:   W&B project name.
        run_name:  Name for this run.
        tags:      List of tags for the W&B run.
        notes:     Notes string for the W&B run.
        config:    Initial config dict (can also be set via log_config).
    """

    def __init__(
        self,
        project:  str = "nanomind",
        run_name: str = "run",
        tags:     list[str] | None = None,
        notes:    str = "",
    ) -> None:
        self.project  = project
        self.run_name = run_name
        self.tags     = tags or []
        self.notes    = notes
        self._run     = None
        self._available = False
        self._pending_config: dict = {}

    def _ensure_init(self, config: dict | None = None) -> None:
        if self._run is not None:
            return
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                tags=self.tags,
                notes=self.notes,
                config=config or self._pending_config,
                reinit=True,
            )
            self._available = True
            log.info(f"W&B run started: {self._run.url}")
        except ImportError:
            log.warning("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            log.warning(f"W&B init failed: {e}. Falling back to console only.")

    def log_config(self, config: dict) -> None:
        self._pending_config = config
        self._ensure_init(config)
        if self._available:
            try:
                import wandb
                wandb.config.update(config, allow_val_change=True)
            except Exception:
                pass

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        self._ensure_init()
        if not self._available:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass

    def log_histogram(self, name: str, values, step: int) -> None:
        self._ensure_init()
        if not self._available:
            return
        try:
            import wandb
            wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
        except Exception:
            pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        self._ensure_init()
        if not self._available:
            return
        try:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if self._available:
            try:
                import wandb
                wandb.finish()
                log.info("W&B run finished.")
            except Exception:
                pass
