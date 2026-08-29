"""
tests/test_logging.py — Tests for NanoMind training logging.
"""

import pytest
from unittest.mock import MagicMock, patch

from nanomind.logging import (
    LogConfig,
    ConsoleLogger,
    TensorBoardLogger,
    WandbLogger,
    MetricsBuffer,
    TrainingLogger,
    build_loggers,
)


# ── LogConfig ─────────────────────────────────────────────────────────────────

class TestLogConfig:
    def test_defaults(self):
        cfg = LogConfig()
        assert cfg.backends == ["console"]
        assert cfg.log_interval == 50

    def test_string_backend_becomes_list(self):
        cfg = LogConfig(backend="tensorboard")
        assert cfg.backends == ["tensorboard"]

    def test_list_backend(self):
        cfg = LogConfig(backend=["console", "tensorboard"])
        assert len(cfg.backends) == 2

    def test_invalid_backend_raises(self):
        with pytest.raises(AssertionError):
            LogConfig(backend="mlflow")

    def test_invalid_log_interval(self):
        with pytest.raises(AssertionError):
            LogConfig(log_interval=0)


# ── ConsoleLogger ─────────────────────────────────────────────────────────────

class TestConsoleLogger:
    def test_log_config_no_crash(self, capsys):
        lg = ConsoleLogger()
        lg.log_config({"lr": 3e-4, "batch_size": 32})
        out = capsys.readouterr().out
        assert "lr" in out

    def test_log_scalars_no_crash(self, capsys):
        lg = ConsoleLogger()
        lg.log_scalars({"train/loss": 2.3, "lr": 3e-4}, step=100)
        out = capsys.readouterr().out
        assert "step" in out

    def test_finish_no_crash(self, capsys):
        lg = ConsoleLogger()
        lg.finish()

    def test_loss_formatted_to_4dp(self, capsys):
        lg = ConsoleLogger()
        lg.log_scalars({"loss": 2.123456}, step=1)
        out = capsys.readouterr().out
        assert "2.1235" in out

    def test_lr_formatted_scientific(self, capsys):
        lg = ConsoleLogger()
        lg.log_scalars({"lr": 3e-4}, step=1)
        out = capsys.readouterr().out
        assert "e-04" in out or "3.00" in out
