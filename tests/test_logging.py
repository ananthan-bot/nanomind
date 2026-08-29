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


# ── MetricsBuffer ─────────────────────────────────────────────────────────────

class TestMetricsBuffer:
    def test_empty_averages(self):
        buf = MetricsBuffer()
        assert buf.averages() == {}

    def test_single_update(self):
        buf = MetricsBuffer()
        buf.update({"loss": 2.0})
        assert abs(buf.averages()["loss"] - 2.0) < 1e-9

    def test_multiple_updates_averaged(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        buf.update({"loss": 3.0})
        assert abs(buf.averages()["loss"] - 2.0) < 1e-6

    def test_weighted_update(self):
        buf = MetricsBuffer()
        buf.update({"loss": 2.0}, n=10)
        buf.update({"loss": 4.0}, n=10)
        # weighted average: (2*10 + 4*10) / 20 = 3.0
        assert abs(buf.averages()["loss"] - 3.0) < 1e-6

    def test_reset_clears(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        buf.reset()
        assert buf.averages() == {}

    def test_multiple_metrics(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0, "acc": 0.8})
        avgs = buf.averages()
        assert "loss" in avgs and "acc" in avgs

    def test_contains(self):
        buf = MetricsBuffer()
        buf.update({"loss": 1.0})
        assert "loss" in buf
        assert "lr" not in buf

    def test_len(self):
        buf = MetricsBuffer()
        buf.update({"a": 1, "b": 2, "c": 3})
        assert len(buf) == 3
