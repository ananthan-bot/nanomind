"""
tests/test_utils.py — Unit tests for NanoMind utility functions.
"""

import time

import pytest
import torch

from nanomind.utils.device import get_device, device_info, is_cuda
from nanomind.utils.seed import set_seed, get_rng_state, restore_rng_state
from nanomind.utils.timer import Timer, timed, tokens_per_second
from nanomind.utils.logger import get_logger


# ── Device Tests ──────────────────────────────────────────────────────────────

class TestGetDevice:
    def test_auto_returns_device(self):
        d = get_device("auto")
        assert isinstance(d, torch.device)

    def test_cpu_explicit(self):
        d = get_device("cpu")
        assert d.type == "cpu"

    def test_device_info_cpu(self):
        info = device_info(torch.device("cpu"))
        assert "CPU" in info

    def test_is_cuda_false_on_cpu(self):
        assert not is_cuda(torch.device("cpu"))


# ── Seed Tests ────────────────────────────────────────────────────────────────

class TestSetSeed:
    def test_reproducible_tensor(self):
        set_seed(0)
        a = torch.randn(10)
        set_seed(0)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = torch.randn(10)
        set_seed(1)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


class TestRngState:
    def test_save_restore(self):
        set_seed(99)
        state = get_rng_state()
        a = torch.randn(5)
        restore_rng_state(state)
        b = torch.randn(5)
        assert torch.allclose(a, b)


# ── Timer Tests ───────────────────────────────────────────────────────────────

class TestTimer:
    def test_elapsed_is_positive(self):
        t = Timer().start()
        time.sleep(0.01)
        assert t.stop() > 0

    def test_lap(self):
        t = Timer().start()
        time.sleep(0.01)
        lap1 = t.lap()
        time.sleep(0.01)
        lap2 = t.lap()
        assert lap1 > 0
        assert lap2 > 0

    def test_reset(self):
        t = Timer().start()
        t.stop()
        t.reset()
        with pytest.raises(AssertionError):
            _ = t.elapsed


class TestTimed:
    def test_context_manager(self, capsys):
        with timed("test block"):
            time.sleep(0.01)
        out = capsys.readouterr().out
        assert "test block" in out

    def test_silent_mode(self, capsys):
        with timed("quiet", verbose=False):
            pass
        assert capsys.readouterr().out == ""


class TestTokensPerSecond:
    def test_basic(self):
        tps = tokens_per_second(1000, 1.0)
        assert tps == pytest.approx(1000.0)

    def test_zero_time_safe(self):
        tps = tokens_per_second(100, 0.0)
        assert tps > 0


# ── Logger Tests ──────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger(self):
        import logging
        log = get_logger("test")
        assert isinstance(log, logging.Logger)

    def test_same_instance(self):
        log1 = get_logger("mymodule")
        log2 = get_logger("mymodule")
        assert log1 is log2

    def test_no_duplicate_handlers(self):
        log = get_logger("no_dup")
        initial = len(log.handlers)
        get_logger("no_dup")
        assert len(log.handlers) == initial
