"""Tests for nanomind.utils.format."""

import pytest
from nanomind.utils.format import fmt_number, fmt_time, fmt_loss, fmt_lr


class TestFmtNumber:
    def test_billions(self): assert fmt_number(1_500_000_000) == "1.50B"
    def test_millions(self): assert fmt_number(1_200_000) == "1.20M"
    def test_thousands(self): assert fmt_number(5_500) == "5.5K"
    def test_small(self): assert fmt_number(42) == "42"


class TestFmtTime:
    def test_hours(self): assert "h" in fmt_time(3661)
    def test_minutes(self): assert "m" in fmt_time(90)
    def test_seconds(self): assert "s" in fmt_time(1.5)


class TestFmtLoss:
    def test_decimal_places(self): assert len(fmt_loss(1.23456789).split(".")[1]) == 4


class TestFmtLr:
    def test_scientific(self): assert "e" in fmt_lr(3e-4)
