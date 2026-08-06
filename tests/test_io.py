"""Tests for nanomind.utils.io."""

import json
import pytest
from pathlib import Path
from nanomind.utils.io import read_text, write_text, read_json, write_json, ensure_dir


class TestTextIO:
    def test_write_read_roundtrip(self, tmp_path):
        p = tmp_path / "test.txt"
        write_text(p, "hello NanoMind")
        assert read_text(p) == "hello NanoMind"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "c.txt"
        write_text(p, "nested")
        assert p.exists()


class TestJsonIO:
    def test_write_read_roundtrip(self, tmp_path):
        p = tmp_path / "data.json"
        data = {"key": [1, 2, 3], "flag": True}
        write_json(p, data)
        loaded = read_json(p)
        assert loaded == data


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        d = ensure_dir(tmp_path / "new_dir")
        assert d.is_dir()

    def test_returns_path(self, tmp_path):
        result = ensure_dir(tmp_path / "x")
        assert isinstance(result, Path)
