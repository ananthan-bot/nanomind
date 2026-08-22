"""
tests/test_cli.py — Tests for the NanoMind CLI argument parser and config I/O.
"""

import json
import pytest
from pathlib import Path

from nanomind.cli.args import build_parser
from nanomind.cli.config_io import load_config, save_config, merge_cli_overrides
from nanomind.config import NanoMindConfig


# ── Argument parser ───────────────────────────────────────────────────────────

class TestArgParser:
    def test_train_subcommand_parsed(self):
        parser = build_parser()
        args   = parser.parse_args(["train"])
        assert args.command == "train"

    def test_generate_requires_checkpoint(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate"])   # missing --checkpoint

    def test_generate_defaults(self):
        parser = build_parser()
        args   = parser.parse_args(["generate", "--checkpoint", "ckpt.pt"])
        assert args.max_new_tokens == 200
        assert args.strategy == "temperature"
        assert args.temperature == 0.8

    def test_eval_requires_checkpoint_and_data(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["eval"])

    def test_generate_stream_flag(self):
        parser = build_parser()
        args   = parser.parse_args(["generate", "--checkpoint", "ckpt.pt", "--stream"])
        assert args.stream is True

    def test_train_d_model_override(self):
        parser = build_parser()
        args   = parser.parse_args(["train", "--d-model", "256"])
        assert args.d_model == 256


# ── Config I/O ────────────────────────────────────────────────────────────────

class TestConfigIO:
    def test_json_roundtrip(self, tmp_path):
        cfg  = NanoMindConfig()
        path = tmp_path / "cfg.json"
        save_config(cfg, path)
        cfg2 = load_config(path)
        assert cfg.model.d_model == cfg2.model.d_model
        assert cfg.train.max_iters == cfg2.train.max_iters

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text("[model]
d_model = 128
")
        with pytest.raises(ValueError):
            load_config(p)

    def test_json_contains_model_section(self, tmp_path):
        cfg  = NanoMindConfig()
        path = tmp_path / "cfg.json"
        save_config(cfg, path)
        data = json.loads(path.read_text())
        assert "model" in data
        assert "train" in data
        assert "checkpoint" in data
