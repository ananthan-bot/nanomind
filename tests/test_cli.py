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
