"""
tests/test_public_api.py — Smoke tests for the NanoMind public API.

Verifies that all advertised public symbols are importable and usable.
"""

import torch


def test_import_nanomind():
    import nanomind
    assert nanomind.__version__ == "1.0.0"


def test_model_config_import():
    from nanomind import ModelConfig
    cfg = ModelConfig()
    assert cfg.d_model == 128


def test_nanomind_model_import():
    from nanomind import NanoMind, ModelConfig
    cfg   = ModelConfig(vocab_size=32, block_size=8, d_model=32, n_layers=1, n_heads=2)
    model = NanoMind(cfg)
    assert model.num_parameters() > 0


def test_nanomind_config_import():
    from nanomind import NanoMindConfig
    cfg = NanoMindConfig()
    assert cfg.run_name == "nanomind_run"


def test_generate_package():
    from nanomind.generate import Generator, GenerationConfig
    cfg = GenerationConfig()
    assert cfg.strategy == "temperature"


def test_eval_package():
    from nanomind.eval import perplexity, bits_per_character
    import math
    assert abs(perplexity(0.0) - 1.0) < 1e-9
    assert abs(bits_per_character(math.log(2)) - 1.0) < 1e-9


def test_checkpoint_package():
    from nanomind.checkpoint import CheckpointConfig, CheckpointManager
    cfg = CheckpointConfig()
    assert cfg.keep_last_n == 3


def test_optim_package():
    from nanomind.optim import get_lr_scheduler, list_schedules
    sched = get_lr_scheduler("constant", lr=1e-3)
    assert sched(0) == 1e-3
    assert "warmup_cosine" in list_schedules()


def test_trainer_package():
    from nanomind.trainer import Trainer, TrainConfig
    cfg = TrainConfig()
    assert cfg.max_iters == 5000
