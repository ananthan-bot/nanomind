"""
tests/test_integration.py — End-to-end integration tests for NanoMind.

Tests the full pipeline: tokenizer → dataset → model → train step →
checkpoint → load → generate.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.trainer import Trainer, TrainConfig
from nanomind.optim import get_optimizer
from nanomind.checkpoint import CheckpointManager, CheckpointConfig, auto_resume
from nanomind.generate import Generator, GenerationConfig
from nanomind.eval import Evaluator, EvalConfig

VOCAB  = 32
TEXT   = "abcdefghijklmnopqrstuvwxyz " * 20
BLOCK  = 8
D      = 32


def make_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                      d_model=D, n_layers=2, n_heads=2, dropout=0.0)
    return NanoMind(cfg), cfg


def make_loader(ids, block=BLOCK, batch=4):
    tokens = torch.tensor(ids)
    xs = torch.stack([tokens[i:i+block]     for i in range(len(ids) - block - 1)])
    ys = torch.stack([tokens[i+1:i+block+1] for i in range(len(ids) - block - 1)])
    return DataLoader(TensorDataset(xs, ys), batch_size=batch, drop_last=True)


class TestEndToEnd:
    def test_tokenize_and_encode_decode(self):
        tok = CharTokenizer().build(TEXT)
        ids = tok.encode(TEXT[:20])
        assert tok.decode(ids) == TEXT[:20]

    def test_model_forward_with_loss(self):
        model, _ = make_model()
        x = torch.randint(0, VOCAB, (2, BLOCK))
        y = torch.randint(0, VOCAB, (2, BLOCK))
        logits, loss = model(x, y)
        assert logits.shape == (2, BLOCK, VOCAB)
        assert loss.item() > 0

    def test_train_reduces_loss(self):
        model, _ = make_model()
        tok    = CharTokenizer().build(TEXT)
        ids    = tok.encode(TEXT)
        loader = make_loader(ids)
        opt = get_optimizer(model, lr=3e-3)
        cfg = TrainConfig(max_iters=20, eval_interval=10, log_interval=10)
        trainer = Trainer(model, opt, loader, loader, cfg, torch.device("cpu"))
        x, y = next(iter(loader))
        loss_before = trainer.eval_step(x, y)
        trainer.train()
        loss_after = trainer.eval_step(x, y)
        assert loss_after < loss_before

    def test_checkpoint_and_reload(self, tmp_path):
        model, cfg_obj = make_model()
        ckpt_cfg = CheckpointConfig(out_dir=str(tmp_path), keep_last_n=1)
        mgr = CheckpointManager(ckpt_cfg)
        mgr.save(model, step=10, val_loss=1.0, model_config=cfg_obj.to_dict())

        model2, _ = make_model()
        for p in model2.parameters():
            torch.nn.init.normal_(p)
        start, meta = auto_resume(str(tmp_path), model2)
        assert start == 11
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_generate_after_training(self):
        model, _ = make_model()
        tok = CharTokenizer().build(TEXT)
        gen_cfg = GenerationConfig(max_new_tokens=10, strategy="greedy")
        generator = Generator(model, tok, device=torch.device("cpu"))
        out = generator.generate("abc", gen_cfg)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_eval_after_training(self):
        model, _ = make_model()
        tok    = CharTokenizer().build(TEXT)
        ids    = tok.encode(TEXT)
        loader = make_loader(ids)
        ev     = Evaluator(model, EvalConfig(max_batches=2))
        result = ev.full_eval(loader)
        assert result.ppl > 1.0
        assert 0.0 <= result.accuracy <= 1.0
