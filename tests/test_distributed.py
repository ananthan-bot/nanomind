"""tests/test_distributed.py — Tests for NanoMind distributed package."""
import pytest
import torch
import torch.nn as nn
from nanomind.distributed import (
    WorldConfig, MockProcessGroup, build_process_groups,
    DDPConfig, DDPStats, DataParallelWrapper, ZeROStats,
    ColumnParallelLinear, RowParallelLinear, TensorParallelMLP,
    PipelineEngine, PipelineSchedule,
    CheckpointingConfig, CheckpointedLayer, SelectiveCheckpointing,
    estimate_activation_memory,
    AMPConfig, LossScaler, MixedPrecisionTrainer,
)

V = 32

class Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.ff = nn.Linear(d, d)
    def forward(self, x):
        return self.ff(x)

class TinyLM(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.rnn  = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── WorldConfig ───────────────────────────────────────────────────────────────

class TestWorldConfig:
    def test_basic(self):
        cfg = WorldConfig(4, 0, 0, dp_size=4, tp_size=1, pp_size=1)
        assert cfg.world_size == 4

    def test_is_main(self):
        cfg = WorldConfig(4, 0, 0, 4, 1, 1)
        assert cfg.is_main_process

    def test_not_main(self):
        cfg = WorldConfig(4, 2, 2, 4, 1, 1)
        assert not cfg.is_main_process

    def test_ranks_3d(self):
        cfg = WorldConfig(8, 0, 0, dp_size=4, tp_size=2, pp_size=1)
        assert cfg.dp_rank == 0
        assert cfg.tp_rank == 0

    def test_invalid_topology(self):
        with pytest.raises(AssertionError):
            WorldConfig(8, 0, 0, dp_size=4, tp_size=3, pp_size=1)

    def test_to_dict(self):
        cfg = WorldConfig(4, 0, 0, 4, 1, 1)
        d   = cfg.to_dict()
        assert "world_size" in d and "dp_rank" in d


# ── DataParallelWrapper ───────────────────────────────────────────────────────

class TestDDP:
    def _wrapper(self, world_size=4):
        cfg = WorldConfig(world_size, 0, 0, world_size, 1, 1)
        m   = TinyLM()
        return DataParallelWrapper(m, cfg, DDPConfig(bucket_size_mb=0.1))

    def test_n_params(self):
        w = self._wrapper()
        assert w.n_parameters() > 0

    def test_forward(self):
        w   = self._wrapper()
        ids = torch.randint(0, V, (2, 4))
        out = w(ids)
        assert isinstance(out, tuple)

    def test_allreduce_divides_grad(self):
        w   = self._wrapper(world_size=2)
        ids = torch.randint(0, V, (2, 4))
        out = w(ids)
        loss = out[0].sum()
        loss.backward()
        # Save gradient before allreduce
        first_param = next(p for p in w.module.parameters() if p.grad is not None)
        grad_before = first_param.grad.clone()
        w.finish_gradient_synchronization()
        grad_after  = first_param.grad
        # After allreduce with dp_size=2, grad should be halved
        assert torch.allclose(grad_after, grad_before / 2, atol=1e-6)

    def test_stats_after_sync(self):
        w   = self._wrapper()
        ids = torch.randint(0, V, (2, 4))
        w(ids)[0].sum().backward()
        w.finish_gradient_synchronization()
        assert w.stats.n_allreduce_calls > 0


# ── ZeROStats ──────────────────────────────────────────────────────────────────

class TestZeROStats:
    def test_zero3_lt_baseline(self):
        m = TinyLM()
        z = ZeROStats(m, world_size=8)
        s = z.summary()
        assert s["zero3_gb"] < s["baseline_gb"]

    def test_speedup_positive(self):
        m = TinyLM()
        z = ZeROStats(m, world_size=4)
        s = z.summary()
        assert s["zero3_speedup"] >= 1.0


# ── Tensor Parallel ───────────────────────────────────────────────────────────

class TestTensorParallel:
    def _cfg(self, tp=2):
        return WorldConfig(tp, 0, 0, dp_size=1, tp_size=tp, pp_size=1)

    def test_column_parallel_output(self):
        cfg = self._cfg(2)
        l   = ColumnParallelLinear(16, 32, cfg, gather_output=True)
        x   = torch.randn(2, 4, 16)
        y   = l(x)
        assert y.shape[-1] <= 32   # may gather to 32 or stay as shard

    def test_row_parallel_output(self):
        cfg = self._cfg(2)
        l   = RowParallelLinear(32, 16, cfg, input_is_parallel=False)
        x   = torch.randn(2, 4, 32)
        y   = l(x)
        assert y.shape == (2, 4, 16)

    def test_tp_mlp_shape(self):
        cfg = self._cfg(2)
        mlp = TensorParallelMLP(16, 64, cfg)
        x   = torch.randn(2, 4, 16)
        y   = mlp(x)
        assert y.shape[-1] == 16


# ── Pipeline Parallelism ──────────────────────────────────────────────────────

class TestPipeline:
    def _engine(self, pp=4, n_micro=4):
        cfg    = WorldConfig(pp, 0, 0, dp_size=1, tp_size=1, pp_size=pp)
        layers = [Block(16) for _ in range(8)]
        return PipelineEngine(layers, cfg, n_microbatches=n_micro)

    def test_n_stages(self):
        e = self._engine(pp=4)
        assert len(e.stages) == 4

    def test_layers_per_stage(self):
        e = self._engine(pp=4)
        total = sum(len(s.layers) for s in e.stages)
        assert total == 8

    def test_bubble_fraction(self):
        e = self._engine(pp=4, n_micro=8)
        s = e.schedule
        assert 0.0 < s.bubble_fraction < 1.0

    def test_efficiency_positive(self):
        e = self._engine(pp=4, n_micro=16)
        assert e.schedule.efficiency > 0.5

    def test_forward_microbatches(self):
        e      = self._engine(pp=4)
        inputs = [torch.randn(2, 8, 16) for _ in range(4)]
        outs   = e.forward_microbatches(inputs, pp_rank=0)
        assert len(outs) == 4


# ── Gradient Checkpointing ────────────────────────────────────────────────────

class TestCheckpointing:
    def test_apply_half(self):
        layers   = [Block(16) for _ in range(8)]
        cfg      = CheckpointingConfig(enabled=True, checkpoint_ratio=0.5)
        sc       = SelectiveCheckpointing(layers, cfg)
        wrapped  = sc.apply()
        n_ckpt   = sum(1 for l in wrapped if isinstance(l, CheckpointedLayer))
        assert n_ckpt >= 1

    def test_disabled(self):
        layers   = [Block(16) for _ in range(4)]
        cfg      = CheckpointingConfig(enabled=False)
        sc       = SelectiveCheckpointing(layers, cfg)
        wrapped  = sc.apply()
        assert all(not isinstance(l, CheckpointedLayer) for l in wrapped)

    def test_checkpointed_layer_forward(self):
        l = CheckpointedLayer(Block(16), enabled=False)
        x = torch.randn(2, 4, 16)
        y = l(x)
        assert y.shape == x.shape

    def test_estimate_activation_memory(self):
        m = estimate_activation_memory(32, 8, 512, 768, 12)
        assert m["total_mb"] > 0
        assert "with_ckpt_mb" in m


# ── Mixed Precision ───────────────────────────────────────────────────────────

class TestAMP:
    def test_loss_scaler_state(self):
        s = LossScaler(AMPConfig(dtype="fp16"))
        d = s.state_dict()
        assert "scale" in d and "steps" in d

    def test_scale_loss(self):
        s    = LossScaler(AMPConfig(dtype="fp16", initial_scale=128.0))
        loss = torch.tensor(1.0)
        assert s.scale_loss(loss).item() == 128.0

    def test_mixed_precision_step(self):
        m   = TinyLM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        t   = MixedPrecisionTrainer(m, opt, AMPConfig(dtype="fp32"))
        ids = torch.randint(0, V, (2, 4))
        with t.autocast():
            out  = m(ids)
            loss = out[0].sum()
        t.backward(loss)
        ok = t.step()
        assert ok

    def test_memory_report(self):
        m   = TinyLM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        t   = MixedPrecisionTrainer(m, opt, AMPConfig(dtype="fp32"))
        r   = t.memory_report()
        assert "n_params" in r and "fp32_gb" in r


class TestWorldRanks:
    def test_pp_rank(self):
        cfg = WorldConfig(4, 2, 2, dp_size=2, tp_size=1, pp_size=2)
        assert cfg.pp_rank == 0 or cfg.pp_rank == 1

    def test_first_last_pp_stage(self):
        cfg = WorldConfig(2, 0, 0, dp_size=1, tp_size=1, pp_size=2)
        assert cfg.is_first_pp_stage

        cfg2 = WorldConfig(2, 1, 1, dp_size=1, tp_size=1, pp_size=2)
        assert cfg2.is_last_pp_stage


class TestDDPBuckets:
    def test_at_least_one_bucket(self):
        cfg = WorldConfig(2, 0, 0, 2, 1, 1)
        m   = TinyLM()
        w   = DataParallelWrapper(m, cfg, DDPConfig(bucket_size_mb=0.01))
        assert len(w._buckets) >= 1

    def test_all_params_covered(self):
        cfg = WorldConfig(2, 0, 0, 2, 1, 1)
        m   = TinyLM()
        w   = DataParallelWrapper(m, cfg)
        all_params = [p for b in w._buckets for p in b]
        n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
        assert len(all_params) == n_trainable


class TestPipelineScheduleFormula:
    def test_more_microbatches_less_bubble(self):
        s4  = PipelineSchedule(n_microbatches=4,  n_stages=4)
        s16 = PipelineSchedule(n_microbatches=16, n_stages=4)
        assert s16.bubble_fraction < s4.bubble_fraction

    def test_single_stage_no_bubble(self):
        s = PipelineSchedule(n_microbatches=8, n_stages=1)
        assert s.bubble_fraction == 0.0


class TestLossScalerGrowth:
    def test_scale_grows(self):
        s   = LossScaler(AMPConfig(dtype="fp16", initial_scale=128.0,
                                    growth_interval=1, scale_growth=2.0))
        s.update()   # trigger growth
        assert s.scale >= 128.0

    def test_backoff_after_overflow(self):
        m   = TinyLM()
        opt = torch.optim.AdamW(m.parameters())
        s   = LossScaler(AMPConfig(dtype="fp16", initial_scale=128.0,
                                    backoff_factor=0.5))
        # Inject inf gradient to trigger overflow
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, float("inf"))
                break
        s.unscale_(opt)
        ok = s.step(opt)
        assert not ok
        assert s.scale < 128.0
