"""
examples/distributed_demo.py — NanoMind Distributed Training demo.

Usage:
    python examples/distributed_demo.py
"""
import torch
import torch.nn as nn
from nanomind.distributed import (
    WorldConfig, MockProcessGroup, build_process_groups,
    DDPConfig, DDPStats, DataParallelWrapper, ZeROStats,
    ColumnParallelLinear, RowParallelLinear, TensorParallelMLP,
    PipelineStage, PipelineEngine, PipelineSchedule,
    CheckpointingConfig, CheckpointedLayer, SelectiveCheckpointing,
    estimate_activation_memory,
    AMPConfig, LossScaler, MixedPrecisionTrainer,
)

V = 64

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.ff(self.norm(x)) + x

class TinyLM(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.rnn  = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind Distributed Training Demo")
print("=" * 60)

# ── World Config ──────────────────────────────────────────────────────────────
print("
── 3D Parallelism World Config ──")
configs = [
    WorldConfig(world_size=8,   rank=0, local_rank=0, dp_size=4, tp_size=2, pp_size=1),
    WorldConfig(world_size=16,  rank=0, local_rank=0, dp_size=4, tp_size=2, pp_size=2),
    WorldConfig(world_size=512, rank=0, local_rank=0, dp_size=64, tp_size=4, pp_size=2),
]
for cfg in configs:
    print(f"  {cfg.world_size} GPUs: DP×TP×PP = "
          f"{cfg.dp_size}×{cfg.tp_size}×{cfg.pp_size}, "
          f"is_main={cfg.is_main_process}")

cfg8 = WorldConfig(world_size=8, rank=0, local_rank=0, dp_size=4, tp_size=2, pp_size=1)
groups = build_process_groups(cfg8)
print(f"  Process groups: {list(groups.keys())}")

# ── Data Parallel (DDP) ───────────────────────────────────────────────────────
print("
── Data Parallel (DDP) ──")
dp_cfg  = WorldConfig(world_size=4, rank=0, local_rank=0, dp_size=4, tp_size=1, pp_size=1)
model   = TinyLM()
wrapper = DataParallelWrapper(model, dp_cfg, DDPConfig(bucket_size_mb=1.0))
print(f"  Model params: {wrapper.n_parameters():,}")
print(f"  DDP buckets:  {len(wrapper._buckets)}")

# Simulate backward + gradient sync
ids  = torch.randint(0, V, (4, 8))
out  = wrapper(ids)
loss = out[0].sum()
loss.backward()
wrapper.finish_gradient_synchronization()
print(f"  After allreduce: {wrapper.stats.to_dict()}")

# ZeRO memory analysis
zero = ZeROStats(model, world_size=8)
z    = zero.summary()
print(f"  ZeRO-3 (8 GPUs): {z['zero3_gb']:.3f} GB vs {z['baseline_gb']:.3f} GB baseline "
      f"({z['zero3_speedup']}× reduction)")

# ── Tensor Parallelism ────────────────────────────────────────────────────────
print("
── Tensor Parallelism (Megatron-LM) ──")
tp_cfg  = WorldConfig(world_size=4, rank=0, local_rank=0, dp_size=1, tp_size=4, pp_size=1)
col_lin = ColumnParallelLinear(64, 256, tp_cfg, bias=True, gather_output=True)
row_lin = RowParallelLinear   (256, 64, tp_cfg, bias=True, input_is_parallel=False)
tp_mlp  = TensorParallelMLP   (64, 256, tp_cfg)

x = torch.randn(2, 8, 64)
print(f"  ColumnParallel: in={col_lin.weight_shape}, out={col_lin(x).shape}")
print(f"  RowParallel:    in={row_lin.weight_shape}, out={row_lin(col_lin(x)).shape}")
print(f"  TP-MLP:         {tp_mlp(x).shape}")

# ── Pipeline Parallelism ──────────────────────────────────────────────────────
print("
── Pipeline Parallelism ──")
pp_cfg = WorldConfig(world_size=4, rank=0, local_rank=0, dp_size=1, tp_size=1, pp_size=4)
layers = [TinyTransformerBlock(32) for _ in range(8)]
engine = PipelineEngine(layers, pp_cfg, n_microbatches=4)

sched  = engine.schedule
print(f"  Stages: {engine.pp_size}, Micro-batches: {engine.n_microbatches}")
print(f"  Schedule: {sched.to_dict()}")
print(f"  Memory: {engine.memory_per_stage(32, 16, 8)}")

micro_inputs = [torch.randn(2, 16, 32) for _ in range(4)]
outputs = engine.forward_microbatches(micro_inputs, pp_rank=0)
print(f"  Micro-batch outputs: {[tuple(o.shape) for o in outputs[:2]]}...")

# ── Gradient Checkpointing ────────────────────────────────────────────────────
print("
── Gradient Checkpointing ──")
ckpt_cfg = CheckpointingConfig(enabled=True, checkpoint_ratio=0.5)
sc       = SelectiveCheckpointing(layers, ckpt_cfg)
ckpt_layers = sc.apply()
n_ckpt = sum(1 for l in ckpt_layers if isinstance(l, CheckpointedLayer))
print(f"  Layers: {len(layers)}, Checkpointed: {n_ckpt}/{len(layers)}")
print(f"  Memory savings: {sc.memory_savings(512.0)}")

mem = estimate_activation_memory(32, 8, 512, 768, 12)
print(f"  Activation memory (32L, 768D, T=512): {mem}")

# CheckpointedLayer
x_ckpt = torch.randn(2, 16, 32, requires_grad=True)
ckpt_l = CheckpointedLayer(layers[0], enabled=True)
y_ckpt = ckpt_l(x_ckpt)
print(f"  CheckpointedLayer output: {tuple(y_ckpt.shape)}")

# ── Mixed Precision ───────────────────────────────────────────────────────────
print("
── Mixed Precision Training (AMP) ──")
amp_model = TinyLM(32)
opt       = torch.optim.AdamW(amp_model.parameters(), lr=1e-4)
amp_cfg   = AMPConfig(dtype="fp32")  # fp32 for CPU demo
trainer   = MixedPrecisionTrainer(amp_model, opt, amp_cfg)

with trainer.autocast():
    out  = amp_model(torch.randint(0, V, (4, 8)))
    loss = out[0].sum()

trainer.backward(loss)
trainer.step(max_grad_norm=1.0)
print(f"  Memory report: {trainer.memory_report()}")

# LossScaler
scaler = LossScaler(AMPConfig(dtype="fp16", initial_scale=2**15))
print(f"  Initial scale: {scaler.scale:.0f}")
scaler.update()
print(f"  Scaler state: {scaler.state_dict()}")

print("
Distributed training demo complete!")
