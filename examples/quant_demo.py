"""
examples/quant_demo.py — NanoMind Quantization & Compression demo.

Demonstrates:
  1. QuantConfig: configure bits, scheme, granularity
  2. TensorQuantizer: per-tensor/channel/group quantization
  3. RTNQuantizer: round-to-nearest PTQ
  4. GPTQQuantizer: Hessian-compensated quantization
  5. AWQQuantizer: activation-aware weight quantization
  6. QAT: quantization-aware training with STE
  7. ModelCalibrator: sensitivity analysis, mixed precision
  8. Model size comparison: FP32 vs INT8 vs INT4

Usage:
    python examples/quant_demo.py
"""
import torch
import torch.nn as nn
from nanomind.quant import (
    QuantConfig, TensorQuantizer, compute_scale_zero,
    quantize, dequantize, quantize_dequantize,
    RTNQuantizer, GPTQQuantizer, HessianCollector,
    AWQQuantizer, ActivationScaleCollector,
    QATLinear, convert_to_qat,
    ModelCalibrator, LayerStats,
)

print("=" * 60)
print("NanoMind Quantization & Compression Demo")
print("=" * 60)

# ── QuantConfig ───────────────────────────────────────────────────────────────
print("
── QuantConfig ──")
for bits in [8, 4, 2]:
    cfg = QuantConfig(bits=bits, scheme="symmetric", granularity="per_channel")
    print(f"  INT{bits}: {cfg.to_dict()}")

# ── TensorQuantizer ───────────────────────────────────────────────────────────
print("
── TensorQuantizer (per-channel INT8) ──")
w    = torch.randn(64, 32)
cfg8 = QuantConfig(bits=8, granularity="per_channel")
q8   = TensorQuantizer(cfg8)
q8.calibrate(w)
w_int = q8.quantize(w)
w_rec = q8.dequantize(w_int)
err   = q8.quantization_error(w)
print(f"  Weight shape:     {tuple(w.shape)}")
print(f"  Quantized dtype:  {w_int.dtype}")
print(f"  Error: MSE={err['mse']:.8f}, SNR={err['snr_db']:.1f}dB")

print("
── TensorQuantizer (per-group INT4) ──")
cfg4 = QuantConfig(bits=4, granularity="per_group", group_size=16)
q4   = TensorQuantizer(cfg4)
err4 = q4.quantization_error(w)
print(f"  INT4 per-group: MSE={err4['mse']:.6f}, SNR={err4['snr_db']:.1f}dB")

# ── RTN ───────────────────────────────────────────────────────────────────────
print("
── RTN (Round-To-Nearest PTQ) ──")
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32, 64)
        self.l2 = nn.Linear(64, 16)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

model = TinyModel()
fp32_params = sum(p.numel() for p in model.parameters()) * 4
cfg_rtn = QuantConfig(bits=4, granularity="per_channel", method="rtn")
rtn     = RTNQuantizer(model, cfg_rtn)
results = rtn.quantize()
for name, err in results.items():
    print(f"  {name}: SNR={err['snr_db']:.1f}dB, MSE={err['mse']:.8f}")
int4_size = rtn.model_size_bytes()
print(f"  FP32 size: {fp32_params:,} bytes | INT4 size: {int4_size:,} bytes "
      f"({fp32_params/int4_size:.1f}x compression)")
for s in rtn.layer_stats():
    print(f"  Layer: {s}")

# ── GPTQ ──────────────────────────────────────────────────────────────────────
print("
── GPTQ (Hessian-compensated quantization) ──")
W = torch.randn(16, 32)
X = torch.randn(64, 32)   # calibration activations
H = X.T @ X / 64          # Hessian
cfg_gptq = QuantConfig(bits=4, granularity="per_channel")
gptq     = GPTQQuantizer(W, H, cfg_gptq)
W_q      = gptq.quantize()
print(f"  Original W MSE: {(W - W_q).pow(2).mean().item():.6f}")
print(f"  GPTQ error: {gptq.error:.6f}")

# ── AWQ ───────────────────────────────────────────────────────────────────────
print("
── AWQ (Activation-Aware Weight Quantization) ──")
act_scales = torch.rand(32) * 2 + 0.5   # simulate activation magnitudes
cfg_awq    = QuantConfig(bits=4, granularity="per_channel")
awq        = AWQQuantizer(W, act_scales, cfg_awq, alpha=0.5)
W_awq, s   = awq.quantize()
err_awq    = awq.error()
print(f"  AWQ error: MSE={err_awq['mse']:.6f}, scale_mean={err_awq['scale_mean']:.4f}")
print(f"  Scale range: [{s.min():.4f}, {s.max():.4f}]")

# ── QAT ───────────────────────────────────────────────────────────────────────
print("
── Quantization-Aware Training (QAT) ──")
cfg_qat  = QuantConfig(bits=4, method="qat")
qat_model = convert_to_qat(TinyModel(), cfg_qat)
x   = torch.randn(4, 32)
out = qat_model(x)
print(f"  QAT output shape: {tuple(out.shape)}")
# Verify gradients flow
loss = out.sum()
loss.backward()
has_grad = any(p.grad is not None for p in qat_model.parameters())
print(f"  Gradients flow through fake-quant (STE): {has_grad}")
# Show QATLinear layers
n_qat = sum(1 for m in qat_model.modules() if isinstance(m, QATLinear))
print(f"  QATLinear layers: {n_qat}")

# ── ModelCalibrator ───────────────────────────────────────────────────────────
print("
── Model Calibration & Sensitivity ──")
calibrator = ModelCalibrator(model, cfg4)
stats      = calibrator.run()
report     = calibrator.report()
print(f"  Calibration report: {report}")
sensitivity = calibrator.sensitivity_analysis()
print(f"  Most sensitive layer: {sensitivity[0]['name']} (SNR={sensitivity[0]['weight_snr']:.1f}dB)")
mp_plan = calibrator.mixed_precision_suggestion(high_bits=8, low_bits=4, threshold_snr=25.0)
print(f"  Mixed precision plan: {mp_plan}")

print("
Quantization demo complete!")
