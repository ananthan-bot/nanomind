"""NanoMind SSM sub-package — Structured State Space Models (S4/Mamba).

Implements the full SSM research stack:
  1. SSMConfig            — d_model, d_state, expand, dt_min/max
  2. make_hippo_matrix    — HiPPO-LegS A matrix initialization
  3. DiscretizedSSM       — ZOH discretization, conv (FFT) + recurrent modes
  4. SelectiveSSM         — Mamba-style input-dependent Δ, B, C
  5. MambaBlock           — expand → conv → SSM → gate → project
  6. MambaLM              — full Mamba language model (weight tied)
  7. S4Layer              — independent per-channel SSMs, FFT convolution
  8. S4Block              — S4 + FFN + residual (pre-norm)
  9. S4Model              — full S4 language model
  10. LinearAttention     — O(T) causal linear attention with recurrent_step
  11. RetentiveLayer      — Retentive Networks (γ-decay mask)
  12. LinearAttnConfig    — linear attention configuration
  13. ComplexityComparison — FLOPs/memory comparison dataclass
  14. ARCHITECTURE_COMPLEXITIES — list of all architectures
  15. compute_ssm_impulse_response — K[t] = C Ā^t B̄
  16. effective_memory_length — last t where |K[t]| > threshold
  17. parameter_count_comparison — Transformer vs S4 vs Mamba params
  18. flops_comparison    — GFLOPs for one forward pass
  19. HybridConfig        — hybrid model config
  20. HybridSSMTransformer — Mamba + periodic local attention (Jamba-style)

Primary exports:
    - :class:`SSMConfig`             — SSM configuration
    - :func:`make_hippo_matrix`      — HiPPO-LegS A matrix
    - :class:`DiscretizedSSM`        — ZOH discretized SSM (S4)
    - :class:`SelectiveSSM`          — Mamba selective state space
    - :class:`MambaBlock`            — full Mamba block
    - :class:`MambaLM`               — Mamba language model
    - :class:`S4Layer`               — S4 per-channel SSMs
    - :class:`S4Block`               — S4 block with FFN
    - :class:`S4Model`               — S4 language model
    - :class:`LinearAttention`       — O(T) linear attention
    - :class:`RetentiveLayer`        — Retentive Networks
    - :func:`flops_comparison`       — GFLOPs analysis
    - :func:`parameter_count_comparison` — param count analysis
    - :class:`HybridSSMTransformer`  — hybrid Mamba+attention model
"""

from nanomind.ssm.core import SSMConfig, make_hippo_matrix, DiscretizedSSM
from nanomind.ssm.mamba import SelectiveSSM, MambaBlock, MambaLM
from nanomind.ssm.s4 import S4Layer, S4Block, S4Model
from nanomind.ssm.linear_attn import (
    LinearAttnConfig, LinearAttention, RetentiveLayer,
)
from nanomind.ssm.analysis import (
    ComplexityComparison, ARCHITECTURE_COMPLEXITIES,
    compute_ssm_impulse_response, effective_memory_length,
    parameter_count_comparison, flops_comparison,
)
from nanomind.ssm.hybrid import HybridConfig, HybridSSMTransformer, LocalAttentionLayer

__all__ = [
    "SSMConfig", "make_hippo_matrix", "DiscretizedSSM",
    "SelectiveSSM", "MambaBlock", "MambaLM",
    "S4Layer", "S4Block", "S4Model",
    "LinearAttnConfig", "LinearAttention", "RetentiveLayer",
    "ComplexityComparison", "ARCHITECTURE_COMPLEXITIES",
    "compute_ssm_impulse_response", "effective_memory_length",
    "parameter_count_comparison", "flops_comparison",
    "HybridConfig", "HybridSSMTransformer", "LocalAttentionLayer",
]
