"""NanoMind MoE++ sub-package — Advanced Mixture of Experts.

Implements the full MoE stack for large language models:
  1. MoEConfig            — n_experts, top_k, capacity_factor, router_type
  2. Expert / ExpertBank  — GELU/SwiGLU FFN experts, shared experts (DeepSeek)
  3. TopKRouter           — noisy top-K, load-balance auxiliary loss
  4. ExpertChoiceRouter   — experts choose tokens (perfect balance)
  5. HashRouter           — deterministic, no learnable parameters
  6. CapacityBuffer       — token overflow management, capacity stats
  7. MoELayer             — dispatch/compute/combine forward pass
  8. MoETransformerBlock  — attention + MoE FFN with residuals
  9. SparseMoETransformer — full LM with top-K sparse FFNs
  10. load_balance_loss   — Switch Transformer auxiliary loss
  11. z_loss              — ST-MoE router logit regularisation
  12. combined_moe_loss   — load_balance + z_loss combined

Primary exports:
    - :class:`MoEConfig`            — configuration dataclass
    - :class:`Expert`               — single FFN expert (GELU/SwiGLU)
    - :class:`ExpertBank`           — bank of N experts + shared
    - :class:`TopKRouter`           — noisy top-K routing + LB loss
    - :class:`ExpertChoiceRouter`   — expert-selects-tokens routing
    - :class:`HashRouter`           — deterministic hash routing
    - :class:`RoutingOutput`        — indices, weights, aux_loss, probs
    - :class:`CapacityBuffer`       — capacity(n), masks, stats
    - :class:`CapacityStats`        — overflow statistics
    - :class:`MoELayer`             — full dispatch/combine layer
    - :class:`MoETransformerBlock`  — attn + MoE FFN block
    - :class:`SparseMoETransformer` — full sparse LM, n_params, n_active_params
    - :func:`load_balance_loss`     — Switch LB auxiliary loss
    - :func:`z_loss`                — ST-MoE z-loss
    - :func:`entropy_loss`          — routing entropy regularisation
    - :func:`combined_moe_loss`     — lb + z combined
"""

from nanomind.moe_v2.config import MoEConfig
from nanomind.moe_v2.expert import Expert, ExpertBank
from nanomind.moe_v2.router import TopKRouter, ExpertChoiceRouter, HashRouter, RoutingOutput
from nanomind.moe_v2.capacity import CapacityBuffer, CapacityStats
from nanomind.moe_v2.layer import MoELayer
from nanomind.moe_v2.model import MoETransformerBlock, SparseMoETransformer
from nanomind.moe_v2.losses import load_balance_loss, z_loss, entropy_loss, combined_moe_loss

__all__ = [
    "MoEConfig",
    "Expert", "ExpertBank",
    "TopKRouter", "ExpertChoiceRouter", "HashRouter", "RoutingOutput",
    "CapacityBuffer", "CapacityStats",
    "MoELayer",
    "MoETransformerBlock", "SparseMoETransformer",
    "load_balance_loss", "z_loss", "entropy_loss", "combined_moe_loss",
]
