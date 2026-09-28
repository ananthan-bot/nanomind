"""NanoMind Speculative Decoding sub-package — Fast inference via speculation.

Implements the full speculative decoding stack:
  1. DraftModel           — abstract draft model interface
  2. NgramDraftModel      — n-gram lookup draft model (no parameters)
  3. SmallModelDraft      — neural small model draft wrapper
  4. SpeculativeSampler   — reject/accept + greedy_verify
  5. SpeculativeResult    — accepted_ids, acceptance_rate, efficiency_gain
  6. SpeculativeDecoder   — K-step speculation loop, GenerationStats
  7. GenerationStats      — speedup, mean_accepted, acceptance_rate
  8. MedusaHead           — SiLU MLP lookahead head
  9. MedusaModel          — K Medusa heads, speculate(), medusa_loss()
  10. TreeNode / TokenTree — candidate tree, all_paths, verify_paths
  11. LookaheadDecoder    — Jacobi single-model speculation

Primary exports:
    - :class:`DraftModel`           — ABC: draft(), logits()
    - :class:`NgramDraftModel`      — n-gram lookup
    - :class:`SmallModelDraft`      — neural draft
    - :class:`SpeculativeSampler`   — verify(), greedy_verify()
    - :class:`SpeculativeResult`    — accepted_ids, acceptance_rate
    - :class:`SpeculativeDecoder`   — generate(), generate_naive(), stats
    - :class:`GenerationStats`      — speedup, tokens_per_second
    - :class:`MedusaHead`           — single lookahead head
    - :class:`MedusaModel`          — full Medusa model, medusa_loss
    - :class:`TokenTree`            — tree builder, all_paths, verify_paths
    - :class:`TreeNode`             — tree node with parent/children
    - :class:`LookaheadDecoder`     — single-model Jacobi decoding
"""

from nanomind.specd.draft import DraftModel, NgramDraftModel, SmallModelDraft
from nanomind.specd.sampler import SpeculativeSampler, SpeculativeResult
from nanomind.specd.engine import SpeculativeDecoder, GenerationStats
from nanomind.specd.medusa import MedusaHead, MedusaModel
from nanomind.specd.tree import TreeNode, TokenTree
from nanomind.specd.lookahead import LookaheadDecoder, LookaheadState

__all__ = [
    "DraftModel", "NgramDraftModel", "SmallModelDraft",
    "SpeculativeSampler", "SpeculativeResult",
    "SpeculativeDecoder", "GenerationStats",
    "MedusaHead", "MedusaModel",
    "TreeNode", "TokenTree",
    "LookaheadDecoder", "LookaheadState",
]
