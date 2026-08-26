"""NanoMind speculative decoding sub-package.

Speculative decoding accelerates autoregressive generation 2-4x by:
  1. Using a small *draft* model to cheaply generate K candidate tokens
  2. Running the large *target* model once to verify all K tokens in parallel
  3. Accepting tokens where distributions agree; sampling corrections where they don't
  4. Guaranteed to produce the EXACT SAME distribution as pure target-model sampling

Primary exports:
    - :class:`SpeculativeGenerator`     — high-level generate() API
    - :class:`SpeculativeConfig`        — n_draft, temperature, top_k/p, max_new_tokens
    - :func:`speculative_decode`        — low-level speculative decoding loop
    - :func:`generate_draft`            — K draft tokens from small model
    - :func:`verify_draft`              — target model verification in one pass
    - :func:`rejection_sample`          — token accept/reject with correction sampling
    - :class:`SpeculativeStats`         — running statistics tracker
    - :func:`print_speculative_report`  — pretty-print stats dict
    - :func:`benchmark_speculative_vs_autoregressive` — speed comparison
"""

from nanomind.speculative.config import SpeculativeConfig
from nanomind.speculative.generator import SpeculativeGenerator
from nanomind.speculative.decode import speculative_decode
from nanomind.speculative.draft import generate_draft
from nanomind.speculative.verify import verify_draft
from nanomind.speculative.sampling import rejection_sample
from nanomind.speculative.tracker import SpeculativeStats, print_speculative_report
from nanomind.speculative.benchmark import benchmark_speculative_vs_autoregressive

__all__ = [
    "SpeculativeConfig",
    "SpeculativeGenerator",
    "speculative_decode",
    "generate_draft",
    "verify_draft",
    "rejection_sample",
    "SpeculativeStats",
    "print_speculative_report",
    "benchmark_speculative_vs_autoregressive",
]
