"""NanoMind Interpretability sub-package — attention, saliency, probing, circuits.

Implements the full interpretability toolbox:
  1. AttentionExtractor  — hook-based attention weight capture
  2. GradientSaliency    — vanilla, grad×input, integrated gradients
  3. LinearProbe         — linear probing on frozen representations
  4. LogitLens           — project intermediate hiddens to vocabulary
  5. HeadAblator         — zero-ablate heads to find critical circuits
  6. OcclusionAttributor — leave-one-out token attribution
  7. ShapleyAttributor   — approximate Shapley value attribution
  8. ActivationPatcher   — causal tracing via activation patching

Primary exports:
    - :class:`AttentionMap`        — layer attention weights + entropy + rollout
    - :class:`AttentionExtractor`  — register hooks, extract()
    - :class:`SaliencyMap`         — per-token saliency, normalised(), top_k_tokens()
    - :class:`GradientSaliency`    — vanilla, grad_times_input, integrated_gradients
    - :class:`LinearProbe`         — fit(), train + test accuracy
    - :class:`ProbeResult`         — layer, task, accuracy, loss
    - :class:`LayerwiseProber`     — probe_all_layers(), extract_hiddens()
    - :class:`LogitLensResult`     — layer_probs, rank_of_correct, prediction_change
    - :class:`LogitLens`           — analyse()
    - :class:`AblationResult`      — loss_delta, importance
    - :class:`HeadAblator`         — ablate_head, ablate_all
    - :class:`Attribution`         — scores, normalised, top_k
    - :class:`OcclusionAttributor` — attribute()
    - :class:`ShapleyAttributor`   — attribute()
    - :class:`PatchResult`         — recovery score
    - :class:`ActivationPatcher`   — trace()
"""

from nanomind.interpret.attention import AttentionMap, AttentionExtractor
from nanomind.interpret.saliency import SaliencyMap, GradientSaliency
from nanomind.interpret.probing import LinearProbe, ProbeResult, LayerwiseProber
from nanomind.interpret.logit_lens import LogitLensResult, LogitLens
from nanomind.interpret.circuits import AblationResult, HeadAblator
from nanomind.interpret.attribution import Attribution, OcclusionAttributor, ShapleyAttributor
from nanomind.interpret.patching import PatchResult, ActivationPatcher

__all__ = [
    "AttentionMap", "AttentionExtractor",
    "SaliencyMap", "GradientSaliency",
    "LinearProbe", "ProbeResult", "LayerwiseProber",
    "LogitLensResult", "LogitLens",
    "AblationResult", "HeadAblator",
    "Attribution", "OcclusionAttributor", "ShapleyAttributor",
    "PatchResult", "ActivationPatcher",
]
