"""NanoMind DPO sub-package — Direct Preference Optimization.

DPO eliminates the separate reward model of RLHF by directly optimising
the policy to prefer chosen completions over rejected ones.

Primary exports:
    - :class:`DPOTrainer`  — train_step() + train_epoch() with frozen reference
    - :class:`DPODataset`  — (prompt, chosen, rejected) dataset with masks
    - :class:`DPOConfig`   — beta, label_smoothing, loss_type (sigmoid/ipo)
    - :func:`dpo_loss`     — core DPO/IPO loss function
    - :func:`compute_log_probs` — masked sequence log-probabilities
    - :func:`reference_free_dpo_loss` — DPO without reference model
"""

from nanomind.dpo.config import DPOConfig
from nanomind.dpo.loss import dpo_loss, compute_log_probs, reference_free_dpo_loss
from nanomind.dpo.dataset import DPODataset
from nanomind.dpo.trainer import DPOTrainer

__all__ = [
    "DPOConfig", "DPOTrainer", "DPODataset",
    "dpo_loss", "compute_log_probs", "reference_free_dpo_loss",
]
