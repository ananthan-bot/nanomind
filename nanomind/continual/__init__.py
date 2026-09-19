"""NanoMind Continual Learning sub-package — learn sequentially without forgetting.

Implements the full continual learning toolbox:
  1. ContinualConfig      — strategy, ewc_lambda, replay_buffer, packnet_ratio
  2. EWC                  — Fisher Information, register_task, penalty()
  3. SynapticIntelligence — online importance via gradient×delta
  4. ReplayBuffer         — reservoir sampling, add/sample batch, DER++ logits
  5. PackNet              — prune_and_pack, apply_mask, freeze_past_weights
  6. ContinualMetrics     — AA, BWT, forgetting score
  7. ContinualEvaluator   — evaluate_task, record, compute()
  8. ContinualTrainer     — unified train_task() for all strategies

Primary exports:
    - :class:`ContinualConfig`      — strategy, ewc_lambda, replay settings
    - :class:`EWC`                  — Fisher IM, register_task, penalty()
    - :class:`SynapticIntelligence` — begin/end_task, update_importances, penalty()
    - :class:`ReplayBuffer`         — reservoir sampling, sample_batch, task_counts
    - :class:`ReplayEntry`          — x, y, task_id, logits (DER++)
    - :class:`PackNet`              — prune_and_pack, apply_mask, free_ratio
    - :class:`ContinualMetrics`     — average_accuracy, backward_transfer, forgetting
    - :class:`ContinualEvaluator`   — record, evaluate_task, compute()
    - :class:`ContinualTrainer`     — train_task, task_logs
"""

from nanomind.continual.config import ContinualConfig
from nanomind.continual.ewc import EWC
from nanomind.continual.si import SynapticIntelligence
from nanomind.continual.replay import ReplayBuffer, ReplayEntry
from nanomind.continual.packnet import PackNet
from nanomind.continual.evaluator import ContinualMetrics, ContinualEvaluator
from nanomind.continual.trainer import ContinualTrainer

__all__ = [
    "ContinualConfig",
    "EWC",
    "SynapticIntelligence",
    "ReplayBuffer", "ReplayEntry",
    "PackNet",
    "ContinualMetrics", "ContinualEvaluator",
    "ContinualTrainer",
]
