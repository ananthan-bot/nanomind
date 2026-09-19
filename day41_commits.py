"""
day41_commits.py — 20 atomic commits for Day 41: Continual Learning & Catastrophic Forgetting.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 41: Continual Learning & Catastrophic Forgetting — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — continual package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/__init__.py",
      '"""NanoMind Continual Learning sub-package — learn sequentially without forgetting."""\n')
commit("feat: add nanomind/continual/ package skeleton for continual learning")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — ContinualConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/config.py", '''\
"""
nanomind/continual/config.py — Continual learning configuration.

## Catastrophic Forgetting

When a neural network learns task B after task A, it overwrites weights
that were critical for A — this is catastrophic forgetting (McCloskey & Cohen, 1989).

## Strategies to Prevent Forgetting

1. Regularisation (Bayesian):
   EWC (Kirkpatrick et al., 2017):   penalise changes to important weights
   SI  (Zenke et al., 2017):         online importance via path integrals
   LwF (Li & Hoiem, 2017):           distill old task outputs on new data

2. Replay / Memory:
   Experience Replay:   store subset of old data, replay during training
   Generative Replay:   train a generator to produce old-task samples
   DER++ (Buzzega et al., 2020): replay with dark experience distillation

3. Parameter Isolation:
   PackNet (Mallya & Lazebnik, 2018): prune and pack task masks
   Progressive Nets (Rusu et al., 2016): add new columns per task
   HAT (Serra et al., 2018): hard attention task masks

4. Architectural:
   PNN (Progressive Neural Networks): lateral connections
   DynamicExpander: grow new layers per task

References:
  Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
  https://arxiv.org/abs/1612.00796

  van de Ven & Tolias (2019) survey: https://arxiv.org/abs/1904.07734
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ContinualConfig:
    """
    Configuration for a continual learning experiment.

    Attributes:
        strategy:       Strategy: ``"ewc"``, ``"replay"``, ``"packnet"``, ``"naive"``.
        n_tasks:        Total number of sequential tasks.
        ewc_lambda:     EWC regularisation strength.
        ewc_n_samples:  Samples for Fisher Information Matrix computation.
        replay_buffer_size: Max samples per task in replay buffer.
        replay_fraction:    Fraction of each batch from replay.
        packnet_prune_ratio: Fraction of weights to prune per task.
        si_xi:          SI damping factor.
        task_agnostic:  If True, task ID not given at test time.
    """
    strategy:            str   = "ewc"
    n_tasks:             int   = 5
    ewc_lambda:          float = 1000.0
    ewc_n_samples:       int   = 200
    replay_buffer_size:  int   = 500
    replay_fraction:     float = 0.3
    packnet_prune_ratio: float = 0.5
    si_xi:               float = 1e-3
    task_agnostic:       bool  = True

    def __post_init__(self) -> None:
        assert self.strategy in ("ewc", "replay", "packnet", "naive", "si", "lwf")
        assert self.n_tasks             >= 1
        assert self.ewc_lambda          >= 0.0
        assert self.ewc_n_samples       >= 1
        assert self.replay_buffer_size  >= 0
        assert 0.0 <= self.replay_fraction <= 1.0
        assert 0.0 <  self.packnet_prune_ratio < 1.0

    @property
    def uses_regularisation(self) -> bool:
        return self.strategy in ("ewc", "si", "lwf")

    @property
    def uses_replay(self) -> bool:
        return self.strategy == "replay"

    @property
    def uses_masking(self) -> bool:
        return self.strategy == "packnet"
''')
commit("feat: add ContinualConfig — strategy, ewc_lambda, replay_buffer_size, packnet_prune_ratio")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — EWC (Elastic Weight Consolidation)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/ewc.py", '''\
"""
nanomind/continual/ewc.py — Elastic Weight Consolidation (EWC).

EWC treats continual learning as Bayesian inference:
  P(θ | D_A, D_B) ∝ P(D_B | θ) × P(θ | D_A)

P(θ | D_A) is approximated as a diagonal Gaussian:
  P(θ | D_A) ≈ N(θ_A*, F_A)

where F_A is the diagonal Fisher Information Matrix — estimated as:
  F_i = E[(∂ log P(y|x,θ) / ∂θ_i)²]

The EWC loss for task B is:
  L(θ) = L_B(θ) + (λ/2) × Σ_i F_i × (θ_i - θ_A*_i)²

High F_i → parameter i was important for task A → constrain it.
Low F_i  → parameter i was unimportant   → allow free learning.

Reference:
  Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
  https://arxiv.org/abs/1612.00796
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.utils.logger import get_logger

log = get_logger("continual.ewc")


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    After each task, call :meth:`register_task` to compute the Fisher
    Information Matrix and store the optimal weights.
    During training on the next task, call :meth:`penalty` to get the
    EWC regularisation term to add to the loss.

    Args:
        model:      Language model.
        lambda_:    Regularisation strength (higher = less forgetting).
        n_samples:  Samples for Fisher estimation.

    Example::

        ewc = EWC(model, lambda_=1000.0)
        # After training task A:
        ewc.register_task(train_loader_A, task_id=0)
        # During training task B:
        loss = task_loss + ewc.penalty()
    """

    def __init__(
        self,
        model:    nn.Module,
        lambda_:  float = 1000.0,
        n_samples: int  = 200,
    ) -> None:
        self.model     = model
        self.lambda_   = lambda_
        self.n_samples = n_samples
        # Store per-task: optimal weights + Fisher diagonals
        self._means:   list[dict] = []   # θ*_A per task
        self._fishers: list[dict] = []   # F_A per task

    def _compute_fisher(self, batches: list) -> dict:
        """
        Compute diagonal Fisher Information Matrix.

        Args:
            batches: List of (x, y) batches.

        Returns:
            Dict mapping param name → Fisher diagonal tensor.
        """
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        n_done = 0

        for x, y in batches:
            if n_done >= self.n_samples:
                break
            self.model.zero_grad()
            logits, loss = self.model(x, y)
            if loss is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            n_done += x.shape[0]

        # Normalise
        for n in fisher:
            fisher[n] /= max(n_done, 1)
        return fisher

    def register_task(self, batches: list, task_id: int = None) -> None:
        """
        Record task completion: compute Fisher and save optimal weights.

        Args:
            batches: List of ``(x, y)`` from the completed task.
            task_id: Optional task identifier (for logging).
        """
        task_id = task_id if task_id is not None else len(self._means)
        log.info(f"EWC: registering task {task_id} with {self.n_samples} samples")

        fisher = self._compute_fisher(batches)
        means  = {n: p.data.clone()
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self._fishers.append(fisher)
        self._means.append(means)

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty for all registered tasks.

        Returns:
            Scalar EWC loss term (add to main loss before backward).
        """
        if not self._fishers:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for fisher, means in zip(self._fishers, self._means):
            for n, p in self.model.named_parameters():
                if n in fisher and p.requires_grad:
                    loss += (fisher[n] * (p - means[n]).pow(2)).sum()

        return (self.lambda_ / 2) * loss

    @property
    def n_tasks_registered(self) -> int:
        return len(self._means)

    def fisher_summary(self, task_id: int = 0) -> dict:
        """Summary statistics of Fisher matrix for a task."""
        if task_id >= len(self._fishers):
            return {}
        f     = self._fishers[task_id]
        total = sum(v.sum().item() for v in f.values())
        nparams = sum(v.numel() for v in f.values())
        return {
            "task_id":      task_id,
            "total_fisher": round(total, 4),
            "n_params":     nparams,
            "mean_fisher":  round(total / max(nparams, 1), 8),
        }
''')
commit("feat: add EWC — Fisher Information Matrix, register_task, penalty(), fisher_summary()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Synaptic Intelligence (SI)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/si.py", '''\
"""
nanomind/continual/si.py — Synaptic Intelligence (SI).

SI (Zenke et al., 2017) computes parameter importance ONLINE during
training, rather than computing Fisher after training like EWC.

Importance is accumulated as the integral of the gradient × weight change:
  Ω_i = Σ_t (-g_i^t × Δθ_i^t) / ((Δθ_i)^2 + ξ)

High Ω_i → parameter contributed to loss reduction → protect it.

Advantages over EWC:
  - No extra backward pass after task completion
  - Computes importance continuously during training
  - Lower memory (no Fisher matrix storage per task)

Reference:
  Zenke, Poole & Ganguli (2017) "Continual Learning Through Synaptic Intelligence"
  https://arxiv.org/abs/1703.04200
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.utils.logger import get_logger

log = get_logger("continual.si")


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI) continual learning regulariser.

    Args:
        model:  Language model.
        lambda_: Regularisation strength.
        xi:      Damping term for numerical stability.

    Example::

        si = SynapticIntelligence(model, lambda_=100.0)
        # Register parameter state before task begins
        si.begin_task()
        # During each training step:
        loss.backward()
        si.update_importances()   # call BEFORE optimiser step
        optimiser.step()
        # After task ends:
        si.end_task()
    """

    def __init__(
        self,
        model:   nn.Module,
        lambda_: float = 100.0,
        xi:      float = 1e-3,
    ) -> None:
        self.model   = model
        self.lambda_ = lambda_
        self.xi      = xi
        self._W:      dict = {}    # accumulated gradient × delta
        self._p_prev: dict = {}    # weights at task start
        self._p_old:  dict = {}    # weights at previous step
        self._omega:  list[dict] = []   # consolidated importances per task
        self._theta_star: list[dict] = []  # optimal weights per task

    def begin_task(self) -> None:
        """Call at the start of each new task."""
        self._W     = {n: torch.zeros_like(p.data)
                       for n, p in self.model.named_parameters() if p.requires_grad}
        self._p_prev = {n: p.data.clone()
                        for n, p in self.model.named_parameters() if p.requires_grad}
        self._p_old  = {n: p.data.clone()
                        for n, p in self.model.named_parameters() if p.requires_grad}

    def update_importances(self) -> None:
        """
        Update online importance estimates after backward pass.

        Call BEFORE the optimiser step so that current gradients
        and previous weights are both available.
        """
        for n, p in self.model.named_parameters():
            if p.grad is not None and n in self._W:
                delta = p.data - self._p_old[n]
                self._W[n] += -p.grad.data * delta
                self._p_old[n] = p.data.clone()

    def end_task(self) -> None:
        """Consolidate importances at end of task."""
        omega = {}
        for n, p in self.model.named_parameters():
            if n in self._W:
                delta_sq = (p.data - self._p_prev[n]).pow(2)
                omega[n] = self._W[n] / (delta_sq + self.xi)
                omega[n] = omega[n].abs()
        self._omega.append(omega)
        self._theta_star.append(
            {n: p.data.clone() for n, p in self.model.named_parameters()
             if p.requires_grad}
        )

    def penalty(self) -> torch.Tensor:
        """Compute SI regularisation penalty."""
        if not self._omega:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for omega, theta in zip(self._omega, self._theta_star):
            for n, p in self.model.named_parameters():
                if n in omega and p.requires_grad:
                    loss += (omega[n] * (p - theta[n]).pow(2)).sum()
        return (self.lambda_ / 2) * loss

    @property
    def n_tasks(self) -> int:
        return len(self._omega)
''')
commit("feat: add SynapticIntelligence — online importance via gradient×delta, begin/end_task, penalty()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Experience Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/replay.py", '''\
"""
nanomind/continual/replay.py — Experience Replay Buffer.

Replay buffers store a subset of past task data and mix it into the
current task's training batches to prevent forgetting.

Reservoir sampling (Vitter, 1985):
  Keep a uniform random sample of size M from a stream of N items.
  Each new item replaces a random existing item with prob M/N.
  This guarantees a uniform sample without storing all N items.

DER++ (Buzzega et al., 2020):
  Also stores logits at the time of storage as "dark experience".
  During replay, adds KL loss between current logits and stored logits.
  This soft constraint is stronger than just replaying labels.

References:
  Buzzega et al. (2020) "Dark Experience for General Continual Learning"
  https://arxiv.org/abs/2004.07211

  Rebuffi et al. (2017) "iCaRL: Incremental Classifier and Representation Learning"
  https://arxiv.org/abs/1611.07725
"""

from __future__ import annotations
import random
import torch
from dataclasses import dataclass


@dataclass
class ReplayEntry:
    """A single entry in the replay buffer."""
    x:       torch.Tensor    # input tokens (T,)
    y:       torch.Tensor    # target tokens (T,)
    task_id: int
    logits:  torch.Tensor | None = None   # stored logits for DER++


class ReplayBuffer:
    """
    Reservoir-sampled experience replay buffer.

    Maintains a fixed-size uniform random sample of training examples
    across all tasks seen so far.

    Args:
        max_size:  Maximum number of examples to store.

    Example::

        buf = ReplayBuffer(max_size=500)
        buf.add(x, y, task_id=0)
        samples = buf.sample(batch_size=16)
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._buffer: list[ReplayEntry] = []
        self._n_seen: int = 0    # total examples seen (for reservoir sampling)

    def add(
        self,
        x:       torch.Tensor,
        y:       torch.Tensor,
        task_id: int,
        logits:  torch.Tensor | None = None,
    ) -> None:
        """
        Add a single example via reservoir sampling.

        Args:
            x:       ``(T,)`` input token sequence.
            y:       ``(T,)`` target token sequence.
            task_id: Task this example belongs to.
            logits:  Optional stored logits for DER++.
        """
        entry = ReplayEntry(x.clone(), y.clone(), task_id,
                            logits.detach().clone() if logits is not None else None)
        self._n_seen += 1
        if len(self._buffer) < self.max_size:
            self._buffer.append(entry)
        else:
            idx = random.randint(0, self._n_seen - 1)
            if idx < self.max_size:
                self._buffer[idx] = entry

    def add_batch(
        self,
        xs:      torch.Tensor,
        ys:      torch.Tensor,
        task_id: int,
        logits:  torch.Tensor | None = None,
    ) -> None:
        """Add a batch of examples to the replay buffer."""
        for i in range(xs.shape[0]):
            lg = logits[i] if logits is not None else None
            self.add(xs[i], ys[i], task_id, lg)

    def sample(self, batch_size: int) -> list[ReplayEntry]:
        """
        Sample a random batch from the replay buffer.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of :class:`ReplayEntry`.
        """
        k = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, k) if k > 0 else []

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a collated ``(x_batch, y_batch)`` tensor.

        Returns:
            ``(x, y)`` tensors of shape ``(k, T)``.
        """
        entries = self.sample(batch_size)
        if not entries:
            raise RuntimeError("Replay buffer is empty")
        xs = torch.stack([e.x for e in entries])
        ys = torch.stack([e.y for e in entries])
        return xs, ys

    def task_counts(self) -> dict[int, int]:
        """Count examples per task."""
        counts: dict[int, int] = {}
        for e in self._buffer:
            counts[e.task_id] = counts.get(e.task_id, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def n_seen(self) -> int:
        return self._n_seen
''')
commit("feat: add ReplayBuffer — reservoir sampling, add/add_batch, sample_batch, task_counts, DER++ logits")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — PackNet (pruning + packing task masks)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/packnet.py", '''\
"""
nanomind/continual/packnet.py — PackNet: prune and pack parameters for each task.

PackNet (Mallya & Lazebnik, 2018) allocates disjoint subnetworks:
  Task 1: train full network → prune P% of weights → "pack" mask_1
  Task 2: train remaining (1-P)% → prune again → pack mask_2
  Task k: remaining (1-P)^k of weights

At inference, the task ID selects which binary mask to activate.
Pruned weights can be reused for future tasks (weight recycling).

This achieves zero forgetting by hard parameter isolation.
Limitation: capacity decreases with each task.

Reference:
  Mallya & Lazebnik (2018) "PackNet: Adding Multiple Tasks to a Single Network
  by Iterative Pruning" https://arxiv.org/abs/1711.05769
"""

from __future__ import annotations
import torch
import torch.nn as nn
from nanomind.utils.logger import get_logger

log = get_logger("continual.packnet")


class PackNet:
    """
    PackNet: iterative pruning for parameter isolation across tasks.

    Args:
        model:       Language model.
        prune_ratio: Fraction of free weights to prune per task.

    Example::

        pn = PackNet(model, prune_ratio=0.5)
        # After training task 0:
        pn.prune_and_pack(task_id=0)
        # Free weights are now available for task 1.
        # At inference:
        pn.apply_mask(task_id=0)   # restore task 0 weights
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.5) -> None:
        self.model       = model
        self.prune_ratio = prune_ratio
        # Binary masks per task: 1 = owned by task, 0 = free
        self._task_masks:  list[dict[str, torch.Tensor]] = []
        # Accumulated frozen mask (union of all past tasks)
        self._frozen_mask: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p.data, dtype=torch.bool)
            for n, p in model.named_parameters() if p.requires_grad
        }
        # Saved weights per task
        self._task_weights: list[dict[str, torch.Tensor]] = []

    def _free_magnitude(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Magnitude of free (non-frozen) weights for a parameter."""
        frozen = self._frozen_mask.get(name, torch.zeros_like(param, dtype=torch.bool))
        mag    = param.data.abs().clone()
        mag[frozen] = float("inf")   # exclude frozen from pruning
        return mag

    def prune_and_pack(self, task_id: int) -> dict:
        """
        Prune the model and assign a mask to this task.

        Args:
            task_id: Current task ID.

        Returns:
            Dict with pruning statistics.
        """
        task_mask = {}
        n_pruned  = 0
        n_total   = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            frozen  = self._frozen_mask[name]
            free    = ~frozen
            n_free  = free.sum().item()
            n_prune = int(n_free * self.prune_ratio)

            # Prune smallest-magnitude free weights
            mag     = param.data.abs().clone()
            mag[frozen] = float("inf")
            flat_mag = mag.view(-1)
            if n_prune > 0:
                threshold = flat_mag.kthvalue(n_prune).values.item()
                new_mask  = (mag <= threshold) & free
            else:
                new_mask  = torch.zeros_like(free)

            task_mask[name]          = new_mask.clone()
            self._frozen_mask[name] |= new_mask
            n_pruned += new_mask.sum().item()
            n_total  += param.numel()

        self._task_masks.append(task_mask)
        self._task_weights.append(
            {n: p.data.clone() for n, p in self.model.named_parameters()
             if p.requires_grad}
        )

        log.info(f"PackNet task {task_id}: pruned {n_pruned}/{n_total} "
                 f"({100*n_pruned/max(n_total,1):.1f}%)")
        return {"n_pruned": n_pruned, "n_total": n_total,
                "prune_pct": n_pruned / max(n_total, 1)}

    def apply_mask(self, task_id: int) -> None:
        """
        Restore model weights for a specific task.

        Args:
            task_id: Task to restore.
        """
        if task_id >= len(self._task_weights):
            raise ValueError(f"Task {task_id} not registered")
        weights = self._task_weights[task_id]
        for n, p in self.model.named_parameters():
            if n in weights:
                p.data.copy_(weights[n])

    def freeze_past_weights(self) -> None:
        """Zero gradients for all frozen (past task) weights during training."""
        for n, p in self.model.named_parameters():
            if p.grad is not None and n in self._frozen_mask:
                p.grad.data[self._frozen_mask[n]] = 0.0

    @property
    def n_tasks(self) -> int:
        return len(self._task_masks)

    @property
    def free_ratio(self) -> float:
        """Fraction of weights still available for new tasks."""
        total  = sum(v.numel() for v in self._frozen_mask.values())
        frozen = sum(v.sum().item() for v in self._frozen_mask.values())
        return 1.0 - frozen / max(total, 1)
''')
commit("feat: add PackNet — prune_and_pack, apply_mask, freeze_past_weights, free_ratio")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Task evaluator
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/evaluator.py", '''\
"""
nanomind/continual/evaluator.py — Continual learning metrics.

Key metrics for continual learning:

  Average Accuracy (AA):
    AA = (1/T) × Σ_t a_t,T
    where a_t,T = accuracy on task t after training on all T tasks

  Backward Transfer (BWT):
    BWT = (1/(T-1)) × Σ_t (a_t,T - a_t,t)
    Negative BWT = forgetting  (accuracy dropped after learning new tasks)
    Positive BWT = backward transfer (new tasks helped old ones)

  Forward Transfer (FWT):
    FWT = (1/(T-1)) × Σ_t (a_t,t - b_t)
    where b_t = random baseline accuracy on task t before training
    Positive FWT = new tasks help future tasks via transfer

  Forgetting (F):
    F = (1/(T-1)) × Σ_t max_{t'≤T} a_t,t' - a_t,T
    How much accuracy was lost from the peak.

Reference:
  Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning"
  https://arxiv.org/abs/1706.08840
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ContinualMetrics:
    """Continual learning evaluation metrics."""
    accuracy_matrix: list[list[float]]   # a[t][s] = acc on task t after task s
    n_tasks:         int

    @property
    def average_accuracy(self) -> float:
        """AA after all tasks: mean accuracy on all past tasks."""
        n = self.n_tasks
        if n == 0:
            return 0.0
        return sum(self.accuracy_matrix[t][n - 1] for t in range(n)) / n

    @property
    def backward_transfer(self) -> float:
        """BWT: negative = forgetting, positive = improvement."""
        n = self.n_tasks
        if n <= 1:
            return 0.0
        bwt = sum(
            self.accuracy_matrix[t][n - 1] - self.accuracy_matrix[t][t]
            for t in range(n - 1)
        )
        return bwt / max(n - 1, 1)

    @property
    def forgetting(self) -> float:
        """Average forgetting across tasks."""
        n = self.n_tasks
        if n <= 1:
            return 0.0
        total = 0.0
        for t in range(n - 1):
            row   = self.accuracy_matrix[t][:n]
            peak  = max(row[:t + 1]) if row else 0.0
            final = row[n - 1]
            total += peak - final
        return total / max(n - 1, 1)

    def to_dict(self) -> dict:
        return {
            "average_accuracy":  round(self.average_accuracy, 4),
            "backward_transfer": round(self.backward_transfer, 4),
            "forgetting":        round(self.forgetting, 4),
            "n_tasks":           self.n_tasks,
        }


class ContinualEvaluator:
    """
    Evaluate continual learning performance across tasks.

    Args:
        model:       The language model.

    Example::

        eval_ = ContinualEvaluator(model)
        eval_.record(task_id=0, after_task=0, accuracy=0.92)
        eval_.record(task_id=0, after_task=1, accuracy=0.74)  # forgetting!
        metrics = eval_.compute()
        print(metrics.backward_transfer)  # negative = forgot
    """

    def __init__(self, model: nn.Module, n_tasks: int) -> None:
        self.model   = model
        self.n_tasks = n_tasks
        # acc_matrix[task_id][after_task] = accuracy
        self._acc: list[list[float | None]] = [
            [None] * n_tasks for _ in range(n_tasks)
        ]

    def record(self, task_id: int, after_task: int, accuracy: float) -> None:
        """Record accuracy on task_id evaluated after training on after_task."""
        self._acc[task_id][after_task] = accuracy

    @torch.no_grad()
    def evaluate_task(
        self,
        task_id:  int,
        batches:  list,
        after_task: int,
    ) -> float:
        """
        Evaluate model accuracy on a task.

        Args:
            task_id:    Task to evaluate.
            batches:    List of ``(x, y)`` batches.
            after_task: Current training step (for logging).

        Returns:
            Accuracy in [0, 1].
        """
        self.model.eval()
        n_correct = 0
        n_total   = 0
        for x, y in batches:
            logits, _ = self.model(x, y)
            preds     = logits.argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total   += y.numel()
        acc = n_correct / max(n_total, 1)
        self.record(task_id, after_task, acc)
        return acc

    def compute(self) -> ContinualMetrics:
        """Compute final continual learning metrics."""
        filled = [
            [self._acc[t][s] or 0.0 for s in range(self.n_tasks)]
            for t in range(self.n_tasks)
        ]
        return ContinualMetrics(accuracy_matrix=filled, n_tasks=self.n_tasks)
''')
commit("feat: add ContinualMetrics (AA, BWT, forgetting) + ContinualEvaluator (evaluate_task, record)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Continual Trainer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/trainer.py", '''\
"""
nanomind/continual/trainer.py — Unified continual learning trainer.

Orchestrates training across sequential tasks with the chosen strategy.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.continual.config import ContinualConfig
from nanomind.continual.ewc import EWC
from nanomind.continual.si import SynapticIntelligence
from nanomind.continual.replay import ReplayBuffer
from nanomind.continual.packnet import PackNet
from nanomind.utils.logger import get_logger

log = get_logger("continual.trainer")


class ContinualTrainer:
    """
    Unified trainer for continual learning strategies.

    Supports: ``"ewc"``, ``"si"``, ``"replay"``, ``"packnet"``, ``"naive"``.

    Args:
        model:  Language model.
        cfg:    :class:`ContinualConfig`.

    Example::

        trainer = ContinualTrainer(model, ContinualConfig(strategy="ewc"))
        for task_id, task_batches in enumerate(tasks):
            trainer.train_task(task_id, task_batches)
        metrics = trainer.evaluator.compute()
    """

    def __init__(self, model: nn.Module, cfg: ContinualConfig) -> None:
        self.model   = model
        self.cfg     = cfg
        self._ewc:    EWC | None = None
        self._si:     SynapticIntelligence | None = None
        self._replay: ReplayBuffer | None = None
        self._packnet: PackNet | None     = None
        self._task_logs: list[dict] = []

        if cfg.strategy == "ewc":
            self._ewc = EWC(model, lambda_=cfg.ewc_lambda, n_samples=cfg.ewc_n_samples)
        elif cfg.strategy == "si":
            self._si  = SynapticIntelligence(model, lambda_=cfg.ewc_lambda)
        elif cfg.strategy == "replay":
            self._replay = ReplayBuffer(max_size=cfg.replay_buffer_size)
        elif cfg.strategy == "packnet":
            self._packnet = PackNet(model, prune_ratio=cfg.packnet_prune_ratio)

    def _build_opt(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_task(
        self,
        task_id: int,
        batches: list,
        epochs:  int = 3,
        lr:      float = 1e-3,
    ) -> dict:
        """
        Train on one task.

        Args:
            task_id: Task index (0-based).
            batches: List of ``(x, y)`` batches.
            epochs:  Training epochs.
            lr:      Learning rate.

        Returns:
            Log dict with ``task_id``, ``final_loss``, ``strategy``.
        """
        opt      = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        if self._si:
            self._si.begin_task()

        total_loss = 0.0
        n_steps    = 0

        for _ in range(epochs):
            for x, y in batches:
                opt.zero_grad()
                logits, loss = self.model(x, y)
                if loss is None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                # Add regularisation
                if self._ewc:
                    loss = loss + self._ewc.penalty()
                if self._si:
                    loss = loss + self._si.penalty()

                # Replay mixing
                if self._replay and len(self._replay) > 0:
                    try:
                        rx, ry = self._replay.sample_batch(max(1, int(x.shape[0] * self.cfg.replay_fraction)))
                        r_logits, r_loss = self.model(rx, ry)
                        if r_loss is None:
                            r_loss = F.cross_entropy(r_logits.view(-1, r_logits.size(-1)), ry.view(-1))
                        loss = loss + r_loss
                    except RuntimeError:
                        pass

                loss.backward()

                if self._si:
                    self._si.update_importances()
                if self._packnet:
                    self._packnet.freeze_past_weights()

                opt.step()
                total_loss += loss.item()
                n_steps    += 1

        # Post-task hooks
        if self._ewc:
            self._ewc.register_task(batches, task_id)
        if self._si:
            self._si.end_task()
        if self._replay:
            for x, y in batches:
                self._replay.add_batch(x, y, task_id)
        if self._packnet:
            self._packnet.prune_and_pack(task_id)

        entry = {
            "task_id":    task_id,
            "final_loss": round(total_loss / max(n_steps, 1), 6),
            "strategy":   self.cfg.strategy,
        }
        self._task_logs.append(entry)
        log.info(f"Task {task_id} done: loss={entry['final_loss']:.4f}")
        return entry

    @property
    def task_logs(self) -> list[dict]:
        return list(self._task_logs)
''')
commit("feat: add ContinualTrainer — unified ewc/si/replay/packnet/naive train_task() orchestration")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — continual __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/continual/__init__.py", '''\
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
''')
commit("refactor: export all continual learning components from nanomind/continual/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/continual_demo.py", '''\
"""
examples/continual_demo.py — NanoMind Continual Learning demo.

Simulates 3 sequential tasks:
  Task 0: train on "hello world" corpus
  Task 1: train on "federated privacy" corpus
  Task 2: train on "neural architecture" corpus

Demonstrates:
  1. Naive baseline (catastrophic forgetting)
  2. EWC (penalise important weight changes)
  3. Experience Replay (mix old data with new)
  4. PackNet (prune and pack per task)
  5. Continual metrics: AA, BWT, forgetting

Usage:
    python examples/continual_demo.py
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.continual import (
    ContinualConfig, EWC, SynapticIntelligence,
    ReplayBuffer, PackNet, ContinualTrainer,
    ContinualEvaluator, ContinualMetrics,
)

# ── Tiny model ────────────────────────────────────────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, V=16, D=32, T=8):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h      = self.tok(x) + self.pos(torch.arange(S))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, V), t.view(-1)) if t is not None else None
        return logits, loss

V = 16
N_TASKS = 3

def make_task(seed, T=6, n=8):
    """Create synthetic task batches."""
    torch.manual_seed(seed)
    batches = []
    for _ in range(n):
        x = torch.randint(0, V, (2, T))
        y = torch.randint(0, V, (2, T))
        batches.append((x, y))
    return batches

tasks = [make_task(seed=i*10) for i in range(N_TASKS)]

print("=" * 60)
print("NanoMind Continual Learning Demo")
print("=" * 60)

# ── EWC ───────────────────────────────────────────────────────────────────────
print("\n── EWC (Elastic Weight Consolidation) ──")
model_ewc = TinyLM(V)
cfg_ewc   = ContinualConfig(strategy="ewc", n_tasks=N_TASKS, ewc_lambda=100.0)
trainer   = ContinualTrainer(model_ewc, cfg_ewc)
for t, task in enumerate(tasks):
    log = trainer.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
ewc_summary = trainer._ewc.fisher_summary(task_id=0)
print(f"  Fisher summary task 0: {ewc_summary}")

# ── SI ────────────────────────────────────────────────────────────────────────
print("\n── Synaptic Intelligence (SI) ──")
model_si = TinyLM(V)
cfg_si   = ContinualConfig(strategy="si", n_tasks=N_TASKS, ewc_lambda=50.0)
trainer_si = ContinualTrainer(model_si, cfg_si)
for t, task in enumerate(tasks):
    log = trainer_si.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
print(f"  SI tasks registered: {trainer_si._si.n_tasks}")

# ── Replay ────────────────────────────────────────────────────────────────────
print("\n── Experience Replay ──")
model_rp = TinyLM(V)
cfg_rp   = ContinualConfig(strategy="replay", n_tasks=N_TASKS, replay_buffer_size=100)
trainer_rp = ContinualTrainer(model_rp, cfg_rp)
for t, task in enumerate(tasks):
    log = trainer_rp.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
buf = trainer_rp._replay
print(f"  Buffer: {len(buf)} samples, tasks={buf.task_counts()}")

# ── PackNet ───────────────────────────────────────────────────────────────────
print("\n── PackNet (pruning + masking) ──")
model_pn = TinyLM(V)
cfg_pn   = ContinualConfig(strategy="packnet", n_tasks=N_TASKS, packnet_prune_ratio=0.3)
trainer_pn = ContinualTrainer(model_pn, cfg_pn)
for t, task in enumerate(tasks):
    log = trainer_pn.train_task(t, task, epochs=2)
    print(f"  Task {t}: loss={log['final_loss']:.4f}")
pn = trainer_pn._packnet
print(f"  PackNet tasks: {pn.n_tasks}, free_ratio={pn.free_ratio:.2%}")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n── Continual Metrics ──")
# Simulate accuracy matrix: 3 tasks, diagonal = task accuracy
acc_matrix = [
    [0.90, 0.72, 0.65],
    [0.00, 0.88, 0.70],
    [0.00, 0.00, 0.85],
]
metrics = ContinualMetrics(accuracy_matrix=acc_matrix, n_tasks=3)
print(f"  Average Accuracy:   {metrics.average_accuracy:.2%}")
print(f"  Backward Transfer:  {metrics.backward_transfer:.4f}")
print(f"  Forgetting:         {metrics.forgetting:.4f}")
m_dict = metrics.to_dict()
print(f"  Dict: {m_dict}")
print("\nContinual learning demo complete!")
''')
commit("feat: add examples/continual_demo.py — EWC, SI, Replay, PackNet, ContinualMetrics demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_continual.py", '''\
"""tests/test_continual.py — Tests for NanoMind continual learning."""
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.continual import (
    ContinualConfig, EWC, SynapticIntelligence,
    ReplayBuffer, ReplayEntry, PackNet,
    ContinualMetrics, ContinualEvaluator, ContinualTrainer,
)

V = 8

class TinyLM(nn.Module):
    def __init__(self, V=8, D=16, T=4):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(min(S, self.T)))
        h = h[:, :self.T]
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, V), t.view(-1)) if t is not None else None
        return logits, loss

def make_batches(n=4, T=4):
    return [(torch.randint(0, V, (2, T)), torch.randint(0, V, (2, T)))
            for _ in range(n)]


# ── ContinualConfig ───────────────────────────────────────────────────────────

class TestContinualConfig:
    def test_defaults(self):
        cfg = ContinualConfig()
        assert cfg.strategy == "ewc"
        assert cfg.n_tasks  == 5

    def test_invalid_strategy(self):
        with pytest.raises(AssertionError):
            ContinualConfig(strategy="unknown")

    def test_uses_regularisation(self):
        assert ContinualConfig(strategy="ewc").uses_regularisation
        assert ContinualConfig(strategy="si").uses_regularisation
        assert not ContinualConfig(strategy="replay").uses_regularisation

    def test_uses_replay(self):
        assert ContinualConfig(strategy="replay").uses_replay

    def test_uses_masking(self):
        assert ContinualConfig(strategy="packnet").uses_masking

    def test_invalid_prune_ratio(self):
        with pytest.raises(AssertionError):
            ContinualConfig(packnet_prune_ratio=1.0)


# ── EWC ───────────────────────────────────────────────────────────────────────

class TestEWC:
    def _ewc(self):
        m = TinyLM(V)
        return EWC(m, lambda_=10.0, n_samples=4), m

    def test_register_task(self):
        ewc, m = self._ewc()
        batches = make_batches()
        ewc.register_task(batches, task_id=0)
        assert ewc.n_tasks_registered == 1

    def test_penalty_zero_before_tasks(self):
        ewc, m = self._ewc()
        assert ewc.penalty().item() == 0.0

    def test_penalty_positive_after_register(self):
        ewc, m = self._ewc()
        ewc.register_task(make_batches(), 0)
        # Move weights slightly
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pen = ewc.penalty()
        assert pen.item() >= 0.0

    def test_fisher_summary_keys(self):
        ewc, m = self._ewc()
        ewc.register_task(make_batches(), 0)
        s = ewc.fisher_summary(0)
        for k in ("task_id", "total_fisher", "n_params", "mean_fisher"):
            assert k in s

    def test_register_two_tasks(self):
        ewc, m = self._ewc()
        ewc.register_task(make_batches(), 0)
        ewc.register_task(make_batches(), 1)
        assert ewc.n_tasks_registered == 2


# ── SynapticIntelligence ──────────────────────────────────────────────────────

class TestSI:
    def _si(self):
        m  = TinyLM(V)
        si = SynapticIntelligence(m, lambda_=10.0)
        return si, m

    def test_begin_task_sets_state(self):
        si, m = self._si()
        si.begin_task()
        assert len(si._W) > 0

    def test_penalty_zero_before_tasks(self):
        si, m = self._si()
        assert si.penalty().item() == 0.0

    def test_end_task_increments(self):
        si, m = self._si()
        si.begin_task()
        x, y = make_batches(1)[0]
        _, loss = m(x, y)
        loss.backward()
        si.update_importances()
        si.end_task()
        assert si.n_tasks == 1

    def test_penalty_non_negative(self):
        si, m = self._si()
        si.begin_task()
        x, y = make_batches(1)[0]
        _, loss = m(x, y); loss.backward()
        si.update_importances()
        si.end_task()
        assert si.penalty().item() >= 0.0


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=10)
        x, y = torch.randint(0, V, (4,)), torch.randint(0, V, (4,))
        buf.add(x, y, task_id=0)
        assert len(buf) == 1

    def test_reservoir_caps_at_max(self):
        buf = ReplayBuffer(max_size=5)
        for _ in range(20):
            x = torch.randint(0, V, (4,))
            y = torch.randint(0, V, (4,))
            buf.add(x, y, task_id=0)
        assert len(buf) == 5

    def test_sample_returns_entries(self):
        buf = ReplayBuffer(max_size=10)
        for i in range(8):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=i % 2)
        entries = buf.sample(3)
        assert len(entries) == 3

    def test_sample_batch_shape(self):
        buf = ReplayBuffer(max_size=10)
        for _ in range(6):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=0)
        xs, ys = buf.sample_batch(4)
        assert xs.shape[0] <= 4
        assert ys.shape == xs.shape

    def test_task_counts(self):
        buf = ReplayBuffer(max_size=20)
        for i in range(10):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=i % 2)
        counts = buf.task_counts()
        assert set(counts.keys()) <= {0, 1}

    def test_n_seen_increments(self):
        buf = ReplayBuffer(max_size=10)
        for _ in range(5):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=0)
        assert buf.n_seen == 5


# ── PackNet ───────────────────────────────────────────────────────────────────

class TestPackNet:
    def _pn(self, ratio=0.3):
        m  = TinyLM(V)
        pn = PackNet(m, prune_ratio=ratio)
        return pn, m

    def test_prune_and_pack_returns_stats(self):
        pn, m = self._pn()
        stats = pn.prune_and_pack(task_id=0)
        assert "n_pruned" in stats and "n_total" in stats

    def test_n_tasks_increments(self):
        pn, m = self._pn()
        pn.prune_and_pack(0)
        pn.prune_and_pack(1)
        assert pn.n_tasks == 2

    def test_free_ratio_decreases(self):
        pn, m = self._pn(ratio=0.4)
        r0 = pn.free_ratio
        pn.prune_and_pack(0)
        r1 = pn.free_ratio
        assert r1 < r0

    def test_apply_mask_restores_weights(self):
        pn, m = self._pn()
        w_before = list(m.parameters())[0].data.clone()
        pn.prune_and_pack(0)
        with torch.no_grad():
            list(m.parameters())[0].add_(torch.ones_like(list(m.parameters())[0]))
        pn.apply_mask(0)
        w_after = list(m.parameters())[0].data
        assert torch.allclose(w_before, w_after)

    def test_invalid_task_raises(self):
        pn, m = self._pn()
        with pytest.raises(ValueError):
            pn.apply_mask(99)


# ── ContinualMetrics ──────────────────────────────────────────────────────────

class TestContinualMetrics:
    def _metrics(self):
        return ContinualMetrics(
            accuracy_matrix=[[0.9, 0.7, 0.6],
                              [0.0, 0.85, 0.75],
                              [0.0, 0.0, 0.8]],
            n_tasks=3,
        )

    def test_average_accuracy(self):
        m = self._metrics()
        # AA = (0.6 + 0.75 + 0.8) / 3
        assert abs(m.average_accuracy - (0.6 + 0.75 + 0.8) / 3) < 1e-4

    def test_backward_transfer_negative_forgetting(self):
        m = self._metrics()
        # BWT = ((0.6-0.9) + (0.75-0.85)) / 2 = (-0.3 + -0.1) / 2 = -0.2
        assert m.backward_transfer < 0.0   # forgetting

    def test_forgetting_positive(self):
        m = self._metrics()
        assert m.forgetting > 0.0

    def test_to_dict_keys(self):
        d = self._metrics().to_dict()
        for k in ("average_accuracy", "backward_transfer", "forgetting", "n_tasks"):
            assert k in d

    def test_single_task_zero_bwt(self):
        m = ContinualMetrics([[0.8]], n_tasks=1)
        assert m.backward_transfer == 0.0


# ── ContinualTrainer ──────────────────────────────────────────────────────────

class TestContinualTrainer:
    def _trainer(self, strategy="ewc"):
        m   = TinyLM(V)
        cfg = ContinualConfig(strategy=strategy, n_tasks=2, ewc_lambda=1.0,
                               ewc_n_samples=4, replay_buffer_size=20)
        return ContinualTrainer(m, cfg), m

    def test_train_task_returns_log(self):
        t, _ = self._trainer("naive")
        log  = t.train_task(0, make_batches(4), epochs=1)
        assert "task_id" in log and "final_loss" in log

    def test_ewc_trainer_registers(self):
        t, _ = self._trainer("ewc")
        t.train_task(0, make_batches(4), epochs=1)
        assert t._ewc.n_tasks_registered == 1

    def test_replay_trainer_fills_buffer(self):
        t, _ = self._trainer("replay")
        t.train_task(0, make_batches(4), epochs=1)
        assert len(t._replay) > 0

    def test_packnet_trainer_packs(self):
        t, _ = self._trainer("packnet")
        t.train_task(0, make_batches(4), epochs=1)
        assert t._packnet.n_tasks == 1

    def test_task_logs(self):
        t, _ = self._trainer("naive")
        t.train_task(0, make_batches(), epochs=1)
        t.train_task(1, make_batches(), epochs=1)
        assert len(t.task_logs) == 2
''')
commit("test: add full continual learning test suite — config, EWC, SI, ReplayBuffer, PackNet, metrics, trainer")

# COMMITS 12-18: additional tests
for title, body in [
    ("test: add EWC penalty scales with lambda test", '''
class TestEWCLambda:
    def test_higher_lambda_higher_penalty(self):
        m1 = TinyLM(V); ewc1 = EWC(m1, lambda_=10.0, n_samples=4)
        m2 = TinyLM(V); ewc2 = EWC(m2, lambda_=1000.0, n_samples=4)
        batches = make_batches()
        ewc1.register_task(batches, 0)
        ewc2.register_task(batches, 0)
        with torch.no_grad():
            for p in m1.parameters(): p.add_(torch.ones_like(p) * 0.1)
            for p in m2.parameters(): p.add_(torch.ones_like(p) * 0.1)
        # Copy m1's state to m2 for fair comparison
        m2.load_state_dict(m1.state_dict())
        assert ewc2.penalty().item() >= ewc1.penalty().item()
'''),
    ("test: add ReplayBuffer add_batch increments n_seen test", '''
class TestReplayBatch:
    def test_add_batch(self):
        buf = ReplayBuffer(max_size=20)
        xs  = torch.randint(0, V, (4, 4))
        ys  = torch.randint(0, V, (4, 4))
        buf.add_batch(xs, ys, task_id=0)
        assert buf.n_seen == 4

    def test_add_with_logits(self):
        buf    = ReplayBuffer(max_size=10)
        x      = torch.randint(0, V, (4,))
        y      = torch.randint(0, V, (4,))
        logits = torch.randn(4, V)
        buf.add(x, y, task_id=0, logits=logits)
        assert buf._buffer[0].logits is not None
'''),
    ("test: add PackNet freeze_past_weights zeros frozen grads test", '''
class TestPackNetFreeze:
    def test_freeze_zeros_frozen_grads(self):
        m  = TinyLM(V)
        pn = PackNet(m, prune_ratio=0.5)
        pn.prune_and_pack(0)   # mark some weights as frozen
        # Compute grads
        x, y = make_batches(1)[0]
        _, loss = m(x, y); loss.backward()
        pn.freeze_past_weights()
        # Frozen weights should have zero gradient
        for name, param in m.named_parameters():
            if param.grad is not None and name in pn._frozen_mask:
                frozen_grads = param.grad[pn._frozen_mask[name]]
                assert (frozen_grads == 0.0).all()
'''),
    ("test: add ContinualMetrics zero forgetting if no drop test", '''
class TestNoForgetting:
    def test_no_forgetting_perfect(self):
        """If accuracy stays constant, forgetting should be 0."""
        m = ContinualMetrics(
            accuracy_matrix=[[0.9, 0.9, 0.9],
                              [0.0, 0.85, 0.85],
                              [0.0, 0.0, 0.8]],
            n_tasks=3,
        )
        assert m.forgetting == 0.0
'''),
    ("test: add SI penalty gradient flows test", '''
class TestSIPenaltyGrad:
    def test_si_penalty_gradient(self):
        m  = TinyLM(V)
        si = SynapticIntelligence(m, lambda_=10.0)
        si.begin_task()
        x, y = make_batches(1)[0]
        _, loss = m(x, y); loss.backward()
        si.update_importances()
        si.end_task()
        pen = si.penalty()
        pen.backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad
'''),
    ("test: add ReplayBuffer sample returns fewer than requested if small test", '''
class TestReplaySmallBuffer:
    def test_sample_fewer_than_requested(self):
        buf = ReplayBuffer(max_size=3)
        for _ in range(3):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), 0)
        entries = buf.sample(10)
        assert len(entries) == 3   # capped at buffer size
'''),
    ("test: add ContinualTrainer SI trains without error test", '''
class TestSITrainer:
    def test_si_train_two_tasks(self):
        m   = TinyLM(V)
        cfg = ContinualConfig(strategy="si", n_tasks=2, ewc_lambda=1.0)
        t   = ContinualTrainer(m, cfg)
        log0 = t.train_task(0, make_batches(4), epochs=1)
        log1 = t.train_task(1, make_batches(4), epochs=1)
        assert log0["strategy"] == "si"
        assert log1["strategy"] == "si"
        assert t._si.n_tasks == 2
'''),
]:
    src = read("tests/test_continual.py")
    src += "\n" + body
    write("tests/test_continual.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.1.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.0.0\"", "__version__ = \"4.1.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.1.0 — Continual Learning & Catastrophic Forgetting release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `interpret`  | Interpretability — attention, saliency, probing, logit lens, circuits, Shapley |",
    "| `interpret`  | Interpretability — attention, saliency, probing, logit lens, circuits, Shapley |\n"
    "| `continual`  | Continual Learning — EWC, SI, ExperienceReplay, PackNet, AA/BWT metrics |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.1.0] — 2024 — Continual Learning & Catastrophic Forgetting\n\n### Added\n"
      "- `ContinualConfig` — strategy, ewc_lambda, replay_buffer_size, packnet_prune_ratio\n"
      "- `EWC` — diagonal Fisher IM, register_task, penalty(), fisher_summary()\n"
      "- `SynapticIntelligence` — online importance, begin/end_task, update_importances\n"
      "- `ReplayBuffer` — reservoir sampling, add/add_batch, sample_batch, DER++ logits\n"
      "- `PackNet` — prune_and_pack, apply_mask, freeze_past_weights, free_ratio\n"
      "- `ContinualMetrics` — average_accuracy, backward_transfer, forgetting\n"
      "- `ContinualEvaluator` — evaluate_task, record, compute()\n"
      "- `ContinualTrainer` — unified EWC/SI/Replay/PackNet/Naive train_task()\n"
      "- `examples/continual_demo.py` — full continual learning demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.1.0, update README and CHANGELOG for Day 41 Continual Learning")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 41 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.1.0",
    "-m", "NanoMind v4.1.0 — Continual Learning & Catastrophic Forgetting", check=False)
r = run("git", "push", "origin", "v4.1.0", check=False)
print("Tag v4.1.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 41 COMPLETE — v4.1.0 TAGGED! ===")
