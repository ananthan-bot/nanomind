"""
day57_commits.py — 20 atomic commits for Day 57: Reasoning Models & Test-Time Compute (o1/R1/PRM/MCTS/GRPO).
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

print("\n=== DAY 57: Reasoning Models & Test-Time Compute (o1 / DeepSeek-R1 / PRM / MCTS / GRPO) — 20 commits, v5.7.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — Reasoning package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/__init__.py",
      '"""NanoMind Reasoning sub-package — Test-Time Compute, PRMs, MCTS, GRPO, and Self-Correction."""\n')
commit("feat: add nanomind/reasoning/ package skeleton for Test-Time Compute & Reasoning")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Configuration classes
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/config.py", '''\
"""
nanomind/reasoning/config.py — Configuration dataclasses for reasoning and search.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PRMConfig:
    """Configuration for Process Reward Models (PRMs)."""
    step_tag: str = "\\n\\n"
    hidden_dim: int = 128
    num_classes: int = 2  # 2: incorrect/correct, 3: negative/neutral/positive
    aggregation: str = "product"  # "product", "min", "last", "mean"
    threshold: float = 0.5
    loss_reduction: str = "mean"


@dataclass
class SearchConfig:
    """Configuration for test-time tree and beam search."""
    algorithm: str = "mcts"  # "mcts", "beam", "best_of_n", "tot"
    num_simulations: int = 50
    c_puct: float = 1.414
    max_depth: int = 10
    beam_width: int = 4
    num_rollouts: int = 8
    temperature: float = 0.7
    discount: float = 0.99
    early_stop_on_boxed: bool = True


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization (DeepSeek-R1 style)."""
    group_size: int = 4  # G: outputs sampled per prompt
    clip_eps: float = 0.2
    kl_weight: float = 0.04
    lr: float = 1e-5
    reward_baseline: str = "group_mean"  # "group_mean" or "zero"
    eps: float = 1e-6


@dataclass
class ReasoningConfig:
    """Master configuration for reasoning engine."""
    prm: PRMConfig = field(default_factory=PRMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    max_reasoning_tokens: int = 2048
    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"
    answer_tag: str = "\\\\boxed"
    enable_reflection: bool = True
    max_backtracks: int = 3
''')
commit("feat: add ReasoningConfig, PRMConfig, SearchConfig, and GRPOConfig in nanomind/reasoning/config.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Process Reward Model (PRM)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/prm.py", '''\
"""
nanomind/reasoning/prm.py — Process Reward Model (PRM800K style step-level verification).
"""
import math
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.reasoning.config import PRMConfig


class StepDelimiter:
    """Utility to split reasoning traces into atomic verification steps."""

    def __init__(self, delimiter: str = "\\n\\n"):
        self.delimiter = delimiter

    def split(self, text: str) -> List[str]:
        """Split reasoning text into steps, stripping empty segments."""
        raw = text.split(self.delimiter)
        steps = [s.strip() for s in raw if s.strip()]
        return steps if steps else [text.strip()]

    def join(self, steps: List[str]) -> str:
        """Recombine steps with the delimiter."""
        return self.delimiter.join(steps)


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model that predicts correctness probabilities at each reasoning step.
    Evaluates tokens at step boundaries and projects hidden states to step probabilities.
    """

    def __init__(self, d_model: int = 128, num_classes: int = 2, config: Optional[PRMConfig] = None):
        super().__init__()
        self.config = config or PRMConfig(hidden_dim=d_model, num_classes=num_classes)
        self.d_model = d_model
        self.num_classes = num_classes

        # Step value / classification head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, hidden_states: torch.Tensor, step_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        hidden_states: (B, T, D) or (T, D)
        step_indices: token indices corresponding to step delimiters.
        Returns:
            step_logits / step_probs: (num_steps,) or (B, num_steps)
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # (1, T, D)

        B, T, D = hidden_states.shape

        if step_indices is None or len(step_indices) == 0:
            # Score the terminal token if no explicit step indices given
            step_indices = [T - 1]

        # Extract embeddings at step positions
        step_idx_tensor = torch.tensor(step_indices, device=hidden_states.device, dtype=torch.long)
        selected_states = hidden_states[:, step_idx_tensor, :]  # (B, num_steps, D)

        logits = self.value_head(selected_states).squeeze(-1)  # (B, num_steps)
        if self.num_classes == 2:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def score_step_embeddings(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Score pre-pooled step embeddings (num_steps, D). Returns probabilities in [0, 1]."""
        logits = self.value_head(step_embeddings).squeeze(-1)
        return torch.sigmoid(logits)


def aggregate_step_scores(scores: List[float], method: str = "product") -> float:
    """
    Aggregate step-level probabilities into an overall trajectory score.
    Methods:
        - 'product': P(all steps correct) = prod_t p_t
        - 'min': Weakest link principle = min_t p_t
        - 'last': Final outcome score = p_T
        - 'mean': Arithmetic average = mean_t p_t
    """
    if not scores:
        return 0.0

    if method == "product":
        prod = 1.0
        for s in scores:
            prod *= max(1e-7, float(s))
        return float(prod)
    elif method == "min":
        return float(min(scores))
    elif method == "last":
        return float(scores[-1])
    elif method == "mean":
        return float(sum(scores) / len(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


class PRMLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for PRM training at step boundaries.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, step_probs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        step_probs: (B, num_steps) probabilities in (0, 1)
        targets: (B, num_steps) binary labels {0, 1}
        mask: (B, num_steps) boolean mask of valid steps
        """
        bce = F.binary_cross_entropy(step_probs.clamp(1e-7, 1 - 1e-7), targets.float(), reduction="none")
        if mask is not None:
            bce = bce * mask.float()
            if self.reduction == "mean":
                denom = mask.sum().clamp(min=1.0)
                return bce.sum() / denom
            elif self.reduction == "sum":
                return bce.sum()
        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        return bce
''')
commit("feat: implement Process Reward Model (PRM) with step delimiter scoring in nanomind/reasoning/prm.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Outcome Reward Model (ORM) & Calibration
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/orm.py", '''\
"""
nanomind/reasoning/orm.py — Outcome Reward Model (ORM), pairwise ranking loss, and calibration metrics.
"""
import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OutcomeRewardModel(nn.Module):
    """
    Outcome Reward Model (ORM) that scores an entire solution at the final token.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, final_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        final_hidden_state: (B, D) terminal token representation.
        Returns: scalar reward (B,)
        """
        return self.score_head(final_hidden_state).squeeze(-1)


class MarginRankingLoss(nn.Module):
    """
    Pairwise ranking loss for trajectory preference optimization (Bradley-Terry formulation).
    Loss = -log sigmoid(reward_chosen - reward_rejected - margin)
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        diff = chosen_rewards - rejected_rewards - self.margin
        return -F.logsigmoid(diff).mean()


def compute_brier_score(preds: List[float], targets: List[int]) -> float:
    """
    Compute Brier score: mean squared error of probabilistic predictions.
    Lower is better (0 = perfect calibration and discrimination).
    """
    if not preds or len(preds) != len(targets):
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(preds, targets)) / len(preds)


def compute_calibration_error(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures difference between predicted confidence and empirical accuracy across bins.
    """
    if not probs or len(probs) != len(labels):
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        b_idx = min(n_bins - 1, int(p * n_bins))
        bins[b_idx].append((p, y))

    ece = 0.0
    total = len(probs)
    for b in bins:
        if not b:
            continue
        bin_size = len(b)
        bin_conf = sum(item[0] for item in b) / bin_size
        bin_acc = sum(item[1] for item in b) / bin_size
        ece += (bin_size / total) * abs(bin_conf - bin_acc)

    return float(ece)


def compare_prm_vs_orm(step_scores: List[float], orm_score: float) -> Dict[str, float]:
    """
    Compare PRM step metrics against ORM terminal outcome score.
    Detects whether failure was sudden (weakest link) or uniform degradation.
    """
    from nanomind.reasoning.prm import aggregate_step_scores
    prm_prod = aggregate_step_scores(step_scores, "product")
    prm_min = aggregate_step_scores(step_scores, "min")
    prm_mean = aggregate_step_scores(step_scores, "mean")

    return {
        "prm_product": prm_prod,
        "prm_min": prm_min,
        "prm_mean": prm_mean,
        "orm_score": float(orm_score),
        "discrepancy": abs(prm_prod - float(orm_score))
    }
''')
commit("feat: implement Outcome Reward Model (ORM) and verifier calibration in nanomind/reasoning/orm.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Monte Carlo Tree Search (MCTS) for Reasoning
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/tree_search.py", '''\
"""
nanomind/reasoning/tree_search.py — Monte Carlo Tree Search (MCTS) & Tree-of-Thought (ToT) for reasoning.
"""
import math
from typing import List, Dict, Optional, Tuple, Callable, Any
from nanomind.reasoning.config import SearchConfig


class ReasoningNode:
    """
    A search tree node representing a reasoning state.
    """

    def __init__(
        self,
        state_text: str,
        step_text: str = "",
        parent: Optional["ReasoningNode"] = None,
        prior_p: float = 1.0,
        is_terminal: bool = False,
    ):
        self.state_text = state_text  # Cumulative reasoning history
        self.step_text = step_text    # Step text transition from parent
        self.parent = parent
        self.prior_p = prior_p
        self.is_terminal = is_terminal

        self.children: Dict[str, "ReasoningNode"] = {}
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.step_reward: float = 0.0

    @property
    def q_value(self) -> float:
        """Mean estimated value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_leaf(self) -> bool:
        """True if the node has not yet been expanded."""
        return len(self.children) == 0

    def add_child(self, step_text: str, prior_p: float = 1.0, is_terminal: bool = False) -> "ReasoningNode":
        """Attach a new child reasoning state."""
        new_state = (self.state_text + "\\n\\n" + step_text).strip() if self.state_text else step_text
        child = ReasoningNode(
            state_text=new_state,
            step_text=step_text,
            parent=self,
            prior_p=prior_p,
            is_terminal=is_terminal,
        )
        self.children[step_text] = child
        return child


def puct_score(child: ReasoningNode, parent_visits: int, c_puct: float = 1.414) -> float:
    """
    Polynomial Upper Confidence Trees (PUCT) score:
    PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    """
    exploration = c_puct * child.prior_p * (math.sqrt(parent_visits) / (1 + child.visits))
    return child.q_value + exploration


class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search for planning optimal step-by-step reasoning trajectories.
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()

    def select(self, node: ReasoningNode) -> ReasoningNode:
        """Traverse down tree using PUCT until reaching an unexpanded leaf or terminal node."""
        curr = node
        while not curr.is_leaf() and not curr.is_terminal:
            best_score = -float("inf")
            best_child = None
            for child in curr.children.values():
                score = puct_score(child, curr.visits, self.config.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            curr = best_child
        return curr

    def backpropagate(self, node: ReasoningNode, value: float):
        """Propagate simulated value back up to root with optional discounting."""
        curr = node
        v = value
        while curr is not None:
            curr.visits += 1
            curr.value_sum += v
            v *= self.config.discount
            curr = curr.parent

    def search(
        self,
        root_prompt: str,
        step_generator_fn: Callable[[str], List[Tuple[str, float, bool]]],
        verifier_fn: Callable[[str, str], float],
        num_simulations: Optional[int] = None,
    ) -> Tuple[List[str], ReasoningNode]:
        """
        Execute MCTS.
        step_generator_fn(state) -> [(step_candidate, prior_p, is_terminal), ...]
        verifier_fn(state, step) -> step_value in [0, 1]
        Returns:
            best_trajectory: List of reasoning step strings
            root: Root of the search tree
        """
        sims = num_simulations or self.config.num_simulations
        root = ReasoningNode(state_text=root_prompt)

        # Initial expansion of root
        root_candidates = step_generator_fn(root.state_text)
        for step_cand, prior, is_term in root_candidates:
            root.add_child(step_cand, prior_p=prior, is_terminal=is_term)

        for _ in range(sims):
            # 1. Selection
            leaf = self.select(root)

            # 2. Expansion
            if not leaf.is_terminal and leaf.visits > 0 and leaf.is_leaf():
                cands = step_generator_fn(leaf.state_text)
                for step_cand, prior, is_term in cands:
                    leaf.add_child(step_cand, prior_p=prior, is_terminal=is_term)
                if not leaf.is_leaf():
                    leaf = next(iter(leaf.children.values()))

            # 3. Evaluation (Simulation / PRM Value)
            value = verifier_fn(leaf.parent.state_text if leaf.parent else "", leaf.step_text)
            leaf.step_reward = value

            # 4. Backpropagation
            self.backpropagate(leaf, value)

        # Extract best trajectory by following highest visit count (most robust)
        trajectory = []
        curr = root
        while not curr.is_leaf():
            best_child = max(curr.children.values(), key=lambda c: c.visits)
            trajectory.append(best_child.step_text)
            curr = best_child
            if curr.is_terminal:
                break

        return trajectory, root
''')
commit("feat: implement Monte Carlo Tree Search (MCTS) for reasoning trajectories in nanomind/reasoning/tree_search.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Tree-of-Thought (ToT) Multi-Path Reasoning
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/tot.py", '''\
"""
nanomind/reasoning/tot.py — Tree of Thoughts (ToT) breadth-first and depth-first exploration.
"""
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass


@dataclass
class ThoughtStep:
    thought: str
    evaluation: str  # "sure", "likely", "impossible"
    score: float


class TreeOfThoughts:
    """
    Tree-of-Thoughts solver using BFS exploration with threshold pruning.
    Evaluates intermediate thoughts (sure/likely/impossible) before proceeding.
    """

    def __init__(self, max_depth: int = 5, beam_width: int = 3, min_score_threshold: float = 0.4):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_score_threshold = min_score_threshold

    def solve(
        self,
        problem: str,
        thought_generator_fn: Callable[[str, int], List[str]],
        thought_evaluator_fn: Callable[[str, str], Tuple[str, float]],
        is_solution_fn: Callable[[str], bool],
    ) -> List[Dict[str, Any]]:
        """
        BFS exploration of reasoning thoughts.
        Returns all valid reasoning trajectories that yield a solution.
        """
        # Active frontier: list of (current_reasoning_string, cumulative_score, depth)
        frontier: List[Tuple[str, float, int]] = [(problem, 1.0, 0)]
        successful_trajectories: List[Dict[str, Any]] = []

        for depth in range(self.max_depth):
            if not frontier:
                break

            candidates: List[Tuple[str, float, int]] = []
            for path_text, cum_score, _ in frontier:
                # If this path already reached solution
                if is_solution_fn(path_text):
                    successful_trajectories.append({
                        "path": path_text,
                        "score": cum_score,
                        "depth": depth,
                    })
                    continue

                # Generate k thoughts for this branch
                next_thoughts = thought_generator_fn(path_text, self.beam_width)
                for thought in next_thoughts:
                    verdict, score = thought_evaluator_fn(path_text, thought)
                    if verdict == "impossible" or score < self.min_score_threshold:
                        continue  # Prune branch

                    new_path = path_text + "\\n\\n" + thought
                    candidates.append((new_path, cum_score * score, depth + 1))

            # Keep top-K candidates (beam pruning)
            candidates.sort(key=lambda x: x[1], reverse=True)
            frontier = candidates[:self.beam_width]

        # Check any remaining solutions in frontier
        for path_text, cum_score, d in frontier:
            if is_solution_fn(path_text):
                successful_trajectories.append({
                    "path": path_text,
                    "score": cum_score,
                    "depth": d,
                })

        return successful_trajectories
''')
commit("feat: implement Tree-of-Thought (ToT) multi-path reasoning in nanomind/reasoning/tot.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — StepBeamSearch & Best-of-N Rejection Sampling
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/beam_search.py", '''\
"""
nanomind/reasoning/beam_search.py — StepBeamSearch and Best-of-N verifier reranking.
"""
from typing import List, Tuple, Dict, Any, Callable, Optional
from collections import Counter
import re


class StepBeamSearch:
    """
    Step-level Beam Search guided by Process Reward Model scores.
    Prunes low-probability reasoning branches at each step delimiter.
    """

    def __init__(self, beam_width: int = 4, max_steps: int = 8, step_tag: str = "\\n\\n"):
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.step_tag = step_tag

    def search(
        self,
        prompt: str,
        step_proposer_fn: Callable[[str, int], List[str]],
        prm_step_scorer_fn: Callable[[str, str], float],
        is_terminal_fn: Callable[[str], bool],
    ) -> List[Dict[str, Any]]:
        """
        Runs step-level beam search.
        Returns ranked list of beam trajectories with cumulative PRM scores.
        """
        # Beam tuple: (full_text, list_of_steps, step_scores, cumulative_score)
        beams = [(prompt, [], [], 1.0)]

        completed_beams = []

        for step_idx in range(self.max_steps):
            if not beams:
                break

            candidates = []
            for full_text, steps, scores, cum_score in beams:
                if is_terminal_fn(full_text):
                    completed_beams.append({
                        "text": full_text,
                        "steps": steps,
                        "scores": scores,
                        "cumulative_score": cum_score,
                    })
                    continue

                # Generate candidates for next step
                proposals = step_proposer_fn(full_text, self.beam_width)
                for prop in proposals:
                    prop_clean = prop.strip()
                    score = prm_step_scorer_fn(full_text, prop_clean)
                    new_text = full_text + self.step_tag + prop_clean
                    new_steps = steps + [prop_clean]
                    new_scores = scores + [score]
                    new_cum = cum_score * score
                    candidates.append((new_text, new_steps, new_scores, new_cum))

            if not candidates:
                break

            # Sort by cumulative score and select top beam_width
            candidates.sort(key=lambda c: c[3], reverse=True)
            beams = candidates[:self.beam_width]

        for full_text, steps, scores, cum_score in beams:
            completed_beams.append({
                "text": full_text,
                "steps": steps,
                "scores": scores,
                "cumulative_score": cum_score,
            })

        completed_beams.sort(key=lambda b: b["cumulative_score"], reverse=True)
        return completed_beams


class BestOfNVerifier:
    """
    Best-of-N Rejection Sampling and Majority Voting using PRM/ORM rewards.
    """

    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples

    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        """Extract content inside \\boxed{...} or \\boxed ..."""
        match = re.search(r"\\\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip()
        match_simple = re.search(r"\\\\boxed\s*([0-9a-zA-Z\.\-]+)", text)
        if match_simple:
            return match_simple.group(1).strip()
        return None

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        scoring_key: str = "reward"
    ) -> Dict[str, Any]:
        """
        Select highest scoring candidate.
        """
        if not candidates:
            raise ValueError("No candidates provided")
        best = max(candidates, key=lambda c: c.get(scoring_key, 0.0))
        return best

    def majority_vote(
        self,
        candidates: List[Dict[str, Any]],
        weighted_by_score: bool = True,
        scoring_key: str = "reward"
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Extract answers and tally votes (weighted or unweighted).
        Returns: (winning_answer, confidence_fraction, vote_distribution)
        """
        answer_votes: Dict[str, float] = {}
        total_weight = 0.0

        for cand in candidates:
            text = cand.get("text", "")
            ans = self.extract_boxed_answer(text)
            if ans is None:
                continue

            weight = cand.get(scoring_key, 1.0) if weighted_by_score else 1.0
            answer_votes[ans] = answer_votes.get(ans, 0.0) + weight
            total_weight += weight

        if not answer_votes or total_weight == 0.0:
            return None, 0.0, {}

        winner = max(answer_votes.items(), key=lambda item: item[1])
        winning_ans = winner[0]
        confidence = winner[1] / total_weight

        norm_distribution = {k: v / total_weight for k, v in answer_votes.items()}
        return winning_ans, confidence, norm_distribution
''')
commit("feat: implement StepBeamSearch and PRM Best-of-N reranking in nanomind/reasoning/beam_search.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Self-Correction & Backtracking
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/self_correction.py", '''\
"""
nanomind/reasoning/self_correction.py — Self-Reflective reasoning and backtracking detection.
"""
import re
from typing import List, Dict, Any, Tuple, Optional


class SelfReflectiveReasoner:
    """
    Reasoning engine that prompts the model to reflect, check for errors, and backtrack.
    Detects reasoning shifts like 'Wait, let me rethink', 'Hold on, that is incorrect'.
    """

    BACKTRACK_TRIGGERS = [
        "wait,",
        "wait!",
        "hold on,",
        "actually, let me rethink",
        "that's incorrect",
        "that is incorrect",
        "re-evaluating",
        "let me double check",
        "let's check that",
        "on second thought",
    ]

    def __init__(
        self,
        think_start_tag: str = "<think>",
        think_end_tag: str = "</think>",
        answer_tag: str = "\\\\boxed",
    ):
        self.think_start_tag = think_start_tag
        self.think_end_tag = think_end_tag
        self.answer_tag = answer_tag

    def format_prompt(self, problem: str) -> str:
        """Format problem with thinking instructions."""
        return (
            f"Solve the following problem step by step.\\n"
            f"Enclose your reasoning inside {self.think_start_tag} and {self.think_end_tag}.\\n"
            f"If you notice an error in your intermediate steps, explicitly correct yourself and rethink.\\n"
            f"Provide the final answer inside {self.answer_tag}{{...}}.\\n\\n"
            f"Problem: {problem}\\n"
        )

    def parse_reasoning_trace(self, output_text: str) -> Dict[str, Any]:
        """
        Extracts thought section, solution section, backtrack occurrences, and boxed answer.
        """
        # Extract <think> ... </think>
        think_pattern = re.escape(self.think_start_tag) + r"(.*?)" + re.escape(self.think_end_tag)
        match = re.search(think_pattern, output_text, re.DOTALL)
        if match:
            thoughts = match.group(1).strip()
            solution = output_text[match.end():].strip()
        else:
            thoughts = output_text
            solution = output_text

        # Detect backtracks
        backtrack_points = []
        lower_thoughts = thoughts.lower()
        for trig in self.BACKTRACK_TRIGGERS:
            pos = 0
            while True:
                idx = lower_thoughts.find(trig, pos)
                if idx == -1:
                    break
                snippet = thoughts[max(0, idx - 20): min(len(thoughts), idx + len(trig) + 40)]
                backtrack_points.append({"trigger": trig, "position": idx, "context": snippet.strip()})
                pos = idx + len(trig)

        # Extract answer
        boxed_match = re.search(r"\\\\boxed\{([^}]+)\}", output_text)
        final_answer = boxed_match.group(1).strip() if boxed_match else None

        return {
            "has_think_tags": match is not None,
            "thoughts": thoughts,
            "solution": solution,
            "backtrack_count": len(backtrack_points),
            "backtrack_events": backtrack_points,
            "final_answer": final_answer,
        }

    def compute_reflection_metrics(self, traces: List[str]) -> Dict[str, float]:
        """
        Aggregate reflection statistics across multiple sampled reasoning traces.
        """
        if not traces:
            return {"backtrack_rate": 0.0, "avg_backtracks": 0.0, "think_tag_compliance": 0.0}

        total = len(traces)
        traces_with_backtrack = 0
        total_backtracks = 0
        compliant_tags = 0

        for t in traces:
            parsed = self.parse_reasoning_trace(t)
            if parsed["backtrack_count"] > 0:
                traces_with_backtrack += 1
            total_backtracks += parsed["backtrack_count"]
            if parsed["has_think_tags"]:
                compliant_tags += 1

        return {
            "backtrack_rate": traces_with_backtrack / total,
            "avg_backtracks": total_backtracks / total,
            "think_tag_compliance": compliant_tags / total,
        }
''')
commit("feat: implement SelfReflectiveReasoner with backtracking in nanomind/reasoning/self_correction.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — Group Relative Policy Optimization (GRPO)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/grpo.py", '''\
"""
nanomind/reasoning/grpo.py — Group Relative Policy Optimization (DeepSeek-R1 core RL engine).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from nanomind.reasoning.config import GRPOConfig


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes group normalized relative advantage:
    A_i = (r_i - mean(r)) / (std(r) + eps)
    rewards: (B, G) or (G,) where G is group size per prompt.
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)  # (1, G)

    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    advantages = (rewards - mean) / (std + eps)
    return advantages.squeeze(0) if advantages.shape[0] == 1 else advantages


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_weight: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Group Relative Policy Optimization objective (DeepSeek-R1):
    L_GRPO = - 1/G sum_{i=1}^G [ min(r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i) - beta * D_KL(pi_theta || pi_ref) ]

    log_probs: (G, T) log probs under current policy pi_theta
    old_log_probs: (G, T) log probs under rollout policy pi_old
    ref_log_probs: (G, T) log probs under frozen reference model pi_ref
    advantages: (G,) scalar relative group advantages
    mask: (G, T) binary token completion mask (1 for generated tokens, 0 for prompt/pad)
    """
    G, T = log_probs.shape

    # Expand advantages to token shape: (G, 1) -> (G, T)
    adv = advantages.unsqueeze(-1)

    # Importance sampling ratio: pi_theta / pi_old
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2)

    # Unbiased Schulman estimator of KL divergence: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
    # or forward KL: log_pi - log_ref
    kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0

    total_per_token = policy_loss + kl_weight * kl

    if mask is not None:
        valid_tokens = mask.sum().clamp(min=1.0)
        total_loss = (total_per_token * mask).sum() / valid_tokens
        p_loss_scalar = (policy_loss * mask).sum() / valid_tokens
        kl_scalar = (kl * mask).sum() / valid_tokens
    else:
        total_loss = total_per_token.mean()
        p_loss_scalar = policy_loss.mean()
        kl_scalar = kl.mean()

    metrics = {
        "loss": float(total_loss.item()),
        "policy_loss": float(p_loss_scalar.item()),
        "kl_divergence": float(kl_scalar.item()),
        "mean_ratio": float(ratio.mean().item()),
    }

    return total_loss, metrics


class GRPOTrainer:
    """
    Self-contained GRPO optimization step manager.
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()

    def step(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Runs one step of GRPO advantage normalization and loss computation.
        """
        advantages = compute_group_advantages(rewards, eps=self.config.eps)
        loss, metrics = grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            mask=mask,
            clip_eps=self.config.clip_eps,
            kl_weight=self.config.kl_weight,
        )
        return loss, metrics
''')
commit("feat: implement Group Relative Policy Optimization (GRPO) in nanomind/reasoning/grpo.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — Self-Taught Reasoner (STaR)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/star.py", '''\
"""
nanomind/reasoning/star.py — Self-Taught Reasoner (STaR) rationale bootstrapping.
"""
from typing import List, Dict, Any, Tuple, Optional, Callable


class STaR:
    """
    Self-Taught Reasoner (STaR: Bootstrapping Reasoning With Reasoning).
    Iteratively generates rationales; saves correct ones; uses hints to rationalize failures.
    """

    def __init__(self, max_rationalizations: int = 1):
        self.max_rationalizations = max_rationalizations
        self.training_buffer: List[Dict[str, str]] = []

    def bootstrap_iteration(
        self,
        dataset: List[Dict[str, str]],
        generate_fn: Callable[[str], str],
        evaluate_fn: Callable[[str, str], bool],
        rationalize_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Dict[str, Any]:
        """
        One STaR bootstrap cycle over a question-answer dataset.
        dataset: list of {"question": ..., "answer": ...}
        generate_fn: (question) -> model_reasoning_and_output
        evaluate_fn: (generated_output, true_answer) -> bool
        rationalize_fn: (question, true_answer) -> hint_assisted_reasoning
        """
        initial_correct = 0
        rationalized_correct = 0
        new_examples = []

        for item in dataset:
            q = item["question"]
            gt = item["answer"]

            # 1. Direct generation attempt
            pred = generate_fn(q)
            is_correct = evaluate_fn(pred, gt)

            if is_correct:
                initial_correct += 1
                example = {"question": q, "rationale": pred, "answer": gt, "source": "direct"}
                new_examples.append(example)
                self.training_buffer.append(example)
            elif rationalize_fn is not None:
                # 2. Rationalization with hint/ground-truth
                hinted_pred = rationalize_fn(q, gt)
                if evaluate_fn(hinted_pred, gt):
                    rationalized_correct += 1
                    example = {"question": q, "rationale": hinted_pred, "answer": gt, "source": "rationalized"}
                    new_examples.append(example)
                    self.training_buffer.append(example)

        total = len(dataset)
        return {
            "total_questions": total,
            "direct_accuracy": initial_correct / max(1, total),
            "rationalized_count": rationalized_correct,
            "total_new_training_examples": len(new_examples),
            "buffer_size": len(self.training_buffer),
        }
''')
commit("feat: implement Self-Taught Reasoner (STaR) rationale bootstrap in nanomind/reasoning/star.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — Test-Time Compute Scaling Laws
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/scaling.py", '''\
"""
nanomind/reasoning/scaling.py — Test-Time Compute Scaling laws, FLOP estimator, and Pass@k analysis.
"""
import math
from typing import List, Dict, Any, Tuple, Optional


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of Pass@k (Chen et al. 2021).
    Probability that at least 1 out of k randomly selected samples is correct,
    given n total generated samples with c correct ones.
    Formula: 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0

    res = 1.0
    for i in range(k):
        res *= (n - c - i) / (n - i)
    return float(1.0 - res)


def estimate_test_time_flops(
    num_tokens: int,
    num_rollouts: int,
    model_params: int = 125_000_000,
) -> float:
    """
    Estimate test-time compute in Floating Point Operations (FLOPs).
    Standard Transformer forward pass ~ 2 * P FLOPs per token.
    Total FLOPs = 2 * P * num_tokens * num_rollouts.
    """
    return float(2.0 * model_params * num_tokens * num_rollouts)


class TestTimeScalingSimulator:
    """
    Simulates test-time scaling curves: Compute Budget (FLOPs) vs Task Accuracy.
    Compares Greedy (k=1), Best-of-N, Majority Voting, and Tree Search.
    """

    def __init__(self, p_step_correct: float = 0.85, num_steps: int = 4):
        self.p_step_correct = p_step_correct
        self.num_steps = num_steps

    def simulate_pass_at_k(self, k_values: List[int], n_total: int = 64) -> Dict[int, float]:
        """
        Simulate pass@k for an end-to-end task with compounding step failure.
        p_full = (p_step)^num_steps
        """
        p_full = self.p_step_correct ** self.num_steps
        c_expected = int(round(p_full * n_total))
        c_expected = max(1, min(n_total, c_expected))

        return {k: pass_at_k(n_total, c_expected, k) for k in k_values if k <= n_total}

    def compute_scaling_tradeoff(self, budgets: List[int]) -> List[Dict[str, Any]]:
        """
        Compute estimated pass rates and FLOPs for varying rollout budgets.
        budgets: list of rollout counts [1, 2, 4, 8, 16, 32]
        """
        results = []
        p_full = self.p_step_correct ** self.num_steps

        for b in budgets:
            # Probability of at least one correct path in b rollouts = 1 - (1 - p_full)^b
            acc = 1.0 - ((1.0 - p_full) ** b)
            flops = estimate_test_time_flops(num_tokens=128 * self.num_steps, num_rollouts=b)
            results.append({
                "budget_rollouts": b,
                "accuracy": float(acc),
                "estimated_gflops": flops / 1e9,
            })
        return results
''')
commit("feat: implement test-time compute scaling laws and pass@k analysis in nanomind/reasoning/scaling.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — Reasoning API exposure
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/reasoning/__init__.py", '''\
"""
nanomind.reasoning — Reasoning Models & Test-Time Compute (o1 / DeepSeek-R1 / PRM / MCTS / GRPO).
"""
from nanomind.reasoning.config import (
    ReasoningConfig,
    PRMConfig,
    SearchConfig,
    GRPOConfig,
)
from nanomind.reasoning.prm import (
    StepDelimiter,
    ProcessRewardModel,
    aggregate_step_scores,
    PRMLoss,
)
from nanomind.reasoning.orm import (
    OutcomeRewardModel,
    MarginRankingLoss,
    compute_brier_score,
    compute_calibration_error,
    compare_prm_vs_orm,
)
from nanomind.reasoning.tree_search import (
    ReasoningNode,
    puct_score,
    MonteCarloTreeSearch,
)
from nanomind.reasoning.tot import (
    ThoughtStep,
    TreeOfThoughts,
)
from nanomind.reasoning.beam_search import (
    StepBeamSearch,
    BestOfNVerifier,
)
from nanomind.reasoning.self_correction import (
    SelfReflectiveReasoner,
)
from nanomind.reasoning.grpo import (
    compute_group_advantages,
    grpo_loss,
    GRPOTrainer,
)
from nanomind.reasoning.star import (
    STaR,
)
from nanomind.reasoning.scaling import (
    pass_at_k,
    estimate_test_time_flops,
    TestTimeScalingSimulator,
)

__all__ = [
    "ReasoningConfig",
    "PRMConfig",
    "SearchConfig",
    "GRPOConfig",
    "StepDelimiter",
    "ProcessRewardModel",
    "aggregate_step_scores",
    "PRMLoss",
    "OutcomeRewardModel",
    "MarginRankingLoss",
    "compute_brier_score",
    "compute_calibration_error",
    "compare_prm_vs_orm",
    "ReasoningNode",
    "puct_score",
    "MonteCarloTreeSearch",
    "ThoughtStep",
    "TreeOfThoughts",
    "StepBeamSearch",
    "BestOfNVerifier",
    "SelfReflectiveReasoner",
    "compute_group_advantages",
    "grpo_loss",
    "GRPOTrainer",
    "STaR",
    "pass_at_k",
    "estimate_test_time_flops",
    "TestTimeScalingSimulator",
]
''')

# Also update nanomind/__init__.py to import reasoning
init_py = read("nanomind/__init__.py")
if "from nanomind.reasoning import" not in init_py:
    init_py = init_py.replace(
        "from nanomind.distill import DistillConfig, DistillTrainer, distillation_loss\n",
        "from nanomind.distill import DistillConfig, DistillTrainer, distillation_loss\n"
        "from nanomind.reasoning import ReasoningConfig, ProcessRewardModel, MonteCarloTreeSearch, GRPOTrainer\n"
    )
    init_py = init_py.replace(
        '    "distillation_loss",\n',
        '    "distillation_loss",\n'
        '    "ReasoningConfig",\n'
        '    "ProcessRewardModel",\n'
        '    "MonteCarloTreeSearch",\n'
        '    "GRPOTrainer",\n'
    )
    write("nanomind/__init__.py", init_py)

commit("feat: expose reasoning API in nanomind/reasoning/__init__.py and top-level package")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — Examples Demo
# ══════════════════════════════════════════════════════════════════════════════
write("examples/reasoning_demo.py", '''\
"""
examples/reasoning_demo.py — End-to-end Demonstration of Reasoning & Test-Time Compute.
Demonstrates:
  1. Process Reward Model (PRM) step scoring
  2. Monte Carlo Tree Search (MCTS) exploration of reasoning paths
  3. Group Relative Policy Optimization (GRPO) advantage calculation
  4. Test-Time Compute Scaling Law simulation (Pass@k vs FLOPs)
"""
import torch
from nanomind.reasoning import (
    ReasoningConfig,
    ProcessRewardModel,
    aggregate_step_scores,
    MonteCarloTreeSearch,
    SearchConfig,
    StepBeamSearch,
    BestOfNVerifier,
    SelfReflectiveReasoner,
    compute_group_advantages,
    grpo_loss,
    pass_at_k,
    TestTimeScalingSimulator,
)

def run_demo():
    print("=" * 70)
    print("  NanoMind Day 57: Reasoning Models & Test-Time Compute (o1 / R1)")
    print("=" * 70)

    # 1. Process Reward Model (PRM)
    print("\\n[1] Process Reward Model (PRM) Step Verification:")
    prm = ProcessRewardModel(d_model=64)
    # Simulated 3 steps: step 1 correct (0.95), step 2 correct (0.90), step 3 mistake (0.20)
    sim_step_scores = [0.95, 0.90, 0.20]
    p_prod = aggregate_step_scores(sim_step_scores, "product")
    p_min = aggregate_step_scores(sim_step_scores, "min")
    p_last = aggregate_step_scores(sim_step_scores, "last")
    print(f"  Step probabilities: {sim_step_scores}")
    print(f"  PRM product aggregation: {p_prod:.4f} (detects failure along chain)")
    print(f"  PRM min aggregation (weakest link): {p_min:.4f}")
    print(f"  PRM last aggregation: {p_last:.4f}")

    # 2. Monte Carlo Tree Search (MCTS)
    print("\\n[2] Monte Carlo Tree Search (MCTS) over Math Reasoning Steps:")
    cfg = SearchConfig(num_simulations=20, c_puct=1.414)
    mcts = MonteCarloTreeSearch(cfg)

    # Simple step candidate generator
    def step_gen(state):
        if "Step 2" in state:
            return [("Step 3: Final Answer is \\\\boxed{42}", 0.9, True)]
        elif "Step 1" in state:
            return [
                ("Step 2 (branch A): Multiply by 2", 0.7, False),
                ("Step 2 (branch B): Divide by 2", 0.3, False),
            ]
        else:
            return [("Step 1: Simplify expression", 1.0, False)]

    def verifier(prefix, step):
        if "Multiply by 2" in step or "boxed{42}" in step or "Step 1" in step:
            return 0.95
        return 0.15

    trajectory, root = mcts.search("Problem: Solve 2x + 10 = 94", step_gen, verifier)
    print("  MCTS Discovered Optimal Trajectory:")
    for i, s in enumerate(trajectory, 1):
        print(f"    {i}. {s}")

    # 3. Group Relative Policy Optimization (GRPO)
    print("\\n[3] Group Relative Policy Optimization (GRPO) Advantage Normalization:")
    rewards = torch.tensor([[1.0, 0.0, 0.5, 1.0]])  # Group of G=4 outputs
    adv = compute_group_advantages(rewards)
    print(f"  Group Rewards: {rewards.tolist()[0]}")
    print(f"  Group Relative Advantages (mean=0, std=1): {[round(x, 3) for x in adv.tolist()[0]]}")

    # 4. Test-Time Scaling Laws (Pass@k)
    print("\\n[4] Test-Time Compute Scaling Laws:")
    sim = TestTimeScalingSimulator(p_step_correct=0.88, num_steps=3)
    k_vals = [1, 2, 4, 8, 16]
    tradeoffs = sim.compute_scaling_tradeoff(k_vals)
    print("  Rollouts | Accuracy | Estimated GFLOPs")
    print("  ---------+----------+-----------------")
    for t in tradeoffs:
        print(f"     {t['budget_rollouts']:2d}    |  {t['accuracy']*100:5.1f}%  |  {t['estimated_gflops']:6.2f} GFLOPs")

    print("\\n✓ All Reasoning demos completed successfully!")

if __name__ == "__main__":
    run_demo()
''')
commit("feat: add examples/reasoning_demo.py demonstrating MCTS, PRM, GRPO, and test-time scaling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — Unit Tests Part 1: PRM
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_reasoning.py", '''\
"""
tests/test_reasoning.py — Comprehensive unit tests for Reasoning & Test-Time Compute.
"""
import math
import pytest
import torch
from nanomind.reasoning import (
    PRMConfig,
    StepDelimiter,
    ProcessRewardModel,
    aggregate_step_scores,
    PRMLoss,
    OutcomeRewardModel,
    MarginRankingLoss,
    compute_brier_score,
    compute_calibration_error,
    compare_prm_vs_orm,
    ReasoningNode,
    puct_score,
    MonteCarloTreeSearch,
    TreeOfThoughts,
    StepBeamSearch,
    BestOfNVerifier,
    SelfReflectiveReasoner,
    compute_group_advantages,
    grpo_loss,
    GRPOTrainer,
    STaR,
    pass_at_k,
    estimate_test_time_flops,
    TestTimeScalingSimulator,
)


class TestPRM:
    def test_step_delimiter(self):
        delim = StepDelimiter(delimiter="\\n\\n")
        text = "Step 1: start\\n\\nStep 2: middle\\n\\nStep 3: end"
        steps = delim.split(text)
        assert len(steps) == 3
        assert steps[0] == "Step 1: start"
        rejoined = delim.join(steps)
        assert rejoined == text

    def test_prm_forward(self):
        prm = ProcessRewardModel(d_model=32, num_classes=2)
        h = torch.randn(2, 10, 32)
        step_indices = [3, 7, 9]
        probs = prm(h, step_indices=step_indices)
        assert probs.shape == (2, 3)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_aggregate_scores(self):
        scores = [0.9, 0.8, 0.5]
        prod = aggregate_step_scores(scores, "product")
        assert math.isclose(prod, 0.9 * 0.8 * 0.5, rel_tol=1e-5)

        min_val = aggregate_step_scores(scores, "min")
        assert min_val == 0.5

        mean_val = aggregate_step_scores(scores, "mean")
        assert math.isclose(mean_val, (0.9 + 0.8 + 0.5) / 3, rel_tol=1e-5)

    def test_prm_loss(self):
        loss_fn = PRMLoss()
        probs = torch.tensor([[0.8, 0.2]])
        targets = torch.tensor([[1, 0]])
        loss = loss_fn(probs, targets)
        assert loss.item() > 0.0
''')
commit("test: add tests for PRM step parsing, scoring, and score aggregation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — Unit Tests Part 2: ORM & Calibration
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_reasoning.py")
test_src += '''\


class TestORM:
    def test_orm_forward(self):
        orm = OutcomeRewardModel(d_model=32)
        h = torch.randn(4, 32)
        scores = orm(h)
        assert scores.shape == (4,)

    def test_margin_ranking_loss(self):
        loss_fn = MarginRankingLoss(margin=0.5)
        chosen = torch.tensor([2.0, 1.5])
        rejected = torch.tensor([0.5, 0.0])
        loss = loss_fn(chosen, rejected)
        assert loss.item() > 0.0

    def test_brier_and_calibration(self):
        preds = [0.9, 0.8, 0.1, 0.2]
        targets = [1, 1, 0, 0]
        brier = compute_brier_score(preds, targets)
        assert brier < 0.05  # Highly calibrated and accurate

        ece = compute_calibration_error(preds, targets, n_bins=5)
        assert 0.0 <= ece <= 1.0
'''
write("tests/test_reasoning.py", test_src)
commit("test: add tests for ORM margin ranking loss and calibration metrics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — Unit Tests Part 3: MCTS & ToT
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_reasoning.py")
test_src += '''\


class TestMCTS:
    def test_reasoning_node_and_puct(self):
        root = ReasoningNode(state_text="Question")
        child1 = root.add_child("Step 1", prior_p=0.8)
        child2 = root.add_child("Step 2", prior_p=0.2)

        assert not root.is_leaf()
        assert child1.is_leaf()
        assert child1.visits == 0

        score1 = puct_score(child1, parent_visits=4)
        score2 = puct_score(child2, parent_visits=4)
        assert score1 > score2  # Higher prior gives higher initial PUCT

    def test_mcts_search_convergence(self):
        mcts = MonteCarloTreeSearch()

        def mock_step_gen(state):
            if "Target" in state:
                return [("Done", 1.0, True)]
            return [("Target Step", 0.8, False), ("Wrong Step", 0.2, False)]

        def mock_verifier(prefix, step):
            return 0.95 if "Target" in step or "Done" in step else 0.05

        traj, root = mcts.search("Initial Prompt", mock_step_gen, mock_verifier, num_simulations=15)
        assert len(traj) > 0
        assert "Target Step" in traj[0]

    def test_tree_of_thoughts(self):
        tot = TreeOfThoughts(max_depth=3, beam_width=2)
        def gen(path, k):
            return [f"thought_{i}" for i in range(k)]
        def evaluate(path, thought):
            return ("sure", 0.9) if "0" in thought else ("impossible", 0.1)
        def is_sol(path):
            return "thought_0\\n\\nthought_0" in path

        solutions = tot.solve("Problem", gen, evaluate, is_sol)
        assert isinstance(solutions, list)
'''
write("tests/test_reasoning.py", test_src)
commit("test: add tests for MCTS node expansion, PUCT scoring, and search execution")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — Unit Tests Part 4: Beam Search & Best of N
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_reasoning.py")
test_src += '''\


class TestBeamSearchAndBestOfN:
    def test_step_beam_search(self):
        beam = StepBeamSearch(beam_width=2, max_steps=3)
        def prop(text, k):
            return ["step_a", "step_b"]
        def scorer(text, step):
            return 0.9 if step == "step_a" else 0.3
        def is_term(text):
            return text.count("step_a") >= 2

        results = beam.search("Prompt", prop, scorer, is_term)
        assert len(results) > 0
        assert results[0]["cumulative_score"] >= results[-1]["cumulative_score"]

    def test_best_of_n_majority_vote(self):
        verifier = BestOfNVerifier()
        candidates = [
            {"text": "Reasoning... \\\\boxed{42}", "reward": 0.9},
            {"text": "Reasoning... \\\\boxed{42}", "reward": 0.8},
            {"text": "Reasoning... \\\\boxed{100}", "reward": 0.4},
        ]
        winner, conf, dist = verifier.majority_vote(candidates, weighted_by_score=True)
        assert winner == "42"
        assert conf > 0.5
        assert "42" in dist and "100" in dist
'''
write("tests/test_reasoning.py", test_src)
commit("test: add tests for StepBeamSearch and Best-of-N verifier")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — Unit Tests Part 5: GRPO, Self-Correction, Scaling
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_reasoning.py")
test_src += '''\


class TestGRPOAndScaling:
    def test_group_advantages(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        adv = compute_group_advantages(rewards)
        assert math.isclose(adv.mean().item(), 0.0, abs_tol=1e-5)
        assert adv[0, 3] > adv[0, 0]

    def test_grpo_loss(self):
        G, T = 4, 8
        log_probs = torch.randn(G, T)
        old_log_probs = log_probs.clone()
        ref_log_probs = log_probs.clone()
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

        loss, metrics = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)
        assert isinstance(loss.item(), float)
        assert "policy_loss" in metrics
        assert "kl_divergence" in metrics

    def test_self_reflection_backtracking(self):
        reasoner = SelfReflectiveReasoner()
        text = "<think>Let me compute 7 * 8 = 54. Wait, that's incorrect. Actually 7 * 8 = 56.</think>\\\\boxed{56}"
        parsed = reasoner.parse_reasoning_trace(text)
        assert parsed["has_think_tags"]
        assert parsed["backtrack_count"] >= 1
        assert parsed["final_answer"] == "56"

    def test_star_bootstrap(self):
        star = STaR()
        dataset = [{"question": "2+2", "answer": "4"}, {"question": "3+3", "answer": "6"}]
        def gen(q):
            return "4" if "2+2" in q else "wrong"
        def evaluate(pred, gt):
            return pred == gt

        res = star.bootstrap_iteration(dataset, gen, evaluate)
        assert res["direct_accuracy"] == 0.5
        assert res["buffer_size"] == 1

    def test_pass_at_k_and_scaling(self):
        # 10 samples, 2 correct: pass@1 should be 0.2
        p1 = pass_at_k(n=10, c=2, k=1)
        assert math.isclose(p1, 0.2, rel_tol=1e-4)

        # pass@k should increase with k
        p5 = pass_at_k(n=10, c=2, k=5)
        assert p5 > p1

        flops = estimate_test_time_flops(num_tokens=100, num_rollouts=4, model_params=1_000_000)
        assert flops == 2.0 * 1_000_000 * 100 * 4

        sim = TestTimeScalingSimulator(p_step_correct=0.9, num_steps=2)
        tradeoffs = sim.compute_scaling_tradeoff([1, 2, 4])
        assert len(tradeoffs) == 3
        assert tradeoffs[-1]["accuracy"] >= tradeoffs[0]["accuracy"]
'''
write("tests/test_reasoning.py", test_src)
commit("test: add tests for GRPO advantages, loss, self-correction, and scaling laws")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — Bump to v5.7.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace('__version__ = "5.6.0"', '__version__ = "5.7.0"')
write("nanomind/__init__.py", src)
commit("feat: bump to v5.7.0 — Reasoning Models & Test-Time Compute (o1/R1/PRM/MCTS/GRPO) release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + Push + Tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `ssm`        | State Space Models — S4, Mamba (selective), linear attention, hybrid, analysis |",
    "| `ssm`        | State Space Models — S4, Mamba (selective), linear attention, hybrid, analysis |\n"
    "| `reasoning`  | Reasoning Models & Test-Time Compute — PRM (step verification), MCTS, ToT, GRPO (DeepSeek-R1), STaR, scaling laws |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.7.0] — 2024 — Reasoning Models & Test-Time Compute\\n\\n### Added\\n"
      "- `PRMConfig`, `SearchConfig`, `GRPOConfig`, `ReasoningConfig` — configurations for reasoning\\n"
      "- `StepDelimiter` — token-level reasoning step extraction and joining\\n"
      "- `ProcessRewardModel` — step-level value head predicting step correctness probabilities\\n"
      "- `aggregate_step_scores` — product, min (weakest link), last, and mean step score aggregation\\n"
      "- `PRMLoss` — masked step-level binary cross-entropy loss\\n"
      "- `OutcomeRewardModel` — terminal token outcome evaluation\\n"
      "- `MarginRankingLoss` — Bradley-Terry pairwise preference ranking objective\\n"
      "- `compute_brier_score`, `compute_calibration_error` (ECE) — verifier calibration diagnostics\\n"
      "- `ReasoningNode` & `puct_score` — tree search state with exploration bonuses\\n"
      "- `MonteCarloTreeSearch` — selection, expansion, simulation, backpropagation for math reasoning\\n"
      "- `TreeOfThoughts` (ToT) — BFS/DFS thought exploration with threshold pruning\\n"
      "- `StepBeamSearch` — PRM-guided step-level beam search\\n"
      "- `BestOfNVerifier` — rejection sampling and weighted majority voting on boxed solutions\\n"
      "- `SelfReflectiveReasoner` — thought tag parsing and backtracking trigger detection\\n"
      "- `compute_group_advantages` — critic-free group relative advantage normalization\\n"
      "- `grpo_loss` & `GRPOTrainer` — DeepSeek-R1 style clipped surrogate loss with KL penalty\\n"
      "- `STaR` — Self-Taught Reasoner rationale bootstrap and failure rationalization\\n"
      "- `pass_at_k` — unbiased mathematical estimator of Pass@k accuracy\\n"
      "- `estimate_test_time_flops` & `TestTimeScalingSimulator` — test-time compute vs accuracy scaling curves\\n"
      "- `examples/reasoning_demo.py` — comprehensive end-to-end reasoning demo\\n\\n---\\n\\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.7.0, update README and CHANGELOG for Day 57 Reasoning")

print("\n=== Pushing Day 57 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.7.0", "-m", "NanoMind v5.7.0 — Reasoning & Test-Time Compute (o1/R1/PRM/MCTS/GRPO)", check=False)
r = run("git", "push", "origin", "v5.7.0", check=False)
print("Tag v5.7.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 57 COMPLETE — v5.7.0 TAGGED! ===")
