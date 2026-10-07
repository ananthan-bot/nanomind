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
        new_state = (self.state_text + "\n\n" + step_text).strip() if self.state_text else step_text
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
