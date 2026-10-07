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

                    new_path = path_text + "\n\n" + thought
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
