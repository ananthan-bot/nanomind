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
        answer_tag: str = "\\boxed",
    ):
        self.think_start_tag = think_start_tag
        self.think_end_tag = think_end_tag
        self.answer_tag = answer_tag

    def format_prompt(self, problem: str) -> str:
        """Format problem with thinking instructions."""
        return (
            f"Solve the following problem step by step.\n"
            f"Enclose your reasoning inside {self.think_start_tag} and {self.think_end_tag}.\n"
            f"If you notice an error in your intermediate steps, explicitly correct yourself and rethink.\n"
            f"Provide the final answer inside {self.answer_tag}{{...}}.\n\n"
            f"Problem: {problem}\n"
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
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", output_text)
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
