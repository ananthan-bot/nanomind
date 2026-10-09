"""
nanomind/safety/constitutional.py — Constitutional AI (Anthropic style Critique & Revision loop).
"""
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ConstitutionalPrinciple:
    """A guiding principle in the model's constitution."""
    name: str
    critique_instruction: str
    revision_instruction: str


DEFAULT_CONSTITUTION: List[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="Harmlessness",
        critique_instruction="Identify any ways in which the previous response is harmful, toxic, dangerous, or encourages illegal acts.",
        revision_instruction="Rewrite the response to be completely harmless, polite, and safe while remaining helpful.",
    ),
    ConstitutionalPrinciple(
        name="Truthfulness",
        critique_instruction="Identify any statements in the response that are unsupported speculation or misleading.",
        revision_instruction="Rewrite the response to clearly acknowledge limitations and avoid misleading claims.",
    ),
]


class ConstitutionalEngine:
    """
    Critique-and-Revision engine enforcing constitutional principles through self-correction.
    """

    def __init__(self, constitution: Optional[List[ConstitutionalPrinciple]] = None, max_turns: int = 2):
        self.constitution = constitution or DEFAULT_CONSTITUTION
        self.max_turns = max_turns

    def critique_and_revise(
        self,
        prompt: str,
        initial_response: str,
        critique_fn: Callable[[str, str, str], str],
        revision_fn: Callable[[str, str, str, str], str],
        is_safe_fn: Callable[[str], bool],
    ) -> Dict[str, Any]:
        """
        Runs the constitutional iteration.
        critique_fn(prompt, current_response, critique_instruction) -> critique_text
        revision_fn(prompt, current_response, critique_text, revision_instruction) -> revised_response
        """
        current_response = initial_response
        history = []

        if is_safe_fn(current_response):
            return {
                "final_response": current_response,
                "turns_taken": 0,
                "history": [],
                "revised": False,
            }

        for turn in range(self.max_turns):
            for principle in self.constitution:
                critique = critique_fn(prompt, current_response, principle.critique_instruction)
                revised = revision_fn(prompt, current_response, critique, principle.revision_instruction)

                history.append({
                    "turn": turn + 1,
                    "principle": principle.name,
                    "critique": critique,
                    "revised_response": revised,
                })
                current_response = revised

                if is_safe_fn(current_response):
                    return {
                        "final_response": current_response,
                        "turns_taken": turn + 1,
                        "history": history,
                        "revised": True,
                    }

        return {
            "final_response": current_response,
            "turns_taken": self.max_turns,
            "history": history,
            "revised": True,
        }
