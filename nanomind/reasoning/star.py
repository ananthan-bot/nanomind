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
