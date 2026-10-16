"""
nanomind/edit/eval.py — Standard evaluation benchmark: Edit Efficacy, Generalization, and Specificity (Locality).
"""
from typing import Dict, Any, List


class EditEvaluator:
    """
    Standard Model Editing Benchmark metrics:
    1. Efficacy: P(target_new | prompt) > P(target_true | prompt)
    2. Generalization: P(target_new | paraphrase_prompt)
    3. Locality / Specificity: P(unrelated_target | neighborhood_prompt) is unchanged
    """

    @staticmethod
    def evaluate_edit(
        efficacy_score: float,
        generalization_scores: List[float],
        neighborhood_drift_scores: List[float],
        locality_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        avg_gen = sum(generalization_scores) / max(1, len(generalization_scores))
        max_drift = max(neighborhood_drift_scores) if neighborhood_drift_scores else 0.0

        is_efficacious = efficacy_score >= 0.8
        is_generalizing = avg_gen >= 0.7
        is_local = max_drift <= locality_threshold

        composite_score = (efficacy_score + avg_gen + (1.0 - min(1.0, max_drift))) / 3.0

        return {
            "efficacy": round(efficacy_score, 4),
            "generalization": round(avg_gen, 4),
            "neighborhood_drift": round(max_drift, 4),
            "passed_all_criteria": is_efficacious and is_generalizing and is_local,
            "composite_score": round(composite_score, 4),
        }
