"""
nanomind/safety/classifier.py — Multi-label Safety Classifier (Llama Guard style moderation head).
"""
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.safety.taxonomy import SafetyCategory, SafetyPolicy


class SafetyClassifier(nn.Module):
    """
    Moderation classification model mapping representations to multi-label safety category probabilities.
    """

    def __init__(self, d_model: int = 128, policy: Optional[SafetyPolicy] = None):
        super().__init__()
        self.d_model = d_model
        self.policy = policy or SafetyPolicy()
        self.categories = list(SafetyCategory)
        num_classes = len(self.categories)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, hidden_state: torch.Tensor) -> Dict[SafetyCategory, torch.Tensor]:
        """
        hidden_state: (B, D) pooled sequence representation.
        Returns:
            dict of {SafetyCategory: probability (B,)}
        """
        logits = self.head(hidden_state)  # (B, num_classes)
        probs = torch.sigmoid(logits)     # Multi-label independent probabilities

        result = {}
        for idx, cat in enumerate(self.categories):
            result[cat] = probs[:, idx]
        return result

    def evaluate_safety(self, hidden_state: torch.Tensor) -> Dict[str, Any]:
        """
        Scores input and determines whether generation violates any category policy.
        """
        probs_dict = self.forward(hidden_state)
        violations = []
        scores_summary = {}

        for cat, prob_tensor in probs_dict.items():
            val = float(prob_tensor.squeeze().item()) if prob_tensor.numel() == 1 else float(prob_tensor[0].item())
            scores_summary[cat.value] = round(val, 4)
            if self.policy.is_violation(cat, val):
                violations.append((cat, val))

        is_safe = len(violations) == 0
        refusal = None if is_safe else self.policy.format_refusal(violations[0][0])

        return {
            "is_safe": is_safe,
            "violations": [(v[0].value, round(v[1], 4)) for v in violations],
            "scores": scores_summary,
            "refusal_message": refusal,
        }
