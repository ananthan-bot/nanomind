"""
nanomind/safety/representation.py — Representation Engineering (RepE): Concept Reading & Activation Steering.
"""
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.safety.config import RepEConfig


class ConceptVector:
    """Stores a learned directional concept vector (e.g., refusal direction or honesty direction)."""

    def __init__(self, name: str, vector: torch.Tensor):
        self.name = name
        self.vector = vector / (vector.norm(p=2) + 1e-8)  # Normalized unit direction

    def similarity(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity of hidden states with this concept direction."""
        norm_h = hidden_state / (hidden_state.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(norm_h, self.vector)


class RepESteeringHook:
    """
    Activation steering hook:
    h_steered = h + alpha * concept_vector (to inject a concept)
    or
    h_steered = h - alpha * concept_vector (to ablate / suppress a toxic concept)
    """

    def __init__(self, concept_vector: ConceptVector, alpha: float = 1.0, mode: str = "suppress"):
        self.concept = concept_vector
        self.alpha = alpha
        self.mode = mode  # "suppress" (subtract) or "inject" (add)

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        v = self.concept.vector.to(output.device).type(output.dtype)
        if self.mode == "suppress":
            return output - self.alpha * v
        else:
            return output + self.alpha * v


def extract_concept_vector(
    positive_reps: torch.Tensor,
    negative_reps: torch.Tensor,
    name: str = "refusal"
) -> ConceptVector:
    """
    Extracts difference-of-means concept direction:
    v = mean(positive_representations) - mean(negative_representations)
    positive_reps: (N_pos, D) representations of target concept prompts
    negative_reps: (N_neg, D) representations of neutral / baseline prompts
    """
    mu_pos = positive_reps.mean(dim=0)
    mu_neg = negative_reps.mean(dim=0)
    direction = mu_pos - mu_neg
    return ConceptVector(name=name, vector=direction)
