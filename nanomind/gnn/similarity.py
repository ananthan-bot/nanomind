"""
nanomind/gnn/similarity.py — Code clone detection via graph similarity.

Code clone types:
  Type 1: Exact copy (whitespace/comments may differ)
  Type 2: Renamed variables/methods
  Type 3: Added/removed statements
  Type 4: Semantically equivalent but structurally different

GNN-based clone detection:
  1. Parse both functions to CodeGraphs
  2. Encode each graph to an embedding
  3. Compute cosine similarity between embeddings
  4. Threshold: similarity > τ → clone

This approach naturally handles Types 1–3.
Type 4 requires more sophisticated semantic analysis.

Reference:
  Wang et al. (2020) "Detecting Code Clones with Graph Neural Networks"
  Fang et al. (2020) "Functional Code Clone Detection"
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.gnn.graph import CodeGraph
from nanomind.gnn.encoder import CodeGraphEncoder


class CodeSimilarityModel(nn.Module):
    """
    Siamese GNN for code clone detection.

    Encodes two code graphs and computes their similarity.

    Args:
        encoder:   Shared :class:`CodeGraphEncoder`.
        threshold: Cosine similarity threshold for clone decision.

    Example::

        encoder = CodeGraphEncoder(in_dim=64, out_dim=128)
        model   = CodeSimilarityModel(encoder)
        score   = model.similarity(graph1, graph2)
        is_clone = model.is_clone(graph1, graph2)
    """

    def __init__(
        self,
        encoder:   CodeGraphEncoder,
        threshold: float = 0.85,
    ) -> None:
        super().__init__()
        self.encoder   = encoder
        self.threshold = threshold

    def forward(
        self,
        g1: CodeGraph,
        g2: CodeGraph,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Encode both graphs and compute similarity.

        Returns:
            ``(emb1, emb2, cosine_similarity)``
        """
        e1  = self.encoder(g1)   # (1, D)
        e2  = self.encoder(g2)
        sim = F.cosine_similarity(e1, e2).item()
        return e1, e2, sim

    def similarity(self, g1: CodeGraph, g2: CodeGraph) -> float:
        """Return cosine similarity score in [-1, 1]."""
        _, _, sim = self.forward(g1, g2)
        return sim

    def is_clone(self, g1: CodeGraph, g2: CodeGraph) -> bool:
        """Return True if similarity exceeds threshold."""
        return self.similarity(g1, g2) >= self.threshold

    def embed(self, graph: CodeGraph) -> torch.Tensor:
        """Embed a single graph: ``(1, D)``."""
        with torch.no_grad():
            return self.encoder(graph)


class TripletCodeLoss(nn.Module):
    """
    Triplet margin loss for code similarity learning.

    Pulls anchor and positive (clone) together,
    pushes anchor and negative (non-clone) apart.

    Args:
        margin: Triplet loss margin.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor:   torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   ``(B, D)`` anchor embeddings.
            positive: ``(B, D)`` positive (clone) embeddings.
            negative: ``(B, D)`` negative (non-clone) embeddings.
        """
        d_pos = 1.0 - F.cosine_similarity(anchor, positive)
        d_neg = 1.0 - F.cosine_similarity(anchor, negative)
        loss  = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()
