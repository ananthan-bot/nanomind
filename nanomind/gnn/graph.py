"""
nanomind/gnn/graph.py — Graph data structures for code programs.

## Why GNNs for Code?

Source code has rich graph structure:
  AST (Abstract Syntax Tree):  hierarchical parse structure
  CFG (Control Flow Graph):    how execution flows between blocks
  DFG (Data Flow Graph):       how data values propagate
  PDG (Program Dependence Graph): data + control combined
  Call Graph:                  which function calls which

Text models (Transformers) treat code as a linear sequence.
GNNs capture the non-sequential relational structure:
  - Variable uses across non-adjacent lines
  - Function call chains across files
  - Data dependencies through branches

Used in: CodeBERT+GNN (Microsoft), GGNN (Li et al., 2016),
         GraphCodeBERT (Guo et al., 2021).

References:
  Li et al. (2016) "Gated Graph Sequence Neural Networks"
  https://arxiv.org/abs/1511.05493

  Guo et al. (2021) "GraphCodeBERT"
  https://arxiv.org/abs/2009.08366
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field


@dataclass
class CodeGraph:
    """
    A graph representation of a code program.

    Attributes:
        node_features:  ``(N, D)`` feature tensor for each node.
        edge_index:     ``(2, E)`` tensor of (src, dst) edge pairs.
        edge_types:     ``(E,)`` integer tensor of edge type IDs.
        node_labels:    List of node label strings (e.g., "FunctionDef").
        edge_labels:    List of edge label strings (e.g., "AST_child").
        n_nodes:        Number of nodes.
        n_edges:        Number of edges.
        source:         Original source code string.
    """
    node_features: torch.Tensor
    edge_index:    torch.Tensor
    edge_types:    torch.Tensor | None = None
    node_labels:   list[str]          = field(default_factory=list)
    edge_labels:   list[str]          = field(default_factory=list)
    source:        str                = ""

    @property
    def n_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def to_adjacency(self) -> torch.Tensor:
        """Dense adjacency matrix ``(N, N)``."""
        N   = self.n_nodes
        adj = torch.zeros(N, N)
        if self.n_edges > 0:
            adj[self.edge_index[0], self.edge_index[1]] = 1.0
        return adj

    def add_self_loops(self) -> "CodeGraph":
        """Return graph with self-loops added to each node."""
        N    = self.n_nodes
        self_idx = torch.arange(N).unsqueeze(0).repeat(2, 1)
        ei       = torch.cat([self.edge_index, self_idx], dim=1)
        et       = None
        if self.edge_types is not None:
            self_et = torch.zeros(N, dtype=torch.long)
            et      = torch.cat([self.edge_types, self_et])
        return CodeGraph(self.node_features, ei, et,
                         self.node_labels, self.edge_labels, self.source)

    def degree(self) -> torch.Tensor:
        """Out-degree of each node: ``(N,)``."""
        N = self.n_nodes
        deg = torch.zeros(N, dtype=torch.long)
        if self.n_edges > 0:
            deg.scatter_add_(0, self.edge_index[0], torch.ones(self.n_edges, dtype=torch.long))
        return deg

    def to_dict(self) -> dict:
        return {
            "n_nodes":    self.n_nodes,
            "n_edges":    self.n_edges,
            "feat_dim":   self.node_features.shape[-1],
            "node_labels": self.node_labels[:5],
        }


# Standard edge type IDs
EDGE_AST_CHILD    = 0
EDGE_AST_NEXT     = 1
EDGE_DATA_FLOW    = 2
EDGE_CONTROL_FLOW = 3
EDGE_CALL         = 4
EDGE_RETURN       = 5

EDGE_TYPE_NAMES = {
    EDGE_AST_CHILD:    "ast_child",
    EDGE_AST_NEXT:     "ast_next_token",
    EDGE_DATA_FLOW:    "data_flow",
    EDGE_CONTROL_FLOW: "control_flow",
    EDGE_CALL:         "function_call",
    EDGE_RETURN:       "function_return",
}
