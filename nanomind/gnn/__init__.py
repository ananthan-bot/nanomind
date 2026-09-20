"""NanoMind GNN sub-package — Graph Neural Networks for code understanding.

Implements the full code graph learning pipeline:
  1. CodeGraph         — node features, edge_index, edge types, adjacency
  2. ASTParser         — Python ast → CodeGraph (one-hot node types)
  3. GCNLayer          — Graph Convolutional Network layer
  4. GraphSAGELayer    — GraphSAGE (mean neighbourhood aggregation)
  5. GATLayer          — Graph Attention Network (learned edge weights)
  6. GGNNLayer         — Gated GNN (GRU-based propagation)
  7. CodeGraphEncoder  — multi-layer GNN → graph/node embeddings
  8. CodeSimilarityModel — Siamese GNN for clone detection
  9. TripletCodeLoss   — triplet margin loss for metric learning
  10. DataFlowAnalyser — def-use analysis, DATA_FLOW edge injection
  11. GraphBatch       — batched graphs for efficient training

Primary exports:
    - :class:`CodeGraph`            — n_nodes, n_edges, adjacency, degree
    - :data:`EDGE_TYPE_NAMES`       — edge type constants
    - :class:`ASTParser`            — parse, parse_function
    - :class:`GCNLayer`             — GCN message passing
    - :class:`GraphSAGELayer`       — SAGE mean aggregation
    - :class:`GATLayer`             — attention-weighted aggregation
    - :class:`GGNNLayer`            — GRU-based n-step propagation
    - :class:`CodeGraphEncoder`     — in_dim→out_dim, pooling, n_params
    - :class:`CodeSimilarityModel`  — similarity, is_clone, embed
    - :class:`TripletCodeLoss`      — anchor/positive/negative triplet loss
    - :class:`DataFlowAnalyser`     — analyse, def_use_pairs
    - :class:`GraphBatch`           — from_graphs, mean/max pool
"""

from nanomind.gnn.graph import CodeGraph, EDGE_TYPE_NAMES, EDGE_AST_CHILD, EDGE_DATA_FLOW
from nanomind.gnn.ast_parser import ASTParser, N_NODE_TYPES
from nanomind.gnn.layers import GCNLayer, GraphSAGELayer, GATLayer, GGNNLayer
from nanomind.gnn.encoder import CodeGraphEncoder
from nanomind.gnn.similarity import CodeSimilarityModel, TripletCodeLoss
from nanomind.gnn.dataflow import DataFlowAnalyser
from nanomind.gnn.batch import GraphBatch

__all__ = [
    "CodeGraph", "EDGE_TYPE_NAMES", "EDGE_AST_CHILD", "EDGE_DATA_FLOW",
    "ASTParser", "N_NODE_TYPES",
    "GCNLayer", "GraphSAGELayer", "GATLayer", "GGNNLayer",
    "CodeGraphEncoder",
    "CodeSimilarityModel", "TripletCodeLoss",
    "DataFlowAnalyser",
    "GraphBatch",
]
