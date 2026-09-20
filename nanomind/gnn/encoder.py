"""
nanomind/gnn/encoder.py — Code Graph Encoder: graph → embedding.

Stacks multiple GNN layers to produce node and graph embeddings.
Graph-level embeddings are produced via global pooling:
  - Mean pooling: average all node embeddings
  - Max pooling:  take element-wise max across nodes
  - Attention pooling: learned weighted sum

Used for: code clone detection, bug classification, function similarity.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.gnn.graph import CodeGraph
from nanomind.gnn.layers import GCNLayer, GraphSAGELayer, GATLayer, GGNNLayer


class CodeGraphEncoder(nn.Module):
    """
    Multi-layer GNN encoder for code graphs.

    Args:
        in_dim:    Input node feature dimension.
        hidden:    Hidden dimension.
        out_dim:   Output embedding dimension.
        n_layers:  Number of GNN layers.
        layer_type: ``"gcn"``, ``"sage"``, ``"gat"``, or ``"ggnn"``.
        pooling:   Graph pooling: ``"mean"``, ``"max"``, or ``"attn"``.
        dropout:   Dropout rate.

    Example::

        encoder = CodeGraphEncoder(in_dim=64, hidden=128, out_dim=256)
        graph   = parser.parse("def foo(x): return x + 1")
        emb     = encoder(graph)   # → (1, 256) graph embedding
    """

    def __init__(
        self,
        in_dim:     int,
        hidden:     int   = 128,
        out_dim:    int   = 256,
        n_layers:   int   = 3,
        layer_type: str   = "gcn",
        pooling:    str   = "mean",
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden)
        self.dropout    = nn.Dropout(dropout)

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            d_in  = hidden
            d_out = hidden if i < n_layers - 1 else out_dim
            if layer_type == "gcn":
                self.layers.append(GCNLayer(d_in, d_out))
            elif layer_type == "sage":
                self.layers.append(GraphSAGELayer(d_in, d_out))
            elif layer_type == "gat":
                self.layers.append(GATLayer(d_in, d_out // 4, n_heads=4))
            elif layer_type == "ggnn":
                self.layers.append(GGNNLayer(d_in))
                if d_in != d_out:
                    self.layers.append(nn.Linear(d_in, d_out))
            else:
                raise ValueError(f"Unknown layer type: {layer_type!r}")

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden if i < n_layers - 1 else out_dim)
            for i in range(n_layers)
        ])

        # Attention pooling
        if pooling == "attn":
            self.pool_attn = nn.Linear(out_dim, 1)

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward_nodes(
        self,
        graph: CodeGraph,
    ) -> torch.Tensor:
        """
        Compute per-node embeddings.

        Args:
            graph: :class:`CodeGraph`.

        Returns:
            ``(N, out_dim)`` node embedding tensor.
        """
        h  = self.input_proj(graph.node_features)
        ei = graph.edge_index

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                h = F.relu(layer(h))
            else:
                h = layer(h, ei, graph.n_nodes)
            if i < len(self.layer_norms):
                h = self.layer_norms[i](h)
            h = self.dropout(h)
        return h

    def forward(self, graph: CodeGraph) -> torch.Tensor:
        """
        Compute graph-level embedding via pooling.

        Args:
            graph: :class:`CodeGraph`.

        Returns:
            ``(1, out_dim)`` graph embedding.
        """
        h = self.forward_nodes(graph)   # (N, D)

        if self.pooling == "mean":
            emb = h.mean(dim=0, keepdim=True)
        elif self.pooling == "max":
            emb = h.max(dim=0, keepdim=True).values
        elif self.pooling == "attn":
            scores = F.softmax(self.pool_attn(h), dim=0)   # (N, 1)
            emb    = (scores * h).sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling!r}")

        return self.out_proj(emb)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
