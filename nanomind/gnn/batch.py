"""
nanomind/gnn/batch.py — Graph batching for efficient GNN training.

GNNs require special batching because graphs have variable sizes.
We batch by concatenating node features and edge indices,
using a batch_idx tensor to identify which graph each node belongs to.

This is the standard approach used by PyTorch Geometric and DGL.
"""

from __future__ import annotations
import torch
from nanomind.gnn.graph import CodeGraph


class GraphBatch:
    """
    Batched graph: multiple graphs concatenated into one big graph.

    Each disconnected subgraph corresponds to one input graph.
    The ``batch_idx`` tensor maps each node to its graph index.

    Attributes:
        node_features: ``(N_total, D)`` concatenated node features.
        edge_index:    ``(2, E_total)`` shifted edge indices.
        batch_idx:     ``(N_total,)`` graph index per node.
        n_graphs:      Number of graphs in the batch.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index:    torch.Tensor,
        batch_idx:     torch.Tensor,
        n_graphs:      int,
        edge_types:    torch.Tensor | None = None,
    ) -> None:
        self.node_features = node_features
        self.edge_index    = edge_index
        self.batch_idx     = batch_idx
        self.n_graphs      = n_graphs
        self.edge_types    = edge_types

    @property
    def n_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def global_mean_pool(self, h: torch.Tensor) -> torch.Tensor:
        """
        Global mean pooling: average node embeddings per graph.

        Args:
            h: ``(N_total, D)`` node embeddings.

        Returns:
            ``(n_graphs, D)`` graph embeddings.
        """
        D     = h.shape[-1]
        out   = torch.zeros(self.n_graphs, D)
        count = torch.zeros(self.n_graphs, 1)
        idx   = self.batch_idx.unsqueeze(-1).expand(-1, D)
        out.scatter_add_(0, idx, h)
        count.scatter_add_(0, self.batch_idx.unsqueeze(-1),
                            torch.ones(self.n_nodes, 1))
        return out / (count + 1e-8)

    def global_max_pool(self, h: torch.Tensor) -> torch.Tensor:
        """Global max pooling per graph."""
        D   = h.shape[-1]
        out = torch.full((self.n_graphs, D), fill_value=-1e9)
        idx = self.batch_idx.unsqueeze(-1).expand(-1, D)
        out = torch.scatter_reduce(out, 0, idx, h, reduce="amax")
        return out

    @classmethod
    def from_graphs(cls, graphs: list[CodeGraph]) -> "GraphBatch":
        """
        Create a GraphBatch from a list of CodeGraphs.

        Args:
            graphs: List of :class:`CodeGraph`.

        Returns:
            :class:`GraphBatch`.
        """
        feats  = []
        edges  = []
        etypes = []
        bidx   = []
        offset = 0

        for g_idx, g in enumerate(graphs):
            feats.append(g.node_features)
            if g.n_edges > 0:
                edges.append(g.edge_index + offset)
                if g.edge_types is not None:
                    etypes.append(g.edge_types)
            bidx.append(torch.full((g.n_nodes,), g_idx, dtype=torch.long))
            offset += g.n_nodes

        node_features = torch.cat(feats, dim=0)
        edge_index    = torch.cat(edges, dim=1) if edges else torch.zeros(2, 0, dtype=torch.long)
        edge_types    = torch.cat(etypes) if etypes else None
        batch_idx     = torch.cat(bidx)

        return cls(node_features, edge_index, batch_idx, len(graphs), edge_types)
