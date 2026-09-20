"""
nanomind/gnn/layers.py — GNN layers: GCN, GraphSAGE, GAT, GGNN.

## Graph Neural Network Message Passing

All GNNs follow the message-passing framework (Gilmer et al., 2017):
  h_v^{t+1} = UPDATE(h_v^t, AGGREGATE({h_u^t : u ∈ N(v)}))

Different choices of AGGREGATE and UPDATE give different GNNs:

  GCN (Kipf & Welling, 2017):
    h^{l+1} = σ(D^{-1/2} A D^{-1/2} H^l W)
    Mean aggregation with symmetric normalisation.

  GraphSAGE (Hamilton et al., 2017):
    h_v^{l+1} = σ(W[h_v^l | MEAN({h_u : u ∈ N(v)})])
    Concatenates own features with neighbour mean.

  GAT (Veličković et al., 2018):
    h_v^{l+1} = σ(Σ_{u∈N(v)} α_{vu} W h_u)
    Learned attention weights α_{vu} per edge.

  GGNN (Li et al., 2016):
    Uses GRU to update node states over timesteps.
    Used for code understanding (bug detection, variable misuse).

References:
  Kipf & Welling (2017): https://arxiv.org/abs/1609.02907
  Hamilton et al. (2017): https://arxiv.org/abs/1706.02216
  Veličković et al. (2018): https://arxiv.org/abs/1710.10903
  Li et al. (2016): https://arxiv.org/abs/1511.05493
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    """Scatter mean: average src values for each index."""
    out   = torch.zeros(n, src.shape[-1])
    count = torch.zeros(n, 1)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    count.scatter_add_(0, idx.unsqueeze(-1), torch.ones(idx.shape[0], 1))
    return out / (count + 1e-8)


def scatter_sum(src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    """Scatter sum: sum src values for each index."""
    out = torch.zeros(n, src.shape[-1])
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    return out


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    h^{l+1} = σ(D^{-1/2} A D^{-1/2} H^l W)

    Args:
        in_dim:  Input feature dimension.
        out_dim: Output feature dimension.
        bias:    Include bias term.

    Example::

        gcn  = GCNLayer(64, 128)
        h_out = gcn(h, edge_index, n_nodes=10)
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(
        self,
        h:          torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes:    int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:          ``(N, in_dim)`` node features.
            edge_index: ``(2, E)`` edge pairs.
            n_nodes:    Number of nodes (default: h.shape[0]).

        Returns:
            ``(N, out_dim)`` updated node features.
        """
        N       = n_nodes or h.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Degree normalisation
        deg     = torch.zeros(N).scatter_add_(0, dst, torch.ones(src.shape[0]))
        deg_inv = deg.pow(-0.5).clamp(max=1e6)

        # Aggregate: normalised mean
        h_proj  = self.W(h)
        if edge_index.shape[1] > 0:
            msg     = h_proj[src] * deg_inv[src].unsqueeze(-1)
            agg     = scatter_sum(msg, dst, N)
            h_out   = agg * deg_inv.unsqueeze(-1)
        else:
            h_out   = h_proj
        return F.relu(h_out)


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer (Hamilton et al., 2017).

    Concatenates self features with neighbour mean, then projects.

    Args:
        in_dim:  Input dimension.
        out_dim: Output dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim * 2, out_dim)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(
        self,
        h:          torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes:    int | None = None,
    ) -> torch.Tensor:
        N       = n_nodes or h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        if edge_index.shape[1] > 0:
            neigh_mean = scatter_mean(h[src], dst, N)
        else:
            neigh_mean = torch.zeros_like(h)
        agg    = torch.cat([h, neigh_mean], dim=-1)
        return F.relu(self.W(agg))


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Veličković et al., 2018).

    Computes learned attention weights per edge.

    Args:
        in_dim:  Input dimension.
        out_dim: Output dimension per head.
        n_heads: Number of attention heads.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        n_heads: int   = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = out_dim
        self.W    = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a    = nn.Parameter(torch.empty(2 * out_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(
        self,
        h:          torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes:    int | None = None,
    ) -> torch.Tensor:
        N       = n_nodes or h.shape[0]
        H       = self.n_heads
        D       = self.d_head
        Wh      = self.W(h).view(N, H, D)   # (N, H, D)

        if edge_index.shape[1] == 0:
            return Wh.mean(dim=1)

        src, dst = edge_index[0], edge_index[1]
        # Attention: concat Wh_src and Wh_dst, dot with a
        e_src   = Wh[src]   # (E, H, D)
        e_dst   = Wh[dst]
        e       = torch.cat([e_src, e_dst], dim=-1)       # (E, H, 2D)
        a_coef  = (e * self.a[:2*D]).sum(dim=-1)           # (E, H)
        a_coef  = F.leaky_relu(a_coef, 0.2)

        # Softmax over neighbours for each node (scatter softmax)
        alpha   = torch.zeros(N, H)
        alpha.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), a_coef.exp())
        alpha   = a_coef.exp() / (alpha[dst] + 1e-9)      # (E, H)
        alpha   = self.drop(alpha)

        msg     = Wh[src] * alpha.unsqueeze(-1)            # (E, H, D)
        out     = scatter_sum(msg.view(-1, H * D), dst, N) # (N, H*D)
        return F.elu(out)


class GGNNLayer(nn.Module):
    """
    Gated Graph Neural Network layer (Li et al., 2016).

    Uses GRU to update node states, allowing multi-step propagation.

    Args:
        d_model:  Node feature dimension.
        n_steps:  Propagation steps.
    """

    def __init__(self, d_model: int, n_steps: int = 3) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.W_msg   = nn.Linear(d_model, d_model)
        self.gru     = nn.GRUCell(d_model, d_model)

    def forward(
        self,
        h:          torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes:    int | None = None,
    ) -> torch.Tensor:
        N    = n_nodes or h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        for _ in range(self.n_steps):
            if edge_index.shape[1] > 0:
                msg = self.W_msg(h[src])             # (E, D)
                agg = scatter_mean(msg, dst, N)      # (N, D)
            else:
                agg = torch.zeros_like(h)
            h   = self.gru(agg, h)                   # GRU update
        return h
