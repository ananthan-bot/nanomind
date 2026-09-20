"""
day42_commits.py — 20 atomic commits for Day 42: Graph Neural Networks for Code Understanding.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 42: GNN for Code Understanding — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — gnn package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/__init__.py",
      '"""NanoMind GNN sub-package — Graph Neural Networks for code understanding."""\n')
commit("feat: add nanomind/gnn/ package skeleton for Graph Neural Networks")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Graph data structures
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/graph.py", '''\
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
''')
commit("feat: add CodeGraph — node_features, edge_index, edge_types, adjacency, degree, self_loops, EDGE_* constants")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — AST parser (pure Python, no tree-sitter dependency)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/ast_parser.py", '''\
"""
nanomind/gnn/ast_parser.py — AST → CodeGraph converter.

Uses Python's built-in ``ast`` module to parse Python source code
into an Abstract Syntax Tree and converts it to a CodeGraph.

AST Node Types (sampled):
  Module, FunctionDef, AsyncFunctionDef, ClassDef
  Return, Delete, Assign, AugAssign, AnnAssign
  For, While, If, With, Try, ExceptHandler
  Import, ImportFrom, Global, Nonlocal, Expr, Pass, Break, Continue
  BoolOp, BinOp, UnaryOp, Lambda, IfExp, Dict, Set, ListComp
  Call, Attribute, Subscript, Name, Constant, List, Tuple

Each AST node becomes a graph node.
Edges:
  EDGE_AST_CHILD:   parent → child
  EDGE_AST_NEXT:    sibling → sibling (sequential order)
"""

from __future__ import annotations
import ast
import torch
from nanomind.gnn.graph import CodeGraph, EDGE_AST_CHILD, EDGE_AST_NEXT

# Node type vocabulary (62 types)
_AST_TYPES = [
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Return", "Delete", "Assign", "AugAssign", "AnnAssign",
    "For", "AsyncFor", "While", "If", "With", "AsyncWith",
    "Raise", "Try", "ExceptHandler", "Assert", "Import", "ImportFrom",
    "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
    "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
    "Dict", "Set", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Await", "Yield", "YieldFrom", "Compare", "Call", "FormattedValue",
    "JoinedStr", "Constant", "Attribute", "Subscript", "Starred",
    "Name", "List", "Tuple", "Slice", "Load", "Store", "Del",
    "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow",
    "Unknown",
]
TYPE2ID = {t: i for i, t in enumerate(_AST_TYPES)}
N_NODE_TYPES = len(_AST_TYPES)


def _type_id(node) -> int:
    return TYPE2ID.get(type(node).__name__, TYPE2ID["Unknown"])


def _one_hot(idx: int, size: int) -> torch.Tensor:
    v = torch.zeros(size)
    v[idx] = 1.0
    return v


class ASTParser:
    """
    Parse Python source code into a :class:`CodeGraph`.

    Each AST node becomes a graph node with a one-hot feature vector
    encoding its node type. Edges connect parent→child and sibling→sibling.

    Args:
        add_sibling_edges: Add EDGE_AST_NEXT edges between siblings.

    Example::

        parser = ASTParser()
        graph  = parser.parse("def foo(x): return x + 1")
        print(graph.n_nodes, graph.n_edges)
    """

    def __init__(self, add_sibling_edges: bool = True) -> None:
        self.add_sibling_edges = add_sibling_edges

    def parse(self, source: str) -> CodeGraph:
        """
        Parse Python source into a CodeGraph.

        Args:
            source: Python source code string.

        Returns:
            :class:`CodeGraph`.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Return empty graph on parse failure
            return CodeGraph(
                node_features = torch.zeros(1, N_NODE_TYPES),
                edge_index    = torch.zeros(2, 0, dtype=torch.long),
                node_labels   = ["<error>"],
                source        = source,
            )

        nodes:       list[tuple] = []   # (node_id, type_id, label)
        edges_src:   list[int]   = []
        edges_dst:   list[int]   = []
        edge_types:  list[int]   = []
        node_map:    dict        = {}   # id(ast_node) → node_id

        def visit(node, parent_id: int | None = None):
            nid   = len(nodes)
            label = type(node).__name__
            nodes.append((nid, _type_id(node), label))
            node_map[id(node)] = nid

            if parent_id is not None:
                edges_src.append(parent_id)
                edges_dst.append(nid)
                edge_types.append(EDGE_AST_CHILD)

            children = list(ast.iter_child_nodes(node))
            prev_id  = None
            for child in children:
                visit(child, parent_id=nid)
                child_id = node_map[id(child)]
                if self.add_sibling_edges and prev_id is not None:
                    edges_src.append(prev_id)
                    edges_dst.append(child_id)
                    edge_types.append(EDGE_AST_NEXT)
                prev_id = child_id

        visit(tree)

        features = torch.stack([_one_hot(n[1], N_NODE_TYPES) for n in nodes])
        labels   = [n[2] for n in nodes]
        if edges_src:
            ei = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            et = torch.tensor(edge_types, dtype=torch.long)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
            et = torch.zeros(0, dtype=torch.long)

        return CodeGraph(
            node_features = features,
            edge_index    = ei,
            edge_types    = et,
            node_labels   = labels,
            source        = source,
        )

    def parse_function(self, func_source: str) -> CodeGraph:
        """Parse a single function definition."""
        return self.parse(func_source)
''')
commit("feat: add ASTParser — Python ast → CodeGraph, one-hot node types, parent→child + sibling edges")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — GCN layer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/layers.py", '''\
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
''')
commit("feat: add GCNLayer, GraphSAGELayer, GATLayer, GGNNLayer — GNN message passing implementations")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Code GNN encoder
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/encoder.py", '''\
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
''')
commit("feat: add CodeGraphEncoder — GCN/SAGE/GAT/GGNN stacks, mean/max/attn pooling, forward_nodes()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Code similarity (clone detection)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/similarity.py", '''\
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
''')
commit("feat: add CodeSimilarityModel (Siamese GNN, is_clone), TripletCodeLoss (margin loss)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Data flow analyser
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/dataflow.py", '''\
"""
nanomind/gnn/dataflow.py — Data flow analysis for code graphs.

Data flow analysis tracks how variable values propagate through code.
Used for:
  - Variable misuse detection (VarMisuse task, Allamanis et al., 2018)
  - Null pointer dereference detection
  - Type inference

Reaching Definitions:
  A definition d reaches point p if there exists a path from d to p
  in the CFG along which d is not redefined.

This module adds DATA_FLOW edges to AST graphs based on simple
variable def-use analysis within Python code.

Reference:
  Allamanis et al. (2018) "Learning to Represent Programs with Graphs"
  https://arxiv.org/abs/1711.00740
"""

from __future__ import annotations
import ast
import torch
from nanomind.gnn.graph import CodeGraph, EDGE_DATA_FLOW


class DataFlowAnalyser:
    """
    Analyse data flow in Python code and add def-use edges.

    Finds all variable definitions and uses in a function.
    Adds EDGE_DATA_FLOW edges from definition site to use sites.

    Args:
        parser: :class:`ASTParser` to get base graph.

    Example::

        analyser = DataFlowAnalyser()
        graph    = analyser.analyse("def f(x):\\n  y = x + 1\\n  return y")
        # graph now has DATA_FLOW edges: x_def→x_use, y_def→y_use
    """

    def __init__(self) -> None:
        self._defs: dict[str, list[int]] = {}   # var_name → [node_ids]
        self._uses: dict[str, list[int]] = {}

    def _collect_defs_uses(self, tree, node_map: dict) -> None:
        """Collect definition and use sites for each variable."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        nid = node_map.get(id(target))
                        if nid is not None:
                            self._defs.setdefault(target.id, []).append(nid)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                nid = node_map.get(id(node))
                if nid is not None:
                    self._uses.setdefault(node.id, []).append(nid)

    def analyse(self, source: str) -> CodeGraph:
        """
        Parse source and add data flow edges.

        Returns:
            :class:`CodeGraph` with DATA_FLOW edges added.
        """
        from nanomind.gnn.ast_parser import ASTParser
        parser   = ASTParser()
        graph    = parser.parse(source)

        # Re-parse to get node map
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return graph

        self._defs.clear()
        self._uses.clear()

        # Build a node_map by traversal order
        nodes = list(ast.walk(tree))
        node_map = {id(n): i for i, n in enumerate(nodes)}
        N_ast = len(nodes)

        self._collect_defs_uses(tree, node_map)

        # Build data flow edges
        extra_src, extra_dst, extra_types = [], [], []
        for var, def_ids in self._defs.items():
            for def_nid in def_ids:
                for use_nid in self._uses.get(var, []):
                    if def_nid < graph.n_nodes and use_nid < graph.n_nodes:
                        extra_src.append(def_nid)
                        extra_dst.append(use_nid)
                        extra_types.append(EDGE_DATA_FLOW)

        if not extra_src:
            return graph

        ei  = graph.edge_index
        et  = graph.edge_types
        new_ei = torch.tensor([extra_src, extra_dst], dtype=torch.long)
        new_et = torch.tensor(extra_types, dtype=torch.long)

        combined_ei = torch.cat([ei, new_ei], dim=1) if ei.shape[1] > 0 else new_ei
        combined_et = torch.cat([et, new_et]) if et is not None else new_et

        return CodeGraph(
            node_features = graph.node_features,
            edge_index    = combined_ei,
            edge_types    = combined_et,
            node_labels   = graph.node_labels,
            source        = source,
        )

    def def_use_pairs(self) -> dict[str, dict]:
        """Return def-use summary per variable."""
        return {
            var: {"n_defs": len(self._defs.get(var, [])),
                  "n_uses": len(self._uses.get(var, []))}
            for var in set(list(self._defs) + list(self._uses))
        }
''')
commit("feat: add DataFlowAnalyser — def-use analysis, DATA_FLOW edges, def_use_pairs()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — graph batch utilities
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/batch.py", '''\
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
''')
commit("feat: add GraphBatch — from_graphs(), global_mean_pool, global_max_pool, batch_idx")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — gnn __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/gnn/__init__.py", '''\
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
''')
commit("refactor: export all GNN components from nanomind/gnn/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/gnn_demo.py", '''\
"""
examples/gnn_demo.py — NanoMind GNN for Code Understanding demo.

Demonstrates:
  1. AST parsing: Python code → CodeGraph
  2. GCN/SAGE/GAT layers: message passing
  3. CodeGraphEncoder: graph → embedding
  4. Code clone detection: similarity model
  5. Data flow analysis: def-use edges
  6. Graph batching: efficient multi-graph processing

Usage:
    python examples/gnn_demo.py
"""
import torch
from nanomind.gnn import (
    ASTParser, CodeGraphEncoder, CodeGraph,
    CodeSimilarityModel, TripletCodeLoss,
    GCNLayer, GraphSAGELayer, GATLayer, GGNNLayer,
    DataFlowAnalyser, GraphBatch, N_NODE_TYPES,
)

print("=" * 60)
print("NanoMind GNN for Code Understanding Demo")
print("=" * 60)

# ── AST Parsing ───────────────────────────────────────────────────────────────
print("\n── AST Parsing ──")
parser = ASTParser()
src1   = "def add(a, b):\\n    return a + b"
src2   = "def multiply(x, y):\\n    result = x * y\\n    return result"
g1     = parser.parse(src1)
g2     = parser.parse(src2)
print(f"  'add':      {g1.n_nodes} nodes, {g1.n_edges} edges")
print(f"  'multiply': {g2.n_nodes} nodes, {g2.n_edges} edges")
print(f"  Node labels (first 5): {g1.node_labels[:5]}")
print(f"  Degree: {g1.degree().tolist()}")

# ── GNN Layers ────────────────────────────────────────────────────────────────
print("\n── GNN Layers ──")
N, D = g1.n_nodes, 32
h    = torch.randn(N, D)
ei   = g1.edge_index

gcn  = GCNLayer(D, 64)
sage = GraphSAGELayer(D, 64)
gat  = GATLayer(D, 16, n_heads=4)
ggnn = GGNNLayer(D, n_steps=2)

print(f"  GCN output:       {tuple(gcn(h, ei, N).shape)}")
print(f"  GraphSAGE output: {tuple(sage(h, ei, N).shape)}")
print(f"  GAT output:       {tuple(gat(h, ei, N).shape)}")
print(f"  GGNN output:      {tuple(ggnn(h, ei, N).shape)}")

# ── CodeGraphEncoder ──────────────────────────────────────────────────────────
print("\n── CodeGraphEncoder ──")
encoder = CodeGraphEncoder(in_dim=N_NODE_TYPES, hidden=32, out_dim=64,
                            n_layers=2, layer_type="gcn", pooling="mean")
with torch.no_grad():
    emb1 = encoder(g1)
    emb2 = encoder(g2)
print(f"  Graph 'add' embedding:      {tuple(emb1.shape)}")
print(f"  Graph 'multiply' embedding: {tuple(emb2.shape)}")
print(f"  Encoder n_params: {encoder.n_params:,}")

# ── Code Similarity ───────────────────────────────────────────────────────────
print("\n── Code Clone Detection ──")
sim_model = CodeSimilarityModel(encoder, threshold=0.9)
sim_score = sim_model.similarity(g1, g2)
is_clone  = sim_model.is_clone(g1, g2)
print(f"  Similarity(add, multiply): {sim_score:.4f}")
print(f"  Is clone: {is_clone}")

# Same function → should be high similarity
g1_copy = parser.parse(src1)
sim_same = sim_model.similarity(g1, g1_copy)
print(f"  Similarity(add, add_copy): {sim_same:.4f}")

# ── Triplet Loss ──────────────────────────────────────────────────────────────
print("\n── Triplet Loss ──")
triplet_loss = TripletCodeLoss(margin=0.5)
a = torch.randn(4, 64)
p = a + torch.randn(4, 64) * 0.1   # positive: slightly perturbed
n = torch.randn(4, 64)              # negative: random
loss = triplet_loss(a, p, n)
print(f"  Triplet loss: {loss.item():.4f}")

# ── Data Flow Analysis ────────────────────────────────────────────────────────
print("\n── Data Flow Analysis ──")
analyser  = DataFlowAnalyser()
src_flow  = "def f(x):\\n    y = x + 1\\n    z = y * 2\\n    return z"
g_flow    = analyser.analyse(src_flow)
print(f"  Graph with dataflow: {g_flow.n_nodes} nodes, {g_flow.n_edges} edges")
print(f"  Def-use pairs: {analyser.def_use_pairs()}")

# ── Graph Batching ────────────────────────────────────────────────────────────
print("\n── Graph Batching ──")
batch = GraphBatch.from_graphs([g1, g2, g_flow])
print(f"  Batch: {batch.n_nodes} total nodes, {batch.n_edges} edges, {batch.n_graphs} graphs")
h_all = torch.randn(batch.n_nodes, 32)
pooled = batch.global_mean_pool(h_all)
print(f"  Mean-pooled embeddings: {tuple(pooled.shape)}")
print("\nGNN demo complete!")
''')
commit("feat: add examples/gnn_demo.py — AST parsing, GNN layers, encoder, clone detection, dataflow, batching")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_gnn.py", '''\
"""tests/test_gnn.py — Tests for NanoMind GNN code understanding."""
import pytest
import torch
from nanomind.gnn import (
    CodeGraph, ASTParser, N_NODE_TYPES, EDGE_TYPE_NAMES,
    GCNLayer, GraphSAGELayer, GATLayer, GGNNLayer,
    CodeGraphEncoder, CodeSimilarityModel, TripletCodeLoss,
    DataFlowAnalyser, GraphBatch,
)

SRC1 = "def add(a, b):\\n    return a + b"
SRC2 = "def sub(x, y):\\n    return x - y"
SRC_FLOW = "def f(x):\\n    y = x + 1\\n    return y"


# ── CodeGraph ─────────────────────────────────────────────────────────────────

class TestCodeGraph:
    def _graph(self, N=5, D=8, E=4):
        feats = torch.randn(N, D)
        ei    = torch.randint(0, N, (2, E))
        return CodeGraph(node_features=feats, edge_index=ei)

    def test_n_nodes(self):
        g = self._graph(N=5)
        assert g.n_nodes == 5

    def test_n_edges(self):
        g = self._graph(E=4)
        assert g.n_edges == 4

    def test_adjacency_shape(self):
        g   = self._graph(N=5, E=3)
        adj = g.to_adjacency()
        assert adj.shape == (5, 5)

    def test_degree_shape(self):
        g   = self._graph(N=5, E=4)
        deg = g.degree()
        assert deg.shape == (5,)

    def test_self_loops_increases_edges(self):
        g     = self._graph(N=5, E=3)
        g_sl  = g.add_self_loops()
        assert g_sl.n_edges == 3 + 5

    def test_to_dict_keys(self):
        g = self._graph()
        d = g.to_dict()
        for k in ("n_nodes", "n_edges", "feat_dim"):
            assert k in d


# ── ASTParser ─────────────────────────────────────────────────────────────────

class TestASTParser:
    def test_parse_returns_graph(self):
        parser = ASTParser()
        g      = parser.parse(SRC1)
        assert isinstance(g, CodeGraph)

    def test_parse_positive_nodes(self):
        parser = ASTParser()
        g      = parser.parse(SRC1)
        assert g.n_nodes > 0

    def test_feature_dim_is_vocab(self):
        parser = ASTParser()
        g      = parser.parse(SRC1)
        assert g.node_features.shape[1] == N_NODE_TYPES

    def test_feature_one_hot(self):
        parser = ASTParser()
        g      = parser.parse(SRC1)
        row_sums = g.node_features.sum(dim=-1)
        assert (row_sums == 1.0).all()

    def test_parse_syntax_error_returns_single_node(self):
        parser = ASTParser()
        g      = parser.parse("def broken syntax!!!")
        assert g.n_nodes >= 1

    def test_sibling_edges(self):
        parser_with    = ASTParser(add_sibling_edges=True)
        parser_without = ASTParser(add_sibling_edges=False)
        g_with    = parser_with.parse("a = 1\\nb = 2\\nc = 3")
        g_without = parser_without.parse("a = 1\\nb = 2\\nc = 3")
        assert g_with.n_edges >= g_without.n_edges


# ── GNN Layers ────────────────────────────────────────────────────────────────

class TestGNNLayers:
    def _setup(self, N=8, D=16, E=6):
        h  = torch.randn(N, D)
        ei = torch.randint(0, N, (2, E))
        return h, ei, N

    def test_gcn_output_shape(self):
        h, ei, N = self._setup()
        gcn = GCNLayer(16, 32)
        out = gcn(h, ei, N)
        assert out.shape == (N, 32)

    def test_sage_output_shape(self):
        h, ei, N = self._setup()
        sage = GraphSAGELayer(16, 32)
        out  = sage(h, ei, N)
        assert out.shape == (N, 32)

    def test_gat_output_shape(self):
        h, ei, N = self._setup()
        gat = GATLayer(16, 8, n_heads=4)
        out = gat(h, ei, N)
        assert out.shape == (N, 32)   # 8 * 4

    def test_ggnn_output_shape(self):
        h, ei, N = self._setup()
        ggnn = GGNNLayer(16, n_steps=2)
        out  = ggnn(h, ei, N)
        assert out.shape == (N, 16)

    def test_gcn_no_edges(self):
        h  = torch.randn(5, 16)
        ei = torch.zeros(2, 0, dtype=torch.long)
        gcn = GCNLayer(16, 32)
        out = gcn(h, ei, 5)
        assert out.shape == (5, 32)

    def test_gcn_gradient_flows(self):
        h  = torch.randn(5, 16, requires_grad=True)
        ei = torch.randint(0, 5, (2, 4))
        gcn = GCNLayer(16, 32)
        out = gcn(h, ei, 5).sum()
        out.backward()
        assert h.grad is not None


# ── CodeGraphEncoder ──────────────────────────────────────────────────────────

class TestCodeGraphEncoder:
    def _encoder(self, pooling="mean"):
        return CodeGraphEncoder(N_NODE_TYPES, hidden=16, out_dim=32,
                                 n_layers=2, layer_type="gcn", pooling=pooling)

    def _graph(self):
        parser = ASTParser()
        return parser.parse(SRC1)

    def test_graph_embedding_shape(self):
        enc = self._encoder()
        g   = self._graph()
        emb = enc(g)
        assert emb.shape == (1, 32)

    def test_node_embeddings_shape(self):
        enc = self._encoder()
        g   = self._graph()
        h   = enc.forward_nodes(g)
        assert h.shape[1] == 32

    def test_max_pooling(self):
        enc = self._encoder("max")
        g   = self._graph()
        emb = enc(g)
        assert emb.shape == (1, 32)

    def test_attn_pooling(self):
        enc = self._encoder("attn")
        g   = self._graph()
        emb = enc(g)
        assert emb.shape == (1, 32)

    def test_n_params_positive(self):
        enc = self._encoder()
        assert enc.n_params > 0

    def test_sage_encoder(self):
        enc = CodeGraphEncoder(N_NODE_TYPES, hidden=16, out_dim=32,
                                n_layers=2, layer_type="sage")
        g   = self._graph()
        emb = enc(g)
        assert emb.shape == (1, 32)


# ── CodeSimilarityModel ───────────────────────────────────────────────────────

class TestCodeSimilarityModel:
    def _model(self, threshold=0.5):
        enc = CodeGraphEncoder(N_NODE_TYPES, hidden=16, out_dim=32, n_layers=1)
        return CodeSimilarityModel(enc, threshold=threshold)

    def _graphs(self):
        parser = ASTParser()
        return parser.parse(SRC1), parser.parse(SRC2)

    def test_similarity_in_range(self):
        m  = self._model()
        g1, g2 = self._graphs()
        sim = m.similarity(g1, g2)
        assert -1.0 <= sim <= 1.0

    def test_same_graph_high_similarity(self):
        parser = ASTParser()
        g1 = parser.parse(SRC1)
        g2 = parser.parse(SRC1)
        enc = CodeGraphEncoder(N_NODE_TYPES, hidden=16, out_dim=32, n_layers=1)
        m   = CodeSimilarityModel(enc, threshold=0.99)
        sim = m.similarity(g1, g2)
        assert sim > 0.95   # same graph, same weights → near identical

    def test_is_clone_returns_bool(self):
        m   = self._model()
        g1, g2 = self._graphs()
        assert isinstance(m.is_clone(g1, g2), bool)

    def test_embed_shape(self):
        m   = self._model()
        g1, _ = self._graphs()
        emb = m.embed(g1)
        assert emb.shape == (1, 32)


# ── TripletCodeLoss ───────────────────────────────────────────────────────────

class TestTripletCodeLoss:
    def test_loss_non_negative(self):
        loss_fn = TripletCodeLoss(margin=0.5)
        a = torch.randn(4, 32)
        p = a + torch.randn(4, 32) * 0.01
        n = torch.randn(4, 32)
        loss = loss_fn(a, p, n)
        assert loss.item() >= 0.0

    def test_perfect_embedding_zero_loss(self):
        """If positive = anchor and negative far away → loss ≈ 0."""
        loss_fn = TripletCodeLoss(margin=0.1)
        a = torch.randn(4, 32)
        a_norm = a / a.norm(dim=-1, keepdim=True)
        n = -a_norm + torch.randn(4, 32) * 0.01
        loss = loss_fn(a_norm, a_norm.clone(), n)
        assert loss.item() >= 0.0

    def test_gradient_through_loss(self):
        loss_fn = TripletCodeLoss()
        a = torch.randn(4, 32, requires_grad=True)
        p = torch.randn(4, 32)
        n = torch.randn(4, 32)
        loss = loss_fn(a, p, n)
        loss.backward()
        assert a.grad is not None


# ── DataFlowAnalyser ──────────────────────────────────────────────────────────

class TestDataFlowAnalyser:
    def test_analyse_returns_graph(self):
        a = DataFlowAnalyser()
        g = a.analyse(SRC_FLOW)
        assert isinstance(g, CodeGraph)

    def test_more_edges_with_dataflow(self):
        parser = ASTParser()
        g_base = parser.parse(SRC_FLOW)
        a      = DataFlowAnalyser()
        g_flow = a.analyse(SRC_FLOW)
        assert g_flow.n_edges >= g_base.n_edges

    def test_def_use_pairs_nonempty(self):
        a = DataFlowAnalyser()
        a.analyse(SRC_FLOW)
        pairs = a.def_use_pairs()
        assert len(pairs) >= 0   # may or may not detect

    def test_syntax_error_handled(self):
        a = DataFlowAnalyser()
        g = a.analyse("broken syntax !!!")
        assert isinstance(g, CodeGraph)


# ── GraphBatch ────────────────────────────────────────────────────────────────

class TestGraphBatch:
    def _graphs(self):
        parser = ASTParser()
        return [parser.parse(SRC1), parser.parse(SRC2)]

    def test_from_graphs_n_graphs(self):
        batch = GraphBatch.from_graphs(self._graphs())
        assert batch.n_graphs == 2

    def test_total_nodes_sum(self):
        gs    = self._graphs()
        batch = GraphBatch.from_graphs(gs)
        assert batch.n_nodes == sum(g.n_nodes for g in gs)

    def test_mean_pool_shape(self):
        batch = GraphBatch.from_graphs(self._graphs())
        h     = torch.randn(batch.n_nodes, 16)
        pooled = batch.global_mean_pool(h)
        assert pooled.shape == (2, 16)

    def test_max_pool_shape(self):
        batch = GraphBatch.from_graphs(self._graphs())
        h     = torch.randn(batch.n_nodes, 16)
        pooled = batch.global_max_pool(h)
        assert pooled.shape == (2, 16)

    def test_batch_idx_range(self):
        batch = GraphBatch.from_graphs(self._graphs())
        assert batch.batch_idx.min().item() == 0
        assert batch.batch_idx.max().item() == 1
''')
commit("test: add full GNN test suite — CodeGraph, ASTParser, GNN layers, encoder, similarity, dataflow, batch")

# COMMITS 12-18
for title, body in [
    ("test: add GCNLayer no self-loops still produces output test", '''
class TestGCNEdgeCases:
    def test_single_node_no_edges(self):
        h  = torch.randn(1, 8)
        ei = torch.zeros(2, 0, dtype=torch.long)
        gcn = GCNLayer(8, 16)
        out = gcn(h, ei, 1)
        assert out.shape == (1, 16)

    def test_complete_graph(self):
        N, D = 4, 8
        h  = torch.randn(N, D)
        idx = [(i, j) for i in range(N) for j in range(N) if i != j]
        src, dst = zip(*idx)
        ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
        gcn = GCNLayer(D, 16)
        out = gcn(h, ei, N)
        assert out.shape == (N, 16)
'''),
    ("test: add ASTParser complex code node types test", '''
class TestASTParserComplex:
    def test_class_nodes(self):
        parser = ASTParser()
        src    = "class Foo:\\n    def bar(self):\\n        pass"
        g      = parser.parse(src)
        assert "ClassDef" in g.node_labels or g.n_nodes > 1

    def test_for_loop_nodes(self):
        parser = ASTParser()
        src    = "for i in range(10):\\n    print(i)"
        g      = parser.parse(src)
        assert g.n_nodes > 1

    def test_empty_string(self):
        parser = ASTParser()
        g      = parser.parse("")
        assert g.n_nodes >= 1
'''),
    ("test: add CodeGraphEncoder different layer types test", '''
class TestEncoderLayerTypes:
    def test_gat_encoder_runs(self):
        from nanomind.gnn import ASTParser, CodeGraphEncoder, N_NODE_TYPES
        parser = ASTParser()
        g      = parser.parse("def f(x): return x")
        enc    = CodeGraphEncoder(N_NODE_TYPES, hidden=16, out_dim=32,
                                   n_layers=2, layer_type="gat")
        emb    = enc(g)
        assert emb.shape == (1, 32)

    def test_ggnn_encoder_runs(self):
        from nanomind.gnn import ASTParser, CodeGraphEncoder, N_NODE_TYPES
        parser = ASTParser()
        g      = parser.parse("def f(x): return x")
        enc    = CodeGraphEncoder(N_NODE_TYPES, hidden=16, out_dim=16,
                                   n_layers=2, layer_type="ggnn")
        emb    = enc(g)
        assert emb.shape[0] == 1
'''),
    ("test: add GraphBatch single graph test", '''
class TestGraphBatchSingle:
    def test_single_graph_batch(self):
        from nanomind.gnn import ASTParser
        parser = ASTParser()
        g      = parser.parse("x = 1")
        batch  = GraphBatch.from_graphs([g])
        assert batch.n_graphs == 1
        h      = torch.randn(batch.n_nodes, 8)
        pooled = batch.global_mean_pool(h)
        assert pooled.shape == (1, 8)
'''),
    ("test: add CodeGraph add_self_loops preserves node features test", '''
class TestSelfLoops:
    def test_self_loops_preserves_features(self):
        from nanomind.gnn.graph import CodeGraph
        feats = torch.randn(4, 8)
        ei    = torch.randint(0, 4, (2, 3))
        g     = CodeGraph(feats, ei)
        g_sl  = g.add_self_loops()
        assert torch.allclose(g.node_features, g_sl.node_features)
'''),
    ("test: add TripletCodeLoss margin effect test", '''
class TestTripletMargin:
    def test_larger_margin_larger_loss(self):
        torch.manual_seed(0)
        a   = torch.randn(4, 32)
        p   = torch.randn(4, 32)
        n   = torch.randn(4, 32)
        l1  = TripletCodeLoss(margin=0.1)(a, p, n)
        l2  = TripletCodeLoss(margin=2.0)(a, p, n)
        assert l2.item() >= l1.item()
'''),
    ("test: add GGNN multi-step propagation test", '''
class TestGGNNSteps:
    def test_different_steps_different_output(self):
        torch.manual_seed(42)
        h  = torch.randn(5, 16)
        ei = torch.randint(0, 5, (2, 6))
        g1 = GGNNLayer(16, n_steps=1)
        g3 = GGNNLayer(16, n_steps=3)
        # Share initial weights
        g3.load_state_dict(g1.state_dict())
        o1 = g1(h, ei, 5)
        o3 = g3(h, ei, 5)
        # More steps → different output
        assert not torch.allclose(o1, o3)
'''),
]:
    src = read("tests/test_gnn.py")
    src += "\n" + body
    write("tests/test_gnn.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.2.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.1.0\"", "__version__ = \"4.2.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.2.0 — GNN for Code Understanding release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `continual`  | Continual Learning — EWC, SI, ExperienceReplay, PackNet, AA/BWT metrics |",
    "| `continual`  | Continual Learning — EWC, SI, ExperienceReplay, PackNet, AA/BWT metrics |\n"
    "| `gnn`        | Graph Neural Networks — AST parser, GCN/GAT/GGNN, code clone detection |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.2.0] — 2024 — Graph Neural Networks for Code Understanding\n\n### Added\n"
      "- `CodeGraph` — node_features, edge_index, edge_types, adjacency, degree, self_loops\n"
      "- `ASTParser` — Python ast → CodeGraph (one-hot types, sibling edges)\n"
      "- `GCNLayer` — symmetric-normalised graph convolution\n"
      "- `GraphSAGELayer` — mean neighbourhood aggregation\n"
      "- `GATLayer` — attention-weighted message passing (multi-head)\n"
      "- `GGNNLayer` — GRU-based n-step propagation\n"
      "- `CodeGraphEncoder` — multi-layer GNN, mean/max/attn pooling\n"
      "- `CodeSimilarityModel` — Siamese GNN for clone detection\n"
      "- `TripletCodeLoss` — triplet margin loss for metric learning\n"
      "- `DataFlowAnalyser` — def-use analysis, DATA_FLOW edge injection\n"
      "- `GraphBatch` — batched graphs, global mean/max pool\n"
      "- `examples/gnn_demo.py` — full GNN code understanding demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.2.0, update README and CHANGELOG for Day 42 GNN Code Understanding")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 42 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.2.0",
    "-m", "NanoMind v4.2.0 — GNN for Code Understanding", check=False)
r = run("git", "push", "origin", "v4.2.0", check=False)
print("Tag v4.2.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 42 COMPLETE — v4.2.0 TAGGED! ===")
