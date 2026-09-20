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
print("
── AST Parsing ──")
parser = ASTParser()
src1   = "def add(a, b):\n    return a + b"
src2   = "def multiply(x, y):\n    result = x * y\n    return result"
g1     = parser.parse(src1)
g2     = parser.parse(src2)
print(f"  'add':      {g1.n_nodes} nodes, {g1.n_edges} edges")
print(f"  'multiply': {g2.n_nodes} nodes, {g2.n_edges} edges")
print(f"  Node labels (first 5): {g1.node_labels[:5]}")
print(f"  Degree: {g1.degree().tolist()}")

# ── GNN Layers ────────────────────────────────────────────────────────────────
print("
── GNN Layers ──")
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
print("
── CodeGraphEncoder ──")
encoder = CodeGraphEncoder(in_dim=N_NODE_TYPES, hidden=32, out_dim=64,
                            n_layers=2, layer_type="gcn", pooling="mean")
with torch.no_grad():
    emb1 = encoder(g1)
    emb2 = encoder(g2)
print(f"  Graph 'add' embedding:      {tuple(emb1.shape)}")
print(f"  Graph 'multiply' embedding: {tuple(emb2.shape)}")
print(f"  Encoder n_params: {encoder.n_params:,}")

# ── Code Similarity ───────────────────────────────────────────────────────────
print("
── Code Clone Detection ──")
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
print("
── Triplet Loss ──")
triplet_loss = TripletCodeLoss(margin=0.5)
a = torch.randn(4, 64)
p = a + torch.randn(4, 64) * 0.1   # positive: slightly perturbed
n = torch.randn(4, 64)              # negative: random
loss = triplet_loss(a, p, n)
print(f"  Triplet loss: {loss.item():.4f}")

# ── Data Flow Analysis ────────────────────────────────────────────────────────
print("
── Data Flow Analysis ──")
analyser  = DataFlowAnalyser()
src_flow  = "def f(x):\n    y = x + 1\n    z = y * 2\n    return z"
g_flow    = analyser.analyse(src_flow)
print(f"  Graph with dataflow: {g_flow.n_nodes} nodes, {g_flow.n_edges} edges")
print(f"  Def-use pairs: {analyser.def_use_pairs()}")

# ── Graph Batching ────────────────────────────────────────────────────────────
print("
── Graph Batching ──")
batch = GraphBatch.from_graphs([g1, g2, g_flow])
print(f"  Batch: {batch.n_nodes} total nodes, {batch.n_edges} edges, {batch.n_graphs} graphs")
h_all = torch.randn(batch.n_nodes, 32)
pooled = batch.global_mean_pool(h_all)
print(f"  Mean-pooled embeddings: {tuple(pooled.shape)}")
print("
GNN demo complete!")
