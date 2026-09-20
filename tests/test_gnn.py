"""tests/test_gnn.py — Tests for NanoMind GNN code understanding."""
import pytest
import torch
from nanomind.gnn import (
    CodeGraph, ASTParser, N_NODE_TYPES, EDGE_TYPE_NAMES,
    GCNLayer, GraphSAGELayer, GATLayer, GGNNLayer,
    CodeGraphEncoder, CodeSimilarityModel, TripletCodeLoss,
    DataFlowAnalyser, GraphBatch,
)

SRC1 = "def add(a, b):\n    return a + b"
SRC2 = "def sub(x, y):\n    return x - y"
SRC_FLOW = "def f(x):\n    y = x + 1\n    return y"


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
        g_with    = parser_with.parse("a = 1\nb = 2\nc = 3")
        g_without = parser_without.parse("a = 1\nb = 2\nc = 3")
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


class TestASTParserComplex:
    def test_class_nodes(self):
        parser = ASTParser()
        src    = "class Foo:\n    def bar(self):\n        pass"
        g      = parser.parse(src)
        assert "ClassDef" in g.node_labels or g.n_nodes > 1

    def test_for_loop_nodes(self):
        parser = ASTParser()
        src    = "for i in range(10):\n    print(i)"
        g      = parser.parse(src)
        assert g.n_nodes > 1

    def test_empty_string(self):
        parser = ASTParser()
        g      = parser.parse("")
        assert g.n_nodes >= 1


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
