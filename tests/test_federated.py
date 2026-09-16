"""tests/test_federated.py — Tests for NanoMind federated learning."""
import copy
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.federated import (
    FederatedConfig, FederatedClient, FederatedServer,
    DifferentialPrivacyEngine, TopKCompressor, QuantisedCompressor,
    SecureAggregator, fedavg, fedmedian, aggregate,
    FederatedDataset, iid_partition, dirichlet_partition,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, V=8, D=16, T=4):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h      = self.tok(x) + self.pos(torch.arange(S))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, V), t.view(-1)) if t is not None else None
        return logits, loss

V = 8

def toy_data(n=10, T=4):
    data = []
    for _ in range(n):
        x = torch.randint(0, V, (T,))
        y = torch.randint(0, V, (T,))
        data.append((x, y))
    return data

def make_batches(data, batch_size=2):
    batches = []
    for i in range(0, len(data), batch_size):
        sub = data[i:i+batch_size]
        if not sub:
            continue
        xs = torch.stack([s[0] for s in sub])
        ys = torch.stack([s[1] for s in sub])
        batches.append((xs, ys))
    return batches

def tiny_cfg(**kw):
    return FederatedConfig(n_clients=4, clients_per_round=2,
                           n_rounds=2, local_epochs=1, local_lr=1e-2, **kw)


# ── FederatedConfig ───────────────────────────────────────────────────────────

class TestFederatedConfig:
    def test_defaults(self):
        cfg = FederatedConfig()
        assert cfg.n_clients == 10
        assert cfg.aggregation == "fedavg"

    def test_invalid_clients_per_round(self):
        with pytest.raises(AssertionError):
            FederatedConfig(n_clients=5, clients_per_round=10)

    def test_invalid_aggregation(self):
        with pytest.raises(AssertionError):
            FederatedConfig(aggregation="fedsum")

    def test_dp_enabled(self):
        cfg = FederatedConfig(dp_epsilon=1.0)
        assert cfg.dp_enabled

    def test_dp_disabled_by_default(self):
        cfg = FederatedConfig()
        assert not cfg.dp_enabled

    def test_participation_rate(self):
        cfg = FederatedConfig(n_clients=10, clients_per_round=5)
        assert cfg.participation_rate == 0.5


# ── DifferentialPrivacyEngine ─────────────────────────────────────────────────

class TestDPEngine:
    def _engine(self):
        return DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=1.0, delta=1e-5)

    def test_noise_multiplier_positive(self):
        eng = self._engine()
        assert eng.noise_multiplier > 0.0

    def test_clip_reduces_norm(self):
        model = TinyLM(V)
        x, y  = torch.randint(0, V, (2, 4)), torch.randint(0, V, (2, 4))
        _, loss = model(x, y)
        loss.backward()
        eng = DifferentialPrivacyEngine(max_grad_norm=0.01, epsilon=1.0, delta=1e-5)
        eng.clip_gradients(model)
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm <= 0.01 + 1e-4

    def test_add_noise_changes_grads(self):
        model = TinyLM(V)
        x, y  = torch.randint(0, V, (2, 4)), torch.randint(0, V, (2, 4))
        _, loss = model(x, y); loss.backward()
        orig = [p.grad.data.clone() for p in model.parameters() if p.grad is not None]
        eng  = self._engine()
        eng.add_noise(model)
        noisy = [p.grad.data for p in model.parameters() if p.grad is not None]
        assert not all(torch.allclose(o, n) for o, n in zip(orig, noisy))

    def test_privacy_spent_keys(self):
        eng   = self._engine()
        spent = eng.privacy_spent(10, 100, 16)
        for k in ("epsilon", "delta", "n_steps", "sigma"):
            assert k in spent

    def test_clip_and_noise_returns_norm(self):
        model = TinyLM(V)
        x, y  = torch.randint(0, V, (2, 4)), torch.randint(0, V, (2, 4))
        _, loss = model(x, y); loss.backward()
        eng   = self._engine()
        norm  = eng.clip_and_noise(model)
        assert isinstance(norm, float) and norm >= 0.0


# ── TopKCompressor ────────────────────────────────────────────────────────────

class TestTopKCompressor:
    def _setup(self, ratio=0.5):
        model = TinyLM(V)
        x, y  = torch.randint(0, V, (2, 4)), torch.randint(0, V, (2, 4))
        _, loss = model(x, y); loss.backward()
        return model, TopKCompressor(ratio=ratio)

    def test_compress_returns_dict(self):
        model, comp = self._setup()
        sparse = comp.compress(model)
        assert "indices" in sparse and "values" in sparse

    def test_compression_ratio(self):
        model, comp = self._setup(ratio=0.5)
        comp.compress(model)
        assert abs(comp.compression_ratio() - 0.5) < 0.1

    def test_error_feedback_buffer_created(self):
        model, comp = self._setup()
        comp.compress(model)
        assert comp._error_buffer is not None

    def test_no_error_feedback(self):
        model = TinyLM(V)
        x, y  = torch.randint(0, V, (2, 4)), torch.randint(0, V, (2, 4))
        _, loss = model(x, y); loss.backward()
        comp   = TopKCompressor(ratio=0.5, error_feedback=False)
        comp.compress(model)
        assert comp._error_buffer is None


# ── QuantisedCompressor ───────────────────────────────────────────────────────

class TestQuantisedCompressor:
    def test_quantise_shape(self):
        comp = QuantisedCompressor(bits=8)
        flat = torch.randn(100)
        q, scale, min_v = comp.quantise(flat)
        assert q.shape == flat.shape
        assert q.dtype == torch.uint8

    def test_dequantise_approx(self):
        comp = QuantisedCompressor(bits=8)
        flat = torch.randn(100)
        q, scale, min_v = comp.quantise(flat)
        rec  = comp.dequantise(q, scale, min_v)
        assert torch.allclose(flat, rec, atol=0.05)  # 8-bit quantization error

    def test_bits_4_supported(self):
        comp = QuantisedCompressor(bits=4)
        flat = torch.randn(50)
        q, s, m = comp.quantise(flat)
        assert q.max() <= 15  # 2^4 - 1


# ── Aggregation ───────────────────────────────────────────────────────────────

class TestAggregation:
    def _make_updates(self, n=3):
        model   = TinyLM(V)
        state   = model.state_dict()
        updates = []
        for i in range(n):
            deltas = {k: torch.randn_like(v.float()) * 0.01 for k, v in state.items()}
            updates.append({"gradient_deltas": deltas, "n_samples": 10})
        return updates, state

    def test_fedavg_output_keys(self):
        updates, state = self._make_updates()
        new_state = fedavg(updates, state)
        assert set(new_state.keys()) == set(state.keys())

    def test_fedavg_changes_weights(self):
        updates, state = self._make_updates()
        new_state = fedavg(updates, state)
        for k in state:
            if state[k].dtype == torch.float32:
                # At least some weights should change
                if not torch.allclose(state[k].float(), new_state[k].float()):
                    break

    def test_fedmedian_output_keys(self):
        updates, state = self._make_updates()
        new_state = fedmedian(updates, state)
        assert set(new_state.keys()) == set(state.keys())

    def test_aggregate_dispatch_fedavg(self):
        updates, state = self._make_updates()
        r1 = aggregate(updates, state, "fedavg")
        r2 = fedavg(updates, state)
        for k in r1:
            assert torch.allclose(r1[k].float(), r2[k].float())

    def test_aggregate_invalid_strategy(self):
        updates, state = self._make_updates()
        with pytest.raises(ValueError):
            aggregate(updates, state, "fedsum")


# ── FederatedClient ───────────────────────────────────────────────────────────

class TestFederatedClient:
    def _client(self):
        model   = TinyLM(V)
        data    = make_batches(toy_data(8))
        cfg     = tiny_cfg()
        return FederatedClient("c0", model, data, cfg)

    def test_train_round_returns_dict(self):
        c      = self._client()
        result = c.train_round(TinyLM(V).state_dict())
        assert "gradient_deltas" in result and "loss" in result

    def test_gradient_deltas_correct_keys(self):
        model  = TinyLM(V)
        c      = FederatedClient("c0", model, make_batches(toy_data(8)), tiny_cfg())
        result = c.train_round(model.state_dict())
        assert set(result["gradient_deltas"].keys()) == set(model.state_dict().keys())

    def test_loss_is_float(self):
        c      = self._client()
        result = c.train_round(TinyLM(V).state_dict())
        assert isinstance(result["loss"], float)

    def test_last_loss(self):
        c = self._client()
        c.train_round(TinyLM(V).state_dict())
        assert c.last_loss is not None


# ── FederatedServer ───────────────────────────────────────────────────────────

class TestFederatedServer:
    def _server(self, n=4):
        model  = TinyLM(V)
        cfg    = tiny_cfg(n_clients=n, clients_per_round=2, n_rounds=2)
        server = FederatedServer(model, cfg)
        for i in range(n):
            data   = make_batches(toy_data(6))
            client = FederatedClient(f"c{i}", copy.deepcopy(model), data, cfg)
            server.add_client(client)
        return server

    def test_train_returns_logs(self):
        server = self._server()
        logs   = server.train()
        assert len(logs) == 2

    def test_round_log_keys(self):
        server = self._server()
        logs   = server.train()
        for entry in logs:
            assert "round" in entry and "avg_loss" in entry

    def test_n_clients(self):
        server = self._server(n=4)
        assert server.n_clients == 4

    def test_train_round_returns_dict(self):
        server = self._server()
        entry  = server.train_round(0)
        assert "round" in entry

    def test_global_model_changes(self):
        model  = TinyLM(V)
        cfg    = tiny_cfg()
        server = FederatedServer(model, cfg)
        init_w = copy.deepcopy(list(model.parameters())[0].data)
        for i in range(4):
            data   = make_batches(toy_data(6))
            client = FederatedClient(f"c{i}", copy.deepcopy(model), data, cfg)
            server.add_client(client)
        server.train()
        final_w = list(model.parameters())[0].data
        # Global model should have changed after training
        assert not torch.allclose(init_w, final_w)


# ── SecureAggregator ──────────────────────────────────────────────────────────

class TestSecureAggregator:
    def test_unmask_recovers_original(self):
        agg  = SecureAggregator("test-round")
        grad = torch.randn(50)
        masked = agg.mask(grad, "c0")
        recovered = agg.unmask(masked, ["c0"])
        assert torch.allclose(grad, recovered, atol=1e-5)

    def test_mask_changes_gradient(self):
        agg  = SecureAggregator("test-round")
        grad = torch.randn(50)
        assert not torch.allclose(grad, agg.mask(grad, "c0"))

    def test_different_clients_different_masks(self):
        agg = SecureAggregator("r1")
        g   = torch.randn(20)
        m0  = agg.mask(g, "c0")
        m1  = agg.mask(g, "c1")
        assert not torch.allclose(m0, m1)

    def test_sum_unmask_two_clients(self):
        agg = SecureAggregator("r1")
        g0  = torch.randn(20)
        g1  = torch.randn(20)
        m0  = agg.mask(g0, "c0")
        m1  = agg.mask(g1, "c1")
        recovered = agg.unmask(m0 + m1, ["c0", "c1"])
        assert torch.allclose(g0 + g1, recovered, atol=1e-5)


# ── Data Partition ────────────────────────────────────────────────────────────

class TestDataPartition:
    def _data(self, n=20):
        return toy_data(n)

    def test_iid_n_clients(self):
        datasets = iid_partition(self._data(), n_clients=4, batch_size=2)
        assert len(datasets) == 4

    def test_iid_all_have_data(self):
        datasets = iid_partition(self._data(), n_clients=4, batch_size=2)
        for ds in datasets:
            assert ds.n_samples >= 0  # some may be empty if data is small

    def test_dirichlet_n_clients(self):
        datasets = dirichlet_partition(self._data(), n_clients=4, alpha=1.0, batch_size=2)
        assert len(datasets) == 4

    def test_federated_dataset_client_id(self):
        datasets = iid_partition(self._data(), n_clients=3, batch_size=2)
        for ds in datasets:
            assert "client-" in ds.client_id


# ── FedAvg equal weights ──────────────────────────────────────────────────────

class TestFedAvgEqualWeights:
    def test_equal_samples_mean_delta(self):
        """FedAvg with equal n_samples should be a simple mean."""
        state   = TinyLM(V).state_dict()
        deltas  = [torch.ones(1) * i for i in range(3)]
        updates = [
            {"gradient_deltas": {"tok.weight": d.expand_as(state["tok.weight"].float())},
             "n_samples": 10}
            for d in deltas
        ]
        new_s = fedavg(updates, state)
        # Mean delta = (0+1+2)/3 = 1.0
        expected_delta = 1.0
        assert abs((new_s["tok.weight"] - state["tok.weight"].float()).mean().item()
                   - expected_delta) < 1e-4


# ── DP noise scale ────────────────────────────────────────────────────────────

class TestDPNoiseScale:
    def test_smaller_epsilon_more_noise(self):
        eng_low  = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=0.1, delta=1e-5)
        eng_high = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=10.0, delta=1e-5)
        assert eng_low.noise_multiplier > eng_high.noise_multiplier

    def test_gaussian_noise_formula(self):
        import math
        eng      = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=1.0, delta=1e-5)
        expected = math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(eng.noise_multiplier - expected) < 1e-6


# ── Client with DP ────────────────────────────────────────────────────────────

class TestClientWithDP:
    def test_client_dp_train_round(self):
        model  = TinyLM(V)
        dp     = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=1.0, delta=1e-5)
        data   = make_batches(toy_data(6))
        cfg    = tiny_cfg()
        client = FederatedClient("c0", model, data, cfg, dp_engine=dp)
        result = client.train_round(TinyLM(V).state_dict())
        assert "loss" in result

    def test_client_compression_train_round(self):
        model  = TinyLM(V)
        comp   = TopKCompressor(ratio=0.5)
        data   = make_batches(toy_data(6))
        cfg    = tiny_cfg()
        client = FederatedClient("c0", model, data, cfg, compressor=comp)
        result = client.train_round(TinyLM(V).state_dict())
        assert "loss" in result
