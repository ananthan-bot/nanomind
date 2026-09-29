"""tests/test_serving.py — Tests for NanoMind serving package."""
import time
import pytest
import torch
import torch.nn as nn
from nanomind.serving import (
    KVBlock, KVCacheManager,
    InferenceRequest, SamplingParams, RequestStatus,
    ContinuousBatchingScheduler, SchedulerOutput,
    PrefixCache,
    LLMEngine, EngineConfig, EngineStats,
    BenchmarkConfig, make_requests, run_benchmark,
)

V = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── KVCacheManager ────────────────────────────────────────────────────────────

class TestKVCacheManager:
    def _mgr(self):
        return KVCacheManager(n_blocks=32, block_size=8, n_layers=1, n_kv_heads=1, d_head=8)

    def test_allocate_reduces_free(self):
        m = self._mgr()
        n = m.n_free_blocks
        m.allocate("s1", n_tokens=8)
        assert m.n_free_blocks < n

    def test_free_restores_blocks(self):
        m = self._mgr()
        m.allocate("s1", 8)
        before = m.n_free_blocks
        m.free("s1")
        assert m.n_free_blocks > before

    def test_can_allocate_true(self):
        m = self._mgr()
        assert m.can_allocate(8)

    def test_can_allocate_false_too_many(self):
        m = self._mgr()
        assert not m.can_allocate(9999)

    def test_stats_keys(self):
        m = self._mgr()
        s = m.stats()
        for k in ("total_blocks", "free_blocks", "utilisation"):
            assert k in s

    def test_copy_on_write(self):
        m = self._mgr()
        m.allocate("src", 8)
        m.copy_on_write("src", "dst")
        assert len(m.get_blocks("dst")) > 0

    def test_double_free_no_error(self):
        m = self._mgr()
        m.allocate("s1", 8)
        m.free("s1")
        m.free("s1")  # should not raise

    def test_get_blocks_returns_list(self):
        m = self._mgr()
        m.allocate("s1", 8)
        blocks = m.get_blocks("s1")
        assert isinstance(blocks, list)


# ── InferenceRequest ──────────────────────────────────────────────────────────

class TestInferenceRequest:
    def test_add_token_updates_list(self):
        r = InferenceRequest("r1", [1, 2, 3], SamplingParams(max_new_tokens=4))
        r.add_token(7)
        assert r.generated_tokens == [7]

    def test_finishes_at_max_tokens(self):
        r = InferenceRequest("r1", [1], SamplingParams(max_new_tokens=2))
        r.add_token(1); r.add_token(2)
        assert r.is_finished

    def test_not_finished_before_max(self):
        r = InferenceRequest("r1", [1], SamplingParams(max_new_tokens=4))
        r.add_token(1)
        assert not r.is_finished

    def test_stop_token(self):
        r = InferenceRequest("r1", [1], SamplingParams(max_new_tokens=10, stop_token_id=99))
        r.add_token(7)
        assert not r.is_finished
        r.add_token(99)
        assert r.is_finished

    def test_output_ids(self):
        r = InferenceRequest("r1", [1, 2], SamplingParams())
        r.add_token(3)
        assert r.output_ids == [1, 2, 3]

    def test_abort(self):
        r = InferenceRequest("r1", [1], SamplingParams())
        r.abort()
        assert r.status == RequestStatus.ABORTED
        assert r.is_finished


# ── ContinuousBatchingScheduler ───────────────────────────────────────────────

class TestScheduler:
    def _sched(self):
        kv = KVCacheManager(64, block_size=8, n_layers=1, n_kv_heads=1, d_head=8)
        return ContinuousBatchingScheduler(kv, max_batch_size=4)

    def test_add_and_step_admits(self):
        s = self._sched()
        r = InferenceRequest("r1", [1, 2, 3], SamplingParams(max_new_tokens=4))
        s.add_request(r)
        out = s.step()
        assert len(out.running) >= 1

    def test_queue_shrinks_after_admit(self):
        s = self._sched()
        for i in range(3):
            s.add_request(InferenceRequest(f"r{i}", [1, 2], SamplingParams(max_new_tokens=2)))
        s.step()
        assert s.queue_len() < 3

    def test_has_work_true(self):
        s = self._sched()
        s.add_request(InferenceRequest("r1", [1], SamplingParams()))
        assert s.has_work()

    def test_stats_keys(self):
        s = self._sched()
        d = s.stats()
        assert "running" in d and "waiting" in d


# ── PrefixCache ───────────────────────────────────────────────────────────────

class TestPrefixCache:
    def test_insert_and_lookup_hit(self):
        c = PrefixCache()
        c.insert([1, 2, 3, 4], [0, 1])
        n, blocks = c.lookup([1, 2, 3, 4, 5])
        assert n >= 4
        assert blocks == [0, 1]

    def test_lookup_miss(self):
        c = PrefixCache()
        n, blocks = c.lookup([9, 9, 9])
        assert n == 0

    def test_stats_hit_rate(self):
        c = PrefixCache()
        c.insert([1, 2, 3], [0])
        c.lookup([1, 2, 3, 4])  # hit
        c.lookup([9, 8])         # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert 0 < s["hit_rate"] < 1

    def test_evict_lru(self):
        c = PrefixCache()
        c.insert([1, 2], [0])
        evicted = c.evict_lru(n=1)
        assert evicted >= 0   # may be 0 if ref_count > 0


# ── LLMEngine ─────────────────────────────────────────────────────────────────

class TestLLMEngine:
    def _engine(self):
        cfg = EngineConfig(max_batch_size=4, n_kv_blocks=64, block_size=8,
                            n_layers=1, n_kv_heads=1, d_head=8, vocab_size=V)
        return LLMEngine(TinyLM(), cfg)

    def test_submit_and_run(self):
        e = self._engine()
        e.submit("r1", [1, 2, 3], SamplingParams(max_new_tokens=4))
        finished = e.run_until_done()
        assert len(finished) == 1
        assert finished[0].n_generated == 4

    def test_multiple_requests(self):
        e = self._engine()
        for i in range(3):
            e.submit(f"r{i}", [i+1, i+2], SamplingParams(max_new_tokens=3))
        finished = e.run_until_done()
        assert len(finished) == 3

    def test_stats_populated(self):
        e = self._engine()
        e.submit("r1", [1, 2], SamplingParams(max_new_tokens=2))
        e.run_until_done()
        s = e.engine_stats()
        assert s["n_tokens_generated"] > 0

    def test_throughput_positive(self):
        e = self._engine()
        e.submit("r1", [1, 2, 3], SamplingParams(max_new_tokens=3))
        e.run_until_done()
        assert e.stats.throughput > 0


class TestKVOOM:
    def test_oom_raises(self):
        from nanomind.serving import KVCacheManager
        kv = KVCacheManager(n_blocks=2, block_size=8,
                             n_layers=1, n_kv_heads=1, d_head=4)
        kv.allocate("s1", 8)
        kv.allocate("s2", 8)
        with pytest.raises(MemoryError):
            kv.allocate("s3", 8)


class TestRequestLengths:
    def test_prompt_len(self):
        from nanomind.serving import InferenceRequest, SamplingParams
        r = InferenceRequest("r", [1, 2, 3, 4], SamplingParams())
        assert r.prompt_len == 4

    def test_total_len(self):
        from nanomind.serving import InferenceRequest, SamplingParams
        r = InferenceRequest("r", [1, 2], SamplingParams(max_new_tokens=4))
        r.add_token(5)
        assert r.total_len == 3   # 2 prompt + 1 generated


class TestSamplingParams:
    def test_to_dict_keys(self):
        from nanomind.serving import SamplingParams
        sp = SamplingParams(max_new_tokens=8, temperature=0.7)
        d  = sp.to_dict()
        assert "max_new_tokens" in d
        assert "temperature"    in d


class TestPrefixMultiInsert:
    def test_two_different_prefixes(self):
        from nanomind.serving import PrefixCache
        c = PrefixCache()
        c.insert([1, 2, 3], [0])
        c.insert([4, 5, 6], [1])
        n1, b1 = c.lookup([1, 2, 3, 9])
        n2, b2 = c.lookup([4, 5, 6, 9])
        assert n1 >= 3 and n2 >= 3
        assert b1 == [0] and b2 == [1]


class TestSchedulerMaxBatch:
    def test_max_batch_size(self):
        from nanomind.serving import KVCacheManager, ContinuousBatchingScheduler
        from nanomind.serving import InferenceRequest, SamplingParams
        kv   = KVCacheManager(64, 8, 1, 1, 8)
        sched = ContinuousBatchingScheduler(kv, max_batch_size=2)
        for i in range(5):
            sched.add_request(InferenceRequest(f"r{i}", [1, 2], SamplingParams(max_new_tokens=4)))
        out = sched.step()
        assert len(out.running) <= 2   # max_batch_size respected
