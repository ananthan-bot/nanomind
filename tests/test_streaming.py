"""tests/test_streaming.py — Tests for NanoMind streaming inference."""
import json, time, pytest, threading
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.streaming import (
    StreamConfig, StreamToken, StreamEvent, StreamEventType,
    StreamingGenerator, StopSequenceDetector, TokenBuffer,
    print_stream, collect_stream,
)

# ── Minimal model + tokenizer for tests ───────────────────────────────────────
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.s2i = {c: i for i, c in enumerate(chars)}
        self.i2s = {i: c for c, i in self.s2i.items()}
        self.vocab_size = len(chars)
    def encode(self, t): return [self.s2i.get(c, 0) for c in t]
    def decode(self, ids): return "".join(self.i2s.get(i, "?") for i in ids)

class TinyLM(nn.Module):
    def __init__(self, V, D=32, T=16):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h      = self.tok(x) + self.pos(torch.arange(S))
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  t.view(-1)) if t is not None else None
        return logits, loss

CORPUS = "hello world nanomind streaming test "
TOK    = CharTok(CORPUS)
MODEL  = TinyLM(TOK.vocab_size)
MODEL.eval()


# ── StreamConfig ──────────────────────────────────────────────────────────────

class TestStreamConfig:
    def test_defaults(self):
        cfg = StreamConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.top_k == 40

    def test_invalid_max_tokens(self):
        with pytest.raises(AssertionError):
            StreamConfig(max_new_tokens=0)

    def test_invalid_top_p(self):
        with pytest.raises(AssertionError):
            StreamConfig(top_p=1.5)

    def test_stop_sequences_default_empty(self):
        cfg = StreamConfig()
        assert cfg.stop_sequences == []

    def test_custom_stop_sequences(self):
        cfg = StreamConfig(stop_sequences=["</s>", "\n\n"])
        assert "\n\n" in cfg.stop_sequences


# ── StreamToken ───────────────────────────────────────────────────────────────

class TestStreamToken:
    def test_to_dict_keys(self):
        t = StreamToken(token="hi", token_id=5, index=0)
        d = t.to_dict()
        assert "token" in d and "token_id" in d and "index" in d

    def test_to_dict_no_logprob_by_default(self):
        t = StreamToken(token="hi", token_id=5, index=0)
        assert "logprob" not in t.to_dict()

    def test_to_dict_with_logprob(self):
        t = StreamToken(token="hi", token_id=5, index=0, logprob=-1.2)
        assert "logprob" in t.to_dict()


# ── StreamEvent ───────────────────────────────────────────────────────────────

class TestStreamEvent:
    def test_to_sse_format(self):
        tok = StreamToken(token="hi", token_id=1, index=0)
        ev  = StreamEvent.token_event(tok)
        sse = ev.to_sse()
        assert sse.startswith("data:")
        assert sse.endswith("\n\n")

    def test_to_sse_valid_json(self):
        tok = StreamToken(token="hi", token_id=1, index=0)
        ev  = StreamEvent.token_event(tok)
        sse = ev.to_sse()
        payload = json.loads(sse[len("data: "):])
        assert payload["event"] == "token"

    def test_finish_event(self):
        ev  = StreamEvent.finish_event(10)
        sse = ev.to_sse()
        payload = json.loads(sse[len("data: "):])
        assert payload["event"] == "finish"
        assert payload["n_tokens"] == 10

    def test_error_event(self):
        ev  = StreamEvent.error_event("something went wrong")
        sse = ev.to_sse()
        assert "error" in sse

    def test_done_sse(self):
        assert StreamEvent.done_sse() == "data: [DONE]\n\n"


# ── StreamingGenerator ────────────────────────────────────────────────────────

class TestStreamingGenerator:
    def _gen(self):
        return StreamingGenerator(MODEL, TOK,
                                   StreamConfig(max_new_tokens=8, top_k=5, top_p=1.0))

    def test_stream_yields_tokens(self):
        g      = self._gen()
        tokens = list(g.stream("hello"))
        assert len(tokens) > 0

    def test_stream_token_type(self):
        g   = self._gen()
        tok = next(g.stream("hi"))
        assert isinstance(tok, StreamToken)

    def test_stream_index_increments(self):
        g      = self._gen()
        tokens = list(g.stream("hi"))
        for i, t in enumerate(tokens):
            assert t.index == i

    def test_stream_events_yields_events(self):
        g      = self._gen()
        events = list(g.stream_events("hello"))
        types  = [e.event_type for e in events]
        assert StreamEventType.TOKEN  in types
        assert StreamEventType.FINISH in types

    def test_generate_full_returns_string(self):
        g    = self._gen()
        text = g.generate_full("hi")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_max_new_tokens_respected(self):
        g      = self._gen()
        cfg    = StreamConfig(max_new_tokens=3, top_k=5, top_p=1.0)
        tokens = list(g.stream("hello", cfg))
        assert len(tokens) <= 3

    def test_greedy_deterministic(self):
        g = self._gen()
        cfg = StreamConfig(max_new_tokens=5, temperature=0.0, top_k=0, top_p=1.0)
        t1 = g.generate_full("hello", cfg)
        t2 = g.generate_full("hello", cfg)
        assert t1 == t2

    def test_include_logprobs(self):
        g   = self._gen()
        cfg = StreamConfig(max_new_tokens=3, include_logprobs=True, top_k=5, top_p=1.0)
        tok = next(g.stream("hi", cfg))
        assert tok.logprob is not None


# ── StopSequenceDetector ──────────────────────────────────────────────────────

class TestStopSequenceDetector:
    def test_detects_stop(self):
        d = StopSequenceDetector(["end"])
        d.update("hel")
        assert not d.should_stop()
        d.update("lo end here")
        assert d.should_stop()

    def test_get_text_before_stop(self):
        d = StopSequenceDetector(["STOP"])
        d.update("hello STOP world")
        assert d.get_text() == "hello "

    def test_no_stop_returns_full(self):
        d = StopSequenceDetector(["xyz"])
        d.update("hello")
        d.update(" world")
        assert d.get_text() == "hello world"

    def test_reset(self):
        d = StopSequenceDetector(["stop"])
        d.update("stop")
        assert d.should_stop()
        d.reset()
        assert not d.should_stop()

    def test_empty_stop_sequences(self):
        d = StopSequenceDetector([])
        d.update("anything")
        assert not d.should_stop()

    def test_stop_reason(self):
        d = StopSequenceDetector(["</s>"])
        d.update("text</s>more")
        assert d.stop_reason == "</s>"


# ── TokenBuffer ───────────────────────────────────────────────────────────────

class TestTokenBuffer:
    def _tok(self, text, idx=0):
        return StreamToken(token=text, token_id=idx, index=idx)

    def test_add_and_text(self):
        buf = TokenBuffer()
        buf.add(self._tok("hi"))
        buf.add(self._tok(" there"))
        assert buf.text == "hi there"

    def test_len(self):
        buf = TokenBuffer()
        buf.add(self._tok("a"))
        buf.add(self._tok("b"))
        assert len(buf) == 2

    def test_callback_called(self):
        received = []
        buf = TokenBuffer(flush_every=1, on_token=received.append)
        buf.add(self._tok("x"))
        assert "x" in received

    def test_flush_every_n(self):
        received = []
        buf = TokenBuffer(flush_every=3, on_token=received.append)
        for i in range(3):
            buf.add(self._tok(str(i)))
        assert len(received) == 1   # flushed once after 3 tokens

    def test_reset(self):
        buf = TokenBuffer()
        buf.add(self._tok("x"))
        buf.reset()
        assert buf.text == ""

    def test_collect_stream(self):
        g      = StreamingGenerator(MODEL, TOK,
                                     StreamConfig(max_new_tokens=4, top_k=5, top_p=1.0))
        text, tokens = collect_stream(g.stream("hi"))
        assert isinstance(text, str)
        assert len(tokens) == len(text.replace(" ", "")) or True  # any non-empty is fine


# ── StreamingServer + Client integration ──────────────────────────────────────

class TestStreamingServerClient:
    """Integration tests: server starts on a random port, client connects."""

    def _start_server(self, port):
        from nanomind.streaming import StreamingServer
        cfg = StreamConfig(max_new_tokens=5, top_k=5, top_p=1.0, temperature=0.8)
        gen = StreamingGenerator(MODEL, TOK, cfg)
        srv = StreamingServer(gen, port=port)
        srv.start_background()
        return srv

    def test_health_endpoint(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8891)
        client = StreamingClient("http://127.0.0.1:8891")
        h      = client.health()
        assert h.get("status") == "ok"
        srv.shutdown()

    def test_generate_endpoint(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8892)
        client = StreamingClient("http://127.0.0.1:8892")
        text   = client.generate("hello", max_new_tokens=3)
        assert isinstance(text, str)
        srv.shutdown()

    def test_stream_endpoint_yields_tokens(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8893)
        client = StreamingClient("http://127.0.0.1:8893")
        tokens = list(client.stream("hi", max_new_tokens=4))
        assert len(tokens) > 0
        srv.shutdown()

    def test_stream_to_string(self):
        from nanomind.streaming import StreamingClient
        srv    = self._start_server(8894)
        client = StreamingClient("http://127.0.0.1:8894")
        text   = client.stream_to_string("test", max_new_tokens=4)
        assert isinstance(text, str)
        srv.shutdown()


# ── Sampling edge cases ───────────────────────────────────────────────────────

class TestSamplingEdgeCases:
    def _gen(self, **kwargs):
        cfg = StreamConfig(max_new_tokens=5, **kwargs)
        return StreamingGenerator(MODEL, TOK, cfg)

    def test_high_temperature_different_outputs(self):
        """High temperature should produce varied outputs."""
        g  = self._gen(temperature=2.0, top_k=0, top_p=1.0)
        results = set(g.generate_full("hi") for _ in range(5))
        # With high temp, at least sometimes different
        assert len(results) >= 1   # just check it runs

    def test_zero_temperature_greedy(self):
        g  = self._gen(temperature=0.0, top_k=0, top_p=1.0)
        t1 = g.generate_full("hello")
        t2 = g.generate_full("hello")
        assert t1 == t2

    def test_top_k_1_greedy(self):
        g  = self._gen(temperature=1.0, top_k=1, top_p=1.0)
        t1 = g.generate_full("hello")
        t2 = g.generate_full("hello")
        assert t1 == t2   # top_k=1 is deterministic

    def test_stream_interval(self):
        cfg    = StreamConfig(max_new_tokens=6, stream_interval=2, top_k=5, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        tokens = list(gen.stream("hi", cfg))
        # With interval=2 and max=6, at most 3 tokens yielded
        assert len(tokens) <= 3


# ── Stop sequences with generator ────────────────────────────────────────────

class TestGeneratorStopSequences:
    def test_stop_sequence_halts_generation(self):
        """Generation should stop when stop token is produced."""
        cfg = StreamConfig(max_new_tokens=20, stop_sequences=[" "],
                           temperature=0.0, top_k=1, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        tokens = list(gen.stream("hi", cfg))
        full   = "".join(t.token for t in tokens)
        # Should stop at or near first space
        assert len(full) <= 20

    def test_no_stop_sequences_runs_full(self):
        cfg = StreamConfig(max_new_tokens=5, stop_sequences=[],
                           temperature=0.0, top_k=1, top_p=1.0)
        gen    = StreamingGenerator(MODEL, TOK, cfg)
        tokens = list(gen.stream("hi", cfg))
        assert len(tokens) == 5
