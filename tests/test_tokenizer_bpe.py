"""
tests/test_tokenizer_bpe.py — Tests for BPETokenizer.
"""

import pytest
from nanomind.tokenizer.bpe import BPETokenizer
from nanomind.tokenizer.factory import get_tokenizer

CORPUS = (
    "the cat sat on the mat. "
    "the cat ate the rat. "
    "the rat sat on the mat. "
    "hello world hello hello world "
) * 20   # Repeat to give BPE enough frequency signal


@pytest.fixture
def tok() -> BPETokenizer:
    return BPETokenizer().train(CORPUS, vocab_size=150)


# ── Training ──────────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_self(self):
        t = BPETokenizer()
        result = t.train("hello world", vocab_size=50)
        assert result is t

    def test_vocab_size_respected(self, tok):
        # May be slightly less than target if corpus is small
        assert tok.vocab_size <= 150

    def test_has_special_tokens(self, tok):
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3

    def test_merges_learned(self, tok):
        assert tok.num_merges > 0

    def test_not_trained_raises(self):
        t = BPETokenizer()
        with pytest.raises(RuntimeError):
            _ = t.vocab_size


# ── Encode / Decode ───────────────────────────────────────────────────────────

class TestEncodeDecode:
    def test_encode_returns_list_of_ints(self, tok):
        ids = tok.encode("the cat")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_nonempty(self, tok):
        assert len(tok.encode("hello")) > 0

    def test_decode_returns_string(self, tok):
        ids = tok.encode("the cat")
        assert isinstance(tok.decode(ids), str)

    def test_common_words_roundtrip(self, tok):
        # Words seen in training should survive encode->decode
        for word in ["the", "cat", "sat", "mat"]:
            ids = tok.encode(word)
            decoded = tok.decode(ids)
            assert word in decoded

    def test_empty_string(self, tok):
        assert tok.decode(tok.encode("")) == "" or tok.encode("") == []
