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


# ── Merge rules ───────────────────────────────────────────────────────────────

class TestMergeRules:
    def test_get_word_freqs(self):
        freqs = BPETokenizer._get_word_freqs("hello hello world")
        assert any("h e l l o" in k for k in freqs)
        assert freqs.get("h e l l o</w>", 0) == 2

    def test_get_pairs(self):
        freqs = {"h e l l o</w>": 2}
        pairs = BPETokenizer._get_pairs(freqs)
        assert ("h", "e") in pairs
        assert pairs[("h", "e")] == 2
        assert ("l", "l") in pairs

    def test_merge_reduces_symbol_count(self):
        freqs = {"h e l l o</w>": 1}
        new_freqs = BPETokenizer._merge_pair(("h", "e"), freqs)
        # "h e" should now be merged into "he"
        assert any("he" in k for k in new_freqs)

    def test_frequent_pair_is_merged_first(self, tok):
        # After training on CORPUS, "th" should be one of the first merges
        # since "the" appears frequently
        merged_tokens = [a + b for a, b in tok._merges[:20]]
        assert any("th" in t for t in merged_tokens)


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_file(self, tok, tmp_path):
        p = tmp_path / "bpe.json"
        tok.save(str(p))
        assert p.exists()

    def test_load_roundtrip_vocab_size(self, tok, tmp_path):
        p = tmp_path / "bpe.json"
        tok.save(str(p))
        loaded = BPETokenizer.load(str(p))
        assert loaded.vocab_size == tok.vocab_size

    def test_load_roundtrip_merges(self, tok, tmp_path):
        p = tmp_path / "bpe.json"
        tok.save(str(p))
        loaded = BPETokenizer.load(str(p))
        assert loaded.num_merges == tok.num_merges

    def test_load_roundtrip_encode(self, tok, tmp_path):
        p = tmp_path / "bpe.json"
        tok.save(str(p))
        loaded = BPETokenizer.load(str(p))
        assert loaded.encode("the cat") == tok.encode("the cat")


# ── Factory + repr + len ──────────────────────────────────────────────────────

class TestFactoryReprLen:
    def test_factory_returns_bpe_class(self):
        cls = get_tokenizer("bpe")
        assert cls is BPETokenizer

    def test_repr_untrained(self):
        assert "untrained" in repr(BPETokenizer())

    def test_repr_trained(self, tok):
        r = repr(tok)
        assert "BPETokenizer" in r
        assert str(tok.vocab_size) in r

    def test_len(self, tok):
        assert len(tok) == tok.vocab_size
