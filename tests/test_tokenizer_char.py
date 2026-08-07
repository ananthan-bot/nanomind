"""
tests/test_tokenizer_char.py — Tests for CharTokenizer.
"""

import pytest
from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.factory import get_tokenizer, list_tokenizers

CORPUS = (
    "Hello, World! This is NanoMind.\n"
    "abcdefghijklmnopqrstuvwxyz\n"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
    "0123456789 !@#$\n"
)


@pytest.fixture
def tok() -> CharTokenizer:
    return CharTokenizer().build(CORPUS)


# ── Build ─────────────────────────────────────────────────────────────────────

class TestBuild:
    def test_returns_self(self):
        t = CharTokenizer()
        result = t.build("abc")
        assert result is t

    def test_vocab_includes_all_chars(self, tok):
        for ch in CORPUS:
            assert ch in tok._char_to_id

    def test_special_tokens_have_lowest_ids(self, tok):
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3

    def test_not_built_raises(self):
        t = CharTokenizer()
        with pytest.raises(RuntimeError):
            _ = t.vocab_size


# ── Encode / Decode roundtrip ─────────────────────────────────────────────────

class TestEncodeDecodeRoundtrip:
    def test_simple_string(self, tok):
        text = "Hello"
        assert tok.decode(tok.encode(text)) == text

    def test_full_corpus(self, tok):
        assert tok.decode(tok.encode(CORPUS)) == CORPUS

    def test_empty_string(self, tok):
        assert tok.decode(tok.encode("")) == ""

    def test_single_char(self, tok):
        assert tok.decode(tok.encode("a")) == "a"

    def test_newline(self, tok):
        assert tok.decode(tok.encode("\n")) == "\n"


# ── Special tokens ────────────────────────────────────────────────────────────

class TestSpecialTokens:
    def test_pad_id_is_zero(self, tok):
        assert tok.pad_id == 0

    def test_unk_id_is_one(self, tok):
        assert tok.unk_id == 1

    def test_encode_with_bos(self, tok):
        ids = tok.encode_with_special("hi", add_bos=True)
        assert ids[0] == tok.bos_id

    def test_encode_with_eos(self, tok):
        ids = tok.encode_with_special("hi", add_eos=True)
        assert ids[-1] == tok.eos_id

    def test_encode_with_bos_and_eos(self, tok):
        ids = tok.encode_with_special("hi", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_decode_strips_pad(self, tok):
        ids = [tok.pad_id] + tok.encode("hi") + [tok.pad_id]
        assert tok.decode(ids) == "hi"

    def test_decode_strips_bos_eos(self, tok):
        ids = tok.encode_with_special("hi", add_bos=True, add_eos=True)
        assert tok.decode(ids) == "hi"
