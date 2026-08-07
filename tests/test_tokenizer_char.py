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
