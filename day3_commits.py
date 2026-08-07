"""
day3_commits.py — 20 atomic commits for Day 3: BPE Tokenizer.

Builds a Byte-Pair Encoding tokenizer from scratch, one commit at a time.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

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

print("\n=== DAY 3: BPE Tokenizer — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — BPETokenizer skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/bpe.py", '''\
"""
nanomind/tokenizer/bpe.py — Byte-Pair Encoding (BPE) tokenizer for NanoMind.

BPE is a data-driven subword tokenization algorithm. It starts with a
character-level vocabulary and repeatedly merges the most frequent adjacent
symbol pair until a desired vocabulary size is reached.

Reference: Sennrich et al., 2016 — https://arxiv.org/abs/1508.07909
"""

from __future__ import annotations

from typing import List

from nanomind.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding tokenizer built from scratch.

    Workflow:
        1. ``tok = BPETokenizer()``
        2. ``tok.train(text, vocab_size=500)``   — learn merge rules
        3. ``tok.encode("hello world")``          — encode string
        4. ``tok.decode([...])``                  — decode back
        5. ``tok.save("bpe_vocab.json")``         — persist
        6. ``tok = BPETokenizer.load(...)``       — restore
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SPECIAL_TOKENS: list[str] = [PAD, UNK, BOS, EOS]
    WORD_END = "</w>"          # Marks the end of a word during BPE training

    def __init__(self) -> None:
        self._merges: list[tuple[str, str]] = []   # Ordered merge rules
        self._vocab: dict[str, int] = {}            # token -> id
        self._id_to_token: dict[int, str] = {}      # id -> token
        self._trained: bool = False

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Call .train(text) or .load(path) first.")

    # ── Abstract method stubs (implemented in later commits) ──────────────────

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        raise NotImplementedError
''')
commit("feat: add BPETokenizer class skeleton with special tokens and stubs")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Word frequency counter
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src += '''

    # ── Vocabulary building helpers ───────────────────────────────────────────

    @staticmethod
    def _get_word_freqs(text: str) -> dict[str, int]:
        """
        Count word frequencies in the corpus.

        Each word is represented as a space-separated sequence of characters
        with a special end-of-word marker on the last character.

        Example:
            "hello hello world" ->
            {"h e l l o</w>": 2, "w o r l d</w>": 1}

        Args:
            text: Raw training corpus.

        Returns:
            Dict mapping space-separated character sequences to frequencies.
        """
        word_freqs: dict[str, int] = {}
        for word in text.split():
            # Convert each word to space-separated chars + end-of-word marker
            chars = list(word[:-1]) + [word[-1] + BPETokenizer.WORD_END]
            key = " ".join(chars)
            word_freqs[key] = word_freqs.get(key, 0) + 1
        return word_freqs
'''
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add _get_word_freqs() — word frequency counter for BPE training")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — get_pairs(): extract adjacent symbol pairs
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src += '''
    @staticmethod
    def _get_pairs(word_freqs: dict[str, int]) -> dict[tuple[str, str], int]:
        """
        Count all adjacent symbol pair frequencies across all words.

        Args:
            word_freqs: Output of :meth:`_get_word_freqs`.

        Returns:
            Dict mapping ``(symbol_a, symbol_b)`` pairs to their total count.

        Example:
            {"h e l l o</w>": 2} ->
            {("h","e"):2, ("e","l"):2, ("l","l"):2, ("l","o</w>"):2}
        """
        pairs: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            symbols = word.split()
            for a, b in zip(symbols[:-1], symbols[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + freq
        return pairs
'''
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add _get_pairs() — extract adjacent symbol pair frequencies")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — merge_pair(): apply a single merge rule
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src += '''
    @staticmethod
    def _merge_pair(
        pair: tuple[str, str],
        word_freqs: dict[str, int],
    ) -> dict[str, int]:
        """
        Apply a single BPE merge rule to all words.

        Replaces all occurrences of ``pair[0] + " " + pair[1]`` with
        the merged token ``pair[0] + pair[1]`` in every word.

        Args:
            pair:       The ``(a, b)`` symbol pair to merge.
            word_freqs: Current word frequency table.

        Returns:
            Updated word frequency table with the merge applied.
        """
        import re
        a, b = pair
        pattern = re.compile(r"(?<![\\S])" + re.escape(a) + r" " + re.escape(b) + r"(?![\\S])")
        merged = a + b
        new_freqs: dict[str, int] = {}
        for word, freq in word_freqs.items():
            new_word = pattern.sub(merged, word)
            new_freqs[new_word] = freq
        return new_freqs
'''
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add _merge_pair() — apply a single BPE merge rule to the vocabulary")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — train(): learn N merge rules
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src += '''
    def train(self, text: str, vocab_size: int = 500) -> "BPETokenizer":
        """
        Learn BPE merge rules from a text corpus.

        Starts with a character-level vocabulary and iteratively merges
        the most frequent adjacent symbol pair until ``vocab_size`` is reached.

        Args:
            text:       Raw training corpus.
            vocab_size: Target vocabulary size (including special tokens).

        Returns:
            Self (for method chaining).
        """
        # Step 1: Build initial character vocab + special tokens
        chars = sorted(set("".join(text.split())))
        base_vocab = self.SPECIAL_TOKENS + chars + [c + self.WORD_END for c in chars]
        vocab = {tok: i for i, tok in enumerate(dict.fromkeys(base_vocab))}

        # Step 2: Get initial word frequencies
        word_freqs = self._get_word_freqs(text)
        merges: list[tuple[str, str]] = []

        # Step 3: Iteratively merge most frequent pair
        n_merges = max(0, vocab_size - len(vocab))
        for _ in range(n_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            word_freqs = self._merge_pair(best, word_freqs)
            merged_token = best[0] + best[1]
            if merged_token not in vocab:
                vocab[merged_token] = len(vocab)
            merges.append(best)

        self._merges = merges
        self._vocab = vocab
        self._id_to_token = {i: t for t, i in vocab.items()}
        self._trained = True
        return self
'''
write("nanomind/tokenizer/bpe.py", src)
commit("feat: implement train() — iteratively learn BPE merge rules from corpus")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — _tokenize_word(): apply merges to a single word
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src += '''
    def _tokenize_word(self, word: str) -> list[str]:
        """
        Apply learned BPE merge rules to a single word.

        The word is first split into individual characters (with an end-of-word
        marker on the last one), then merge rules are applied in order.

        Args:
            word: A single word string (no spaces).

        Returns:
            List of BPE subword tokens.
        """
        if not word:
            return []
        # Initialise as character sequence with end-of-word marker
        symbols = list(word[:-1]) + [word[-1] + self.WORD_END]

        # Apply each merge rule in training order
        for a, b in self._merges:
            i = 0
            new_symbols: list[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols
'''
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add _tokenize_word() — apply learned BPE merges to a single word")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — implement encode()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src = src.replace(
    "    def encode(self, text: str) -> List[int]:\n        raise NotImplementedError",
    '''\
    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of BPE token IDs.

        Words are split on whitespace; each word is tokenized with
        :meth:`_tokenize_word`. Unknown subwords map to UNK.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        self._require_trained()
        unk_id = self._vocab.get(self.UNK, 1)
        ids: list[int] = []
        for word in text.split():
            for subword in self._tokenize_word(word):
                ids.append(self._vocab.get(subword, unk_id))
            # Add a space token between words if it exists in the vocab
            if " " in self._vocab:
                ids.append(self._vocab[" "])
        return ids'''
)
write("nanomind/tokenizer/bpe.py", src)
commit("feat: implement encode() — tokenize text using learned BPE merge rules")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — implement decode()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src = src.replace(
    "    def decode(self, ids: List[int]) -> str:\n        raise NotImplementedError",
    '''\
    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of BPE token IDs back to a string.

        End-of-word markers (``</w>``) are replaced with spaces.
        Special tokens (PAD, BOS, EOS) are stripped.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        self._require_trained()
        skip_ids = {
            self._vocab.get(self.PAD, -1),
            self._vocab.get(self.BOS, -2),
            self._vocab.get(self.EOS, -3),
        }
        tokens = [self._id_to_token.get(i, self.UNK) for i in ids if i not in skip_ids]
        text = "".join(tokens)
        text = text.replace(self.WORD_END, " ")
        return text.strip()'''
)
write("nanomind/tokenizer/bpe.py", src)
commit("feat: implement decode() — convert BPE token IDs back to string")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — implement vocab_size property
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src = src.replace(
    "    @property\n    def vocab_size(self) -> int:\n        raise NotImplementedError",
    '''\
    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the BPE vocabulary."""
        self._require_trained()
        return len(self._vocab)'''
)
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add vocab_size property to BPETokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — add special token ID properties
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src = src.replace(
    "    def _require_trained(self) -> None:",
    '''\
    @property
    def pad_id(self) -> int:
        """Integer ID of the PAD token."""
        return self._vocab[self.PAD]

    @property
    def unk_id(self) -> int:
        """Integer ID of the UNK token."""
        return self._vocab[self.UNK]

    @property
    def bos_id(self) -> int:
        """Integer ID of the BOS token."""
        return self._vocab[self.BOS]

    @property
    def eos_id(self) -> int:
        """Integer ID of the EOS token."""
        return self._vocab[self.EOS]

    def _require_trained(self) -> None:'''
)
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add pad_id, unk_id, bos_id, eos_id properties to BPETokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — implement save()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src = src.replace(
    "    def save(self, path: str) -> None:\n        raise NotImplementedError",
    '''\
    def save(self, path: str) -> None:
        """
        Persist the BPE tokenizer (merges + vocab) to a JSON file.

        Args:
            path: Output file path.
        """
        import json
        from pathlib import Path as _P
        self._require_trained()
        data = {
            "merges": [list(m) for m in self._merges],
            "vocab":  self._vocab,
        }
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        _P(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")'''
)
write("nanomind/tokenizer/bpe.py", src)
commit("feat: implement save() — persist BPE merges and vocab to JSON")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — implement load()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src = src.replace(
    "    @classmethod\n    def load(cls, path: str) -> \"BPETokenizer\":\n        raise NotImplementedError",
    '''\
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load a BPETokenizer from a JSON file created by :meth:`save`.

        Args:
            path: Path to a JSON file.

        Returns:
            A ready-to-use :class:`BPETokenizer`.
        """
        import json
        from pathlib import Path as _P
        data = json.loads(_P(path).read_text(encoding="utf-8"))
        tok = cls()
        tok._merges = [tuple(m) for m in data["merges"]]
        tok._vocab = data["vocab"]
        tok._id_to_token = {int(i): t for t, i in tok._vocab.items()}
        tok._trained = True
        return tok'''
)
write("nanomind/tokenizer/bpe.py", src)
commit("feat: implement load() — restore BPETokenizer from JSON file")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — add __repr__, __len__, num_merges
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/bpe.py")
src += '''
    @property
    def num_merges(self) -> int:
        """Number of learned BPE merge rules."""
        return len(self._merges)

    def __len__(self) -> int:
        """Alias for vocab_size — allows len(tokenizer)."""
        return self.vocab_size

    def __repr__(self) -> str:
        if not self._trained:
            return "BPETokenizer(untrained)"
        return (
            f"BPETokenizer("
            f"vocab_size={self.vocab_size}, "
            f"num_merges={self.num_merges})"
        )
'''
write("nanomind/tokenizer/bpe.py", src)
commit("feat: add __repr__, __len__, and num_merges property to BPETokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — register BPETokenizer in factory
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/tokenizer/factory.py")
src = src.replace(
    "from nanomind.tokenizer.char import CharTokenizer",
    "from nanomind.tokenizer.char import CharTokenizer\nfrom nanomind.tokenizer.bpe import BPETokenizer"
)
src = src.replace(
    '"char": CharTokenizer,',
    '"char": CharTokenizer,\n    "bpe":  BPETokenizer,'
)
write("nanomind/tokenizer/factory.py", src)
commit("feat: register BPETokenizer in the tokenizer factory registry")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — test: BPE training runs without error
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_tokenizer_bpe.py", '''\
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
''')
commit("test: add BPE training tests — convergence, vocab size, special tokens")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: encode/decode roundtrip
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_tokenizer_bpe.py")
src += '''

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
'''
write("tests/test_tokenizer_bpe.py", src)
commit("test: add encode/decode tests for BPETokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: merge rule correctness
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_tokenizer_bpe.py")
src += '''

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
'''
write("tests/test_tokenizer_bpe.py", src)
commit("test: add merge rule correctness tests for BPE algorithm")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: persistence (save/load)
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_tokenizer_bpe.py")
src += '''

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
'''
write("tests/test_tokenizer_bpe.py", src)
commit("test: add BPE persistence (save/load roundtrip) tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: factory + repr + len
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_tokenizer_bpe.py")
src += '''

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
'''
write("tests/test_tokenizer_bpe.py", src)
commit("test: add factory, __repr__, and __len__ tests for BPETokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — export BPETokenizer; unify BaseTokenizer; update README + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/__init__.py", '''\
"""NanoMind tokenizer sub-package.

Available tokenizers:
    - ``CharTokenizer`` — character-level, simplest, great for small datasets
    - ``BPETokenizer``  — byte-pair encoding, subword, better compression

Use :func:`get_tokenizer` to look up a tokenizer by name.
"""

from nanomind.tokenizer.base import BaseTokenizer
from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.bpe import BPETokenizer
from nanomind.tokenizer.factory import get_tokenizer, list_tokenizers

__all__ = [
    "BaseTokenizer",
    "CharTokenizer",
    "BPETokenizer",
    "get_tokenizer",
    "list_tokenizers",
]
''')

readme = read("README.md")
readme = readme.replace(
    "| 3 | BPE tokenizer | 🔜 |",
    "| 3 | BPE tokenizer | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Character-level tokenizer with BOS/EOS/PAD/UNK, save/load, factory (Day 2)",
    "- Character-level tokenizer with BOS/EOS/PAD/UNK, save/load, factory (Day 2)\n- BPE tokenizer with merge learning, encode/decode, persistence, factory (Day 3)"
)
write("CHANGELOG.md", cl)
commit("chore: export BPETokenizer from tokenizer package; update README and CHANGELOG for Day 3")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 3 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 3 COMPLETE ===")
