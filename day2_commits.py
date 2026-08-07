"""
day2_commits.py — 20 atomic commits for Day 2: Character-Level Tokenizer.

Each commit writes exactly one piece of the tokenizer, ensuring a clean,
meaningful git history.
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

print("\n=== DAY 2: Character-Level Tokenizer — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — tokenizer package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/__init__.py", '"""NanoMind tokenizer sub-package."""\n')
commit("feat: add nanomind/tokenizer/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — BaseTokenizer abstract class
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/base.py", '''\
"""
nanomind/tokenizer/base.py — Abstract base class for all NanoMind tokenizers.

Defines the interface every tokenizer must implement so downstream code
(data pipeline, generation) can work with any tokenizer interchangeably.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Abstract interface shared by all NanoMind tokenizers."""

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of integer token IDs."""
        ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the tokenizer state to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """Load a tokenizer from a previously saved file."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of strings. Returns a list of token ID lists."""
        return [self.encode(t) for t in texts]

    def batch_decode(self, id_lists: List[List[int]]) -> List[str]:
        """Decode a list of token ID lists. Returns a list of strings."""
        return [self.decode(ids) for ids in id_lists]
''')
commit("feat: add BaseTokenizer abstract class with encode/decode interface")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — CharTokenizer skeleton (class + __init__ only)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/char.py", '''\
"""
nanomind/tokenizer/char.py — Character-level tokenizer for NanoMind.

Maps every unique character in the training corpus to an integer ID.
This is the simplest possible tokenizer and works well for small datasets.
"""

from __future__ import annotations

from typing import List

from nanomind.tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.

    Each unique character in the training text gets a unique integer ID.
    Special tokens occupy the lowest IDs so they are always present.
    """

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._built: bool = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, text: str) -> "CharTokenizer":
        """Build vocabulary from a raw text string."""
        raise NotImplementedError("Coming next commit")

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        raise NotImplementedError

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call .build(text) or .load(path) first.")
''')
commit("feat: add CharTokenizer class skeleton with __init__ and method stubs")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Add special token constants
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    def __init__(self) -> None:",
    '''\
    # ── Special tokens ────────────────────────────────────────────────────────
    PAD = "<PAD>"   # Padding (ID 0)
    UNK = "<UNK>"   # Unknown character (ID 1)
    BOS = "<BOS>"   # Beginning of sequence (ID 2)
    EOS = "<EOS>"   # End of sequence (ID 3)
    SPECIAL_TOKENS: list[str] = [PAD, UNK, BOS, EOS]

    def __init__(self) -> None:'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: add PAD, UNK, BOS, EOS special token constants to CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Implement build() + _build_vocab()
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    '        """Build vocabulary from a raw text string."""\n        raise NotImplementedError("Coming next commit")',
    '''\
        """
        Build the character vocabulary from a raw text string.

        Special tokens are always prepended so they have the lowest IDs:
        PAD=0, UNK=1, BOS=2, EOS=3.

        Args:
            text: The full training corpus as a single string.

        Returns:
            Self (for method chaining).
        """
        vocab = self._build_vocab(text)
        self._char_to_id = {ch: i for i, ch in enumerate(vocab)}
        self._id_to_char = {i: ch for i, ch in enumerate(vocab)}
        self._built = True
        return self

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_vocab(self, text: str) -> list[str]:
        """Return ordered list of tokens: special tokens first, then sorted chars."""
        unique_chars = sorted(set(text) - set(self.SPECIAL_TOKENS))
        return self.SPECIAL_TOKENS + unique_chars'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: implement build() and _build_vocab() in CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Implement encode()
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    def encode(self, text: str) -> List[int]:\n        raise NotImplementedError",
    '''\
    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of integer token IDs.

        Unknown characters (not seen during build) are mapped to UNK (ID 1).

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        self._require_built()
        unk_id = self._char_to_id[self.UNK]
        return [self._char_to_id.get(ch, unk_id) for ch in text]'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: implement encode() — maps characters to integer IDs with UNK fallback")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Implement decode()
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    def decode(self, ids: List[int]) -> str:\n        raise NotImplementedError",
    '''\
    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of integer token IDs back to a string.

        Special tokens (PAD, BOS, EOS) are stripped from the output.
        Unknown IDs are replaced with the UNK token string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        self._require_built()
        skip = {self._char_to_id[self.PAD], self._char_to_id[self.BOS], self._char_to_id[self.EOS]}
        chars = [
            self._id_to_char.get(i, self.UNK)
            for i in ids
            if i not in skip
        ]
        return "".join(chars)'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: implement decode() — maps integer IDs to string, strips special tokens")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Add encode_with_bos_eos() for sequence wrapping
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    # ── Persistence",
    '''\
    def encode_with_special(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text, optionally wrapping with BOS and EOS tokens.

        Args:
            text:    Input string.
            add_bos: Prepend BOS token if True.
            add_eos: Append EOS token if True.

        Returns:
            List of token IDs (possibly with BOS/EOS).
        """
        self._require_built()
        ids = self.encode(text)
        if add_bos:
            ids = [self._char_to_id[self.BOS]] + ids
        if add_eos:
            ids = ids + [self._char_to_id[self.EOS]]
        return ids

    # ── Persistence'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: add encode_with_special() for BOS/EOS sequence wrapping")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — Implement save()
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    def save(self, path: str) -> None:\n        raise NotImplementedError",
    '''\
    def save(self, path: str) -> None:
        """
        Persist the vocabulary to a JSON file.

        Args:
            path: Output file path (e.g. ``checkpoints/vocab.json``).
        """
        import json
        from pathlib import Path as _P
        self._require_built()
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        _P(path).write_text(
            json.dumps({"char_to_id": self._char_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: implement save() — persist vocabulary to JSON file")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — Implement load()
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    @classmethod\n    def load(cls, path: str) -> \"CharTokenizer\":\n        raise NotImplementedError",
    '''\
    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Load a previously saved tokenizer from a JSON vocab file.

        Args:
            path: Path to a JSON file created by :meth:`save`.

        Returns:
            A ready-to-use :class:`CharTokenizer`.
        """
        import json
        from pathlib import Path as _P
        data = json.loads(_P(path).read_text(encoding="utf-8"))
        tok = cls()
        tok._char_to_id = data["char_to_id"]
        tok._id_to_char = {int(i): ch for ch, i in tok._char_to_id.items()}
        tok._built = True
        return tok'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: implement load() — restore tokenizer from JSON vocab file")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — Add vocab_size property
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    @property\n    def vocab_size(self) -> int:\n        raise NotImplementedError",
    '''\
    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary (including special tokens)."""
        self._require_built()
        return len(self._char_to_id)'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: add vocab_size property to CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — Add pad_id, unk_id, bos_id, eos_id properties
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src = src.replace(
    "    def _require_built(self) -> None:",
    '''\
    @property
    def pad_id(self) -> int:
        """Integer ID of the PAD token."""
        return self._char_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        """Integer ID of the UNK token."""
        return self._char_to_id[self.UNK]

    @property
    def bos_id(self) -> int:
        """Integer ID of the BOS (beginning-of-sequence) token."""
        return self._char_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        """Integer ID of the EOS (end-of-sequence) token."""
        return self._char_to_id[self.EOS]

    def _require_built(self) -> None:'''
)
write("nanomind/tokenizer/char.py", src)
commit("feat: add pad_id, unk_id, bos_id, eos_id convenience properties")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — Add __repr__ and __len__
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "nanomind/tokenizer/char.py").read_text(encoding="utf-8")
src += '''\

    def __len__(self) -> int:
        """Alias for vocab_size — allows len(tokenizer)."""
        return self.vocab_size

    def __repr__(self) -> str:
        size = self.vocab_size if self._built else "?"
        return f"CharTokenizer(vocab_size={size})"
'''
write("nanomind/tokenizer/char.py", src)
commit("feat: add __repr__ and __len__ to CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — Add get_tokenizer() factory function
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/factory.py", '''\
"""
nanomind/tokenizer/factory.py — Tokenizer factory for NanoMind.

Provides a single get_tokenizer() function that returns the right tokenizer
by name, so downstream code doesn't need to import specific classes.
"""

from __future__ import annotations

from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.base import BaseTokenizer

_REGISTRY: dict[str, type[BaseTokenizer]] = {
    "char": CharTokenizer,
}


def get_tokenizer(name: str) -> type[BaseTokenizer]:
    """
    Return a tokenizer *class* (not instance) by name.

    Args:
        name: Tokenizer name. Currently supported: ``"char"``.

    Returns:
        The tokenizer class. Call ``.build(text)`` or ``.load(path)`` on it.

    Raises:
        ValueError: If the tokenizer name is not recognised.

    Example::

        TokenizerClass = get_tokenizer("char")
        tok = TokenizerClass().build(text)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown tokenizer '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_tokenizers() -> list[str]:
    """Return a sorted list of all registered tokenizer names."""
    return sorted(_REGISTRY)
''')
commit("feat: add get_tokenizer() factory and tokenizer registry")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — Unit tests: encode/decode roundtrip
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_tokenizer_char.py", '''\
"""
tests/test_tokenizer_char.py — Tests for CharTokenizer.
"""

import pytest
from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.factory import get_tokenizer, list_tokenizers

CORPUS = (
    "Hello, World! This is NanoMind.\\n"
    "abcdefghijklmnopqrstuvwxyz\\n"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ\\n"
    "0123456789 !@#$\\n"
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
        assert tok.decode(tok.encode("\\n")) == "\\n"
''')
commit("test: add encode/decode roundtrip tests for CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — Tests: special token handling
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "tests/test_tokenizer_char.py").read_text(encoding="utf-8")
src += '''

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
'''
write("tests/test_tokenizer_char.py", src)
commit("test: add special token handling tests (BOS, EOS, PAD stripping)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — Tests: unknown character fallback
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "tests/test_tokenizer_char.py").read_text(encoding="utf-8")
src += '''

# ── Unknown characters ────────────────────────────────────────────────────────

class TestUnknownChars:
    def test_unknown_char_maps_to_unk_id(self, tok):
        # Build on limited corpus, then encode something not in vocab
        small_tok = CharTokenizer().build("abc")
        ids = small_tok.encode("xyz")
        assert all(i == small_tok.unk_id for i in ids)

    def test_unknown_id_in_decode(self, tok):
        result = tok.decode([99999])
        assert result == tok.UNK

    def test_known_chars_not_mapped_to_unk(self, tok):
        ids = tok.encode("Hello")
        assert tok.unk_id not in ids
'''
write("tests/test_tokenizer_char.py", src)
commit("test: add unknown character fallback tests for CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — Tests: vocab persistence (save/load)
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "tests/test_tokenizer_char.py").read_text(encoding="utf-8")
src += '''

# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_file(self, tok, tmp_path):
        p = tmp_path / "vocab.json"
        tok.save(str(p))
        assert p.exists()

    def test_load_roundtrip_vocab_size(self, tok, tmp_path):
        p = tmp_path / "vocab.json"
        tok.save(str(p))
        loaded = CharTokenizer.load(str(p))
        assert loaded.vocab_size == tok.vocab_size

    def test_load_roundtrip_encode(self, tok, tmp_path):
        p = tmp_path / "vocab.json"
        tok.save(str(p))
        loaded = CharTokenizer.load(str(p))
        text = "Hello, World!"
        assert loaded.encode(text) == tok.encode(text)

    def test_load_roundtrip_special_ids(self, tok, tmp_path):
        p = tmp_path / "vocab.json"
        tok.save(str(p))
        loaded = CharTokenizer.load(str(p))
        assert loaded.pad_id == tok.pad_id
        assert loaded.unk_id == tok.unk_id
        assert loaded.bos_id == tok.bos_id
        assert loaded.eos_id == tok.eos_id
'''
write("tests/test_tokenizer_char.py", src)
commit("test: add vocab persistence (save/load roundtrip) tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — Tests: factory + repr + len
# ══════════════════════════════════════════════════════════════════════════════
src = (REPO / "tests/test_tokenizer_char.py").read_text(encoding="utf-8")
src += '''

# ── Factory + repr + len ──────────────────────────────────────────────────────

class TestFactory:
    def test_get_tokenizer_char(self):
        cls = get_tokenizer("char")
        assert cls is CharTokenizer

    def test_get_tokenizer_unknown_raises(self):
        with pytest.raises(ValueError):
            get_tokenizer("nonexistent")

    def test_list_tokenizers(self):
        assert "char" in list_tokenizers()


class TestReprAndLen:
    def test_repr_before_build(self):
        t = CharTokenizer()
        assert "?" in repr(t)

    def test_repr_after_build(self, tok):
        r = repr(tok)
        assert "CharTokenizer" in r
        assert str(tok.vocab_size) in r

    def test_len(self, tok):
        assert len(tok) == tok.vocab_size
'''
write("tests/test_tokenizer_char.py", src)
commit("test: add factory, __repr__, and __len__ tests for CharTokenizer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — Export CharTokenizer from package; update README roadmap
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/tokenizer/__init__.py", '''\
"""NanoMind tokenizer sub-package."""

from nanomind.tokenizer.base import BaseTokenizer
from nanomind.tokenizer.char import CharTokenizer
from nanomind.tokenizer.factory import get_tokenizer, list_tokenizers

__all__ = [
    "BaseTokenizer",
    "CharTokenizer",
    "get_tokenizer",
    "list_tokenizers",
]
''')

readme = (REPO / "README.md").read_text(encoding="utf-8")
readme = readme.replace(
    "| 2 | Character-level tokenizer | 🔜 |",
    "| 2 | Character-level tokenizer | ✅ Done — 20 commits |"
)
write("README.md", readme)

# update CHANGELOG
cl = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
cl = cl.replace(
    "- Project scaffold, tooling, and CI pipeline (Day 1)",
    "- Project scaffold, tooling, and CI pipeline (Day 1)\n- Character-level tokenizer with BOS/EOS/PAD/UNK, save/load, factory (Day 2)"
)
write("CHANGELOG.md", cl)
commit("chore: export CharTokenizer from tokenizer package; update README and CHANGELOG for Day 2")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 2 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 2 COMPLETE ===")
