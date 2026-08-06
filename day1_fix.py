"""
day1_fix.py — Make the remaining 15 commits for Day 1 with real incremental changes.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

import winreg
def get_full_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for subkey in [
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            r"Environment"
        ]:
            try:
                key = winreg.OpenKey(hive, subkey)
                val, _ = winreg.QueryValueEx(key, "PATH")
                paths.append(val)
            except Exception:
                pass
    return ";".join(paths)

os.environ["PATH"] = get_full_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"ERR stdout: {r.stdout}\nERR stderr: {r.stderr}")
        sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}")
        return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}")
    return True

print("\n=== DAY 1 FIX: 15 more commits ===\n")

# ── Commit 6: Add LICENSE ─────────────────────────────────────────────────────
(REPO / "LICENSE").write_text(
    "MIT License\n\nCopyright (c) 2026 ananthan-bot\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "of this software and associated documentation files (the \"Software\"), to deal\n"
    "in the Software without restriction, including without limitation the rights\n"
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    "copies of the Software, and to permit persons to whom the Software is\n"
    "furnished to do so, subject to the following conditions:\n\n"
    "The above copyright notice and this permission notice shall be included in all\n"
    "copies or substantial portions of the Software.\n\n"
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
    "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
    "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
    "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
    "SOFTWARE.\n",
    encoding="utf-8"
)
commit("chore: add MIT LICENSE file")

# ── Commit 7: .editorconfig ───────────────────────────────────────────────────
(REPO / ".editorconfig").write_text(
    "root = true\n\n"
    "[*]\n"
    "charset = utf-8\n"
    "end_of_line = lf\n"
    "insert_final_newline = true\n"
    "trim_trailing_whitespace = true\n\n"
    "[*.py]\n"
    "indent_style = space\n"
    "indent_size = 4\n\n"
    "[*.{yaml,yml,toml,json}]\n"
    "indent_style = space\n"
    "indent_size = 2\n\n"
    "[Makefile]\n"
    "indent_style = tab\n",
    encoding="utf-8"
)
commit("chore: add .editorconfig for consistent cross-editor formatting")

# ── Commit 8: CHANGELOG.md ────────────────────────────────────────────────────
(REPO / "CHANGELOG.md").write_text(
    "# Changelog\n\n"
    "All notable changes to NanoMind are documented here.\n"
    "Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).\n\n"
    "## [Unreleased]\n\n"
    "### Added\n"
    "- Project scaffold, tooling, and CI pipeline (Day 1)\n"
    "- Coloured logging utility (`nanomind.utils.logger`)\n"
    "- Reproducibility utilities (`nanomind.utils.seed`)\n"
    "- Device detection (`nanomind.utils.device`)\n"
    "- Benchmarking timer (`nanomind.utils.timer`)\n",
    encoding="utf-8"
)
commit("docs: add CHANGELOG.md")

# ── Commit 9: CONTRIBUTING.md ─────────────────────────────────────────────────
(REPO / "CONTRIBUTING.md").write_text(
    "# Contributing to NanoMind\n\n"
    "Thanks for your interest! Here is how to get started.\n\n"
    "## Setup\n\n"
    "```bash\n"
    "git clone https://github.com/ananthan-bot/nanomind.git\n"
    "cd nanomind\n"
    "make dev   # installs dev deps + pre-commit hooks\n"
    "```\n\n"
    "## Workflow\n\n"
    "1. Create a feature branch: `git checkout -b feat/my-feature`\n"
    "2. Make your changes\n"
    "3. Run `make lint` and `make test`\n"
    "4. Commit with a descriptive message\n"
    "5. Open a Pull Request\n\n"
    "## Commit Style\n\n"
    "Use [Conventional Commits](https://www.conventionalcommits.org/):\n\n"
    "- `feat:` new feature\n"
    "- `fix:` bug fix\n"
    "- `docs:` documentation\n"
    "- `test:` tests\n"
    "- `refactor:` refactoring\n"
    "- `chore:` tooling/config\n",
    encoding="utf-8"
)
commit("docs: add CONTRIBUTING.md with setup and commit style guide")

# ── Commit 10: nanomind/cli/ stub ─────────────────────────────────────────────
cli_dir = REPO / "nanomind" / "cli"
cli_dir.mkdir(exist_ok=True)
(cli_dir / "__init__.py").write_text(
    '"""NanoMind CLI entry points — train, generate, eval, info."""\n',
    encoding="utf-8"
)
commit("feat: add nanomind/cli/ package stub for CLI entry points")

# ── Commit 11: configs/ directory ────────────────────────────────────────────
configs_dir = REPO / "configs"
configs_dir.mkdir(exist_ok=True)
(configs_dir / "small.yaml").write_text(
    "# NanoMind — Small model config (~1M params, fast CPU training)\n"
    "model:\n"
    "  d_model: 128\n"
    "  n_layers: 4\n"
    "  n_heads: 4\n"
    "  block_size: 128\n"
    "  dropout: 0.1\n\n"
    "train:\n"
    "  batch_size: 32\n"
    "  max_iters: 5000\n"
    "  learning_rate: 3.0e-4\n"
    "  min_lr: 3.0e-5\n"
    "  warmup_iters: 100\n"
    "  grad_clip: 1.0\n"
    "  eval_interval: 200\n"
    "  device: auto\n",
    encoding="utf-8"
)
commit("chore: add configs/small.yaml (~1M param, CPU-friendly training config)")

# ── Commit 12: configs/medium.yaml ───────────────────────────────────────────
(configs_dir / "medium.yaml").write_text(
    "# NanoMind — Medium model config (~12M params, GPU recommended)\n"
    "model:\n"
    "  d_model: 256\n"
    "  n_layers: 6\n"
    "  n_heads: 8\n"
    "  block_size: 256\n"
    "  dropout: 0.1\n\n"
    "train:\n"
    "  batch_size: 64\n"
    "  max_iters: 10000\n"
    "  learning_rate: 3.0e-4\n"
    "  min_lr: 3.0e-5\n"
    "  warmup_iters: 200\n"
    "  grad_clip: 1.0\n"
    "  eval_interval: 500\n"
    "  device: auto\n",
    encoding="utf-8"
)
commit("chore: add configs/medium.yaml (~12M param, GPU recommended config)")

# ── Commit 13: nanomind/utils/format.py ──────────────────────────────────────
(REPO / "nanomind" / "utils" / "format.py").write_text(
    '"""\nnanomind/utils/format.py — Human-readable formatting helpers.\n"""\n\n\n'
    "def fmt_number(n: int) -> str:\n"
    '    """Format a large integer with K/M/B suffix.\n\n'
    '    Example: 1_200_000 -> "1.2M"\n'
    '    """\n'
    "    if n >= 1_000_000_000:\n"
    '        return f"{n / 1_000_000_000:.2f}B"\n'
    "    if n >= 1_000_000:\n"
    '        return f"{n / 1_000_000:.2f}M"\n'
    "    if n >= 1_000:\n"
    '        return f"{n / 1_000:.1f}K"\n'
    "    return str(n)\n\n\n"
    "def fmt_time(seconds: float) -> str:\n"
    '    """Format seconds as a human-readable duration.\n\n'
    '    Example: 3661 -> "1h 01m 01s"\n'
    '    """\n'
    "    s = int(seconds)\n"
    "    h, rem = divmod(s, 3600)\n"
    "    m, sec = divmod(rem, 60)\n"
    "    if h:\n"
    '        return f"{h}h {m:02d}m {sec:02d}s"\n'
    "    if m:\n"
    '        return f"{m}m {sec:02d}s"\n'
    '    return f"{seconds:.2f}s"\n\n\n'
    "def fmt_loss(loss: float) -> str:\n"
    '    """Format a loss value with 4 decimal places."""\n'
    '    return f"{loss:.4f}"\n\n\n'
    "def fmt_lr(lr: float) -> str:\n"
    '    """Format a learning rate in scientific notation."""\n'
    '    return f"{lr:.2e}"\n',
    encoding="utf-8"
)
commit("feat: add nanomind/utils/format.py with fmt_number, fmt_time, fmt_loss helpers")

# ── Commit 14: nanomind/utils/io.py ──────────────────────────────────────────
(REPO / "nanomind" / "utils" / "io.py").write_text(
    '"""\nnanomind/utils/io.py — File I/O helpers for NanoMind.\n"""\n\n'
    "import json\n"
    "from pathlib import Path\n"
    "from typing import Any\n\n\n"
    "def read_text(path: str | Path, encoding: str = \"utf-8\") -> str:\n"
    '    """Read a text file and return its contents."""\n'
    "    return Path(path).read_text(encoding=encoding)\n\n\n"
    "def write_text(path: str | Path, text: str, encoding: str = \"utf-8\") -> None:\n"
    '    """Write text to a file, creating parent directories if needed."""\n'
    "    p = Path(path)\n"
    "    p.parent.mkdir(parents=True, exist_ok=True)\n"
    "    p.write_text(text, encoding=encoding)\n\n\n"
    "def read_json(path: str | Path) -> Any:\n"
    '    """Read and parse a JSON file."""\n'
    "    return json.loads(Path(path).read_text(encoding=\"utf-8\"))\n\n\n"
    "def write_json(path: str | Path, data: Any, indent: int = 2) -> None:\n"
    '    """Write data as JSON to a file."""\n'
    "    write_text(path, json.dumps(data, ensure_ascii=False, indent=indent))\n\n\n"
    "def ensure_dir(path: str | Path) -> Path:\n"
    '    """Create a directory (and parents) if it does not exist.\n\n'
    '    Returns the resolved Path.\n'
    '    """\n'
    "    p = Path(path)\n"
    "    p.mkdir(parents=True, exist_ok=True)\n"
    "    return p\n",
    encoding="utf-8"
)
commit("feat: add nanomind/utils/io.py with read/write helpers for text and JSON")

# ── Commit 15: expand utils __init__ with all exports ────────────────────────
(REPO / "nanomind" / "utils" / "__init__.py").write_text(
    '"""NanoMind utilities sub-package."""\n\n'
    "from nanomind.utils.logger import get_logger\n"
    "from nanomind.utils.seed import set_seed, get_rng_state, restore_rng_state\n"
    "from nanomind.utils.device import get_device, device_info, is_cuda\n"
    "from nanomind.utils.timer import Timer, timed, tokens_per_second\n"
    "from nanomind.utils.format import fmt_number, fmt_time, fmt_loss, fmt_lr\n"
    "from nanomind.utils.io import read_text, write_text, read_json, write_json, ensure_dir\n\n"
    "__all__ = [\n"
    '    "get_logger",\n'
    '    "set_seed", "get_rng_state", "restore_rng_state",\n'
    '    "get_device", "device_info", "is_cuda",\n'
    '    "Timer", "timed", "tokens_per_second",\n'
    '    "fmt_number", "fmt_time", "fmt_loss", "fmt_lr",\n'
    '    "read_text", "write_text", "read_json", "write_json", "ensure_dir",\n'
    "]\n",
    encoding="utf-8"
)
commit("refactor: export all utility functions from nanomind/utils/__init__.py")

# ── Commit 16: add test for format utils ─────────────────────────────────────
(REPO / "tests" / "test_format.py").write_text(
    '"""Tests for nanomind.utils.format."""\n\n'
    "import pytest\n"
    "from nanomind.utils.format import fmt_number, fmt_time, fmt_loss, fmt_lr\n\n\n"
    "class TestFmtNumber:\n"
    "    def test_billions(self): assert fmt_number(1_500_000_000) == \"1.50B\"\n"
    "    def test_millions(self): assert fmt_number(1_200_000) == \"1.20M\"\n"
    "    def test_thousands(self): assert fmt_number(5_500) == \"5.5K\"\n"
    "    def test_small(self): assert fmt_number(42) == \"42\"\n\n\n"
    "class TestFmtTime:\n"
    "    def test_hours(self): assert \"h\" in fmt_time(3661)\n"
    "    def test_minutes(self): assert \"m\" in fmt_time(90)\n"
    "    def test_seconds(self): assert \"s\" in fmt_time(1.5)\n\n\n"
    "class TestFmtLoss:\n"
    "    def test_decimal_places(self): assert len(fmt_loss(1.23456789).split(\".\")[1]) == 4\n\n\n"
    "class TestFmtLr:\n"
    "    def test_scientific(self): assert \"e\" in fmt_lr(3e-4)\n",
    encoding="utf-8"
)
commit("test: add tests/test_format.py for formatting utility functions")

# ── Commit 17: add test for io utils ─────────────────────────────────────────
(REPO / "tests" / "test_io.py").write_text(
    '"""Tests for nanomind.utils.io."""\n\n'
    "import json\n"
    "import pytest\n"
    "from pathlib import Path\n"
    "from nanomind.utils.io import read_text, write_text, read_json, write_json, ensure_dir\n\n\n"
    "class TestTextIO:\n"
    "    def test_write_read_roundtrip(self, tmp_path):\n"
    "        p = tmp_path / \"test.txt\"\n"
    "        write_text(p, \"hello NanoMind\")\n"
    "        assert read_text(p) == \"hello NanoMind\"\n\n"
    "    def test_creates_parent_dirs(self, tmp_path):\n"
    "        p = tmp_path / \"a\" / \"b\" / \"c.txt\"\n"
    "        write_text(p, \"nested\")\n"
    "        assert p.exists()\n\n\n"
    "class TestJsonIO:\n"
    "    def test_write_read_roundtrip(self, tmp_path):\n"
    "        p = tmp_path / \"data.json\"\n"
    "        data = {\"key\": [1, 2, 3], \"flag\": True}\n"
    "        write_json(p, data)\n"
    "        loaded = read_json(p)\n"
    "        assert loaded == data\n\n\n"
    "class TestEnsureDir:\n"
    "    def test_creates_directory(self, tmp_path):\n"
    "        d = ensure_dir(tmp_path / \"new_dir\")\n"
    "        assert d.is_dir()\n\n"
    "    def test_returns_path(self, tmp_path):\n"
    "        result = ensure_dir(tmp_path / \"x\")\n"
    "        assert isinstance(result, Path)\n",
    encoding="utf-8"
)
commit("test: add tests/test_io.py for file I/O utility functions")

# ── Commit 18: expand nanomind __init__ with new exports ─────────────────────
(REPO / "nanomind" / "__init__.py").write_text(
    '"""\nNanoMind - A small GPT-style transformer LLM built from scratch in PyTorch.\n"""\n\n'
    '__version__ = "0.1.0"\n'
    '__author__ = "ananthan-bot"\n\n'
    "from nanomind.utils import (\n"
    "    get_logger,\n"
    "    get_device,\n"
    "    set_seed,\n"
    "    fmt_number,\n"
    "    fmt_time,\n"
    ")\n\n"
    "__all__ = [\n"
    '    "__version__",\n'
    '    "get_logger",\n'
    '    "get_device",\n'
    '    "set_seed",\n'
    '    "fmt_number",\n'
    '    "fmt_time",\n'
    "]\n",
    encoding="utf-8"
)
commit("refactor: update nanomind/__init__.py to re-export key utilities")

# ── Commit 19: add py.typed marker (PEP 561) ─────────────────────────────────
(REPO / "nanomind" / "py.typed").write_text("", encoding="utf-8")
commit("chore: add py.typed marker for PEP 561 inline type information")

# ── Commit 20: update README roadmap to show Day 1 done ──────────────────────
readme = (REPO / "README.md").read_text(encoding="utf-8")
readme = readme.replace(
    "| 1 | Project scaffold & tooling | ✅ Done |",
    "| 1 | Project scaffold & tooling | ✅ Done — 20 commits |"
)
(REPO / "README.md").write_text(readme, encoding="utf-8")
commit("docs: mark Day 1 complete in README roadmap")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
if r.returncode != 0:
    print(f"Push failed: {r.stderr}")
else:
    print("Pushed!")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
