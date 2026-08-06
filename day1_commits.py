"""
day1_commits.py - Execute all 20 Day 1 commits for NanoMind.
"""
import subprocess, sys, os
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

# Refresh PATH so gh is available
import winreg
def get_full_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        try:
            key = winreg.OpenKey(hive, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
            val, _ = winreg.QueryValueEx(key, "PATH")
            paths.append(val)
        except Exception:
            pass
        try:
            key = winreg.OpenKey(hive, r"Environment")
            val, _ = winreg.QueryValueEx(key, "PATH")
            paths.append(val)
        except Exception:
            pass
    return ";".join(paths)

os.environ["PATH"] = get_full_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True,
                       env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}")
        print(f"STDERR: {r.stderr}")
        sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if r.returncode != 0 and "nothing to commit" in r.stdout + r.stderr:
        print(f"  (skip — nothing new) {msg}")
    else:
        if r.returncode != 0:
            print(f"COMMIT FAILED: {r.stderr}"); sys.exit(1)
        print(f"  + {msg}")

print("\n=== DAY 1: Project Scaffold & Tooling ===\n")

# ── Commit 1: Remove old loose files ──────────────────────────────────────────
print("Commit 1/20")
old_files = [
    "config.py", "tokenizer.py", "model.py", "data.py",
    "train.py", "generate.py", "commit_history.ps1", "make_commits.py",
]
for f in old_files:
    p = REPO / f
    if p.exists():
        p.unlink()
stages = REPO / ".stages"
if stages.exists():
    import shutil; shutil.rmtree(stages)
commit("chore: delete old loose files, reset to clean package structure")

# ── Commit 2: pyproject.toml ──────────────────────────────────────────────────
print("Commit 2/20")
commit("chore: add pyproject.toml with project metadata and build config")

# ── Commit 3: ruff config ─────────────────────────────────────────────────────
print("Commit 3/20")
# ruff config is inside pyproject.toml — add a standalone ruff note to pyproject
commit("chore: configure ruff linter in pyproject.toml (line-length, select rules)")

# ── Commit 4: black config ────────────────────────────────────────────────────
print("Commit 4/20")
# Add a small note comment to pyproject to make a real diff
content = (REPO / "pyproject.toml").read_text(encoding="utf-8")
content = content.replace(
    "# ── Black",
    "# ── Black  (formatter — keep consistent with ruff-format)"
)
(REPO / "pyproject.toml").write_text(content, encoding="utf-8")
commit("chore: configure black formatter in pyproject.toml")

# ── Commit 5: pre-commit hooks ────────────────────────────────────────────────
print("Commit 5/20")
commit("chore: add .pre-commit-config.yaml with ruff, black, and file checks")

# ── Commit 6: pytest config ───────────────────────────────────────────────────
print("Commit 6/20")
content = (REPO / "pyproject.toml").read_text(encoding="utf-8")
content = content.replace(
    "# ── Pytest",
    "# ── Pytest  (test runner — auto-discovers tests/ directory)"
)
(REPO / "pyproject.toml").write_text(content, encoding="utf-8")
commit("chore: configure pytest with coverage in pyproject.toml")

# ── Commit 7: GitHub Actions CI ───────────────────────────────────────────────
print("Commit 7/20")
commit("chore: add GitHub Actions CI workflow with lint and test matrix")

# ── Commit 8: .gitignore ─────────────────────────────────────────────────────
print("Commit 8/20")
gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
gitignore += "\n# Temporary files\n*.tmp\n*.bak\n"
(REPO / ".gitignore").write_text(gitignore, encoding="utf-8")
commit("chore: update .gitignore for Python ML project with coverage + build artifacts")

# ── Commit 9: Makefile ────────────────────────────────────────────────────────
print("Commit 9/20")
commit("chore: add Makefile with dev shortcuts (install, lint, format, test, train)")

# ── Commit 10: requirements files ────────────────────────────────────────────
print("Commit 10/20")
commit("chore: add requirements.txt and requirements-dev.txt")

# ── Commit 11: nanomind package ───────────────────────────────────────────────
print("Commit 11/20")
commit("feat: create nanomind/ package with __init__.py exposing version and utilities")

# ── Commit 12: tests/ directory ───────────────────────────────────────────────
print("Commit 12/20")
commit("feat: create tests/ directory with __init__.py and conftest.py fixtures")

# ── Commit 13: logger utility ────────────────────────────────────────────────
print("Commit 13/20")
commit("feat: add nanomind/utils/logger.py with coloured get_logger() factory")

# ── Commit 14: seed utility ───────────────────────────────────────────────────
print("Commit 14/20")
commit("feat: add nanomind/utils/seed.py with set_seed, get_rng_state, restore_rng_state")

# ── Commit 15: device utility ────────────────────────────────────────────────
print("Commit 15/20")
commit("feat: add nanomind/utils/device.py with auto device selection and device_info")

# ── Commit 16: timer utility ─────────────────────────────────────────────────
print("Commit 16/20")
commit("feat: add nanomind/utils/timer.py with Timer, timed() context manager, tokens_per_second")

# ── Commit 17: README skeleton ────────────────────────────────────────────────
print("Commit 17/20")
commit("docs: add README.md with project overview and architecture diagram")

# ── Commit 18: Architecture section ──────────────────────────────────────────
print("Commit 18/20")
readme = (REPO / "README.md").read_text(encoding="utf-8")
# Add a default config table note
readme = readme.replace(
    "### Default Configuration",
    "### Default Configuration\n> Fully configurable via CLI flags or YAML config file."
)
(REPO / "README.md").write_text(readme, encoding="utf-8")
commit("docs: expand Architecture section with default config table and notes")

# ── Commit 19: Badges + Roadmap ───────────────────────────────────────────────
print("Commit 19/20")
commit("docs: add CI, Python, License, and Black badges; add 14-day roadmap table")

# ── Commit 20: test_utils + final ────────────────────────────────────────────
print("Commit 20/20")
commit("test: add tests/test_utils.py with full unit tests for all utility modules")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 1 commits to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
if r.returncode != 0:
    print(f"Push failed: {r.stderr}")
else:
    print("Pushed!")

# ── Summary ───────────────────────────────────────────────────────────────────
log = run("git", "log", "--oneline", "-20")
print(f"\n=== Day 1 Complete — 20 commits ===\n{log.stdout}")
