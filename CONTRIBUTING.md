# Contributing to NanoMind

Thank you for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/ananthan-bot/nanomind.git
cd nanomind
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Use `from __future__ import annotations` at the top of every file
- Write docstrings for all public functions and classes
- Keep functions focused and short — prefer composition over complexity
- Use `pathlib.Path` for all file I/O (never raw strings)

## Commit Convention

All commits follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code change without feature/fix
- `test:` — adding or updating tests
- `docs:` — documentation only
- `chore:` — build, CI, tooling

## Pull Requests

1. Fork the repo and create a feature branch
2. Write tests for your changes
3. Ensure all tests pass
4. Open a PR with a clear description
