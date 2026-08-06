# Contributing to NanoMind

Thanks for your interest! Here is how to get started.

## Setup

```bash
git clone https://github.com/ananthan-bot/nanomind.git
cd nanomind
make dev   # installs dev deps + pre-commit hooks
```

## Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Run `make lint` and `make test`
4. Commit with a descriptive message
5. Open a Pull Request

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` tests
- `refactor:` refactoring
- `chore:` tooling/config
