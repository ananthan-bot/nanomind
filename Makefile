.PHONY: install dev test lint format clean train generate

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=nanomind --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python -m nanomind.cli.train --data data.txt

train-fast:
	python -m nanomind.cli.train --data data.txt --max_iters 200 --eval_interval 100

# ── Generation ────────────────────────────────────────────────────────────────
generate:
	python -m nanomind.cli.generate --checkpoint checkpoints/best.pt --prompt "ROMEO:"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
