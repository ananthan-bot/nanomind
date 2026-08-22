"""NanoMind command-line interface sub-package.

Entry point:
    ``nanomind`` (after `pip install -e .`) calls :func:`nanomind.cli.main.main`.

Subcommands:
    train    — train a model from config
    generate — generate text from a checkpoint
    eval     — evaluate model perplexity/accuracy
    info     — display model summary
"""
from nanomind.cli.main import main
__all__ = ["main"]
