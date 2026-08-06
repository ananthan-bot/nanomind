"""
nanomind/utils/io.py — File I/O helpers for NanoMind.
"""

import json
from pathlib import Path
from typing import Any


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents."""
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to a file, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write data as JSON to a file."""
    write_text(path, json.dumps(data, ensure_ascii=False, indent=indent))


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Returns the resolved Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
