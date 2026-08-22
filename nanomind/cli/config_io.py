"""
nanomind/cli/config_io.py — Config file I/O for JSON and YAML formats.
"""

from __future__ import annotations

import json
from pathlib import Path

from nanomind.config import NanoMindConfig


def load_config(path: str | Path) -> NanoMindConfig:
    """
    Load a NanoMindConfig from a JSON or YAML file.

    Supports ``.json`` and ``.yaml`` / ``.yml`` extensions.

    Args:
        path: Path to the config file.

    Returns:
        Parsed :class:`~nanomind.config.NanoMindConfig`.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install it with: pip install pyyaml"
            )
    else:
        raise ValueError(f"Unsupported config format: '{suffix}'. Use .json or .yaml")

    return NanoMindConfig.from_dict(data)


def save_config(cfg: NanoMindConfig, path: str | Path) -> None:
    """
    Save a NanoMindConfig to JSON or YAML.

    Args:
        cfg:  Config to save.
        path: Destination file path (.json or .yaml).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()

    if suffix == ".json":
        p.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            p.write_text(yaml.dump(cfg.to_dict(), default_flow_style=False), encoding="utf-8")
        except ImportError:
            raise ImportError("PyYAML required. pip install pyyaml")
    else:
        raise ValueError(f"Unsupported format: '{suffix}'")


def merge_cli_overrides(cfg: NanoMindConfig, overrides: dict) -> NanoMindConfig:
    """
    Apply CLI argument overrides on top of a file-based config.

    Keys use dot notation: ``"model.d_model"`` overrides ``cfg.model.d_model``.

    Args:
        cfg:       Base config loaded from file.
        overrides: Dict of ``"section.key": value`` overrides from CLI args.

    Returns:
        Updated config (in-place modification + returned for chaining).
    """
    for dotkey, value in overrides.items():
        if value is None:
            continue
        parts = dotkey.split(".", 1)
        if len(parts) == 2:
            section, key = parts
            sub = getattr(cfg, section, None)
            if sub is not None and hasattr(sub, key):
                setattr(sub, key, type(getattr(sub, key))(value))
        elif hasattr(cfg, dotkey):
            setattr(cfg, dotkey, value)
    return cfg
