"""tests/test_version.py — Package version and public API tests."""

import nanomind


def test_version_is_string():
    assert isinstance(nanomind.__version__, str)


def test_version_is_1_0_0():
    assert nanomind.__version__ == "1.0.0"


def test_public_exports():
    assert hasattr(nanomind, "NanoMind")
    assert hasattr(nanomind, "ModelConfig")
    assert hasattr(nanomind, "NanoMindConfig")
