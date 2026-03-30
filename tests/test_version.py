"""Ensure version is in sync across pyproject.toml and ppmlx/__init__.py."""

import tomllib
from pathlib import Path


def test_version_sync():
    root = Path(__file__).parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        toml_ver = tomllib.load(f)["project"]["version"]
    from ppmlx import __version__

    assert __version__ == toml_ver, f"Version mismatch: pyproject.toml={toml_ver}, __init__.py={__version__}"
