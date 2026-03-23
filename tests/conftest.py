"""Stubs out mlx/* at collection time so tests run on any platform."""
from __future__ import annotations
import sys
import types
import pytest


def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


for _p in [
    "mlx", "mlx.core", "mlx.nn",
    "mlx_lm", "mlx_lm.utils",
    "mlx_vlm", "mlx_vlm.utils",
    "mlx_embeddings", "mlx_embeddings.utils",
]:
    if _p not in sys.modules:
        _stub(_p)


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    """Redirect ~/.pp-llm to a temp directory for isolation."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path
