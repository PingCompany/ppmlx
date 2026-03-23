"""Tests for pp_llm/memory.py"""
from __future__ import annotations
import subprocess
from pathlib import Path
import pytest

from pp_llm.memory import (
    get_system_ram_bytes,
    get_system_ram_gb,
    estimate_model_memory_bytes,
    estimate_model_memory_gb,
    check_memory_warning,
    format_size,
)


def test_get_system_ram_bytes():
    result = get_system_ram_bytes()
    assert isinstance(result, int)
    assert result > 0


def test_get_system_ram_gb():
    result = get_system_ram_gb()
    assert isinstance(result, float)
    assert result > 0


def test_estimate_model_memory_bytes(tmp_path):
    # Create files totaling exactly 100 bytes
    f1 = tmp_path / "weights.bin"
    f1.write_bytes(b"x" * 60)
    f2 = tmp_path / "config.json"
    f2.write_bytes(b"y" * 40)

    result = estimate_model_memory_bytes(tmp_path)
    assert result == int(100 * 1.2)  # 120 bytes


def test_estimate_nonexistent_path(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    result = estimate_model_memory_bytes(nonexistent)
    assert result == 0


def test_check_memory_warning_small_model(tmp_path, monkeypatch):
    # Mock system RAM to 16 GB
    monkeypatch.setattr(
        "pp_llm.memory.get_system_ram_bytes",
        lambda: 16 * 1024 ** 3,
    )
    # Create ~1 GB of files (in model_gb terms: 1 GB * 1.2 = 1.2 GB, ratio = 1.2/16 = 0.075)
    # Actually we need the estimated bytes to be ~1 GB after * 1.2
    # So raw files = 1 GB / 1.2 ≈ 0.833 GB ≈ 894 MB
    # For test speed, mock estimate_model_memory_gb directly
    monkeypatch.setattr(
        "pp_llm.memory.estimate_model_memory_gb",
        lambda p: 1.0,
    )
    # 1.0 GB / 16 GB = 0.0625 ratio → no warning
    result = check_memory_warning(tmp_path)
    assert result is None


def test_check_memory_warning_too_large(tmp_path, monkeypatch):
    # Mock system RAM to 8 GB
    monkeypatch.setattr(
        "pp_llm.memory.get_system_ram_bytes",
        lambda: 8 * 1024 ** 3,
    )
    # Mock model size to 10 GB — exceeds RAM
    monkeypatch.setattr(
        "pp_llm.memory.estimate_model_memory_gb",
        lambda p: 10.0,
    )
    result = check_memory_warning(tmp_path)
    assert result is not None
    assert "10.0 GB" in result
    assert "8" in result


def test_format_size_bytes():
    assert format_size(512) == "512 B"


def test_format_size_gb():
    assert format_size(5 * 1024 ** 3) == "5.00 GB"


def test_sysctl_failure_fallback(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("sysctl not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    result = get_system_ram_bytes()
    assert result == 8 * 1024 ** 3
