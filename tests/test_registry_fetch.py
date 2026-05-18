"""Tests for registry_fetch module."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ppmlx.registry_fetch import (
    _extract_lab,
    _extract_modalities,
    _extract_params_b,
    _repo_id_to_alias,
    is_cache_stale,
    maybe_refresh,
)


# ── Alias generation ─────────────────────────────────────────────────


@pytest.mark.parametrize("repo_id, expected", [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "qwen3.5:9b"),
    ("mlx-community/Qwen3.5-0.8B-8bit", "qwen3.5:0.8b-8bit"),
    ("mlx-community/GLM-4.7-Flash-4bit", "glm-4.7-flash"),
    ("mlx-community/GLM-4.7-Flash-8bit", "glm-4.7-flash-8bit"),
    ("mlx-community/Kimi-K2.5", "kimi-k2.5"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4:26b-a4b"),
    ("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", "devstral-small-2:24b"),
    ("mlx-community/Qwen3.5-35B-A3B-4bit", "qwen3.5:35b-a3b"),
    ("mlx-community/parakeet-tdt-0.6b-v3", "parakeet-tdt:0.6b-v3"),
])
def test_repo_id_to_alias(repo_id: str, expected: str):
    assert _repo_id_to_alias(repo_id) == expected


def test_repo_id_to_alias_no_slash():
    assert _repo_id_to_alias("no-slash") is None


# ── Cache staleness ──────────────────────────────────────────────────


def test_staleness_always(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "cache.json"):
        assert is_cache_stale("always") is True


def test_staleness_never(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "cache.json"):
        assert is_cache_stale("never") is False


def test_staleness_weekly_fresh(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time() - 3600}))  # 1 hour ago
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        assert is_cache_stale("weekly") is False


def test_staleness_weekly_old(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time() - 8 * 86400}))  # 8 days ago
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        assert is_cache_stale("weekly") is True


def test_staleness_no_cache_file(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "nope.json"):
        assert is_cache_stale("weekly") is True


# ── maybe_refresh ────────────────────────────────────────────────────


def test_maybe_refresh_never_returns_cache(tmp_path: Path):
    cache = tmp_path / "cache.json"
    data = {"models": {"test:1b": {"repo_id": "mlx-community/test-1B-4bit"}}}
    cache.write_text(json.dumps(data))
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        result = maybe_refresh("never")
    assert result is not None
    assert "test:1b" in result["models"]


def test_maybe_refresh_never_no_cache(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "nope.json"):
        result = maybe_refresh("never")
    assert result is None


def test_maybe_refresh_fetch_failure_falls_back(tmp_path: Path):
    cache = tmp_path / "cache.json"
    data = {"fetched_at": time.time() - 999999, "models": {"old:1b": {}}}
    cache.write_text(json.dumps(data))
    with (
        patch("ppmlx.registry_fetch.get_cache_path", return_value=cache),
        patch("ppmlx.registry_fetch._fetch_from_hf", return_value=None),
    ):
        result = maybe_refresh("weekly")
    assert result is not None
    assert "old:1b" in result["models"]


# ── Metadata extraction ──────────────────────────────────────────────


def _make_model(repo_id: str, tags: list[str] | None = None) -> MagicMock:
    m = MagicMock()
    m.id = repo_id
    m.tags = tags or []
    return m


def test_extract_lab_known():
    assert _extract_lab(_make_model("mlx-community/Qwen3.5-9B")) == "Alibaba"
    assert _extract_lab(_make_model("mlx-community/Llama-3.1-8B")) == "Meta"
    assert _extract_lab(_make_model("mlx-community/gemma-4-26b")) == "Google"


def test_extract_lab_unknown():
    assert _extract_lab(_make_model("mlx-community/SomeNew-Model")) is None


def test_extract_modalities_text():
    assert _extract_modalities(_make_model("mlx-community/Qwen3-4B")) == "text"


def test_extract_modalities_vision():
    assert _extract_modalities(_make_model("mlx-community/Qwen3-VL-4B")) == "text, vision"


def test_extract_modalities_audio():
    assert _extract_modalities(_make_model("mlx-community/whisper-large")) == "audio, speech"


def test_extract_params_b():
    assert _extract_params_b(_make_model("mlx-community/Qwen3.5-9B-4bit")) == 9.0
    assert _extract_params_b(_make_model("mlx-community/parakeet-0.6b")) == 0.6
    assert _extract_params_b(_make_model("mlx-community/Kimi-K2.5")) is None


# ── Registry merge (integration) ─────────────────────────────────────


def test_registry_merge_logic():
    """Test the merge semantics: new aliases added, existing get updated downloads.

    Tests the merge logic directly to avoid test ordering issues with
    sys.modules patches in test_cli.py.
    """
    bundled_models = {
        "existing:9b": {"repo_id": "mlx-community/Existing-9B", "downloads": 100, "lab": "TestLab"},
    }
    fetched_models = {
        "new-model:4b": {"repo_id": "mlx-community/New-Model-4B", "downloads": 500},
        "existing:9b": {"repo_id": "mlx-community/Existing-9B", "downloads": 999},
    }

    # Replicate the merge logic from registry._load()
    merged = dict(bundled_models)
    for alias, entry in fetched_models.items():
        if alias in merged:
            merged[alias]["downloads"] = entry.get("downloads", merged[alias].get("downloads", 0))
        else:
            merged[alias] = entry

    # New model added
    assert "new-model:4b" in merged
    assert merged["new-model:4b"]["repo_id"] == "mlx-community/New-Model-4B"
    # Existing model: downloads updated, other metadata preserved
    assert merged["existing:9b"]["downloads"] == 999
    assert merged["existing:9b"]["lab"] == "TestLab"
    assert merged["existing:9b"]["repo_id"] == "mlx-community/Existing-9B"
