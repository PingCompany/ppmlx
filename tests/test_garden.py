"""Tests for ppmlx garden command and registry garden fields."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

# Mock modules before importing cli (same pattern as test_cli.py)
for mod_name in [
    "ppmlx.models", "ppmlx.engine", "ppmlx.db",
    "ppmlx.config", "ppmlx.memory", "ppmlx.modelfile",
    "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
    "ppmlx.registry",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.cli import app

runner = CliRunner()


# ── Registry tests ────────────────────────────────────────────────────

def _sample_registry_data() -> dict:
    """Build a minimal registry with garden fields for testing."""
    return {
        "version": 1,
        "updated": "2026-03-24",
        "source": "https://huggingface.co/mlx-community",
        "models": {
            "test-small": {
                "repo_id": "mlx-community/test-small-4bit",
                "params_b": 4.0,
                "size_gb": 3.0,
                "type": "dense",
                "lab": "TestLab",
                "modalities": "text",
                "downloads": 100000,
                "created": "2026-01-01",
                "recommended_ram_gb": 6,
                "quality_tier": "lightweight",
                "use_cases": ["chat", "instruction"],
                "notes": "A compact model for quick tasks.",
            },
            "test-medium": {
                "repo_id": "mlx-community/test-medium-4bit",
                "params_b": 9.0,
                "size_gb": 6.0,
                "type": "dense",
                "lab": "TestLab",
                "modalities": "text, vision",
                "downloads": 80000,
                "created": "2026-02-01",
                "recommended_ram_gb": 10,
                "quality_tier": "balanced",
                "use_cases": ["chat", "code", "reasoning"],
                "notes": "Good all-rounder for most Macs.",
            },
            "test-large": {
                "repo_id": "mlx-community/test-large-4bit",
                "params_b": 27.0,
                "size_gb": 16.0,
                "type": "dense",
                "lab": "BigLab",
                "modalities": "text",
                "downloads": 50000,
                "created": "2026-03-01",
                "recommended_ram_gb": 20,
                "quality_tier": "flagship",
                "use_cases": ["chat", "code", "reasoning", "creative"],
                "notes": "High quality for demanding tasks.",
            },
            "test-huge": {
                "repo_id": "mlx-community/test-huge-4bit",
                "params_b": 120.0,
                "size_gb": 63.0,
                "type": "dense",
                "lab": "BigLab",
                "modalities": "text",
                "downloads": 30000,
                "created": "2026-03-01",
                "recommended_ram_gb": 72,
                "quality_tier": "flagship",
                "use_cases": ["chat", "code", "reasoning", "creative"],
                "notes": "Needs high-end hardware.",
            },
            "test-no-garden": {
                "repo_id": "mlx-community/test-no-garden-4bit",
                "params_b": 7.0,
                "size_gb": 4.0,
                "type": "dense",
                "lab": "OtherLab",
                "modalities": "text",
                "downloads": 5000,
                "created": "2025-06-01",
            },
        },
    }


def _registry_entries_from_sample():
    """Return registry_entries()-style dict from sample data, with defaults."""
    defaults = {
        "recommended_ram_gb": None,
        "quality_tier": None,
        "use_cases": [],
        "notes": None,
    }
    result = {}
    for alias, entry in _sample_registry_data()["models"].items():
        e = dict(entry)
        for k, v in defaults.items():
            if k not in e:
                e[k] = v if not isinstance(v, list) else list(v)
        result[alias] = e
    return result


def _setup_garden_mocks(system_ram_gb: float = 24.0):
    """Configure mocks so garden commands work with sample registry data."""
    entries = _registry_entries_from_sample()
    aliases = {alias: e["repo_id"] for alias, e in entries.items()}

    sys.modules["ppmlx.models"].DEFAULT_ALIASES = {}
    sys.modules["ppmlx.models"].load_user_aliases = MagicMock(return_value={})
    sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value=aliases)
    sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=[])
    sys.modules["ppmlx.models"].load_favorites = MagicMock(return_value=[])
    sys.modules["ppmlx.registry"].registry_entries = MagicMock(return_value=entries)
    sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=system_ram_gb)


# ── Registry loading tests ────────────────────────────────────────────


def _load_real_registry():
    """Load the real registry module, bypassing any mocks."""
    import importlib
    import importlib.util

    registry_path = Path(__file__).parent.parent / "ppmlx" / "registry.py"
    spec = importlib.util.spec_from_file_location("_real_registry", registry_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRegistryGardenFields:
    """Test that registry.py handles garden fields correctly."""

    def test_load_with_garden_fields(self, tmp_path):
        """Registry entries with garden fields are returned as-is."""
        reg_mod = _load_real_registry()

        data = _sample_registry_data()
        data_file = tmp_path / "registry_data.json"
        data_file.write_text(json.dumps(data))

        reg_mod._DATA_FILE = data_file
        reg_mod._cache = None

        entries = reg_mod.registry_entries()
        assert entries["test-small"]["recommended_ram_gb"] == 6
        assert entries["test-small"]["quality_tier"] == "lightweight"
        assert entries["test-small"]["use_cases"] == ["chat", "instruction"]
        assert entries["test-small"]["notes"] == "A compact model for quick tasks."

    def test_load_without_garden_fields_uses_defaults(self, tmp_path):
        """Entries missing garden fields get sensible defaults."""
        reg_mod = _load_real_registry()

        data = _sample_registry_data()
        data_file = tmp_path / "registry_data.json"
        data_file.write_text(json.dumps(data))

        reg_mod._DATA_FILE = data_file
        reg_mod._cache = None

        entries = reg_mod.registry_entries()
        # test-no-garden has no garden fields
        no_garden = entries["test-no-garden"]
        assert no_garden["recommended_ram_gb"] is None
        assert no_garden["quality_tier"] is None
        assert no_garden["use_cases"] == []
        assert no_garden["notes"] is None

    def test_lookup_returns_defaults(self, tmp_path):
        """registry_lookup fills in defaults for missing fields."""
        reg_mod = _load_real_registry()

        data = _sample_registry_data()
        data_file = tmp_path / "registry_data.json"
        data_file.write_text(json.dumps(data))

        reg_mod._DATA_FILE = data_file
        reg_mod._cache = None

        entry = reg_mod.registry_lookup("test-no-garden")
        assert entry is not None
        assert entry["use_cases"] == []
        assert entry["quality_tier"] is None

        # Annotated entry keeps its values
        reg_mod._cache = None  # clear cache between lookups
        entry2 = reg_mod.registry_lookup("test-medium")
        assert entry2["quality_tier"] == "balanced"
        assert "code" in entry2["use_cases"]

    def test_lookup_nonexistent_returns_none(self, tmp_path):
        """registry_lookup returns None for unknown aliases."""
        reg_mod = _load_real_registry()

        data = _sample_registry_data()
        data_file = tmp_path / "registry_data.json"
        data_file.write_text(json.dumps(data))

        reg_mod._DATA_FILE = data_file
        reg_mod._cache = None
        assert reg_mod.registry_lookup("nonexistent") is None


# ── Garden CLI command tests ──────────────────────────────────────────


class TestGardenBrowse:
    """Test the 'ppmlx garden' browse command."""

    def test_garden_shows_table(self):
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden"])
        assert result.exit_code == 0
        assert "Model Garden" in result.output
        assert "test-small" in result.output
        assert "test-medium" in result.output

    def test_garden_filter_by_use_case(self):
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden", "--filter", "code"])
        assert result.exit_code == 0
        # test-medium and test-large have "code" use case
        assert "test-medium" in result.output
        assert "test-large" in result.output
        # test-small does NOT have "code"
        assert "test-small" not in result.output

    def test_garden_filter_nonexistent_use_case(self):
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden", "--filter", "nonexistent"])
        assert result.exit_code == 0
        assert "No models match" in result.output

    def test_garden_ram_filter(self):
        _setup_garden_mocks(system_ram_gb=12.0)
        result = runner.invoke(app, ["garden", "--ram"])
        assert result.exit_code == 0
        # Only test-small (6 GB) and test-medium (10 GB) fit in 12 GB
        assert "test-small" in result.output
        assert "test-medium" in result.output
        assert "test-large" not in result.output
        assert "test-huge" not in result.output

    def test_garden_sort_by_size(self):
        _setup_garden_mocks(system_ram_gb=128.0)
        result = runner.invoke(app, ["garden", "--sort", "size"])
        assert result.exit_code == 0
        # All garden models should be present
        assert "test-small" in result.output
        assert "test-huge" in result.output

    def test_garden_sort_by_params(self):
        _setup_garden_mocks(system_ram_gb=128.0)
        result = runner.invoke(app, ["garden", "--sort", "params"])
        assert result.exit_code == 0
        assert "test-huge" in result.output

    def test_garden_excludes_unannotated(self):
        """Models without garden annotations are excluded from garden view."""
        _setup_garden_mocks(system_ram_gb=128.0)
        result = runner.invoke(app, ["garden"])
        assert result.exit_code == 0
        # test-no-garden has no quality_tier or use_cases
        assert "test-no-garden" not in result.output

    def test_garden_color_coding_present(self):
        """RAM color coding markup is present in output."""
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden"])
        assert result.exit_code == 0
        # The output should contain the RAM values
        assert "6 GB" in result.output   # test-small recommended
        assert "10 GB" in result.output  # test-medium recommended


class TestGardenRecommend:
    """Test the 'ppmlx garden recommend' subcommand."""

    def test_recommend_shows_top_models(self):
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden", "recommend"])
        assert result.exit_code == 0
        assert "recommendation" in result.output.lower()
        # Should show at least one model
        assert "test-" in result.output

    def test_recommend_respects_ram(self):
        _setup_garden_mocks(system_ram_gb=8.0)
        result = runner.invoke(app, ["garden", "recommend"])
        assert result.exit_code == 0
        # With 8 GB, only test-small (6 GB needed) fits
        assert "test-small" in result.output
        assert "test-large" not in result.output
        assert "test-huge" not in result.output

    def test_recommend_filter_by_use_case(self):
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden", "recommend", "--use-case", "code"])
        assert result.exit_code == 0
        # test-medium and test-large have "code"; test-large is flagship
        assert "test-large" in result.output

    def test_recommend_no_fitting_models(self):
        _setup_garden_mocks(system_ram_gb=2.0)
        result = runner.invoke(app, ["garden", "recommend"])
        assert result.exit_code == 0
        assert "No curated models fit" in result.output

    def test_recommend_prefers_flagship(self):
        """Flagship models are recommended before balanced/lightweight."""
        _setup_garden_mocks(system_ram_gb=24.0)
        result = runner.invoke(app, ["garden", "recommend", "--count", "1"])
        assert result.exit_code == 0
        # test-large is flagship and fits in 24 GB (needs 20)
        assert "test-large" in result.output

    def test_recommend_count(self):
        """--count limits the number of recommendations."""
        _setup_garden_mocks(system_ram_gb=128.0)
        result = runner.invoke(app, ["garden", "recommend", "--count", "2"])
        assert result.exit_code == 0
        # Count model names in output (rough check)
        model_mentions = sum(1 for name in ["test-small", "test-medium", "test-large", "test-huge"]
                            if name in result.output)
        assert model_mentions <= 3  # at most 2 recommendations + maybe one in footer
