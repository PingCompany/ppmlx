"""Built-in model registry shipped with ppmlx.

The registry provides curated aliases for popular mlx-community models,
sourced from https://huggingface.co/mlx-community. It can be disabled
via config.toml ([registry] enabled = false) or PPMLX_REGISTRY_ENABLED=0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DATA_FILE = Path(__file__).parent / "registry_data.json"
_cache: dict[str, Any] | None = None

# Defaults for garden-specific fields (backward compatible with older entries)
_GARDEN_DEFAULTS: dict[str, Any] = {
    "recommended_ram_gb": None,
    "quality_tier": None,
    "use_cases": [],
    "notes": None,
}


def _load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with open(_DATA_FILE) as f:
            _cache = json.load(f)
    return _cache


def _with_defaults(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *entry* with missing garden fields filled in."""
    result = dict(entry)
    for key, default in _GARDEN_DEFAULTS.items():
        if key not in result:
            result[key] = default if not isinstance(default, list) else list(default)
    return result


def registry_meta() -> dict[str, Any]:
    """Return registry metadata (version, updated, source)."""
    data = _load()
    return {
        "version": data.get("version", 0),
        "updated": data.get("updated", "unknown"),
        "source": data.get("source", ""),
        "count": len(data.get("models", {})),
    }


def registry_aliases() -> dict[str, str]:
    """Return {alias: repo_id} for all registry models."""
    data = _load()
    return {alias: entry["repo_id"] for alias, entry in data.get("models", {}).items()}


def registry_entries() -> dict[str, dict[str, Any]]:
    """Return the full registry entries {alias: {repo_id, params_b, size_gb, ...}}.

    Garden-specific fields (recommended_ram_gb, quality_tier, use_cases, notes)
    are filled with defaults when absent, so callers never need to guard.
    """
    return {
        alias: _with_defaults(entry)
        for alias, entry in _load().get("models", {}).items()
    }


def registry_lookup(alias: str) -> dict[str, Any] | None:
    """Look up a single alias. Returns the entry dict (with defaults) or None."""
    entry = _load().get("models", {}).get(alias)
    if entry is None:
        return None
    return _with_defaults(entry)
