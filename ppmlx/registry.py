"""Built-in model registry shipped with ppmlx.

The registry provides curated aliases for popular mlx-community models,
sourced from https://huggingface.co/mlx-community. It can be disabled
via config.toml ([registry] enabled = false) or PPMLX_REGISTRY_ENABLED=0.

On each load the registry merges:
  1. Bundled ``registry_data.json`` (curated baseline)
  2. Cached trending models from HuggingFace (auto-refreshed per config)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parent / "registry_data.json"
_cache: dict[str, Any] | None = None


def _load_bundled() -> dict[str, Any]:
    """Load the bundled registry_data.json."""
    with open(_DATA_FILE) as f:
        return json.load(f)


def _load() -> dict[str, Any]:
    """Load registry: bundled baseline merged with fetched overlay."""
    global _cache
    if _cache is not None:
        return _cache

    bundled = _load_bundled()

    try:
        from ppmlx.config import load_config
        cfg = load_config()
        if not cfg.registry.enabled:
            _cache = bundled
            return _cache

        from ppmlx.registry_fetch import maybe_refresh
        fetched = maybe_refresh(cfg.registry.refresh)
    except Exception:
        logger.debug("Registry fetch/merge failed, using bundled only", exc_info=True)
        fetched = None

    if fetched and fetched.get("models"):
        merged_models = dict(bundled.get("models", {}))
        for alias, entry in fetched["models"].items():
            if alias in merged_models:
                # Existing alias — only update download count
                merged_models[alias]["downloads"] = entry.get(
                    "downloads", merged_models[alias].get("downloads", 0)
                )
            else:
                # New trending model not in bundled — add it
                merged_models[alias] = entry
        _cache = {
            **bundled,
            "models": merged_models,
        }
    else:
        _cache = bundled

    return _cache


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
    """Return the full registry entries {alias: {repo_id, params_b, size_gb, ...}}."""
    return dict(_load().get("models", {}))


def registry_lookup(alias: str) -> dict[str, Any] | None:
    """Look up a single alias. Returns the entry dict or None."""
    return _load().get("models", {}).get(alias)
