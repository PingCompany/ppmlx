"""Dynamic model registry for ppmlx.

The registry is populated from the top-downloaded ``mlx-community`` models on
HuggingFace and cached locally. It can be disabled via config.toml
([registry] enabled = false) or PPMLX_REGISTRY_ENABLED=0.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, Any] | None = None
_EMPTY_REGISTRY: dict[str, Any] = {
    "version": 1,
    "updated": "unknown",
    "source": "huggingface-api-downloads",
    "models": {},
}


def _load() -> dict[str, Any]:
    """Load the dynamic registry cache; never fall back to bundled data."""
    global _cache
    if _cache is not None:
        return _cache

    try:
        from ppmlx.config import load_config
        cfg = load_config()
        if not cfg.registry.enabled:
            _cache = dict(_EMPTY_REGISTRY)
            return _cache

        from ppmlx.registry_fetch import maybe_refresh
        fetched = maybe_refresh(cfg.registry.refresh)
    except Exception:
        logger.debug("Registry fetch/load failed", exc_info=True)
        fetched = None

    if fetched and fetched.get("models"):
        _cache = fetched
    else:
        _cache = dict(_EMPTY_REGISTRY)

    return _cache


def refresh_registry() -> dict[str, Any]:
    """Force-refresh the dynamic registry cache and reload merged entries."""
    global _cache
    _cache = None
    try:
        from ppmlx.registry_fetch import maybe_refresh
        maybe_refresh("always")
    except Exception:
        logger.debug("Registry force-refresh failed", exc_info=True)
    _cache = None
    return _load()


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
