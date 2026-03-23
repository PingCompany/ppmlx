from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Any


# ── Built-in alias map (exact HuggingFace repo IDs from spec) ──────────

DEFAULT_ALIASES: dict[str, str] = {
    # Qwen 3.5 — released Feb 2026
    "qwen3.5:0.8b":       "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
    "qwen3.5:2b":         "mlx-community/Qwen3.5-2B-MLX-4bit",
    "qwen3.5:4b":         "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen3.5:9b":         "mlx-community/Qwen3.5-9B-MLX-4bit",
    "qwen3.5:27b":        "mlx-community/Qwen3.5-27B-4bit",
    "qwen3.5:35b-a3b":    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "qwen3.5:122b-a10b":  "mlx-community/Qwen3.5-122B-A10B-4bit",
    # GPT-OSS (OpenAI open weights) — released Aug 2025
    "gpt-oss:20b":        "mlx-community/gpt-oss-20b-4bit",
    "gpt-oss:120b":       "mlx-community/gpt-oss-120b-4bit",
}

# Patterns for routing
_VISION_INDICATORS = ["-VL-", "-vlm", "Qwen3.5-"]
_TEXT_ONLY_INDICATORS = ["-text-", "-Text-", "OptiQ"]
_EMBED_PREFIXES = ("embed:",)


class ModelNotFoundError(Exception):
    pass


def _get_ppmlx_dir() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        return get_ppmlx_dir()
    except ImportError:
        return Path.home() / ".ppmlx"


def _get_models_dir() -> Path:
    d = _get_ppmlx_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_aliases_file() -> Path:
    return _get_ppmlx_dir() / "aliases.json"


def load_user_aliases() -> dict[str, str]:
    """Load user-defined aliases from ~/.ppmlx/aliases.json."""
    p = _get_aliases_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_user_alias(name: str, repo_id: str) -> None:
    """Add or update a user alias."""
    aliases = load_user_aliases()
    aliases[name] = repo_id
    p = _get_aliases_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(aliases, indent=2))


def remove_user_alias(name: str) -> bool:
    """Remove a user alias. Returns True if it existed."""
    aliases = load_user_aliases()
    if name in aliases:
        del aliases[name]
        _get_aliases_file().write_text(json.dumps(aliases, indent=2))
        return True
    return False


def all_aliases() -> dict[str, str]:
    """Return merged dict: user aliases override defaults."""
    merged = dict(DEFAULT_ALIASES)
    merged.update(load_user_aliases())
    return merged


def resolve_alias(name: str) -> str:
    """
    Resolve a model name to a HuggingFace repo ID.

    1. If name contains '/' -> direct repo ID, return as-is
    2. Check user aliases
    3. Check DEFAULT_ALIASES
    4. Prefix match (e.g. 'qwen3.5' -> smallest qwen3.5 variant)
    5. Raise ModelNotFoundError with helpful message
    """
    if "/" in name:
        return name

    user = load_user_aliases()
    if name in user:
        return user[name]
    if name in DEFAULT_ALIASES:
        return DEFAULT_ALIASES[name]

    # Prefix match: find smallest variant
    matches = [(k, v) for k, v in DEFAULT_ALIASES.items() if k.startswith(name + ":") or k == name]
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    available = sorted({**DEFAULT_ALIASES, **user}.keys())
    raise ModelNotFoundError(
        f"Unknown model: '{name}'\n"
        f"Available aliases: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}\n"
        f"Or use a HuggingFace repo ID directly (e.g. 'mlx-community/model-name')."
    )


def is_vision_model(repo_id: str) -> bool:
    """Detect if a model should be loaded with mlx-vlm."""
    if any(ind in repo_id for ind in _TEXT_ONLY_INDICATORS):
        return False
    return any(ind in repo_id for ind in _VISION_INDICATORS)


def is_embed_model(alias_or_repo: str) -> bool:
    """Detect if a model is an embedding model."""
    if alias_or_repo.startswith(_EMBED_PREFIXES):
        return True
    lower = alias_or_repo.lower()
    return any(p in lower for p in ["embed", "embedding", "minilm", "bge-", "nomic-"])


def repo_to_local_name(repo_id: str) -> str:
    """Convert 'org/repo' -> 'org--repo' for local directory name."""
    return repo_id.replace("/", "--")


def _get_repo_size(repo_id: str, token: str | None = None) -> int | None:
    """Return total download size in bytes by listing the repo tree. Returns None on failure."""
    try:
        from huggingface_hub import list_repo_tree
        return sum(
            getattr(f, "size", 0) or 0
            for f in list_repo_tree(repo_id, token=token, recursive=True)
        )
    except Exception:
        return None


def _get_hf_token(explicit: str | None = None) -> str | None:
    """Return HF token: explicit arg > config.toml > HF_TOKEN env var."""
    if explicit:
        return explicit
    try:
        import tomllib
        cfg_path = _get_ppmlx_dir() / "config.toml"
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                data = tomllib.load(f)
            tok = data.get("auth", {}).get("hf_token")
            if tok:
                return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or None


def download_model(alias_or_repo: str, token: str | None = None) -> Path:
    """
    Download a model from HuggingFace Hub with a uv-style Rich progress bar.
    Returns local path.
    """
    from rich.progress import (
        Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TextColumn,
    )

    token = _get_hf_token(token)
    repo_id = resolve_alias(alias_or_repo)
    local_name = repo_to_local_name(repo_id)
    local_path = _get_models_dir() / local_name

    if local_path.exists() and any(local_path.iterdir()):
        return local_path

    local_path.mkdir(parents=True, exist_ok=True)
    total = _get_repo_size(repo_id, token)

    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.file_download import base_tqdm as _tqdm_base
    except ImportError:
        from tqdm import tqdm as _tqdm_base  # type: ignore[assignment]

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=10,
        transient=False,
    ) as progress:
        task = progress.add_task(f"↓ {alias_or_repo}", total=total)

        class _RichTqdm(_tqdm_base):  # type: ignore[valid-type]
            """Forwards byte-level updates to the Rich progress bar."""
            def display(self, *args, **kwargs) -> None:
                pass  # suppress tqdm's own output

            def update(self, n: int = 1) -> None:
                super().update(n)
                if n and n > 0:
                    progress.update(task, advance=int(n))

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                token=token,
                ignore_patterns=["*.md", "*.txt", "original/*"],
                tqdm_class=_RichTqdm,
            )
        except Exception as e:
            shutil.rmtree(local_path, ignore_errors=True)
            raise ModelNotFoundError(f"Failed to download '{repo_id}': {e}") from e
        finally:
            if total:
                progress.update(task, completed=total)

    return local_path


def get_model_path(alias_or_repo: str) -> Path | None:
    """Return local path if model exists, else None."""
    try:
        repo_id = resolve_alias(alias_or_repo)
    except ModelNotFoundError:
        repo_id = alias_or_repo

    local_name = repo_to_local_name(repo_id)
    p = _get_models_dir() / local_name
    if p.exists() and any(p.iterdir()):
        return p
    return None


def list_local_models() -> list[dict[str, Any]]:
    """List all locally downloaded models."""
    models_dir = _get_models_dir()
    result = []
    if not models_dir.exists():
        return result

    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024 ** 3)
        repo_id = d.name.replace("--", "/", 1)
        aliases_for_model = [k for k, v in all_aliases().items() if v == repo_id]
        result.append({
            "name": d.name,
            "repo_id": repo_id,
            "alias": aliases_for_model[0] if aliases_for_model else repo_id,
            "size_gb": round(size_gb, 2),
            "path": d,
        })
    return result


def remove_model(alias_or_repo: str) -> bool:
    """Remove a locally downloaded model. Returns True if removed."""
    path = get_model_path(alias_or_repo)
    if path is None:
        return False
    shutil.rmtree(path)
    return True
