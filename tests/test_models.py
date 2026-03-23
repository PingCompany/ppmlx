"""Tests for ppmlx.models — Model Registry."""
from __future__ import annotations
import pytest
from pathlib import Path

from ppmlx.models import (
    DEFAULT_ALIASES,
    ModelNotFoundError,
    all_aliases,
    download_model,
    get_model_path,
    is_embed_model,
    is_vision_model,
    list_local_models,
    load_user_aliases,
    remove_model,
    remove_user_alias,
    repo_to_local_name,
    resolve_alias,
    save_user_alias,
)


def test_default_aliases_count():
    assert len(DEFAULT_ALIASES) >= 5


def test_resolve_alias_direct_repo():
    assert resolve_alias("org/repo") == "org/repo"


def test_resolve_alias_known():
    result = resolve_alias("qwen3.5:0.8b")
    assert result == "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"


def test_resolve_alias_user_override(tmp_home):
    save_user_alias("mymodel", "myorg/myrepo")
    assert resolve_alias("mymodel") == "myorg/myrepo"


def test_resolve_alias_prefix_match():
    # "qwen3.5" should resolve to the smallest qwen3.5 variant (alphabetically first)
    result = resolve_alias("qwen3.5")
    # Should be one of the qwen3.5 variants
    assert "Qwen3.5" in result or "qwen3.5" in result.lower()


def test_resolve_alias_unknown_raises(tmp_home):
    with pytest.raises(ModelNotFoundError) as exc_info:
        resolve_alias("totally-unknown-model-xyz")
    msg = str(exc_info.value)
    assert "totally-unknown-model-xyz" in msg
    assert "Available aliases" in msg


def test_is_vision_model_true():
    # Qwen3.5 models are vision (contain "Qwen3.5-")
    assert is_vision_model("mlx-community/Qwen3.5-4B-MLX-4bit") is True


def test_is_vision_model_false_text_only():
    # Text-only variants have "-text-" or "-Text-"
    assert is_vision_model("mlx-community/gemma-3-text-4b-it-4bit") is False


def test_is_embed_model():
    assert is_embed_model("embed:all-minilm") is True
    assert is_embed_model("qwen3.5:9b") is False


def test_save_load_user_alias(tmp_home):
    save_user_alias("custom:model", "myorg/custom-model")
    aliases = load_user_aliases()
    assert aliases["custom:model"] == "myorg/custom-model"


def test_remove_user_alias(tmp_home):
    save_user_alias("temp:model", "myorg/temp-model")
    result = remove_user_alias("temp:model")
    assert result is True
    aliases = load_user_aliases()
    assert "temp:model" not in aliases


def test_list_local_models_empty(tmp_home):
    result = list_local_models()
    assert result == []


def test_list_local_models_with_model(tmp_home):
    # Create a fake model directory with a file
    models_dir = tmp_home / ".ppmlx" / "models"
    fake_model = models_dir / "mlx-community--Qwen3.5-4B-MLX-4bit"
    fake_model.mkdir(parents=True)
    (fake_model / "config.json").write_text('{"model_type": "qwen"}')

    result = list_local_models()
    assert len(result) == 1
    assert result[0]["name"] == "mlx-community--Qwen3.5-4B-MLX-4bit"
    assert result[0]["repo_id"] == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert result[0]["size_gb"] >= 0
    assert isinstance(result[0]["path"], Path)


def test_remove_model_not_found(tmp_home):
    result = remove_model("nonexistent-model-xyz")
    assert result is False


def test_repo_to_local_name():
    assert repo_to_local_name("org/repo") == "org--repo"
    assert repo_to_local_name("mlx-community/Qwen3.5-4B-MLX-4bit") == "mlx-community--Qwen3.5-4B-MLX-4bit"


def test_download_model_already_exists(tmp_home, monkeypatch):
    # Create a pre-existing model directory with content
    models_dir = tmp_home / ".ppmlx" / "models"
    local_path = models_dir / "mlx-community--Qwen3.5-0.8B-OptiQ-4bit"
    local_path.mkdir(parents=True)
    (local_path / "config.json").write_text("{}")

    # snapshot_download should NOT be called
    called = []
    def fake_snapshot_download(**kwargs):
        called.append(kwargs)
        return str(local_path)

    monkeypatch.setattr("ppmlx.models.snapshot_download", fake_snapshot_download, raising=False)

    # Patch at the huggingface_hub level since download_model imports it
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    result = download_model("qwen3.5:0.8b")
    assert result == local_path
    assert len(called) == 0  # Should not have been called


def test_download_model_calls_snapshot(tmp_home, monkeypatch):
    called = []

    def fake_snapshot_download(repo_id, local_dir, token, ignore_patterns, **kwargs):
        called.append({
            "repo_id": repo_id,
            "local_dir": local_dir,
            "token": token,
            "ignore_patterns": ignore_patterns,
        })
        # Create a file so the directory appears non-empty
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    result = download_model("qwen3.5:0.8b")

    assert len(called) == 1
    assert called[0]["repo_id"] == "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    assert "Qwen3.5-0.8B-OptiQ-4bit" in called[0]["local_dir"]
    assert called[0]["token"] is None
    assert result.exists()
