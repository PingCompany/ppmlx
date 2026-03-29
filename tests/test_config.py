"""Tests for ppmlx.config module."""
from __future__ import annotations

import pytest

from ppmlx.config import (
    Config,
    DefaultsConfig,
    LoggingConfig,
    MemoryConfig,
    ServerConfig,
    ToolAwarenessConfig,
    get_ppmlx_dir,
    load_config,
)


class TestDefaultValues:
    def test_server_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 6767
        assert cfg.cors is True
        assert cfg.max_loaded_models == 2

    def test_defaults_config_defaults(self):
        cfg = DefaultsConfig()
        assert cfg.model == "qwen3.5:0.8b"
        assert cfg.embed_model == "embed:all-minilm"
        assert cfg.temperature == 0.7
        assert cfg.top_p == 1.0
        assert cfg.max_tokens == 2048

    def test_logging_defaults(self):
        cfg = LoggingConfig()
        assert cfg.enabled is True
        assert cfg.snapshot_interval_seconds == 60

    def test_memory_defaults(self):
        cfg = MemoryConfig()
        assert cfg.wired_limit_mb == 0

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.server.port == 6767
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.cors is True
        assert cfg.server.max_loaded_models == 2
        assert cfg.tool_awareness.mode == "no_tools_only"
        assert cfg.analytics.enabled is False
        assert cfg.analytics.provider == "posthog"

    def test_tool_awareness_defaults(self):
        cfg = ToolAwarenessConfig()
        assert cfg.mode == "no_tools_only"


class TestLoadConfigDefaults:
    def test_no_file_returns_defaults(self, tmp_home):
        cfg = load_config()
        assert cfg.server.port == 6767
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.cors is True
        assert cfg.server.max_loaded_models == 2
        assert cfg.defaults.model == "qwen3.5:0.8b"
        assert cfg.defaults.temperature == 0.7
        assert cfg.defaults.max_tokens == 2048
        assert cfg.tool_awareness.mode == "no_tools_only"


class TestTomlLoading:
    def test_load_from_toml(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        toml_content = """
[server]
host = "0.0.0.0"
port = 8080
cors = false
max_loaded_models = 4

[defaults]
model = "llama3:8b"
embed_model = "embed:custom"
temperature = 0.5
top_p = 0.9
max_tokens = 4096

[logging]
enabled = false
snapshot_interval_seconds = 120

[memory]
wired_limit_mb = 1024

[tool_awareness]
mode = "all"

[analytics]
enabled = false
provider = "posthog"
host = "https://stats.example.com"
project_api_key = "phc_test_123"
respect_do_not_track = true
"""
        (config_dir / "config.toml").write_text(toml_content)
        cfg = load_config()
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.port == 8080
        assert cfg.server.cors is False
        assert cfg.server.max_loaded_models == 4
        assert cfg.defaults.model == "llama3:8b"
        assert cfg.defaults.embed_model == "embed:custom"
        assert cfg.defaults.temperature == 0.5
        assert cfg.defaults.top_p == 0.9
        assert cfg.defaults.max_tokens == 4096
        assert cfg.logging.enabled is False
        assert cfg.logging.snapshot_interval_seconds == 120
        assert cfg.memory.wired_limit_mb == 1024
        assert cfg.tool_awareness.mode == "all"
        assert cfg.analytics.enabled is False
        assert cfg.analytics.provider == "posthog"
        assert cfg.analytics.host == "https://stats.example.com"
        assert cfg.analytics.project_api_key == "phc_test_123"
        assert cfg.analytics.respect_do_not_track is True

    def test_partial_toml(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        cfg = load_config()
        assert cfg.server.port == 9000
        assert cfg.server.host == "127.0.0.1"

    def test_malformed_toml_silently_ignored(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("this is not valid toml ][[[")
        cfg = load_config()
        assert cfg.server.port == 6767


class TestEnvVarOverrides:
    def test_port_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_PORT", "9999")
        cfg = load_config()
        assert cfg.server.port == 9999

    def test_host_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_HOST", "0.0.0.0")
        cfg = load_config()
        assert cfg.server.host == "0.0.0.0"

    def test_cors_false_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "false")
        cfg = load_config()
        assert cfg.server.cors is False

    def test_cors_zero_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "0")
        cfg = load_config()
        assert cfg.server.cors is False

    def test_cors_true_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "true")
        cfg = load_config()
        assert cfg.server.cors is True

    def test_cors_no_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "no")
        cfg = load_config()
        assert cfg.server.cors is False

    def test_max_loaded_models_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MAX_LOADED_MODELS", "5")
        cfg = load_config()
        assert cfg.server.max_loaded_models == 5

    def test_default_model_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_DEFAULT_MODEL", "mistral:7b")
        cfg = load_config()
        assert cfg.defaults.model == "mistral:7b"

    def test_temperature_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_TEMP", "0.3")
        cfg = load_config()
        assert cfg.defaults.temperature == 0.3

    def test_max_tokens_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MAX_TOKENS", "512")
        cfg = load_config()
        assert cfg.defaults.max_tokens == 512

    def test_log_enabled_false(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_LOG_ENABLED", "false")
        cfg = load_config()
        assert cfg.logging.enabled is False

    def test_log_snapshot_interval(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_LOG_SNAPSHOT_INTERVAL", "300")
        cfg = load_config()
        assert cfg.logging.snapshot_interval_seconds == 300

    def test_memory_wired_limit(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_WIRED_LIMIT", "2048")
        cfg = load_config()
        assert cfg.memory.wired_limit_mb == 2048

    def test_invalid_env_var_ignored(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_PORT", "not_a_number")
        cfg = load_config()
        assert cfg.server.port == 6767

    def test_tool_awareness_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_INJECT_TOOL_AWARENESS", "all")
        cfg = load_config()
        assert cfg.tool_awareness.mode == "all"

    def test_tool_awareness_env_var_legacy_true_maps_to_all(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_INJECT_TOOL_AWARENESS", "true")
        cfg = load_config()
        assert cfg.tool_awareness.mode == "all"

    def test_tool_awareness_env_var_legacy_false_maps_to_off(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_INJECT_TOOL_AWARENESS", "false")
        cfg = load_config()
        assert cfg.tool_awareness.mode == "off"

    def test_analytics_enabled_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_ENABLED", "false")
        cfg = load_config()
        assert cfg.analytics.enabled is False

    def test_analytics_host_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_HOST", "https://stats.example.com")
        cfg = load_config()
        assert cfg.analytics.host == "https://stats.example.com"

    def test_analytics_project_api_key_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_PROJECT_API_KEY", "phc_test_123")
        cfg = load_config()
        assert cfg.analytics.project_api_key == "phc_test_123"

    def test_analytics_legacy_website_id_env_var_maps_to_project_key(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_WEBSITE_ID", "legacy-site-123")
        cfg = load_config()
        assert cfg.analytics.project_api_key == "legacy-site-123"


class TestCliOverrides:
    def test_port_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"port": 8080})
        assert cfg.server.port == 8080

    def test_host_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"host": "0.0.0.0"})
        assert cfg.server.host == "0.0.0.0"

    def test_cors_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"cors": False})
        assert cfg.server.cors is False

    def test_model_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"model": "phi3:mini"})
        assert cfg.defaults.model == "phi3:mini"

    def test_temperature_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"temperature": 0.1})
        assert cfg.defaults.temperature == 0.1

    def test_max_tokens_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"max_tokens": 100})
        assert cfg.defaults.max_tokens == 100

    def test_none_value_ignored(self, tmp_home):
        cfg = load_config(cli_overrides={"port": None})
        assert cfg.server.port == 6767


class TestPriority:
    def test_cli_overrides_env(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_PORT", "9000")
        cfg = load_config(cli_overrides={"port": 7777})
        assert cfg.server.port == 7777

    def test_env_overrides_toml(self, tmp_home, monkeypatch):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        monkeypatch.setenv("PPMLX_PORT", "8888")
        cfg = load_config()
        assert cfg.server.port == 8888

    def test_cli_overrides_toml(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        cfg = load_config(cli_overrides={"port": 5555})
        assert cfg.server.port == 5555

    def test_cli_overrides_env_overrides_toml(self, tmp_home, monkeypatch):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        monkeypatch.setenv("PPMLX_PORT", "8888")
        cfg = load_config(cli_overrides={"port": 7777})
        assert cfg.server.port == 7777

    def test_toml_overrides_defaults(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        cfg = load_config()
        assert cfg.server.port == 9000
        assert cfg.server.host == "127.0.0.1"


class TestGetPpLlmDir:
    def test_creates_directory(self, tmp_home):
        d = get_ppmlx_dir()
        assert d.exists()
        assert d.is_dir()
        assert d == tmp_home / ".ppmlx"

    def test_idempotent(self, tmp_home):
        d1 = get_ppmlx_dir()
        d2 = get_ppmlx_dir()
        assert d1 == d2
        assert d1.exists()
