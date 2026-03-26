from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6767
    cors: bool = True
    max_loaded_models: int = 2
    ttl_seconds: int = 0  # 0 = disabled; auto-unload idle models after N seconds


@dataclass
class DefaultsConfig:
    model: str = "qwen3.5:0.8b"
    embed_model: str = "embed:all-minilm"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048


@dataclass
class LoggingConfig:
    enabled: bool = True
    snapshot_interval_seconds: int = 60


@dataclass
class MemoryConfig:
    wired_limit_mb: int = 0


@dataclass
class RegistryConfig:
    enabled: bool = True


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)


def get_ppmlx_dir() -> Path:
    """Return ~/.ppmlx, creating it if needed."""
    d = Path.home() / ".ppmlx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_bool(v: str) -> bool:
    return v.lower() not in ("0", "false", "no")


def load_config(cli_overrides: dict[str, Any] | None = None) -> Config:
    """Load config with priority: CLI overrides > env vars > TOML file > defaults."""
    cfg = Config()
    toml_path = Path.home() / ".ppmlx" / "config.toml"
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _apply_toml(cfg, data)
    except Exception:
        pass
    _apply_env(cfg)
    if cli_overrides:
        _apply_cli(cfg, cli_overrides)
    return cfg


def _apply_toml(cfg: Config, data: dict) -> None:
    if "server" in data:
        s = data["server"]
        if "host" in s: cfg.server.host = str(s["host"])
        if "port" in s: cfg.server.port = int(s["port"])
        if "cors" in s: cfg.server.cors = bool(s["cors"])
        if "max_loaded_models" in s: cfg.server.max_loaded_models = int(s["max_loaded_models"])
        if "ttl_seconds" in s: cfg.server.ttl_seconds = int(s["ttl_seconds"])
    if "defaults" in data:
        d = data["defaults"]
        if "model" in d: cfg.defaults.model = str(d["model"])
        if "embed_model" in d: cfg.defaults.embed_model = str(d["embed_model"])
        if "temperature" in d: cfg.defaults.temperature = float(d["temperature"])
        if "top_p" in d: cfg.defaults.top_p = float(d["top_p"])
        if "max_tokens" in d: cfg.defaults.max_tokens = int(d["max_tokens"])
    if "logging" in data:
        lg = data["logging"]
        if "enabled" in lg: cfg.logging.enabled = bool(lg["enabled"])
        if "snapshot_interval_seconds" in lg:
            cfg.logging.snapshot_interval_seconds = int(lg["snapshot_interval_seconds"])
    if "memory" in data:
        m = data["memory"]
        if "wired_limit_mb" in m: cfg.memory.wired_limit_mb = int(m["wired_limit_mb"])
    if "registry" in data:
        r = data["registry"]
        if "enabled" in r: cfg.registry.enabled = bool(r["enabled"])


def _apply_env(cfg: Config) -> None:
    mapping = {
        "PPMLX_HOST": ("server", "host", str),
        "PPMLX_PORT": ("server", "port", int),
        "PPMLX_CORS": ("server", "cors", _parse_bool),
        "PPMLX_MAX_LOADED_MODELS": ("server", "max_loaded_models", int),
        "PPMLX_TTL_SECONDS": ("server", "ttl_seconds", int),
        "PPMLX_DEFAULT_MODEL": ("defaults", "model", str),
        "PPMLX_DEFAULT_EMBED_MODEL": ("defaults", "embed_model", str),
        "PPMLX_TEMP": ("defaults", "temperature", float),
        "PPMLX_TOP_P": ("defaults", "top_p", float),
        "PPMLX_MAX_TOKENS": ("defaults", "max_tokens", int),
        "PPMLX_LOG_ENABLED": ("logging", "enabled", _parse_bool),
        "PPMLX_LOG_SNAPSHOT_INTERVAL": ("logging", "snapshot_interval_seconds", int),
        "PPMLX_MEMORY_WIRED_LIMIT": ("memory", "wired_limit_mb", int),
        "PPMLX_REGISTRY_ENABLED": ("registry", "enabled", _parse_bool),
    }
    for env_key, (section, attr, coerce) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(getattr(cfg, section), attr, coerce(val))
            except (ValueError, AttributeError):
                pass


def _apply_cli(cfg: Config, overrides: dict) -> None:
    for key, val in overrides.items():
        if val is None:
            continue
        if key == "host": cfg.server.host = str(val)
        elif key == "port": cfg.server.port = int(val)
        elif key == "cors": cfg.server.cors = bool(val)
        elif key == "model": cfg.defaults.model = str(val)
        elif key == "temperature": cfg.defaults.temperature = float(val)
        elif key == "max_tokens": cfg.defaults.max_tokens = int(val)
