from __future__ import annotations
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6767
    cors: bool = True
    cors_origins: list[str] = field(default_factory=list)
    max_loaded_models: int = 2
    max_request_body_mb: int = 10
    max_tokens_cap: int = 32768
    max_tools_tokens: int = 12000
    ttl_seconds: int = 0  # 0 = disabled; auto-unload idle models after N seconds


@dataclass
class DefaultsConfig:
    model: str = "qwen3.5:0.8b"
    embed_model: str = "embed:all-minilm"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048
    draft_model: str | None = None
    speculative_tokens: int = 5
    auto_speculative: bool = False  # auto-detect draft models for speculative decoding
    prompt_cache_limit: int = 4    # max prompt KV-cache entries (0 = disabled)


@dataclass
class UIConfig:
    show_stats: bool = False     # show TTFT / tok/s after each response
    markdown: bool = False       # render markdown in assistant output


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
    refresh: str = "weekly"  # "always" | "weekly" | "monthly" | "never"


@dataclass
class ToolAwarenessConfig:
    mode: str = "no_tools_only"


@dataclass
class ThinkingConfig:
    enabled: bool = True
    default_reasoning_budget: int = 2048
    effort_base: int = 256

    def effort_to_budget(self, effort: str) -> int | None:
        """Map reasoning_effort (low/medium/high) to token budget using effort_base."""
        multipliers = {"low": 1, "medium": 4, "high": 32}
        m = multipliers.get(effort.lower())
        return self.effort_base * m if m is not None else None


@dataclass
class RouterConfig:
    enabled: bool = False
    small_model: str = "qwen3.5:0.8b"
    large_model: str = "qwen3.5:9b"
    threshold: int = 3  # complexity score at or above which we use large_model


@dataclass
class AnalyticsConfig:
    enabled: bool = False
    provider: str = "posthog"
    host: str = ""
    project_api_key: str = ""
    respect_do_not_track: bool = True


@dataclass
class VoiceSettings:
    """Voice I/O settings — written to [voice] in config.toml."""
    stt_model: str = "mlx-community/whisper-large-v3-turbo-q4"
    tts_model: str = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
    tts_voice: str | None = None
    tts_speed: float = 1.0
    tts_volume: float = 1.10
    # Push-to-talk: hold ptt_key to record, release to send.
    # Set ptt_mode = true to enable; ptt_key accepts 'space', 'f5', single chars, …
    ptt_mode: bool = False
    ptt_key: str = "space"
    # Auto-silence params (used when ptt_mode = false)
    silence_threshold: float = 0.01
    silence_duration: float = 1.5


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    tool_awareness: ToolAwarenessConfig = field(default_factory=ToolAwarenessConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    voice: VoiceSettings = field(default_factory=VoiceSettings)


def get_ppmlx_dir() -> Path:
    """Return ~/.ppmlx, creating it if needed."""
    d = Path.home() / ".ppmlx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_bool(v: str) -> bool:
    return v.lower() not in ("0", "false", "no", "off", "")


def _normalize_refresh(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in ("always", "weekly", "monthly", "never"):
        return raw
    return "weekly"


def _normalize_tool_awareness_mode(value: Any) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "all",
        "true": "all",
        "yes": "all",
        "all": "all",
        "no_tools_only": "no_tools_only",
    }
    return aliases.get(raw, "no_tools_only")


def _normalize_analytics_provider(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"", "posthog"}:
        return "posthog"
    return raw


def load_config(cli_overrides: dict[str, Any] | None = None) -> Config:
    """Load config with priority: CLI overrides > env vars > TOML file > defaults."""
    cfg = Config()
    toml_path = Path.home() / ".ppmlx" / "config.toml"
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _apply_toml(cfg, data)
    except FileNotFoundError:
        pass  # No config file — use defaults
    except Exception as exc:
        import logging
        logging.getLogger("ppmlx.config").warning(
            "Failed to load %s: %s — using defaults", toml_path, exc,
        )
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
        if "cors_origins" in s:
            origins = s["cors_origins"]
            if isinstance(origins, list):
                cfg.server.cors_origins = [str(o) for o in origins]
        if "max_loaded_models" in s: cfg.server.max_loaded_models = int(s["max_loaded_models"])
        if "max_request_body_mb" in s: cfg.server.max_request_body_mb = int(s["max_request_body_mb"])
        if "max_tokens_cap" in s: cfg.server.max_tokens_cap = int(s["max_tokens_cap"])
        if "max_tools_tokens" in s: cfg.server.max_tools_tokens = int(s["max_tools_tokens"])
        if "ttl_seconds" in s: cfg.server.ttl_seconds = int(s["ttl_seconds"])
    if "defaults" in data:
        d = data["defaults"]
        if "model" in d: cfg.defaults.model = str(d["model"])
        if "embed_model" in d: cfg.defaults.embed_model = str(d["embed_model"])
        if "temperature" in d: cfg.defaults.temperature = float(d["temperature"])
        if "top_p" in d: cfg.defaults.top_p = float(d["top_p"])
        if "max_tokens" in d: cfg.defaults.max_tokens = int(d["max_tokens"])
        if "draft_model" in d: cfg.defaults.draft_model = str(d["draft_model"]) if d["draft_model"] else None
        if "speculative_tokens" in d: cfg.defaults.speculative_tokens = int(d["speculative_tokens"])
        if "auto_speculative" in d: cfg.defaults.auto_speculative = bool(d["auto_speculative"])
        if "prompt_cache_limit" in d: cfg.defaults.prompt_cache_limit = int(d["prompt_cache_limit"])
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
        if "refresh" in r: cfg.registry.refresh = _normalize_refresh(r["refresh"])
    if "tool_awareness" in data:
        ta = data["tool_awareness"]
        if "mode" in ta:
            cfg.tool_awareness.mode = _normalize_tool_awareness_mode(ta["mode"])
    if "thinking" in data:
        th = data["thinking"]
        if "enabled" in th: cfg.thinking.enabled = bool(th["enabled"])
        if "default_reasoning_budget" in th: cfg.thinking.default_reasoning_budget = int(th["default_reasoning_budget"])
        if "effort_base" in th: cfg.thinking.effort_base = int(th["effort_base"])
    if "ui" in data:
        u = data["ui"]
        if "show_stats" in u: cfg.ui.show_stats = bool(u["show_stats"])
        if "markdown" in u: cfg.ui.markdown = bool(u["markdown"])
    if "router" in data:
        rt = data["router"]
        if "enabled" in rt: cfg.router.enabled = bool(rt["enabled"])
        if "small_model" in rt: cfg.router.small_model = str(rt["small_model"])
        if "large_model" in rt: cfg.router.large_model = str(rt["large_model"])
        if "threshold" in rt: cfg.router.threshold = int(rt["threshold"])
    if "analytics" in data:
        an = data["analytics"]
        if "enabled" in an: cfg.analytics.enabled = bool(an["enabled"])
        if "provider" in an: cfg.analytics.provider = _normalize_analytics_provider(an["provider"])
        if "host" in an: cfg.analytics.host = str(an["host"]).strip()
        if "project_api_key" in an:
            cfg.analytics.project_api_key = str(an["project_api_key"]).strip()
        elif "website_id" in an:
            cfg.analytics.project_api_key = str(an["website_id"]).strip()
        if "respect_do_not_track" in an:
            cfg.analytics.respect_do_not_track = bool(an["respect_do_not_track"])
    if "voice" in data:
        v = data["voice"]
        if "stt_model" in v: cfg.voice.stt_model = str(v["stt_model"])
        if "tts_model" in v: cfg.voice.tts_model = str(v["tts_model"])
        if "tts_voice" in v: cfg.voice.tts_voice = str(v["tts_voice"]) if v["tts_voice"] else None
        if "tts_speed" in v: cfg.voice.tts_speed = float(v["tts_speed"])
        if "tts_volume" in v: cfg.voice.tts_volume = float(v["tts_volume"])
        if "ptt_mode" in v: cfg.voice.ptt_mode = bool(v["ptt_mode"])
        if "ptt_key" in v: cfg.voice.ptt_key = str(v["ptt_key"]).strip().lower()
        if "silence_threshold" in v: cfg.voice.silence_threshold = float(v["silence_threshold"])
        if "silence_duration" in v: cfg.voice.silence_duration = float(v["silence_duration"])


def _apply_env(cfg: Config) -> None:
    mapping = {
        "PPMLX_HOST": ("server", "host", str),
        "PPMLX_PORT": ("server", "port", int),
        "PPMLX_CORS": ("server", "cors", _parse_bool),
        "PPMLX_MAX_LOADED_MODELS": ("server", "max_loaded_models", int),
        "PPMLX_MAX_TOOLS_TOKENS": ("server", "max_tools_tokens", int),
        "PPMLX_TTL_SECONDS": ("server", "ttl_seconds", int),
        "PPMLX_DEFAULT_MODEL": ("defaults", "model", str),
        "PPMLX_DEFAULT_EMBED_MODEL": ("defaults", "embed_model", str),
        "PPMLX_TEMP": ("defaults", "temperature", float),
        "PPMLX_TOP_P": ("defaults", "top_p", float),
        "PPMLX_MAX_TOKENS": ("defaults", "max_tokens", int),
        "PPMLX_DRAFT_MODEL": ("defaults", "draft_model", str),
        "PPMLX_SPECULATIVE_TOKENS": ("defaults", "speculative_tokens", int),
        "PPMLX_AUTO_SPECULATIVE": ("defaults", "auto_speculative", _parse_bool),
        "PPMLX_PROMPT_CACHE_LIMIT": ("defaults", "prompt_cache_limit", int),
        "PPMLX_LOG_ENABLED": ("logging", "enabled", _parse_bool),
        "PPMLX_LOG_SNAPSHOT_INTERVAL": ("logging", "snapshot_interval_seconds", int),
        "PPMLX_MEMORY_WIRED_LIMIT": ("memory", "wired_limit_mb", int),
        "PPMLX_REGISTRY_ENABLED": ("registry", "enabled", _parse_bool),
        "PPMLX_REGISTRY_REFRESH": ("registry", "refresh", _normalize_refresh),
        "PPMLX_INJECT_TOOL_AWARENESS": ("tool_awareness", "mode", _normalize_tool_awareness_mode),
        "PPMLX_THINKING_ENABLED": ("thinking", "enabled", _parse_bool),
        "PPMLX_THINKING_BUDGET": ("thinking", "default_reasoning_budget", int),
        "PPMLX_EFFORT_BASE": ("thinking", "effort_base", int),
        "PPMLX_SHOW_STATS": ("ui", "show_stats", _parse_bool),
        "PPMLX_MARKDOWN": ("ui", "markdown", _parse_bool),
        "PPMLX_ROUTER_ENABLED": ("router", "enabled", _parse_bool),
        "PPMLX_ROUTER_SMALL_MODEL": ("router", "small_model", str),
        "PPMLX_ROUTER_LARGE_MODEL": ("router", "large_model", str),
        "PPMLX_ROUTER_THRESHOLD": ("router", "threshold", int),
        "PPMLX_ANALYTICS_ENABLED": ("analytics", "enabled", _parse_bool),
        "PPMLX_ANALYTICS_PROVIDER": ("analytics", "provider", _normalize_analytics_provider),
        "PPMLX_ANALYTICS_HOST": ("analytics", "host", str),
        "PPMLX_ANALYTICS_PROJECT_API_KEY": ("analytics", "project_api_key", str),
        "PPMLX_ANALYTICS_RESPECT_DNT": ("analytics", "respect_do_not_track", _parse_bool),
    }
    for env_key, (section, attr, coerce) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(getattr(cfg, section), attr, coerce(val))
            except (ValueError, AttributeError):
                pass
    legacy_website_id = os.environ.get("PPMLX_ANALYTICS_WEBSITE_ID")
    if legacy_website_id and not cfg.analytics.project_api_key:
        cfg.analytics.project_api_key = legacy_website_id.strip()


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
        elif key == "draft_model": cfg.defaults.draft_model = str(val) if val else None
        elif key == "speculative_tokens": cfg.defaults.speculative_tokens = int(val)
        elif key == "auto_speculative": cfg.defaults.auto_speculative = bool(val)
        elif key == "prompt_cache_limit": cfg.defaults.prompt_cache_limit = int(val)


def check_first_run() -> None:
    """Show analytics opt-in prompt on first run."""
    try:
        marker = get_ppmlx_dir() / ".first_run_done"
        if marker.exists():
            return
        if not sys.stdin.isatty():
            marker.touch()
            return
        from rich.console import Console
        console = Console()
        console.print(
            "\n[bold]ppmlx[/bold] can collect anonymous usage statistics "
            "(command names, version, OS — never prompts or model outputs).\n"
            "This helps improve ppmlx.\n"
        )
        answer = input("Enable anonymous analytics? [y/N] ").strip().lower()
        enabled = answer in ("y", "yes")
        _save_analytics_preference(enabled)
        marker.touch()
    except Exception:
        pass


def _save_analytics_preference(enabled: bool) -> None:
    """Save analytics preference to config.toml."""
    import tomli_w
    cfg_path = get_ppmlx_dir() / "config.toml"
    data: dict = {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        pass
    data.setdefault("analytics", {})["enabled"] = enabled
    with open(cfg_path, "wb") as f:
        tomli_w.dump(data, f)
