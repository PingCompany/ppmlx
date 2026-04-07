"""Config editor TUI using prompt_toolkit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _Item:
    key: str
    label: str
    kind: str  # "cycle" | "toggle" | "text"
    options: list[Any] | None = None
    labels: dict[Any, str] | None = None
    suffix: str = ""  # appended to display value, e.g. " tokens"


@dataclass
class _Group:
    title: str


_Row = _Item | _Group


def config_menu() -> None:
    """Show an interactive config editor."""
    import tomllib

    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    from ppmlx.config import get_ppmlx_dir
    from ppmlx.tui._style import get_style, header_text

    cfg_path = get_ppmlx_dir() / "config.toml"

    data: dict = {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        pass

    # ── Option definitions ───────────────────────────────────────────

    ta_modes = ["off", "no_tools_only", "all"]
    refresh_modes = ["always", "weekly", "monthly", "never"]
    budget_options = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    effort_base_options = [64, 128, 256, 512, 1024]
    tools_options = [0, 3000, 6000, 12000, 24000]
    max_models_options = [1, 2, 3, 4, 5, 6, 8]
    ttl_options = [0, 60, 300, 600, 1800, 3600]
    temp_options = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    max_tokens_options = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    def _ensure_in(options: list, val: Any) -> list:
        if val not in options:
            options = sorted(options + [val])
        return options

    # Read current values
    srv = data.get("server", {})
    defs = data.get("defaults", {})
    think = data.get("thinking", {})
    reg = data.get("registry", {})

    cur_host = srv.get("host", "127.0.0.1")
    cur_port = srv.get("port", 6767)
    cur_max_models = srv.get("max_loaded_models", 2)
    cur_ttl = srv.get("ttl_seconds", 0)
    cur_tools = srv.get("max_tools_tokens", 12000)
    cur_model = defs.get("model", "qwen3.5:0.8b")
    cur_temp = defs.get("temperature", 0.7)
    cur_max_tokens = defs.get("max_tokens", 2048)
    cur_ta = data.get("tool_awareness", {}).get("mode", "no_tools_only")
    cur_thinking = think.get("enabled", True)
    cur_budget = think.get("default_reasoning_budget", 2048)
    cur_effort = think.get("effort_base", 256)
    cur_reg_enabled = reg.get("enabled", True)
    cur_refresh = reg.get("refresh", "weekly")
    cur_hf_token = data.get("auth", {}).get("hf_token", "")
    cur_logging = data.get("logging", {}).get("enabled", True)
    cur_analytics = data.get("analytics", {}).get("enabled", False)

    if cur_ta not in ta_modes:
        cur_ta = "no_tools_only"
    if cur_refresh not in refresh_modes:
        cur_refresh = "weekly"

    max_models_options = _ensure_in(max_models_options, cur_max_models)
    ttl_options = _ensure_in(ttl_options, cur_ttl)
    tools_options = _ensure_in(tools_options, cur_tools)
    budget_options = _ensure_in(budget_options, cur_budget)
    effort_base_options = _ensure_in(effort_base_options, cur_effort)
    temp_options = _ensure_in(temp_options, cur_temp)
    max_tokens_options = _ensure_in(max_tokens_options, cur_max_tokens)

    # ── Layout (groups + items) ──────────────────────────────────────

    rows: list[_Row] = [
        _Group("Server"),
        _Item("host", "Host", "text"),
        _Item("port", "Port", "text"),
        _Item("max_models", "Max Loaded Models", "cycle",
              max_models_options),
        _Item("ttl", "Model TTL", "cycle",
              ttl_options,
              {0: "Disabled", **{v: f"{v}s" for v in ttl_options if v > 0}}),
        _Item("tools_tokens", "Max Tools Tokens", "cycle",
              tools_options,
              {0: "Unlimited", **{v: f"{v} tokens" for v in tools_options if v > 0}}),

        _Group("Defaults"),
        _Item("model", "Default Model", "text"),
        _Item("temperature", "Temperature", "cycle", temp_options),
        _Item("max_tokens", "Max Tokens", "cycle",
              max_tokens_options, suffix=" tokens"),

        _Group("Thinking"),
        _Item("thinking", "Thinking", "toggle",
              labels={True: "Enabled", False: "Disabled"}),
        _Item("budget", "Reasoning Budget", "cycle",
              budget_options,
              {0: "Off", **{v: f"{v} tokens" for v in budget_options if v > 0}}),
        _Item("effort_base", "Effort Base", "cycle",
              effort_base_options,
              {v: f"{v} (low={v}, med={v*4}, high={v*32})" for v in effort_base_options}),

        _Group("Tools"),
        _Item("ta", "Tool Awareness", "cycle",
              ta_modes,
              {"off": "Off", "no_tools_only": "No Tools Only", "all": "All"}),

        _Group("Registry"),
        _Item("reg_enabled", "Registry", "toggle",
              labels={True: "Enabled", False: "Disabled"}),
        _Item("refresh", "Auto-Refresh", "cycle",
              refresh_modes,
              {"always": "Always", "weekly": "Weekly", "monthly": "Monthly", "never": "Never"}),

        _Group("Other"),
        _Item("hf_token", "HuggingFace Token", "text"),
        _Item("logging", "Request Logging", "toggle",
              labels={True: "Enabled", False: "Disabled"}),
        _Item("analytics", "Usage Analytics", "toggle",
              labels={True: "Enabled", False: "Disabled"}),
    ]

    # Build selectable items index (skip groups)
    selectable: list[int] = [i for i, r in enumerate(rows) if isinstance(r, _Item)]

    state: dict[str, Any] = {
        "cursor": 0,  # index into selectable
        "dirty": False,
        "saved_flash": False,
        "editing": False,
        "edit_buf": "",
        # Values
        "host": cur_host,
        "port": str(cur_port),
        "max_models": max_models_options.index(cur_max_models),
        "ttl": ttl_options.index(cur_ttl),
        "tools_tokens": tools_options.index(cur_tools),
        "model": cur_model,
        "temperature": temp_options.index(cur_temp),
        "max_tokens": max_tokens_options.index(cur_max_tokens),
        "ta": ta_modes.index(cur_ta),
        "thinking": cur_thinking,
        "budget": budget_options.index(cur_budget),
        "effort_base": effort_base_options.index(cur_effort),
        "reg_enabled": cur_reg_enabled,
        "refresh": refresh_modes.index(cur_refresh),
        "hf_token": cur_hf_token,
        "logging": cur_logging,
        "analytics": cur_analytics,
    }

    LABEL_W = 22

    def _mask_token(token: str) -> str:
        if not token:
            return "(not set)"
        if len(token) <= 4:
            return "\u2022" * len(token)
        return "\u2022" * (len(token) - 4) + token[-4:]

    def _display_value(item: _Item) -> str:
        val = state[item.key]
        if item.kind == "text":
            if item.key == "hf_token":
                return _mask_token(val)
            return str(val) if val else "(not set)"
        if item.kind == "toggle":
            return (item.labels or {True: "Yes", False: "No"})[val]
        # cycle
        opt = item.options[val] if isinstance(val, int) else val
        if item.labels and opt in item.labels:
            return item.labels[opt]
        return f"{opt}{item.suffix}"

    def _get_text():
        fragments = list(header_text("ppmlx config"))
        fragments.append(("class:dim", f"  {cfg_path}\n\n"))

        cur_sel = selectable[state["cursor"]]
        cur_item = rows[cur_sel]

        for i, row in enumerate(rows):
            if isinstance(row, _Group):
                fragments.append(("class:section", f"  {row.title}\n"))
                continue

            is_cursor = i == cur_sel
            prefix = "  \u25b8 " if is_cursor else "    "
            style = "class:cursor" if is_cursor else ""

            label = row.label.ljust(LABEL_W)

            if is_cursor and state["editing"]:
                fragments.append((style, f"{prefix}{label}"))
                fragments.append(("class:value", state["edit_buf"]))
                fragments.append(("class:value", "\u2588"))
                fragments.append(("", "\n"))
            elif row.kind == "text":
                fragments.append((style, f"{prefix}{label}"))
                dv = _display_value(row)
                fragments.append(("class:dim" if not is_cursor else style, dv))
                fragments.append(("", "\n"))
            else:
                fragments.append((style, f"{prefix}{label}"))
                dv = _display_value(row)
                arrows = f"\u25c0 {dv} \u25b6"
                fragments.append(("class:value" if not is_cursor else style, arrows))
                fragments.append(("", "\n"))

        fragments.append(("", "\n"))
        if state["saved_flash"]:
            fragments.append(("class:checked", "                         \u2713 saved\n"))
        elif state["dirty"]:
            fragments.append(("class:unsaved", "                         \u2022 unsaved changes\n"))
        else:
            fragments.append(("", "\n"))

        fragments.append(("", "\n"))
        if state["editing"]:
            fragments.append(("class:footer", "type value \u2022 enter confirm \u2022 esc cancel"))
        else:
            fragments.append(("class:footer", "\u2191\u2193 navigate \u2022 \u2190\u2192 cycle \u2022 enter edit \u2022 s save \u2022 esc quit"))
        return fragments

    def _cur_item() -> _Item:
        return rows[selectable[state["cursor"]]]  # type: ignore[return-value]

    def _cycle(delta: int) -> None:
        item = _cur_item()
        state["saved_flash"] = False
        if item.kind == "toggle":
            state[item.key] = not state[item.key]
            state["dirty"] = True
        elif item.kind == "cycle":
            n = len(item.options)  # type: ignore[arg-type]
            state[item.key] = (state[item.key] + delta) % n
            state["dirty"] = True

    def _save():
        import tomli_w

        data.setdefault("server", {})["host"] = state["host"]
        try:
            data["server"]["port"] = int(state["port"])
        except ValueError:
            pass
        data["server"]["max_loaded_models"] = max_models_options[state["max_models"]]
        data["server"]["ttl_seconds"] = ttl_options[state["ttl"]]
        data["server"]["max_tools_tokens"] = tools_options[state["tools_tokens"]]
        data.setdefault("defaults", {})["model"] = state["model"]
        data["defaults"]["temperature"] = temp_options[state["temperature"]]
        data["defaults"]["max_tokens"] = max_tokens_options[state["max_tokens"]]
        data.setdefault("tool_awareness", {})["mode"] = ta_modes[state["ta"]]
        data.setdefault("thinking", {})["enabled"] = state["thinking"]
        data["thinking"]["default_reasoning_budget"] = budget_options[state["budget"]]
        data["thinking"]["effort_base"] = effort_base_options[state["effort_base"]]
        data.setdefault("registry", {})["enabled"] = state["reg_enabled"]
        data["registry"]["refresh"] = refresh_modes[state["refresh"]]
        data.setdefault("auth", {})["hf_token"] = state["hf_token"]
        data.setdefault("logging", {})["enabled"] = state["logging"]
        data.setdefault("analytics", {})["enabled"] = state["analytics"]
        with open(cfg_path, "wb") as f:
            tomli_w.dump(data, f)
        state["dirty"] = False
        state["saved_flash"] = True

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        if state["editing"]:
            return
        if state["cursor"] > 0:
            state["cursor"] -= 1
            state["saved_flash"] = False

    @kb.add("down")
    def _down(event):
        if state["editing"]:
            return
        if state["cursor"] < len(selectable) - 1:
            state["cursor"] += 1
            state["saved_flash"] = False

    @kb.add("left")
    def _left(event):
        if state["editing"]:
            return
        _cycle(-1)

    @kb.add("right")
    def _right(event):
        if state["editing"]:
            return
        _cycle(1)

    @kb.add("enter")
    def _enter(event):
        state["saved_flash"] = False
        item = _cur_item()
        if item.kind == "text":
            if state["editing"]:
                state[item.key] = state["edit_buf"]
                state["editing"] = False
                state["dirty"] = True
            else:
                state["editing"] = True
                state["edit_buf"] = str(state[item.key]) if state[item.key] else ""
        elif item.kind in ("toggle", "cycle"):
            _cycle(1)

    @kb.add("escape")
    def _escape(event):
        if state["editing"]:
            state["editing"] = False
            return
        if state["dirty"]:
            _save()
        event.app.exit(result=None)

    @kb.add("backspace")
    def _backspace(event):
        if state["editing"]:
            state["edit_buf"] = state["edit_buf"][:-1]

    @kb.add("s")
    def _save_key(event):
        if state["editing"]:
            state["edit_buf"] += "s"
            return
        _save()

    @kb.add("<any>")
    def _char(event):
        ch = event.data
        if state["editing"] and ch.isprintable() and len(ch) == 1:
            state["edit_buf"] += ch

    body = Window(
        content=FormattedTextControl(_get_text),
        always_hide_cursor=True,
    )

    app = Application(
        layout=Layout(body),
        key_bindings=kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )

    app.run()
