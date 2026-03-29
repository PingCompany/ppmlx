"""Config editor TUI using prompt_toolkit."""
from __future__ import annotations


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

    # Load existing config
    data: dict = {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        pass

    hf_token = data.get("auth", {}).get("hf_token", "")
    ta_modes = ["off", "no_tools_only", "all"]
    ta_current = data.get("tool_awareness", {}).get("mode", "no_tools_only")
    if ta_current not in ta_modes:
        ta_current = "no_tools_only"
    analytics_enabled = data.get("analytics", {}).get("enabled", False)

    ta_labels = {"off": "Off", "no_tools_only": "No Tools Only", "all": "All"}
    analytics_labels = {True: "Enabled", False: "Disabled"}

    state = {
        "cursor": 0,
        "hf_token": hf_token,
        "ta_index": ta_modes.index(ta_current),
        "analytics": analytics_enabled,
        "dirty": False,
        "editing_token": False,
        "token_buf": "",
        "saved_flash": False,
    }

    items = ["hf_token", "tool_awareness", "analytics"]

    def _mask_token(token: str) -> str:
        if not token:
            return "(not set)"
        if len(token) <= 4:
            return "\u2022" * len(token)
        return "\u2022" * (len(token) - 4) + token[-4:]

    def _get_text():
        fragments = list(header_text("ppmlx config"))

        # Config path
        fragments.append(("class:dim", f"  {cfg_path}\n\n"))

        # HF Token row
        is_cursor = state["cursor"] == 0
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        if state["editing_token"]:
            fragments.append((style, f"{prefix}HuggingFace Token    "))
            fragments.append(("class:value", state["token_buf"]))
            fragments.append(("class:value", "\u2588"))
            fragments.append(("", "\n"))
        else:
            masked = _mask_token(state["hf_token"])
            fragments.append((style, f"{prefix}HuggingFace Token    "))
            fragments.append(("class:dim" if not is_cursor else style, masked))
            fragments.append(("", "\n"))

        # Tool Awareness row
        is_cursor = state["cursor"] == 1
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        ta_label = ta_labels[ta_modes[state["ta_index"]]]
        fragments.append((style, f"{prefix}Tool Awareness       "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {ta_label} \u25b6"))
        fragments.append(("", "\n"))

        # Analytics row
        is_cursor = state["cursor"] == 2
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        an_label = analytics_labels[state["analytics"]]
        fragments.append((style, f"{prefix}Usage Analytics       "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {an_label} \u25b6"))
        fragments.append(("", "\n"))

        fragments.append(("", "\n"))

        if state["saved_flash"]:
            fragments.append(("class:checked", "                         \u2713 saved\n"))
        elif state["dirty"]:
            fragments.append(("class:unsaved", "                         \u2022 unsaved changes\n"))
        else:
            fragments.append(("", "\n"))

        fragments.append(("", "\n"))
        if state["editing_token"]:
            fragments.append(("class:footer", "type token \u2022 enter confirm \u2022 esc cancel"))
        else:
            fragments.append(("class:footer", "\u2191\u2193 navigate \u2022 \u2190\u2192 cycle \u2022 enter edit token \u2022 s save \u2022 esc quit"))
        return fragments

    def _save():
        import tomli_w

        data.setdefault("auth", {})["hf_token"] = state["hf_token"]
        data.setdefault("tool_awareness", {})["mode"] = ta_modes[state["ta_index"]]
        data.setdefault("analytics", {})["enabled"] = state["analytics"]
        with open(cfg_path, "wb") as f:
            tomli_w.dump(data, f)
        state["dirty"] = False
        state["saved_flash"] = True

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        if state["editing_token"]:
            return
        if state["cursor"] > 0:
            state["cursor"] -= 1
            state["saved_flash"] = False

    @kb.add("down")
    def _down(event):
        if state["editing_token"]:
            return
        if state["cursor"] < len(items) - 1:
            state["cursor"] += 1
            state["saved_flash"] = False

    @kb.add("left")
    def _left(event):
        if state["editing_token"]:
            return
        state["saved_flash"] = False
        if state["cursor"] == 1:
            state["ta_index"] = (state["ta_index"] - 1) % len(ta_modes)
            state["dirty"] = True
        elif state["cursor"] == 2:
            state["analytics"] = not state["analytics"]
            state["dirty"] = True

    @kb.add("right")
    def _right(event):
        if state["editing_token"]:
            return
        state["saved_flash"] = False
        if state["cursor"] == 1:
            state["ta_index"] = (state["ta_index"] + 1) % len(ta_modes)
            state["dirty"] = True
        elif state["cursor"] == 2:
            state["analytics"] = not state["analytics"]
            state["dirty"] = True

    @kb.add("enter")
    def _enter(event):
        state["saved_flash"] = False
        if state["cursor"] == 0:
            if state["editing_token"]:
                # Confirm token edit
                state["hf_token"] = state["token_buf"]
                state["editing_token"] = False
                state["dirty"] = True
            else:
                # Start editing token
                state["editing_token"] = True
                state["token_buf"] = state["hf_token"]

    @kb.add("escape")
    def _escape(event):
        if state["editing_token"]:
            state["editing_token"] = False
            return
        if state["dirty"]:
            _save()
        event.app.exit(result=None)

    @kb.add("backspace")
    def _backspace(event):
        if state["editing_token"]:
            state["token_buf"] = state["token_buf"][:-1]

    @kb.add("s")
    def _save_key(event):
        if state["editing_token"]:
            state["token_buf"] += "s"
            return
        _save()

    @kb.add("<any>")
    def _char(event):
        ch = event.data
        if state["editing_token"] and ch.isprintable() and len(ch) == 1:
            state["token_buf"] += ch

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
