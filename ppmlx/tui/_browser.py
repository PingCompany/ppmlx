"""Read-only interactive model browser using prompt_toolkit."""
from __future__ import annotations


def browse_models(
    rows: list,
    *,
    title: str = "Models",
    command_str: str = "ppmlx list",
    footer_extra: str = "",
) -> None:
    """Show an interactive read-only browser for model rows. Supports search and navigation."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window, ScrollOffsets
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.data_structures import Point

    from ppmlx.cli import _visible_rows
    from ppmlx.tui._style import (
        get_style, header_text,
        render_model_row, render_table_header, render_section_title,
    )

    state = {"cursor": 0, "search": ""}

    def _selectable_indices(r):
        return [i for i, row in enumerate(r) if row.section_header is None]

    def _filtered():
        return _visible_rows(rows, state["search"])

    def _clamp_cursor(r):
        indices = _selectable_indices(r)
        if not indices:
            state["cursor"] = 0
            return
        if state["cursor"] not in indices:
            state["cursor"] = indices[0]

    def _get_header():
        fragments = list(header_text(command_str))
        fragments.append(("", "Search: "))
        fragments.append(("class:value", state["search"]))
        fragments.append(("class:value", "\u2588"))
        fragments.append(("", "\n"))
        return fragments

    def _cursor_line():
        """Return visual line number of cursor (accounting for section titles + table headers)."""
        r = _filtered()
        line = 0
        for i, row in enumerate(r):
            if row.section_header is not None:
                line += 4  # blank + title + header + separator
                continue
            if i == state["cursor"]:
                return line
            line += 1
        return 0

    def _get_list():
        r = _filtered()
        _clamp_cursor(r)
        fragments = []

        if not r:
            fragments.append(("class:dim", "  No models found.\n"))
            return fragments

        for i, row in enumerate(r):
            if row.section_header is not None:
                fragments.extend(render_section_title(row.section_header))
                fragments.extend(render_table_header())
                continue
            fragments.extend(render_model_row(row, is_cursor=i == state["cursor"]))

        return fragments

    def _get_footer():
        parts = []
        if footer_extra:
            parts.append(("class:dim", f"  {footer_extra}  "))
        parts.append(("class:footer", "\u2191\u2193 navigate \u2022 type to search \u2022 esc/q quit"))
        return parts

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        r = _filtered()
        indices = _selectable_indices(r)
        if not indices:
            return
        try:
            pos = indices.index(state["cursor"])
            if pos > 0:
                state["cursor"] = indices[pos - 1]
        except ValueError:
            state["cursor"] = indices[0]

    @kb.add("down")
    def _down(event):
        r = _filtered()
        indices = _selectable_indices(r)
        if not indices:
            return
        try:
            pos = indices.index(state["cursor"])
            if pos < len(indices) - 1:
                state["cursor"] = indices[pos + 1]
        except ValueError:
            state["cursor"] = indices[0]

    @kb.add("escape")
    def _escape(event):
        event.app.exit()

    @kb.add("q")
    def _quit(event):
        if not state["search"]:
            event.app.exit()
        else:
            state["search"] += "q"
            state["cursor"] = 0

    @kb.add("backspace")
    def _backspace(event):
        if state["search"]:
            state["search"] = state["search"][:-1]
            state["cursor"] = 0

    @kb.add("<any>")
    def _char(event):
        ch = event.data
        if ch.isprintable() and len(ch) == 1:
            state["search"] += ch
            state["cursor"] = 0

    header_window = Window(
        content=FormattedTextControl(_get_header),
        height=4,
        always_hide_cursor=True,
    )

    list_control = FormattedTextControl(_get_list)
    list_control.get_cursor_position = lambda: Point(x=0, y=_cursor_line())

    list_window = Window(
        content=list_control,
        always_hide_cursor=True,
        scroll_offsets=ScrollOffsets(top=2, bottom=2),
    )

    footer_window = Window(
        content=FormattedTextControl(_get_footer),
        height=1,
        always_hide_cursor=True,
    )

    app = Application(
        layout=Layout(HSplit([header_window, list_window, footer_window])),
        key_bindings=kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )

    app.run()
