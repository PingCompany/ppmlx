from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
try:
    import setproctitle as _setproctitle_mod
except ImportError:
    _setproctitle_mod = None  # type: ignore[assignment]

app = typer.Typer(
    name="ppmlx",
    help="Run LLMs locally on Apple Silicon via MLX",
    no_args_is_help=True,
)
console = Console()


@dataclass
class _LaunchItem:
    key: str    # "run" | "claude" | "codex" | "opencode" | "pi"
    label: str
    desc: str
    cmd: str    # executable to check with shutil.which; "" = always installed


@dataclass
class _PickerRow:
    alias: str
    size_gb: float | None
    downloaded: bool
    section_header: str | None  # non-None → non-selectable section label


_LAUNCH_ITEMS: list[_LaunchItem] = [
    _LaunchItem("run",      "Run a model",        "Start an interactive chat with a model",       ""),
    _LaunchItem("serve",    "Start API server",   "OpenAI-compatible server on :6767",            ""),
    _LaunchItem("claude",   "Launch Claude Code", "Agentic coding across large codebases",        "claude"),
    _LaunchItem("codex",    "Launch Codex",       "OpenAI's open-source coding agent",            "codex"),
    _LaunchItem("opencode", "Launch Opencode",    "Anomaly's open-source coding agent",           "opencode"),
    _LaunchItem("openwebui","Launch Open WebUI",  "Web-based chat UI with multimodal support",    "open-webui"),
    _LaunchItem("pi",       "Launch Pi",          "Minimal AI agent toolkit with plugin support", "pi"),
]


def _track_usage(event: str, data: dict | None = None, *, context: str = "cli") -> None:
    try:
        from ppmlx.analytics import track_async

        track_async(event, data, context=context)
    except Exception:
        pass


# ── Unified model record & table builder ─────────────────────────────

@dataclass
class ModelRecord:
    alias: str
    repo_id: str
    is_favorite: bool = False
    is_downloaded: bool = False
    is_loaded: bool = False
    size_gb: float | None = None
    local_path: Path | None = None
    params_b: float | None = None
    model_type: str | None = None   # "dense" | "sparse"
    lab: str | None = None
    modalities: str | None = None
    downloads: int | None = None
    released: str | None = None
    source: str = "registry"        # "built-in" | "custom" | "registry" | "local-only"


def _build_model_records(
    *,
    filter_downloaded: bool | None = None,
    filter_favorites: bool = False,
    filter_text: str | None = None,
    filter_lab: str | None = None,
    filter_modality: str | None = None,
    filter_type: str | None = None,
    sort_by: str = "alias",
    limit: int | None = None,
    exclude_embed: bool = True,
) -> list[ModelRecord]:
    """Build a unified list of model records from all sources."""
    from ppmlx.models import (
        all_aliases, DEFAULT_ALIASES, load_user_aliases,
        list_local_models, load_favorites,
    )

    try:
        from ppmlx.registry import registry_entries
        reg = registry_entries()
    except Exception:
        reg = {}

    aliases = all_aliases()
    user_aliases = load_user_aliases()
    local_models = list_local_models()
    favorites_list = load_favorites()
    fav_set = set(favorites_list)

    local_by_repo: dict[str, dict] = {}
    for m in local_models:
        local_by_repo[m["repo_id"]] = m

    # Source priority: custom > built-in > registry
    _SOURCE_PRIO = {"custom": 0, "built-in": 1, "registry": 2, "local-only": 3}

    # Group by repo_id, keeping only the best alias per repo.
    by_repo: dict[str, ModelRecord] = {}
    for alias, repo_id in aliases.items():
        lm = local_by_repo.get(repo_id)
        reg_entry = reg.get(alias, {})

        source = "registry"
        if alias in user_aliases:
            source = "custom"
        elif alias in DEFAULT_ALIASES:
            source = "built-in"

        candidate = ModelRecord(
            alias=alias,
            repo_id=repo_id,
            is_favorite=alias in fav_set,
            is_downloaded=lm is not None,
            size_gb=lm["size_gb"] if lm else reg_entry.get("size_gb"),
            local_path=lm["path"] if lm else None,
            params_b=reg_entry.get("params_b") or None,
            model_type=reg_entry.get("type"),
            lab=reg_entry.get("lab"),
            modalities=reg_entry.get("modalities"),
            downloads=reg_entry.get("downloads"),
            released=reg_entry.get("created"),
            source=source,
        )

        existing = by_repo.get(repo_id)
        if existing is None:
            by_repo[repo_id] = candidate
        else:
            # Pick best alias: higher source priority wins, then shorter name.
            e_prio = _SOURCE_PRIO.get(existing.source, 9)
            c_prio = _SOURCE_PRIO.get(candidate.source, 9)
            if (c_prio, len(alias)) < (e_prio, len(existing.alias)):
                # Carry over fields the new candidate might lack.
                if candidate.params_b is None:
                    candidate.params_b = existing.params_b
                if candidate.size_gb is None:
                    candidate.size_gb = existing.size_gb
                if candidate.lab is None:
                    candidate.lab = existing.lab
                if not candidate.is_favorite and existing.is_favorite:
                    candidate.is_favorite = True
                by_repo[repo_id] = candidate
            else:
                # Keep existing but merge any missing metadata from candidate.
                if existing.params_b is None:
                    existing.params_b = candidate.params_b
                if existing.size_gb is None:
                    existing.size_gb = candidate.size_gb
                if existing.lab is None:
                    existing.lab = candidate.lab
                if not existing.is_favorite and candidate.is_favorite:
                    existing.is_favorite = True

    records = dict(by_repo)

    # Local-only models (downloaded but not in any alias map)
    alias_repos = set(aliases.values())
    for m in local_models:
        if m["repo_id"] not in alias_repos:
            alias = m.get("alias") or m["repo_id"]
            if m["repo_id"] not in records:
                records[m["repo_id"]] = ModelRecord(
                    alias=alias,
                    repo_id=m["repo_id"],
                    is_favorite=alias in fav_set,
                    is_downloaded=True,
                    size_gb=m["size_gb"],
                    local_path=m["path"],
                    source="local-only",
                )

    result = list(records.values())

    if exclude_embed:
        result = [r for r in result if not r.alias.startswith("embed:")]
    if filter_downloaded is True:
        result = [r for r in result if r.is_downloaded]
    elif filter_downloaded is False:
        result = [r for r in result if not r.is_downloaded]
    if filter_favorites:
        result = [r for r in result if r.is_favorite]
    if filter_text:
        q = filter_text.lower()
        result = [r for r in result if q in f"{r.alias} {r.repo_id} {r.lab or ''}".lower()]
    if filter_lab:
        q = filter_lab.lower()
        result = [r for r in result if q in (r.lab or "").lower()]
    if filter_modality:
        q = filter_modality.lower()
        result = [r for r in result if q in (r.modalities or "").lower()]
    if filter_type:
        q = filter_type.lower()
        result = [r for r in result if (r.model_type or "").lower() == q]

    sort_keys = {
        "alias":     lambda r: r.alias,
        "downloads": lambda r: -(r.downloads or 0),
        "size":      lambda r: -(r.size_gb or 0),
        "params":    lambda r: -(r.params_b or 0),
        "created":   lambda r: r.released or "",
        "name":      lambda r: r.alias,
    }
    result.sort(key=lambda r: (not r.is_favorite, sort_keys.get(sort_by, sort_keys["alias"])(r)))

    if limit:
        result = result[:limit]
    return result


def _model_table(
    records: list[ModelRecord],
    *,
    title: str = "Models",
    show_status: bool = True,
    show_params: bool = False,
    show_size: bool = True,
    show_type: bool = False,
    show_lab: bool = False,
    show_modalities: bool = False,
    show_downloads: bool = False,
    show_released: bool = False,
    show_repo: bool = False,
    show_source: bool = False,
    show_path: bool = False,
) -> Table:
    """Build a Rich Table from ModelRecords with configurable columns."""
    table = Table(title=title, show_header=True)

    if show_status:
        table.add_column("", width=3)
    table.add_column("Alias", style="cyan", no_wrap=True)
    if show_params:
        table.add_column("Params", justify="right", style="magenta")
    if show_size:
        table.add_column("Size", justify="right")
    if show_type:
        table.add_column("Type", style="dim")
    if show_lab:
        table.add_column("Lab", style="green")
    if show_modalities:
        table.add_column("Modalities", style="blue")
    if show_downloads:
        table.add_column("Downloads", justify="right")
    if show_released:
        table.add_column("Released", style="dim")
    if show_repo:
        table.add_column("HuggingFace Repo", style="green")
    if show_source:
        table.add_column("Source", style="dim")
    if show_path:
        table.add_column("Path", style="dim")

    for r in records:
        row: list[str] = []
        if show_status:
            flags = ""
            if r.is_favorite:
                flags += "★"
            if r.is_downloaded:
                flags += "✓"
            if r.is_loaded:
                flags += "●"
            row.append(flags)
        row.append(r.alias)
        if show_params:
            row.append(f"{r.params_b}B" if r.params_b else "—")
        if show_size:
            row.append(f"{r.size_gb:.1f} GB" if r.size_gb else "—")
        if show_type:
            row.append(r.model_type or "—")
        if show_lab:
            row.append(r.lab or "—")
        if show_modalities:
            row.append(r.modalities or "—")
        if show_downloads:
            row.append(f"{r.downloads:,}" if r.downloads else "—")
        if show_released:
            row.append(r.released or "—")
        if show_repo:
            row.append(r.repo_id)
        if show_source:
            row.append(r.source)
        if show_path:
            row.append(str(r.local_path) if r.local_path else "—")
        style = "yellow" if r.source == "custom" else ""
        table.add_row(*row, style=style)

    return table


# ── Picker helpers ───────────────────────────────────────────────────

def _build_picker_rows(*, local_only: bool = False) -> list[_PickerRow]:
    records = _build_model_records()
    rows: list[_PickerRow] = []

    fav_records = [r for r in records if r.is_favorite and (not local_only or r.is_downloaded)]
    dl_records = [r for r in records if r.is_downloaded and not r.is_favorite]
    avail_records = [r for r in records if not r.is_downloaded and not r.is_favorite]

    if fav_records:
        rows.append(_PickerRow("", None, True, "★ Favorites"))
        for r in fav_records:
            rows.append(_PickerRow(f"★ {r.alias}", r.size_gb, r.is_downloaded, None))

    if dl_records:
        rows.append(_PickerRow("", None, True, "Downloaded"))
        for r in sorted(dl_records, key=lambda r: r.alias):
            rows.append(_PickerRow(r.alias, r.size_gb, True, None))

    if avail_records and not local_only:
        rows.append(_PickerRow("", None, False, "Available"))
        for r in sorted(avail_records, key=lambda r: r.alias):
            rows.append(_PickerRow(r.alias, None, False, None))

    return rows


def _visible_rows(rows: list[_PickerRow], ft: str) -> list[_PickerRow]:
    if not ft:
        return rows
    result: list[_PickerRow] = []
    pending_header: _PickerRow | None = None
    for row in rows:
        if row.section_header is not None:
            pending_header = row
            continue
        if ft.lower() in row.alias.lower():
            if pending_header is not None:
                result.append(pending_header)
                pending_header = None
            result.append(row)
    return result


def _model_picker_curses(
    stdscr: "curses.window",
    all_rows: list[_PickerRow],
    *,
    command_str: str = "ppmlx",
) -> str | None:
    """Shared curses model picker. Returns selected alias or None on cancel."""
    import curses
    from ppmlx import __version__

    curses.curs_set(1)
    filter_text = ""
    _init_sel = [i for i, r in enumerate(all_rows) if r.section_header is None]
    picker_idx = _init_sel[0] if _init_sel else 0

    while True:
        visible = _visible_rows(all_rows, filter_text)
        sel = [i for i, r in enumerate(visible) if r.section_header is None]

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        try:
            stdscr.addstr(0, 0, f"ppmlx {__version__}", curses.A_BOLD)
        except curses.error:
            pass
        try:
            stdscr.addstr(1, 0, f"$ {command_str}", curses.A_DIM)
        except curses.error:
            pass
        try:
            stdscr.addstr(3, 0, "Search: ", curses.A_BOLD)
            stdscr.addstr(3, 8, (filter_text + "\u258c")[: w - 9])
        except curses.error:
            pass

        row = 5
        for i, r in enumerate(visible):
            if row >= h - 2:
                break
            if r.section_header is not None:
                try:
                    stdscr.addstr(row, 2, r.section_header, curses.A_DIM | curses.A_BOLD)
                except curses.error:
                    pass
                row += 1
                continue

            is_sel = i == picker_idx
            prefix = "\u25b6 " if is_sel else "  "
            attr = curses.A_BOLD if is_sel else curses.A_NORMAL
            try:
                stdscr.addstr(row, 0, prefix + r.alias, attr)
                if r.size_gb is not None:
                    size_str = f"{r.size_gb:.1f} GB"
                    col = w - len(size_str) - 1
                    if col > len(prefix + r.alias) + 1:
                        stdscr.addstr(row, col, size_str, curses.A_DIM)
            except curses.error:
                pass
            row += 1

        if not sel:
            try:
                stdscr.addstr(row, 2, "No models match.", curses.A_DIM)
            except curses.error:
                pass

        status = "\u2191/\u2193 navigate  \u2022  enter select  \u2022  esc cancel"
        try:
            stdscr.addstr(h - 1, 0, status[: w - 1], curses.A_DIM)
        except curses.error:
            pass
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            return None

        if key == curses.KEY_UP:
            if picker_idx in sel:
                pos = sel.index(picker_idx)
                if pos > 0:
                    picker_idx = sel[pos - 1]
            elif sel:
                picker_idx = sel[0]
        elif key == curses.KEY_DOWN:
            if picker_idx in sel:
                pos = sel.index(picker_idx)
                if pos < len(sel) - 1:
                    picker_idx = sel[pos + 1]
            elif sel:
                picker_idx = sel[0]
        elif key in (curses.KEY_ENTER, 10, 13):
            if sel and picker_idx in sel:
                picked = visible[picker_idx].alias
                return picked.removeprefix("\u2605 ")
        elif key in (curses.KEY_LEFT, 27):  # Esc
            return None
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            filter_text = filter_text[:-1]
            new_vis = _visible_rows(all_rows, filter_text)
            new_sel = [i for i, r in enumerate(new_vis) if r.section_header is None]
            if new_sel and picker_idx not in new_sel:
                picker_idx = new_sel[0]
        elif 32 <= key <= 126:
            filter_text += chr(key)
            new_vis = _visible_rows(all_rows, filter_text)
            new_sel = [i for i, r in enumerate(new_vis) if r.section_header is None]
            if new_sel:
                picker_idx = new_sel[0]


def _launch_tui(*, preset_action: str | None = None, command_str: str = "ppmlx launch") -> tuple[str | None, str | None]:
    """TUI launcher. Returns (action_key, model_alias) or (None, None).

    When *preset_action* is given the TUI opens directly in the model picker
    and returns (preset_action, model) without showing the action menu.
    """
    import curses
    from ppmlx import __version__

    installed = [bool(not item.cmd or shutil.which(item.cmd)) for item in _LAUNCH_ITEMS]
    all_rows = _build_picker_rows()

    model: list[str | None] = [None]
    action: list[str | None] = [None]
    main_idx = [next((i for i, inst in enumerate(installed) if inst), 0)]

    def _draw_main(stdscr: "curses.window") -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 8 or w < 30:
            try:
                stdscr.addstr(0, 0, "Terminal too small")
            except curses.error:
                pass
            return

        try:
            stdscr.addstr(0, 0, f"ppmlx {__version__}", curses.A_BOLD)
        except curses.error:
            pass
        try:
            stdscr.addstr(1, 0, f"$ {command_str}", curses.A_DIM)
        except curses.error:
            pass

        row = 3
        for i, item in enumerate(_LAUNCH_ITEMS):
            if row >= h - 2:
                break
            is_sel = i == main_idx[0]
            is_inst = installed[i]
            prefix = "\u25b6 " if is_sel else "  "

            if not is_inst:
                right_tag = "(not installed)"
            elif model[0]:
                right_tag = f"({model[0]})"
            else:
                right_tag = ""

            label_attr = (
                curses.A_BOLD if (is_sel and is_inst)
                else curses.A_DIM if not is_inst
                else curses.A_NORMAL
            )
            try:
                stdscr.addstr(row, 0, prefix + item.label, label_attr)
                if right_tag:
                    col = w - len(right_tag) - 1
                    if col > len(prefix + item.label) + 1:
                        stdscr.addstr(row, col, right_tag, curses.A_DIM)
            except curses.error:
                pass

            row += 1
            if row < h - 2:
                try:
                    stdscr.addstr(row, 4, item.desc[: w - 5], curses.A_DIM)
                except curses.error:
                    pass
            row += 2

        status = "\u2191/\u2193 navigate  \u2022  enter launch  \u2022  \u2192 change model  \u2022  esc quit"
        try:
            stdscr.addstr(h - 1, 0, status[: w - 1], curses.A_DIM)
        except curses.error:
            pass
        stdscr.refresh()

    def _curses_main(stdscr: "curses.window") -> None:
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()

        if preset_action:
            selected = _model_picker_curses(stdscr, all_rows, command_str=command_str)
            if selected:
                model[0] = selected
                action[0] = preset_action
            else:
                action[0] = "cancelled"
            return

        while action[0] is None:
            curses.curs_set(0)
            _draw_main(stdscr)
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                action[0] = "cancelled"
                break

            if key == curses.KEY_UP:
                idx = main_idx[0] - 1
                while idx >= 0 and not installed[idx]:
                    idx -= 1
                if idx >= 0:
                    main_idx[0] = idx
            elif key == curses.KEY_DOWN:
                idx = main_idx[0] + 1
                while idx < len(_LAUNCH_ITEMS) and not installed[idx]:
                    idx += 1
                if idx < len(_LAUNCH_ITEMS):
                    main_idx[0] = idx
            elif key == curses.KEY_RIGHT:
                selected = _model_picker_curses(stdscr, all_rows, command_str=command_str)
                if selected:
                    model[0] = selected
            elif key in (curses.KEY_ENTER, 10, 13):
                if installed[main_idx[0]]:
                    action[0] = _LAUNCH_ITEMS[main_idx[0]].key
            elif key in (27, ord("q"), ord("Q")):
                action[0] = "cancelled"

    curses.wrapper(_curses_main)
    if action[0] is None or action[0] == "cancelled":
        return None, None
    return action[0], model[0]


def _start_server_bg(model: str, host: str, port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "ppmlx.cli", "serve", "--host", host, "--port", str(port)]
    if model:
        cmd += ["--model", model]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _wait_server_ready(host: str, port: int, proc: subprocess.Popen, timeout: int = 30) -> bool:
    import httpx
    import time
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _port_in_use(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _flush_port(host: str, port: int) -> None:
    """Kill any process listening on the given port and wait until it's free."""
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip()
    if not pids:
        return
    for pid in pids.splitlines():
        pid = pid.strip()
        if pid:
            try:
                os.kill(int(pid), 9)
                console.print(f"[yellow]Killed process {pid} on port {port}[/yellow]")
            except (ProcessLookupError, ValueError):
                pass
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _port_in_use(host, port):
            return
        time.sleep(0.3)
    console.print(f"[red]Port {port} still in use after killing processes.[/red]")


def _launch_coding_tool(action: str, model: str, host: str, port: int) -> None:
    if _port_in_use(host, port):
        console.print(f"[red]Port {port} is already in use.[/red]")
        console.print(f"[dim]Try: ppmlx launch -a {action} -m {model} --port {port + 1}[/dim]")
        raise typer.Exit(1)

    proc = _start_server_bg(model, host, port)
    console.print(f"[dim]Starting ppmlx server on {host}:{port}...[/dim]")

    try:
        ready = _wait_server_ready(host, port, proc)
    except KeyboardInterrupt:
        proc.terminate()
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(1)

    if not ready:
        stderr_output = ""
        if proc.stderr:
            stderr_output = proc.stderr.read().decode(errors="replace").strip()
        proc.terminate()
        console.print("[red]Server failed to start within 30 seconds.[/red]")
        if stderr_output:
            console.print(f"[dim]{stderr_output[-500:]}[/dim]")
        raise typer.Exit(1)

    base_url = f"http://{host}:{port}/v1"
    env = os.environ.copy()

    if action == "claude":
        base = f"http://{host}:{port}"
        cmd = ["claude", "--model", model]
        env["ANTHROPIC_BASE_URL"] = base
        env["ANTHROPIC_API_KEY"] = "local"
    elif action == "codex":
        cmd = [
            "codex", "--model", model,
            "-c", 'model_provider="ppmlx"',
            "-c", 'model_providers.ppmlx.name="ppmlx"',
            "-c", f'model_providers.ppmlx.base_url="{base_url}"',
            "-c", 'model_providers.ppmlx.env_key="OPENAI_API_KEY"',
            "-c", 'model_providers.ppmlx.wire_api="responses"',
        ]
        env["OPENAI_API_KEY"] = "local"
    elif action == "opencode":
        cmd = ["opencode"]
        env["OPENAI_API_KEY"] = "local"
        env["OPENAI_BASE_URL"] = base_url
    elif action == "pi":
        models_file = Path.home() / ".pi" / "agent" / "models.json"
        models_file.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(models_file.read_text()) if models_file.exists() else {}
        if isinstance(existing, dict) and "providers" in existing:
            existing["providers"]["ppmlx"] = {
                "api": "openai-completions",
                "apiKey": "local",
                "baseUrl": base_url,
                "models": [{
                    "_launch": True,
                    "contextWindow": 262144,
                    "id": model,
                    "input": ["text"],
                    "reasoning": True,
                }],
            }
        else:
            if isinstance(existing, list):
                entries = [e for e in existing if isinstance(e, dict) and e.get("id") != "ppmlx"]
            else:
                entries = []
            entries.append({
                "id": "ppmlx",
                "name": f"ppmlx ({model})",
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": "local",
            })
            existing = entries
        models_file.write_text(json.dumps(existing, indent=2))
        cmd = ["pi", "--model", f"ppmlx/{model}"]
    elif action == "openwebui":
        import time, webbrowser
        owui_env = env.copy()
        owui_env["OPENAI_API_BASE_URLS"] = base_url
        owui_env["OPENAI_API_KEYS"] = "local"
        owui_env["OLLAMA_BASE_URLS"] = ""
        owui_env["ENABLE_OLLAMA_API"] = "false"
        owui_env["DEFAULT_MODELS"] = model
        owui_port = 8080
        owui_proc = subprocess.Popen(
            ["open-webui", "serve", "--port", str(owui_port)],
            env=owui_env,
        )
        owui_url = f"http://localhost:{owui_port}"
        console.print(f"[dim]Waiting for Open WebUI on {owui_url}...[/dim]")
        import httpx
        deadline = time.monotonic() + 60
        ready = False
        while time.monotonic() < deadline:
            if owui_proc.poll() is not None:
                break
            try:
                if httpx.get(owui_url, timeout=1.0).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if ready:
            webbrowser.open(owui_url)
            console.print(f"[green]Open WebUI ready at {owui_url}[/green]")
            try:
                owui_proc.wait()
            except KeyboardInterrupt:
                pass
        else:
            console.print("[red]Open WebUI failed to start within 60 seconds.[/red]")
        owui_proc.terminate()
        try:
            owui_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owui_proc.kill()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    else:
        proc.terminate()
        return

    try:
        subprocess.run(cmd, env=env)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _pick_model_tui(*, command_str: str = "ppmlx", local_only: bool = False, allow_none: bool = False) -> str | None:
    """Unified curses model picker. Returns alias, None (lazy-load), or raises typer.Exit on cancel."""
    import curses

    all_rows = _build_picker_rows(local_only=local_only)

    if allow_none:
        all_rows.insert(0, _PickerRow("(none \u2014 lazy load)", None, True, None))

    if not any(r.section_header is None for r in all_rows):
        if local_only:
            console.print("[yellow]No local models found. Run: ppmlx pull <model>[/yellow]")
        else:
            console.print("[yellow]No models available.[/yellow]")
        raise typer.Exit(1)

    try:
        selected = curses.wrapper(lambda stdscr: _model_picker_curses(stdscr, all_rows, command_str=command_str))
    except KeyboardInterrupt:
        selected = None

    if selected is None:
        raise typer.Exit()
    if selected == "(none \u2014 lazy load)":
        return None
    return selected


def _pick_model(local_only: bool = False, multi: bool = False) -> "str | list[str]":
    """Model picker. Single-select uses curses TUI, multi-select uses questionary.

    Returns a single alias string (multi=False) or list of aliases (multi=True).
    Raises typer.Exit if cancelled or nothing selected.
    """
    if not multi:
        result = _pick_model_tui(local_only=local_only)
        if result is None:
            raise typer.Exit()
        return result

    import questionary

    records = _build_model_records(
        filter_downloaded=True if local_only else None,
    )

    if not records:
        if local_only:
            console.print("[yellow]No local models found. Run: ppmlx pull <model>[/yellow]")
        else:
            console.print("[yellow]No models available.[/yellow]")
        raise typer.Exit(1)

    choices = []
    for r in records:
        prefix = "★ " if r.is_favorite else "  "
        size = f"{r.size_gb:.1f} GB" if r.size_gb else ""
        dl = "  ✓" if r.is_downloaded and not local_only else ""
        if size:
            label = f"{prefix}{r.alias:<36} {size}{dl}"
        else:
            label = f"{prefix}{r.alias:<36} {r.repo_id}"
        choices.append(questionary.Choice(label, value=r.alias))

    selected = questionary.checkbox(
        "Select models  (Space=toggle, Enter=confirm, Ctrl-C=cancel):",
        choices=choices,
    ).ask()
    if not selected:
        raise typer.Exit()
    return selected


def _version_callback(value: bool):
    if value:
        from ppmlx import __version__
        console.print(f"ppmlx {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    )
):
    """ppmlx: Run LLMs on Apple Silicon via MLX."""


@app.command()
def launch(
    action: Optional[str] = typer.Argument(None, help="Action: run, serve, claude, codex, opencode, openwebui, pi"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name or alias"),
    host: Optional[str] = typer.Option(None, help="Bind host"),
    port: Optional[int] = typer.Option(None, help="Bind port (default: 6767)"),
    no_cors: bool = typer.Option(False, "--no-cors", help="Disable CORS"),
    flush: bool = typer.Option(False, "--flush", "-f", help="Kill any process using the port before starting"),
):
    """Select an action and model, then launch.

    Without arguments, opens an interactive TUI picker.
    With ACTION and MODEL, launches directly (non-interactive).
    """
    from ppmlx.config import load_config

    valid_actions = {item.key for item in _LAUNCH_ITEMS}
    _track_usage("launch_invoked", {"interactive": not bool(action and model)})

    overrides = {k: v for k, v in [("host", host), ("port", port)] if v}
    cfg = load_config(cli_overrides=overrides)
    effective_host = host or cfg.server.host
    effective_port = port or cfg.server.port

    if flush:
        _flush_port(effective_host, effective_port)

    if action and model:
        if action not in valid_actions:
            console.print(f"[red]Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}[/red]")
            raise typer.Exit(1)
    elif action and not model:
        if action in valid_actions:
            action, model = _launch_tui(
                preset_action=action,
                command_str=f"ppmlx launch {action}",
            )
        else:
            action, model = _launch_tui()
    else:
        action, model = _launch_tui()

    if not action:
        raise typer.Exit()

    if not model:
        console.print("[yellow]No model selected. Press \u2192 in the menu to pick one.[/yellow]")
        raise typer.Exit(1)

    if action == "run":
        run(model=model, system=None, max_kv_size=None, temperature=None, max_tokens=None)
    elif action == "serve":
        serve(model=model, host=effective_host, port=effective_port, interactive=False, no_cors=no_cors)
    else:
        _launch_coding_tool(action, model, effective_host, effective_port)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host"),
    port: Optional[int] = typer.Option(None, help="Bind port (default: 6767)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Pre-load a model on startup"),
    embed_model: Optional[str] = typer.Option(None, "--embed-model", help="Pre-load an embedding model"),
    no_cors: bool = typer.Option(False, "--no-cors", help="Disable CORS"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactively select a model to serve"),
):
    """Start the OpenAI-compatible API server."""
    import uvicorn
    from ppmlx.config import load_config
    from ppmlx import __version__

    overrides = {}
    if host: overrides["host"] = host
    if port: overrides["port"] = port
    if no_cors: overrides["cors"] = False
    cfg = load_config(cli_overrides=overrides)
    _track_usage(
        "serve_started",
        {
            "interactive": interactive,
            "preload_model": bool(model),
            "embed_model": bool(embed_model),
            "cors": cfg.server.cors,
        },
        context="server",
    )

    effective_host = host or cfg.server.host
    effective_port = port or cfg.server.port

    # Interactive model selection
    if interactive and model is None:
        model = _pick_model_tui(
            command_str="ppmlx serve --interactive",
            local_only=True,
            allow_none=True,
        )

    console.print(Panel(
        f"[bold green]ppmlx server v{__version__}[/bold green]\n"
        f"   Listening on [link]http://{effective_host}:{effective_port}[/link]\n"
        f"   Endpoints:\n"
        f"     POST /v1/chat/completions\n"
        f"     POST /v1/completions\n"
        f"     POST /v1/embeddings\n"
        f"     GET  /v1/models\n"
        f"     GET  /health\n"
        f"     GET  /metrics\n"
        f"   SQLite log: ~/.ppmlx/ppmlx.db",
        title="ppmlx",
        border_style="green",
    ))

    # IDE connection hint
    selected_model = model or "(any — set model in your IDE)"
    console.print(Panel(
        f"[bold]API base:[/bold]  http://{effective_host}:{effective_port}/v1\n"
        f"[bold]Model:[/bold]     {selected_model}\n"
        f"[bold]API key:[/bold]   (not required — use any string)\n\n"
        f"[dim]Cursor[/dim]   → Settings › AI › OpenAI-compatible\n"
        f"[dim]Continue[/dim] → config.json: provider 'openai', apiBase above\n"
        f"[dim]Aider[/dim]    → --openai-api-base http://{effective_host}:{effective_port}/v1",
        title="Connect your IDE",
        border_style="blue",
    ))

    if _setproctitle_mod:
        _setproctitle_mod.setproctitle(f"ppmlx: server ({effective_host}:{effective_port})")

    uvicorn.run(
        "ppmlx.server:app",
        host=effective_host,
        port=effective_port,
        log_level="info",
        reload=False,
    )


@app.command()
def run(
    model: Optional[str] = typer.Argument(None, help="Model name or alias"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size", help="Max KV cache tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
):
    """Start an interactive chat REPL with a model."""
    _track_usage("repl_started", {"interactive_model_pick": model is None})
    if not model:
        model = _pick_model()
    from ppmlx.models import get_model_path, download_model, resolve_alias, ModelNotFoundError
    from ppmlx.engine import get_engine
    from ppmlx.memory import check_memory_warning

    try:
        repo_id = resolve_alias(model)
    except ModelNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    local_path = get_model_path(repo_id)
    if not local_path:
        console.print(f"[yellow]Model not found locally. Downloading {model}...[/yellow]")
        try:
            local_path = download_model(model)
        except KeyboardInterrupt:
            console.print("\n[yellow]Download cancelled.[/yellow]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            raise typer.Exit(1)

    warning = check_memory_warning(local_path)
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")

    engine = get_engine()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    # Session state
    history_enabled = True
    wordwrap = True
    verbose = False
    format_json = False
    think = False

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.filters import emacs_insert_mode

    _kb = KeyBindings()

    @_kb.add("c-left")   # Ctrl+Left  — jump word left
    @_kb.add("escape", "b")  # Alt/Option+B
    def _word_left(event):
        event.current_buffer.cursor_left(count=len(
            event.current_buffer.document.get_word_before_cursor(WORD=True) or " "
        ))

    @_kb.add("c-right")  # Ctrl+Right — jump word right
    @_kb.add("escape", "f")  # Alt/Option+F
    def _word_right(event):
        event.current_buffer.cursor_right(count=len(
            event.current_buffer.document.get_word_after_cursor(WORD=True) or " "
        ))

    @_kb.add("s-left")   # Shift+Left  — extend selection left (move + select)
    def _sel_left(event):
        buf = event.current_buffer
        if buf.selection_state is None:
            buf.start_selection()
        buf.cursor_left()

    @_kb.add("s-right")  # Shift+Right — extend selection right
    def _sel_right(event):
        buf = event.current_buffer
        if buf.selection_state is None:
            buf.start_selection()
        buf.cursor_right()

    @_kb.add("c-a")      # Ctrl+A — beginning of line (also Cmd+Left in most macOS terminals)
    def _bol(event):
        event.current_buffer.cursor_position = 0

    @_kb.add("c-e")      # Ctrl+E — end of line (also Cmd+Right)
    def _eol(event):
        event.current_buffer.cursor_position = len(event.current_buffer.text)

    @_kb.add("c-k")      # Ctrl+K — delete to end of line
    def _kill_eol(event):
        buf = event.current_buffer
        buf.delete(count=len(buf.document.get_text_after_cursor()))

    @_kb.add("c-u")      # Ctrl+U — delete to beginning of line
    def _kill_bol(event):
        buf = event.current_buffer
        deleted = buf.cursor_position
        buf.cursor_position = 0
        buf.delete(count=deleted)

    _session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        key_bindings=_kb,
        mouse_support=False,
        enable_open_in_editor=False,
    )
    _prompt = ANSI("\033[1;34mYou\033[0m: ")

    console.print(f"[green]Chatting with [bold]{model}[/bold]. Type /help or /? for commands, /bye to exit.[/green]")

    while True:
        try:
            user_input = _session.prompt(_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/bye", "/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        elif user_input in ("/help", "/?"):
            console.print("[bold]REPL commands:[/bold]")
            console.print("  [cyan]/set parameter <key> <val>[/cyan]  Set a parameter (temperature, max_tokens)")
            console.print("  [cyan]/set system <string>[/cyan]        Set system message")
            console.print("  [cyan]/set history[/cyan]                Enable history")
            console.print("  [cyan]/set nohistory[/cyan]              Disable history")
            console.print("  [cyan]/set wordwrap[/cyan]               Enable word wrap")
            console.print("  [cyan]/set nowordwrap[/cyan]             Disable word wrap")
            console.print("  [cyan]/set format json[/cyan]            Enable JSON mode")
            console.print("  [cyan]/set noformat[/cyan]               Disable formatting")
            console.print("  [cyan]/set verbose[/cyan]                Show LLM stats after each reply")
            console.print("  [cyan]/set quiet[/cyan]                  Disable LLM stats")
            console.print("  [cyan]/set think[/cyan]                  Show thinking blocks")
            console.print("  [cyan]/set nothink[/cyan]                Hide thinking blocks")
            console.print("  [cyan]/show info[/cyan]                  Show model details")
            console.print("  [cyan]/show license[/cyan]               Show model license")
            console.print("  [cyan]/show modelfile[/cyan]             Show model config")
            console.print("  [cyan]/show parameters[/cyan]            Show generation parameters")
            console.print("  [cyan]/show system[/cyan]                Show system message")
            console.print("  [cyan]/show template[/cyan]              Show chat template")
            console.print("  [cyan]/clear[/cyan]                      Clear conversation history")
            console.print("  [cyan]/model <name>[/cyan]               Switch to a different model")
            console.print("  [cyan]/bye[/cyan]                        Exit")

        elif user_input.startswith("/set"):
            arg = user_input[4:].strip()
            parts = arg.split(None, 2)
            sub = parts[0] if parts else ""

            def _set_help() -> None:
                console.print("[bold]/set options:[/bold]")
                console.print("  [cyan]system <string>[/cyan]         Set system message")
                console.print("  [cyan]parameter <key> <value>[/cyan] Set a generation parameter")
                console.print("  [cyan]history[/cyan]                 Enable history")
                console.print("  [cyan]nohistory[/cyan]               Disable history")
                console.print("  [cyan]wordwrap[/cyan]                Enable word wrap")
                console.print("  [cyan]nowordwrap[/cyan]              Disable word wrap")
                console.print("  [cyan]format json[/cyan]             Enable JSON mode")
                console.print("  [cyan]noformat[/cyan]                Disable formatting")
                console.print("  [cyan]verbose[/cyan]                 Show LLM stats after each reply")
                console.print("  [cyan]quiet[/cyan]                   Disable LLM stats")
                console.print("  [cyan]think[/cyan]                   Show thinking blocks")
                console.print("  [cyan]nothink[/cyan]                 Hide thinking blocks")

            def _set_parameter_help() -> None:
                console.print("[bold]/set parameter options:[/bold]")
                console.print("  [cyan]temperature <float>[/cyan]   Sampling temperature (e.g. 0.7)")
                console.print("  [cyan]max_tokens <int>[/cyan]      Max tokens to generate (e.g. 2048)")
                console.print("  [cyan]num_predict <int>[/cyan]     Alias for max_tokens")

            if not sub or sub == "?":
                _set_help()
            elif sub == "system":
                if len(parts) < 2 or parts[1] == "?":
                    console.print("[bold]/set system[/bold] — set the system message sent before the conversation.")
                    console.print("  Usage: [cyan]/set system <your prompt here>[/cyan]")
                else:
                    new_system = arg[len("system"):].strip()
                    messages = [m for m in messages if m["role"] != "system"]
                    if new_system:
                        messages.insert(0, {"role": "system", "content": new_system})
                    console.print("[dim]System message updated.[/dim]")
            elif sub == "history":
                history_enabled = True
                console.print("[dim]History enabled.[/dim]")
            elif sub == "nohistory":
                history_enabled = False
                console.print("[dim]History disabled.[/dim]")
            elif sub == "wordwrap":
                wordwrap = True
                console.print("[dim]Word wrap enabled.[/dim]")
            elif sub == "nowordwrap":
                wordwrap = False
                console.print("[dim]Word wrap disabled.[/dim]")
            elif sub == "format":
                if len(parts) < 2 or parts[1] == "?":
                    console.print("[bold]/set format[/bold] — enable a response format.")
                    console.print("  [cyan]/set format json[/cyan]  Respond only with valid JSON")
                elif parts[1] == "json":
                    format_json = True
                    console.print("[dim]JSON mode enabled.[/dim]")
                else:
                    console.print(f"[red]Unknown format: {parts[1]}. Supported: json[/red]")
            elif sub == "noformat":
                format_json = False
                console.print("[dim]Formatting disabled.[/dim]")
            elif sub == "verbose":
                verbose = True
                console.print("[dim]Verbose mode enabled.[/dim]")
            elif sub == "quiet":
                verbose = False
                console.print("[dim]Quiet mode enabled.[/dim]")
            elif sub == "think":
                think = True
                console.print("[dim]Thinking enabled.[/dim]")
            elif sub == "nothink":
                think = False
                console.print("[dim]Thinking disabled.[/dim]")
            elif sub == "parameter":
                if len(parts) < 3 or parts[1] == "?":
                    _set_parameter_help()
                else:
                    key, val = parts[1], parts[2]
                    if key == "temperature":
                        try:
                            temperature = float(val)
                            console.print(f"[dim]temperature = {temperature}[/dim]")
                        except ValueError:
                            console.print(f"[red]Invalid value: {val}[/red]")
                    elif key in ("max_tokens", "num_predict"):
                        try:
                            max_tokens = int(val)
                            console.print(f"[dim]max_tokens = {max_tokens}[/dim]")
                        except ValueError:
                            console.print(f"[red]Invalid value: {val}[/red]")
                    else:
                        console.print(f"[red]Unknown parameter: {key}[/red]")
                        _set_parameter_help()
            else:
                console.print(f"[red]Unknown /set option: {sub}[/red]")
                _set_help()

        elif user_input.startswith("/show"):
            sub = user_input[5:].strip()
            sys_msgs = [m for m in messages if m["role"] == "system"]
            sys_prompt = sys_msgs[0]["content"] if sys_msgs else "(none)"
            t = temperature if temperature is not None else 0.7
            mt = max_tokens if max_tokens is not None else 2048

            if sub in ("", "info"):
                console.print(f"  [bold]model[/bold]       {model}  ({repo_id})")
                console.print(f"  [bold]path[/bold]        {local_path or '(not cached)'}")
                console.print(f"  [bold]system[/bold]      {sys_prompt[:80]}")
                console.print(f"  [bold]temperature[/bold] {t}")
                console.print(f"  [bold]max_tokens[/bold]  {mt}")
                console.print(f"  [bold]history[/bold]     {'on' if history_enabled else 'off'}")
                console.print(f"  [bold]wordwrap[/bold]    {'on' if wordwrap else 'off'}")
                console.print(f"  [bold]verbose[/bold]     {'on' if verbose else 'off'}")
                console.print(f"  [bold]think[/bold]       {'on' if think else 'off'}")
                console.print(f"  [bold]json[/bold]        {'on' if format_json else 'off'}")

            elif sub == "system":
                console.print(sys_prompt)

            elif sub in ("parameters", "params"):
                console.print(f"  temperature  {t}")
                console.print(f"  max_tokens   {mt}")

            elif sub == "license":
                if local_path:
                    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "license.txt"):
                        lic = Path(local_path) / name
                        if lic.exists():
                            console.print(lic.read_text())
                            break
                    else:
                        console.print("[dim]No LICENSE file found in model directory.[/dim]")
                else:
                    console.print("[dim]Model not downloaded yet.[/dim]")

            elif sub == "modelfile":
                console.print(f"  FROM {repo_id}")
                if sys_prompt != "(none)":
                    console.print(f"  SYSTEM {sys_prompt}")
                console.print(f"  PARAMETER temperature {t}")
                console.print(f"  PARAMETER num_predict {mt}")
                if local_path:
                    cfg = Path(local_path) / "config.json"
                    if cfg.exists():
                        try:
                            data = json.loads(cfg.read_text())
                            for k in ("model_type", "architectures", "quantization_config"):
                                if k in data:
                                    console.print(f"  # {k}: {data[k]}")
                        except Exception:
                            pass

            elif sub == "template":
                if local_path:
                    tc = Path(local_path) / "tokenizer_config.json"
                    if tc.exists():
                        try:
                            data = json.loads(tc.read_text())
                            tmpl = data.get("chat_template")
                            if tmpl:
                                console.print(tmpl)
                            else:
                                console.print("[dim]No chat_template in tokenizer_config.json.[/dim]")
                        except Exception as exc:
                            console.print(f"[red]Could not read tokenizer_config.json: {exc}[/red]")
                    else:
                        console.print("[dim]tokenizer_config.json not found.[/dim]")
                else:
                    console.print("[dim]Model not downloaded yet.[/dim]")

            elif sub == "?":
                console.print("[bold]/show options:[/bold]")
                console.print("  [cyan]info[/cyan]        Show model details and session state")
                console.print("  [cyan]license[/cyan]     Show model license")
                console.print("  [cyan]modelfile[/cyan]   Show model config (FROM, SYSTEM, PARAMETER)")
                console.print("  [cyan]parameters[/cyan]  Show generation parameters")
                console.print("  [cyan]system[/cyan]      Show system message")
                console.print("  [cyan]template[/cyan]    Show chat template")
            else:
                console.print(f"[red]Unknown /show option: {sub}[/red]")
                console.print("[bold]/show options:[/bold]")
                console.print("  [cyan]info[/cyan]        Show model details and session state")
                console.print("  [cyan]license[/cyan]     Show model license")
                console.print("  [cyan]modelfile[/cyan]   Show model config (FROM, SYSTEM, PARAMETER)")
                console.print("  [cyan]parameters[/cyan]  Show generation parameters")
                console.print("  [cyan]system[/cyan]      Show system message")
                console.print("  [cyan]template[/cyan]    Show chat template")

        elif user_input == "/clear":
            sys_msgs = [m for m in messages if m["role"] == "system"]
            messages = sys_msgs
            console.print("[dim]Conversation cleared.[/dim]")

        elif user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            try:
                repo_id = resolve_alias(new_model)
                local_path = get_model_path(repo_id)
                model = new_model
                console.print(f"[dim]Switched to {model}[/dim]")
            except ModelNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")

        else:
            # Parse image references from input: [/path/to/img.jpg] or bare paths
            import re as _re
            _IMG_EXTS = r"\.(?:jpg|jpeg|png|gif|webp|bmp)"
            _bracket = _re.compile(r"\[([^\[\]]+?" + _IMG_EXTS + r")\]", _re.IGNORECASE)
            _bare    = _re.compile(r"(?<!\S)((?:/|~/)[\S]+" + _IMG_EXTS + r")(?!\S)", _re.IGNORECASE)

            image_paths = [m.group(1) for m in _bracket.finditer(user_input)]
            clean_input = _bracket.sub("", user_input)
            for m in _bare.finditer(clean_input):
                image_paths.append(m.group(1))
            clean_input = _bare.sub("", clean_input).strip()
            text_input  = clean_input or user_input  # fallback if only path was typed

            # Build user content: structured if images present, plain string otherwise
            if image_paths:
                user_content: object = [{"type": "text", "text": text_input}] + [
                    {"type": "image_url", "image_url": {"url": p}} for p in image_paths
                ]
            else:
                user_content = user_input

            if history_enabled:
                messages.append({"role": "user", "content": user_content})

            # Build messages to send (inject JSON instruction if needed)
            send_msgs = list(messages) if history_enabled else []
            if not history_enabled:
                sys_msgs = [m for m in messages if m["role"] == "system"]
                send_msgs = sys_msgs + [{"role": "user", "content": user_content}]
            if format_json:
                if send_msgs and send_msgs[0]["role"] == "system":
                    send_msgs[0] = {
                        "role": "system",
                        "content": send_msgs[0]["content"] + "\n\nRespond only with valid JSON.",
                    }
                else:
                    send_msgs.insert(0, {"role": "system", "content": "Respond only with valid JSON."})

            console.print("[bold green]Assistant:[/bold green] ", end="")
            full_response = ""
            try:
                if image_paths:
                    # Vision path — no streaming support in mlx-vlm
                    from ppmlx.engine_vlm import get_vision_engine
                    import time as _time
                    t0 = _time.monotonic()
                    text, prompt_toks, completion_toks = get_vision_engine().generate(
                        repo_id, send_msgs,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 2048,
                    )
                    elapsed = _time.monotonic() - t0
                    console.print(text, no_wrap=not wordwrap)
                    full_response = text
                    if verbose:
                        tps = completion_toks / elapsed if elapsed > 0 else 0
                        console.print(
                            f"[dim]prompt {prompt_toks} tokens  "
                            f"completion {completion_toks} tokens  "
                            f"{tps:.1f} tok/s  {elapsed:.2f}s[/dim]"
                        )
                elif verbose:
                    import time as _time
                    t0 = _time.monotonic()
                    text, reasoning, prompt_toks, completion_toks = engine.generate(
                        repo_id, send_msgs,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 2048,
                        strip_thinking=not think,
                        enable_thinking=think,
                    )
                    elapsed = _time.monotonic() - t0
                    if think and reasoning:
                        console.print()
                        console.print(f"[dim italic]{reasoning}[/dim italic]")
                        console.print()
                    console.print(text, no_wrap=not wordwrap)
                    full_response = text
                    tps = completion_toks / elapsed if elapsed > 0 else 0
                    console.print(
                        f"[dim]prompt {prompt_toks} tokens  "
                        f"completion {completion_toks} tokens  "
                        f"{tps:.1f} tok/s  {elapsed:.2f}s[/dim]"
                    )
                else:
                    # Streaming with think-tag handling
                    in_think = False
                    for chunk in engine.stream_generate(
                        repo_id, send_msgs,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 2048,
                        enable_thinking=think,
                    ):
                        if "<think>" in chunk and not in_think:
                            before, _, after = chunk.partition("<think>")
                            if before:
                                console.print(before, end="")
                                full_response += before
                            in_think = True
                            chunk = after
                        if "</think>" in chunk and in_think:
                            inside, _, after = chunk.partition("</think>")
                            if think and inside:
                                console.print(f"[dim italic]{inside}[/dim italic]", end="")
                            in_think = False
                            chunk = after
                        if chunk:
                            if in_think:
                                if think:
                                    console.print(chunk, end="", style="dim italic")
                            else:
                                console.print(chunk, end="")
                                full_response += chunk
                    console.print()
            except KeyboardInterrupt:
                console.print("\n[dim]Generation interrupted.[/dim]")
                if history_enabled:
                    messages.pop()
                continue
            except Exception as exc:
                console.print(f"\n[red]Error: {exc}[/red]")
                if history_enabled:
                    messages.pop()
                continue

            if history_enabled:
                messages.append({"role": "assistant", "content": full_response})
            continue

        continue


def _do_pull(model: str, token: Optional[str]) -> bool:
    """Download a single model and print result. Returns True on success."""
    from ppmlx.models import download_model, resolve_alias, ModelNotFoundError
    from ppmlx.memory import check_memory_warning

    try:
        repo_id = resolve_alias(model)
    except ModelNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return False

    console.print(f"[blue]Pulling [bold]{model}[/bold] ({repo_id})[/blue]")
    try:
        local_path = download_model(model, token=token)
        console.print(f"[green]✓ Downloaded to {local_path}[/green]")
        _track_usage("model_pulled", {"used_token": bool(token)})
        warning = check_memory_warning(local_path)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
        return True
    except KeyboardInterrupt:
        console.print("\n[yellow]Download cancelled.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Pull failed: {e}[/red]")
        return False


@app.command()
def pull(
    model: Optional[str] = typer.Argument(None, help="Model alias or HuggingFace repo ID (omit for interactive selector)"),
    token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token"),
):
    """Download a model from HuggingFace Hub (interactive multiselect when no model given)."""
    if model is None:
        import questionary

        records = _build_model_records(filter_downloaded=False)
        choices = []
        for r in records:
            label = f"{r.alias:<40} {r.repo_id}"
            choices.append(questionary.Choice(label, value=r.alias))

        if not choices:
            console.print("[green]All available models are already downloaded.[/green]")
            return

        selected = questionary.checkbox(
            "Select models to download  (Space=toggle, Enter=confirm, Ctrl-C=cancel):",
            choices=choices,
        ).ask()

        if not selected:
            console.print("[dim]Nothing selected.[/dim]")
            return

        for m in selected:
            _do_pull(m, token)
        return

    if not _do_pull(model, token):
        raise typer.Exit(1)


@app.command(name="list")
def list_models(
    all_models: bool = typer.Option(False, "--all", "-a", help="Show all models (local + registry)"),
    show_path: bool = typer.Option(False, "--path", help="Show local file paths"),
):
    """List models. Shows downloaded models by default, --all includes registry."""
    _track_usage("list_models", {"all_models": all_models, "show_path": show_path})
    records = _build_model_records(
        filter_downloaded=True if not all_models else None,
    )
    if not records:
        console.print("[dim]No models downloaded yet. Run: ppmlx pull <model>[/dim]")
        return

    title = "Models" if all_models else "Local Models"
    table = _model_table(
        records,
        title=title,
        show_params=True,
        show_path=show_path,
    )
    console.print(table)


@app.command()
def rm(
    model: Optional[str] = typer.Argument(None, help="Model alias or name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a locally downloaded model."""
    _track_usage("remove_model", {"force": force, "interactive_model_pick": model is None})
    from ppmlx.models import remove_model, get_model_path, resolve_alias

    models_to_remove: list[str] = []
    if not model:
        selected = _pick_model(local_only=True, multi=True)
        models_to_remove = selected  # type: ignore[assignment]
    else:
        models_to_remove = [model]

    for m in models_to_remove:
        try:
            repo_id = resolve_alias(m)
        except Exception:
            repo_id = m

        path = get_model_path(repo_id)
        if not path:
            console.print(f"[yellow]Model '{m}' not found locally.[/yellow]")
            continue

        if not force:
            confirm = typer.confirm(f"Remove '{m}' from {path}?")
            if not confirm:
                continue

        if remove_model(m):
            console.print(f"[green]Removed {m}[/green]")
        else:
            console.print(f"[red]Failed to remove {m}[/red]")


@app.command()
def ps():
    """Show currently loaded models and memory usage."""
    _track_usage("ps_checked")
    import httpx
    from ppmlx.config import load_config

    cfg = load_config()
    url = f"http://{cfg.server.host}:{cfg.server.port}/health"

    try:
        response = httpx.get(url, timeout=3.0)
        data = response.json()
        loaded = data.get("loaded_models", [])
        uptime = data.get("uptime_seconds", 0)

        if not loaded:
            console.print("[dim]No models currently loaded. Start server: ppmlx serve[/dim]")
            return

        table = Table(title="Loaded Models")
        table.add_column("Model", style="cyan")
        for m in loaded:
            table.add_row(m)
        console.print(table)
        console.print(f"[dim]Server uptime: {uptime}s[/dim]")
    except Exception:
        console.print("[yellow]Server not running. Start it with: ppmlx serve[/yellow]")


@app.command()
def quantize(
    model: Optional[str] = typer.Argument(None, help="HuggingFace repo ID or alias"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (2,3,4,6,8)"),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    upload: Optional[str] = typer.Option(None, "--upload-repo", help="HF repo to upload to"),
    token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token"),
):
    """Convert and quantize a HuggingFace model to MLX format."""
    _track_usage(
        "quantize_started",
        {"bits": bits, "group_size": group_size, "upload": bool(upload)},
    )
    if not model:
        model = _pick_model()
    from ppmlx.quantize import quantize as do_quantize, QuantizeConfig

    cfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        output_path=Path(output) if output else None,
        upload_repo=upload,
        hf_token=token,
    )

    try:
        path = do_quantize(model, cfg, progress_callback=lambda msg: console.print(f"[blue]{msg}[/blue]"))
        console.print(f"[green]Quantized model saved to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Quantization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="config")
def config_cmd(
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="Set HuggingFace token"),
    analytics: Optional[bool] = typer.Option(
        None,
        "--analytics/--no-analytics",
        help="Enable or disable anonymous usage analytics.",
    ),
):
    """View or interactively set ppmlx configuration (HF token, defaults, etc.)."""
    import tomllib
    import tomli_w  # type: ignore[import]
    import questionary
    from ppmlx.config import get_ppmlx_dir, load_config

    cfg_path = get_ppmlx_dir() / "config.toml"
    cfg = load_config()
    current_mode = getattr(getattr(cfg, "tool_awareness", None), "mode", "no_tools_only")
    analytics_cfg = getattr(cfg, "analytics", None)
    analytics_enabled = bool(getattr(analytics_cfg, "enabled", True))
    analytics_host = str(getattr(analytics_cfg, "host", "") or "")
    analytics_key = str(getattr(analytics_cfg, "project_api_key", "") or "")
    mode_meta = {
        "off": ("Off", "Never inject tool-awareness hints."),
        "no_tools_only": ("No Tools Only", "Inject only when the request has no tools."),
        "all": ("All Requests", "Inject for requests with and without tools."),
    }

    # Load existing config
    try:
        with open(cfg_path, "rb") as f:
            data: dict = tomllib.load(f)
    except Exception:
        data = {}

    if hf_token is not None:
        # Non-interactive: set token directly
        data.setdefault("auth", {})["hf_token"] = hf_token
        if analytics is not None:
            data.setdefault("analytics", {})["enabled"] = analytics
        cfg_path.write_bytes(tomli_w.dumps(data).encode())
        console.print(f"[green]HuggingFace token saved to {cfg_path}[/green]")
        return

    if analytics is not None:
        data.setdefault("analytics", {})["enabled"] = analytics
        try:
            cfg_path.write_bytes(tomli_w.dumps(data).encode())
        except Exception as exc:
            console.print(f"[red]Failed to write config: {exc}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Anonymous analytics {'enabled' if analytics else 'disabled'}.[/green]")
        return

    # Interactive flow
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.formatted_text import ANSI

    console.print(f"[bold]ppmlx config[/bold]  ({cfg_path})\n")

    current_token = data.get("auth", {}).get("hf_token") or os.environ.get("HF_TOKEN", "")
    masked = ("*" * (len(current_token) - 4) + current_token[-4:]) if len(current_token) > 4 else ("(not set)" if not current_token else current_token)
    console.print(f"  HuggingFace token: [dim]{masked}[/dim]")
    console.print(f"  Tool awareness: [dim]{mode_meta.get(current_mode, ('No Tools Only', ''))[0]}[/dim]")
    analytics_status = "enabled" if analytics_enabled else "disabled"
    analytics_sink = "configured" if analytics_host and analytics_key else "not configured"
    console.print(f"  Usage analytics: [dim]{analytics_status} ({analytics_sink})[/dim]")
    console.print()

    new_token = pt_prompt(
        ANSI("\033[1mNew HF token\033[0m (leave blank to keep current, 'clear' to remove): "),
    ).strip()

    changed = False
    if new_token == "clear":
        data.setdefault("auth", {}).pop("hf_token", None)
        console.print("[dim]Token cleared.[/dim]")
        changed = True
    elif new_token:
        data.setdefault("auth", {})["hf_token"] = new_token
        console.print("[green]Token saved.[/green]")
        changed = True
    else:
        console.print("[dim]No change.[/dim]")

    selected_mode = questionary.select(
        "Inject tool awareness into /v1/chat/completions?",
        choices=[
            questionary.Choice(f"{label:<14} {desc}", value=value)
            for value, (label, desc) in mode_meta.items()
        ],
        default=current_mode,
    ).ask()

    if selected_mode is None:
        raise typer.Exit()

    if selected_mode != current_mode:
        data.setdefault("tool_awareness", {})["mode"] = selected_mode
        console.print(f"[green]Tool awareness set to {selected_mode}.[/green]")
        changed = True
    else:
        console.print("[dim]Tool awareness unchanged.[/dim]")

    selected_analytics = questionary.select(
        "Anonymous usage analytics?",
        choices=[
            questionary.Choice("Enabled  Send anonymous product events (never prompts or file content).", value=True),
            questionary.Choice("Disabled Do not send any usage analytics.", value=False),
        ],
        default=analytics_enabled,
    ).ask()

    if selected_analytics is None:
        raise typer.Exit()

    if selected_analytics != analytics_enabled:
        data.setdefault("analytics", {})["enabled"] = selected_analytics
        console.print(
            f"[green]Anonymous analytics {'enabled' if selected_analytics else 'disabled'}.[/green]"
        )
        changed = True
    else:
        console.print("[dim]Analytics setting unchanged.[/dim]")

    if not changed:
        console.print("[dim]No config changes to write.[/dim]")
        return

    try:
        cfg_path.write_bytes(tomli_w.dumps(data).encode())
    except Exception as exc:
        console.print(f"[red]Failed to write config: {exc}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
