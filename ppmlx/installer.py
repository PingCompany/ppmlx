"""Interactive component installer for ppmlx.

Manages optional heavy dependencies (voice, embeddings) that are not
included in the base install to keep the footprint small.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.table import Table

console = Console()

_CHECKMARK = "[green]✓[/green]"
_CROSS     = "[red]✗[/red]"
_WARN      = "[yellow]![/yellow]"


@dataclass
class Component:
    """An optional installable component."""
    key: str
    label: str
    description: str
    packages: list[str]            # pip packages to install
    check_imports: list[str]       # modules to import to verify install
    size_hint: str                 # rough download size hint
    extras_key: str | None = None  # pyproject extras group name
    requires_brew: list[str] = field(default_factory=list)  # brew formulae needed first


COMPONENTS: list[Component] = [
    Component(
        key="voice",
        label="Voice I/O",
        description="Push-to-talk input (Whisper STT) + spoken output (Voxtral TTS)",
        packages=["mlx-whisper>=0.4", "mlx-audio>=0.4", "sounddevice>=0.4", "soundfile>=0.12"],
        check_imports=["mlx_whisper", "mlx_audio", "sounddevice", "soundfile"],
        size_hint="~2 GB  (models downloaded on first use)",
        extras_key="voice",
        requires_brew=["portaudio"],
    ),
    Component(
        key="embeddings",
        label="Embeddings",
        description="Local text embedding models (semantic search, RAG)",
        packages=["mlx-embeddings>=0.0.5"],
        check_imports=["mlx_embeddings"],
        size_hint="~200 MB",
        extras_key="embeddings",
    ),
]

_COMPONENT_MAP = {c.key: c for c in COMPONENTS}


def _is_installed(component: Component) -> bool:
    """Return True if all required imports succeed."""
    for mod in component.check_imports:
        try:
            importlib.import_module(mod.replace("-", "_"))
        except ImportError:
            return False
    return True


def _brew_installed(formula: str) -> bool:
    """Return True if a Homebrew formula is installed."""
    try:
        result = subprocess.run(
            ["brew", "list", formula],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _install_component(component: Component) -> bool:
    """Install a component's packages. Returns True on success."""
    # Check brew deps first
    for formula in component.requires_brew:
        if not _brew_installed(formula):
            console.print(f"  Installing brew dep: [cyan]{formula}[/cyan]")
            result = subprocess.run(["brew", "install", formula], timeout=120)
            if result.returncode != 0:
                console.print(f"  [red]brew install {formula} failed[/red]")
                return False

    # Install pip packages
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + component.packages
    console.print(f"  Running: [dim]{' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, timeout=300)
    return result.returncode == 0


def status_table() -> None:
    """Print a table showing which components are installed."""
    table = Table(title="ppmlx components", box=None, padding=(0, 2))
    table.add_column("", width=2)
    table.add_column("Component", style="bold")
    table.add_column("Description", style="dim")
    table.add_column("Size", style="dim", justify="right")

    for comp in COMPONENTS:
        ok = _is_installed(comp)
        mark = _CHECKMARK if ok else _CROSS
        table.add_row(mark, comp.label, comp.description, comp.size_hint)

    console.print(table)


def install_interactive() -> None:
    """Interactive installation wizard."""
    import questionary

    console.print("\n[bold]ppmlx component installer[/bold]")
    console.print("[dim]Base install is lean (~400 MB). Add components as needed.\n[/dim]")

    choices = []
    for comp in COMPONENTS:
        ok = _is_installed(comp)
        prefix = "✓ " if ok else "  "
        choices.append(
            questionary.Choice(
                title=f"{prefix}{comp.label}  [dim]{comp.size_hint}[/dim]  — {comp.description}",
                value=comp.key,
                checked=ok,
                disabled="installed" if ok else None,
            )
        )

    selected_keys: list[str] = questionary.checkbox(
        "Select components to install:",
        choices=choices,
        style=questionary.Style([
            ("checkbox-selected", "fg:green"),
            ("selected", "fg:green"),
        ]),
    ).ask()

    if selected_keys is None:
        console.print("[dim]Cancelled.[/dim]")
        return

    to_install = [_COMPONENT_MAP[k] for k in selected_keys if not _is_installed(_COMPONENT_MAP[k])]

    if not to_install:
        console.print("[green]Nothing to install.[/green]")
        return

    for comp in to_install:
        console.print(f"\nInstalling [bold]{comp.label}[/bold]…")
        success = _install_component(comp)
        if success:
            console.print(f"  {_CHECKMARK} [green]{comp.label} installed[/green]")
        else:
            console.print(f"  {_CROSS} [red]{comp.label} installation failed[/red]")
            console.print(f"  [dim]Try manually: pip install {' '.join(comp.packages)}[/dim]")

    console.print("\n[green]Done.[/green] Run [bold]ppmlx install --status[/bold] to verify.")


def install_component(key: str) -> bool:
    """Install a single component by key. Returns True on success."""
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        console.print(f"[red]Unknown component: {key!r}[/red]")
        console.print(f"Available: {', '.join(_COMPONENT_MAP)}")
        return False

    if _is_installed(comp):
        console.print(f"[green]{comp.label} is already installed.[/green]")
        return True

    console.print(f"Installing [bold]{comp.label}[/bold]…")
    success = _install_component(comp)
    if success:
        console.print(f"{_CHECKMARK} [green]{comp.label} installed[/green]")
    else:
        console.print(f"{_CROSS} [red]Installation failed[/red]")
        console.print(f"[dim]Try manually: pip install {' '.join(comp.packages)}[/dim]")
    return success


def prompt_install_if_missing(key: str, feature_name: str) -> bool:
    """If a component is missing, prompt the user to install it.

    Returns True if the component is available (installed now or already was).
    Call this at the entry point of any feature that needs optional deps.
    """
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        return False

    if _is_installed(comp):
        return True

    import questionary
    console.print(f"\n[yellow]{feature_name} requires the [bold]{comp.label}[/bold] component ({comp.size_hint}).[/yellow]")
    console.print(f"[dim]{comp.description}[/dim]\n")

    answer = questionary.confirm(f"Install {comp.label} now?", default=True).ask()
    if not answer:
        console.print(f"[dim]Skipped. Install later with: ppmlx install {key}[/dim]")
        return False

    return install_component(key)
