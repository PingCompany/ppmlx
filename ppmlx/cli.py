from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="ppmlx",
    help="Run LLMs locally on Apple Silicon via MLX",
    no_args_is_help=True,
)
console = Console()


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

    effective_host = host or cfg.server.host
    effective_port = port or cfg.server.port

    # Interactive model selection
    if interactive and model is None:
        import questionary
        from ppmlx.models import list_local_models
        local = list_local_models()
        if local:
            choices = [questionary.Choice("(none — lazy load on first request)", value=None)]
            for m in local:
                label = f"{m['alias']:<22} {m['size_gb']:.1f} GB"
                choices.append(questionary.Choice(label, value=m["alias"]))
            model = questionary.select("Select model to pre-load:", choices=choices).ask()
        else:
            console.print("[dim]No local models found. Download one first: ppmlx pull[/dim]")

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

    uvicorn.run(
        "ppmlx.server:app",
        host=effective_host,
        port=effective_port,
        log_level="info",
        reload=False,
    )


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or alias"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size", help="Max KV cache tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
):
    """Start an interactive chat REPL with a model."""
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

    console.print(f"[green]Chatting with [bold]{model}[/bold]. Type /help or /? for commands, /bye to exit.[/green]")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
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
            console.print("  [cyan]/help[/cyan] or [cyan]/?[/cyan]        Show this help")
            console.print("  [cyan]/clear[/cyan]            Clear conversation history")
            console.print("  [cyan]/system <prompt>[/cyan]  Set a new system prompt")
            console.print("  [cyan]/model <name>[/cyan]     Switch to a different model")
            console.print("  [cyan]/bye[/cyan]              Exit")
            continue
        elif user_input.startswith("/system "):
            new_system = user_input[8:].strip()
            messages = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": new_system})
            console.print(f"[dim]System prompt updated.[/dim]")
            continue
        elif user_input == "/clear":
            system_msgs = [m for m in messages if m["role"] == "system"]
            messages = system_msgs
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        elif user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            try:
                repo_id = resolve_alias(new_model)
                model = new_model
                console.print(f"[dim]Switched to {model}[/dim]")
            except ModelNotFoundError as e:
                console.print(f"[red]{e}[/red]")
            continue

        messages.append({"role": "user", "content": user_input})

        console.print("[bold green]Assistant:[/bold green] ", end="")
        full_response = ""
        try:
            for chunk in engine.stream_generate(
                repo_id, messages,
                temperature=temperature or 0.7,
                max_tokens=max_tokens or 2048,
            ):
                console.print(chunk, end="")
                full_response += chunk
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            continue
        console.print()  # newline after response
        messages.append({"role": "assistant", "content": full_response})


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
        warning = check_memory_warning(local_path)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
        return True
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
        from ppmlx.models import DEFAULT_ALIASES, list_local_models

        local_repos = {m["repo_id"] for m in list_local_models()}
        choices = []
        for alias, repo in DEFAULT_ALIASES.items():
            if alias.startswith("embed:"):
                continue
            tick = " ✓" if repo in local_repos else ""
            label = f"{alias:<24} {repo}{tick}"
            choices.append(questionary.Choice(label, value=alias))

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
def list_models():
    """List locally downloaded models."""
    from ppmlx.models import list_local_models

    models = list_local_models()
    if not models:
        console.print("[dim]No models downloaded yet. Run: ppmlx pull <model>[/dim]")
        return

    table = Table(title="Local Models", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Alias", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    for m in models:
        table.add_row(
            m.get("name", ""),
            m.get("alias", m.get("repo_id", "")),
            f"{m.get('size_gb', 0):.2f} GB",
            str(m.get("path", "")),
        )
    console.print(table)


@app.command()
def rm(
    model: str = typer.Argument(..., help="Model alias or name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a locally downloaded model."""
    from ppmlx.models import remove_model, get_model_path, resolve_alias

    try:
        repo_id = resolve_alias(model)
    except Exception:
        repo_id = model

    path = get_model_path(repo_id)
    if not path:
        console.print(f"[yellow]Model '{model}' not found locally.[/yellow]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Remove model '{model}' from {path}?")
        if not confirm:
            raise typer.Abort()

    removed = remove_model(model)
    if removed:
        console.print(f"[green]Removed {model}[/green]")
    else:
        console.print(f"[red]Failed to remove {model}[/red]")
        raise typer.Exit(1)


@app.command(name="alias")
def add_alias(
    name: str = typer.Argument(..., help="Alias name (e.g. my-model)"),
    repo: str = typer.Argument(..., help="HuggingFace repo ID (e.g. org/model)"),
):
    """Add a custom model alias."""
    from ppmlx.models import save_user_alias
    save_user_alias(name, repo)
    console.print(f"[green]Alias created: [bold]{name}[/bold] -> {repo}[/green]")


@app.command()
def aliases():
    """Show all model aliases (built-in + custom)."""
    from ppmlx.models import DEFAULT_ALIASES, load_user_aliases

    user_aliases = load_user_aliases()

    table = Table(title="Model Aliases", show_header=True)
    table.add_column("Alias", style="cyan")
    table.add_column("HuggingFace Repo", style="green")
    table.add_column("Source", style="dim")

    for alias, repo in sorted(DEFAULT_ALIASES.items()):
        source = "custom" if alias in user_aliases else "built-in"
        style = "yellow" if alias in user_aliases else ""
        table.add_row(alias, repo, source, style=style)

    for alias, repo in sorted(user_aliases.items()):
        if alias not in DEFAULT_ALIASES:
            table.add_row(alias, repo, "custom", style="yellow")

    console.print(table)


@app.command()
def ps():
    """Show currently loaded models and memory usage."""
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
    model: str = typer.Argument(..., help="HuggingFace repo ID or alias"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (2,3,4,6,8)"),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    upload: Optional[str] = typer.Option(None, "--upload-repo", help="HF repo to upload to"),
    token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token"),
):
    """Convert and quantize a HuggingFace model to MLX format."""
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


@app.command()
def create(
    name: str = typer.Argument(..., help="Name for the new model"),
    file: str = typer.Option("Modelfile", "-f", help="Path to Modelfile"),
):
    """Create a custom model from a Modelfile."""
    from ppmlx.modelfile import parse_modelfile, save_modelfile, ModelfileParseError

    mf_path = Path(file)
    if not mf_path.exists():
        console.print(f"[red]Modelfile not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        text = mf_path.read_text()
        cfg = parse_modelfile(text, name=name)
        saved_path = save_modelfile(name, cfg)
        console.print(f"[green]Created model [bold]{name}[/bold] from {cfg.from_model}[/green]")
        console.print(f"[dim]Config saved: {saved_path}[/dim]")
    except ModelfileParseError as e:
        console.print(f"[red]Modelfile parse error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def logs(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries to show"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model"),
    since: Optional[str] = typer.Option(None, "--since", help="Time window (e.g. 1h, 24h)"),
    errors: bool = typer.Option(False, "--errors", help="Show only errors"),
    stats: bool = typer.Option(False, "--stats", help="Show aggregate statistics"),
    export: Optional[str] = typer.Option(None, "--export", help="Export format: csv"),
    slow: Optional[float] = typer.Option(None, "--slow", help="Show requests slower than N ms"),
):
    """Query the request log database."""
    from ppmlx.db import get_db

    db = get_db()
    db.init()

    if stats:
        s = db.get_stats()
        table = Table(title="ppmlx Statistics (last 24h)")
        table.add_column("Model", style="cyan")
        table.add_column("Requests", justify="right")
        table.add_column("Avg tok/s", justify="right")
        table.add_column("Avg TTFT", justify="right")
        table.add_column("Errors", justify="right", style="red")
        for m in s.get("by_model", []):
            tps = f"{m['avg_tps']:.1f}" if m.get("avg_tps") else "—"
            ttft = f"{m['avg_ttft']:.0f}ms" if m.get("avg_ttft") else "—"
            table.add_row(m["model"], str(m["count"]), tps, ttft, str(m["errors"]))
        console.print(table)
        console.print(f"[dim]Total requests: {s['total_requests']}[/dim]")
        return

    since_hours = None
    if since:
        try:
            if since.endswith("h"):
                since_hours = float(since[:-1])
            elif since.endswith("m"):
                since_hours = float(since[:-1]) / 60
        except ValueError:
            pass

    rows = db.query_requests(
        limit=limit,
        model=model,
        since_hours=since_hours,
        errors_only=errors,
        min_duration_ms=slow,
    )

    if not rows:
        console.print("[dim]No log entries found.[/dim]")
        return

    if export == "csv":
        import csv, io
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        console.print(buf.getvalue())
        return

    table = Table(title=f"Recent Requests (last {limit})", show_header=True)
    table.add_column("Time", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")

    for row in rows:
        ts = str(row.get("timestamp", ""))[:19]
        mdl = str(row.get("model_alias", ""))
        status = row.get("status", "ok")
        status_str = f"[green]{status}[/green]" if status == "ok" else f"[red]{status}[/red]"
        dur = f"{row['total_duration_ms']:.0f}ms" if row.get("total_duration_ms") else "—"
        toks = str(row.get("total_tokens", "—"))
        table.add_row(ts, mdl, status_str, dur, toks)

    console.print(table)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model alias or repo ID"),
):
    """Show detailed model information."""
    from ppmlx.models import resolve_alias, get_model_path, ModelNotFoundError
    from ppmlx.memory import estimate_model_memory_gb, get_system_ram_gb

    try:
        repo_id = resolve_alias(model)
    except ModelNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    local_path = get_model_path(repo_id)

    table = Table(title=f"Model: {model}", show_header=False)
    table.add_column("Property", style="dim")
    table.add_column("Value", style="cyan")

    table.add_row("Alias", model)
    table.add_row("HF Repo", repo_id)
    table.add_row("Downloaded", "Yes" if local_path else "No")

    if local_path:
        size_gb = estimate_model_memory_gb(local_path)
        table.add_row("Estimated RAM", f"{size_gb:.1f} GB")
        table.add_row("Local Path", str(local_path))
        ram_gb = get_system_ram_gb()
        table.add_row("System RAM", f"{ram_gb:.0f} GB")

    console.print(table)


@app.command()
def estimate(
    model: str = typer.Argument(..., help="Model alias or repo ID"),
):
    """Estimate RAM requirements before downloading."""
    from ppmlx.models import resolve_alias, get_model_path, ModelNotFoundError
    from ppmlx.memory import estimate_model_memory_gb, get_system_ram_gb, check_memory_warning

    try:
        repo_id = resolve_alias(model)
    except ModelNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    local_path = get_model_path(repo_id)
    ram_gb = get_system_ram_gb()

    if local_path:
        model_gb = estimate_model_memory_gb(local_path)
        console.print(f"[cyan]Model:[/cyan] {model} ({repo_id})")
        console.print(f"[cyan]Estimated RAM:[/cyan] {model_gb:.1f} GB")
        console.print(f"[cyan]System RAM:[/cyan] {ram_gb:.0f} GB")
        warning = check_memory_warning(local_path)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
        else:
            console.print("[green]Should fit comfortably in RAM.[/green]")
    else:
        console.print(f"[yellow]Model not downloaded. Cannot estimate size accurately.[/yellow]")
        console.print(f"[cyan]System RAM:[/cyan] {ram_gb:.0f} GB")
        console.print(f"[dim]Pull the model first: ppmlx pull {model}[/dim]")


if __name__ == "__main__":
    app()
