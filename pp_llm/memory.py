from __future__ import annotations
import subprocess
from pathlib import Path


def get_system_ram_bytes() -> int:
    """
    Return total system RAM in bytes using sysctl.
    Falls back to 8 GB if sysctl fails.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return 8 * 1024 ** 3  # fallback: 8 GB


def get_system_ram_gb() -> float:
    """Return total system RAM in GB."""
    return get_system_ram_bytes() / (1024 ** 3)


def estimate_model_memory_bytes(model_path: Path) -> int:
    """
    Estimate memory required to run a model by summing file sizes * 1.2.
    Returns bytes.
    """
    if not model_path.exists():
        return 0
    total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    return int(total * 1.2)


def estimate_model_memory_gb(model_path: Path) -> float:
    """Return estimated memory in GB."""
    return estimate_model_memory_bytes(model_path) / (1024 ** 3)


def check_memory_warning(model_path: Path) -> str | None:
    """
    Return a warning string if the model likely won't fit in RAM.
    Returns None if memory is sufficient.
    """
    system_ram_gb = get_system_ram_gb()
    model_gb = estimate_model_memory_gb(model_path)

    if model_gb == 0:
        return None

    ratio = model_gb / system_ram_gb

    if ratio > 1.0:
        return (
            f"⚠️  This model requires ~{model_gb:.1f} GB but your Mac only has {system_ram_gb:.0f} GB RAM.\n"
            f"   Inference will be extremely slow due to memory swapping.\n"
            f"   Consider a smaller model."
        )
    elif ratio > 0.8:
        return (
            f"⚠️  This model requires ~{model_gb:.1f} GB ({ratio*100:.0f}% of your {system_ram_gb:.0f} GB RAM).\n"
            f"   Performance may be reduced. Consider a smaller model."
        )
    return None


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"
