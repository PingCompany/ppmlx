from __future__ import annotations
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


@dataclass
class QuantizeConfig:
    """Configuration for model quantization."""
    bits: Literal[2, 3, 4, 6, 8] = 4
    group_size: int = 64
    output_path: Path | None = None
    upload_repo: str | None = None
    hf_token: str | None = None


class QuantizationError(Exception):
    pass


def _get_output_path(hf_path: str, bits: int) -> Path:
    """Generate default output path for a quantized model."""
    try:
        from ppmlx.config import get_ppmlx_dir
        models_dir = get_ppmlx_dir() / "models"
    except ImportError:
        models_dir = Path.home() / ".ppmlx" / "models"
    safe_name = hf_path.replace("/", "--")
    return models_dir / f"{safe_name}-{bits}bit"


def _resolve_source(hf_path_or_alias: str) -> str:
    """Resolve alias to repo ID if needed."""
    try:
        from ppmlx.models import resolve_alias
        return resolve_alias(hf_path_or_alias)
    except Exception:
        return hf_path_or_alias


def is_already_quantized(model_path: Path) -> bool:
    """Check if a model directory already contains quantized weights."""
    if not model_path.is_dir():
        return False
    # Quantized MLX models typically have a config.json with quantization info
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text())
            if data.get("quantization"):
                return True
        except Exception:
            pass
    # Also check for quantized weight file naming patterns
    for f in model_path.iterdir():
        if f.suffix == ".safetensors" and "quantized" in f.name.lower():
            return True
    return False


def check_disk_space(output_path: Path, required_bytes: int | None = None) -> str | None:
    """Check available disk space. Returns a warning string if low, else None."""
    try:
        usage = shutil.disk_usage(output_path.parent if output_path.parent.exists() else Path.home())
        free_gb = usage.free / (1024 ** 3)
        if required_bytes:
            required_gb = required_bytes / (1024 ** 3)
            if free_gb < required_gb * 1.5:
                return (
                    f"Low disk space: {free_gb:.1f} GB free, "
                    f"estimated {required_gb:.1f} GB needed for quantized model."
                )
        elif free_gb < 5:
            return f"Low disk space: only {free_gb:.1f} GB free."
    except Exception:
        pass
    return None


def quantize(
    hf_path_or_alias: str,
    config: QuantizeConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
    *,
    local_path: Path | None = None,
) -> Path:
    """
    Quantize a model to MLX format.

    When *local_path* is provided the model is read from that directory instead
    of being downloaded from HuggingFace.  *hf_path_or_alias* is still used to
    derive the output path and alias name.

    Returns the local path where the quantized model is saved.
    """
    if config is None:
        config = QuantizeConfig()

    repo_id = _resolve_source(hf_path_or_alias)
    output_path = config.output_path or _get_output_path(repo_id, config.bits)
    output_path.mkdir(parents=True, exist_ok=True)

    source = str(local_path) if local_path else repo_id

    source_path = local_path or Path(source)
    if source_path.is_dir() and is_already_quantized(source_path):
        if progress_callback:
            progress_callback("Model appears to be already quantized, proceeding anyway...")

    # Warn about disk space
    disk_warning = check_disk_space(output_path)
    if disk_warning and progress_callback:
        progress_callback(f"Warning: {disk_warning}")

    if progress_callback:
        progress_callback(f"Converting {repo_id} to {config.bits}-bit MLX format...")

    success = _try_python_api(source, output_path, config)
    if not success:
        success = _try_subprocess(source, output_path, config, progress_callback)

    if not success:
        raise QuantizationError(
            f"Failed to quantize {repo_id}. "
            "Ensure mlx-lm is installed and the model is accessible."
        )

    if progress_callback:
        progress_callback(f"Saved to {output_path}")

    if config.upload_repo:
        _upload_to_hub(output_path, config.upload_repo, config.hf_token, progress_callback)

    _create_alias(repo_id, output_path, config.bits)

    return output_path


def _try_python_api(source: str, output_path: Path, config: QuantizeConfig) -> bool:
    """Try calling mlx_lm.convert Python API."""
    try:
        import mlx_lm
        convert_fn = getattr(mlx_lm, "convert", None)
        if convert_fn is None:
            return False
        convert_fn(
            hf_path=source,
            mlx_path=str(output_path),
            quantize=True,
            q_bits=config.bits,
            q_group_size=config.group_size,
        )
        return True
    except Exception as e:
        print(f"[ppmlx] Python API failed: {e}, trying subprocess...", file=sys.stderr)
        return False


def _try_subprocess(
    source: str,
    output_path: Path,
    config: QuantizeConfig,
    progress_callback: Callable[[str], None] | None,
) -> bool:
    """Fall back to subprocess: python -m mlx_lm.convert ..."""
    try:
        cmd = [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", source,
            "--mlx-path", str(output_path),
            "--quantize",
            "--q-bits", str(config.bits),
            "--q-group-size", str(config.group_size),
        ]
        if config.hf_token:
            cmd += ["--hf-token", config.hf_token]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ppmlx] mlx_lm.convert error: {result.stderr}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[ppmlx] Subprocess fallback failed: {e}", file=sys.stderr)
        return False


def _upload_to_hub(
    local_path: Path,
    repo_id: str,
    token: str | None,
    progress_callback: Callable[[str], None] | None,
) -> None:
    """Upload quantized model to HuggingFace Hub."""
    try:
        from huggingface_hub import upload_folder
        if progress_callback:
            progress_callback(f"Uploading to {repo_id}...")
        upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            token=token,
        )
    except Exception as e:
        print(f"[ppmlx] Upload failed: {e}", file=sys.stderr)


def _create_alias(repo_id: str, output_path: Path, bits: int) -> None:
    """Auto-create a user alias for the quantized model."""
    try:
        from ppmlx.models import save_user_alias
        short = repo_id.split("/")[-1].lower()
        alias = f"{short}-{bits}bit"
        save_user_alias(alias, str(output_path))
    except Exception:
        pass
