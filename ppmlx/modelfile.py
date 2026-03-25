from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ModelfileParseError(ValueError):
    """Raised when a Modelfile cannot be parsed."""


@dataclass
class ModelfileConfig:
    """Parsed representation of a Modelfile."""
    name: str                                    # user-provided name (set when saving)
    from_model: str                              # base model alias or HF repo
    system: str | None = None                    # system prompt
    template: str | None = None                  # custom chat template
    license: str | None = None                   # license text
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "from_model": self.from_model,
            "system": self.system,
            "template": self.template,
            "license": self.license,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelfileConfig":
        obj = cls(name=data["name"], from_model=data["from_model"])
        obj.system = data.get("system")
        obj.template = data.get("template")
        obj.license = data.get("license")
        obj.parameters = data.get("parameters", {})
        return obj


def _parse_value(raw: str) -> Any:
    """Parse a PARAMETER value with type coercion."""
    raw = raw.strip()
    # JSON list/dict
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    # Quoted string
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    # Bool
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    # Int
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _read_block(lines: list[str], start: int) -> tuple[str, int]:
    """
    Read a triple-quoted block starting at line `start`.
    The opening \"\"\" must be present on lines[start] (after the directive keyword).
    Returns (content, next_line_index).
    """
    content_lines = []
    i = start
    first = lines[i]
    after_open = first[first.find('"""') + 3:]
    i += 1

    if '"""' in after_open:
        content = after_open[:after_open.find('"""')]
        return content.strip(), i

    content_lines.append(after_open)
    while i < len(lines):
        line = lines[i]
        if '"""' in line:
            before_close = line[:line.find('"""')]
            content_lines.append(before_close)
            i += 1
            break
        content_lines.append(line)
        i += 1

    return "\n".join(content_lines).strip(), i


def parse_modelfile(text: str, name: str = "") -> ModelfileConfig:
    """
    Parse a Modelfile text.

    Directives (case-insensitive):
      FROM <model>
      SYSTEM \"\"\"...\"\"\"  or  SYSTEM <single-line>
      PARAMETER <key> <value>
      TEMPLATE \"\"\"...\"\"\"
      LICENSE \"\"\"...\"\"\"

    Comments: lines starting with #
    """
    lines = text.splitlines()
    from_model: str | None = None
    system: str | None = None
    template: str | None = None
    license_text: str | None = None
    parameters: dict[str, Any] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        upper = stripped.upper()

        if upper.startswith("FROM "):
            from_model = stripped[5:].strip()
            i += 1

        elif upper.startswith("SYSTEM"):
            rest = stripped[6:].strip()
            if rest.startswith('"""'):
                content, i = _read_block(lines, i)
                system = content
            elif rest:
                system = rest
                i += 1
            else:
                i += 1

        elif upper.startswith("PARAMETER "):
            rest = stripped[10:].strip()
            parts = rest.split(None, 1)
            if len(parts) == 2:
                key, val_raw = parts
                parameters[key.lower()] = _parse_value(val_raw)
            i += 1

        elif upper.startswith("TEMPLATE"):
            rest = stripped[8:].strip()
            if rest.startswith('"""'):
                content, i = _read_block(lines, i)
                template = content
            elif rest:
                template = rest
                i += 1
            else:
                i += 1

        elif upper.startswith("LICENSE"):
            rest = stripped[7:].strip()
            if rest.startswith('"""'):
                content, i = _read_block(lines, i)
                license_text = content
            elif rest:
                license_text = rest
                i += 1
            else:
                i += 1

        else:
            print(f"[ppmlx] Warning: unknown Modelfile directive: {stripped!r}", file=sys.stderr)
            i += 1

    if from_model is None:
        raise ModelfileParseError("Modelfile is missing required FROM directive.")

    cfg = ModelfileConfig(name=name, from_model=from_model)
    cfg.system = system
    cfg.template = template
    cfg.license = license_text
    cfg.parameters = parameters
    return cfg


def _get_modelfiles_dir() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        d = get_ppmlx_dir() / "modelfiles"
    except ImportError:
        d = Path.home() / ".ppmlx" / "modelfiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_modelfile(name: str, config: ModelfileConfig) -> Path:
    """Save a ModelfileConfig as JSON to ~/.ppmlx/modelfiles/<name>.json."""
    config.name = name
    p = _get_modelfiles_dir() / f"{name}.json"
    p.write_text(json.dumps(config.to_dict(), indent=2))
    return p


def load_modelfile(name: str) -> ModelfileConfig | None:
    """Load a saved ModelfileConfig by name. Returns None if not found."""
    p = _get_modelfiles_dir() / f"{name}.json"
    if not p.exists():
        return None
    try:
        return ModelfileConfig.from_dict(json.loads(p.read_text()))
    except Exception:
        return None


def list_modelfiles() -> list[str]:
    """Return names of all saved modelfiles."""
    d = _get_modelfiles_dir()
    return [p.stem for p in sorted(d.glob("*.json"))]


def delete_modelfile(name: str) -> bool:
    """Delete a saved modelfile. Returns True if it existed."""
    p = _get_modelfiles_dir() / f"{name}.json"
    if p.exists():
        p.unlink()
        return True
    return False
