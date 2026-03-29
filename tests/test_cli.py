import sys
from unittest.mock import MagicMock
from typer.testing import CliRunner

# Mock all ppmlx modules before importing cli
for mod_name in ["ppmlx.models", "ppmlx.engine", "ppmlx.db",
                  "ppmlx.config", "ppmlx.memory",
                  "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
                  "ppmlx.registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.cli import app

runner = CliRunner()


def test_version():
    """--version returns current version and exits 0."""
    from ppmlx import __version__
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help():
    """--help exits 0 and mentions ppmlx."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ppmlx" in result.output


def _setup_model_mocks(
    defaults=None,
    user_aliases=None,
    local_models=None,
    favorites=None,
):
    """Configure ppmlx.models mocks for commands using _build_model_records."""
    if defaults is None:
        defaults = {}
    if user_aliases is None:
        user_aliases = {}
    if local_models is None:
        local_models = []
    if favorites is None:
        favorites = []
    merged = {**defaults, **user_aliases}
    sys.modules["ppmlx.models"].DEFAULT_ALIASES = defaults
    sys.modules["ppmlx.models"].load_user_aliases = MagicMock(return_value=user_aliases)
    sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value=merged)
    sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=local_models)
    sys.modules["ppmlx.models"].load_favorites = MagicMock(return_value=favorites)
    sys.modules["ppmlx.registry"].registry_entries = MagicMock(return_value={})


def test_list_command_empty():
    """list command shows 'No models' message when no models are downloaded."""
    _setup_model_mocks()

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models" in result.output


def test_list_command_with_model():
    """list command opens TUI browser when models are present."""
    from unittest.mock import patch

    mock_models = [
        {
            "name": "Meta-Llama-3-8B-Instruct-4bit",
            "alias": "llama3",
            "repo_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "size_gb": 4.5,
            "path": "/Users/test/.ppmlx/models/llama3",
        }
    ]
    _setup_model_mocks(
        defaults={"llama3": "mlx-community/Meta-Llama-3-8B-Instruct-4bit"},
        local_models=mock_models,
    )

    with patch("ppmlx.tui.browse_models") as mock_browse:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_browse.assert_called_once()
        rows = mock_browse.call_args[0][0]
        aliases = [r.alias for r in rows if r.section_header is None]
        assert "llama3" in aliases


def test_pull_command():
    """pull command calls download_model with the correct model name."""
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["ppmlx.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(return_value="mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    sys.modules["ppmlx.models"].download_model = MagicMock(return_value="/tmp/mistral")
    sys.modules["ppmlx.memory"].check_memory_warning = MagicMock(return_value=None)
    sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

    result = runner.invoke(app, ["pull", "mistral"])
    assert result.exit_code == 0
    sys.modules["ppmlx.models"].download_model.assert_called_once()
    call_args = sys.modules["ppmlx.models"].download_model.call_args
    assert call_args[0][0] == "mistral" or call_args[1].get("model") == "mistral" or "mistral" in str(call_args)


def test_pull_unknown_model():
    """pull command exits with code 1 when model is not found."""
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["ppmlx.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=ModelNotFoundError("not found"))

    result = runner.invoke(app, ["pull", "nonexistent-model-xyz"])
    assert result.exit_code == 1


def test_rm_with_force():
    """rm --force skips confirmation and calls remove_model."""
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(return_value="mlx-community/some-model")
    sys.modules["ppmlx.models"].get_model_path = MagicMock(return_value="/tmp/some-model")
    sys.modules["ppmlx.models"].remove_model = MagicMock(return_value=True)

    result = runner.invoke(app, ["rm", "some-model", "--force"])
    assert result.exit_code == 0
    sys.modules["ppmlx.models"].remove_model.assert_called_once_with("some-model")


def test_serve_help():
    """serve --help shows available options."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output or "host" in result.output
    assert "--port" in result.output or "port" in result.output
