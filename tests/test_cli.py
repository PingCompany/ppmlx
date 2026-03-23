import sys
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

# Mock all pp_llm modules before importing cli
for mod_name in ["pp_llm.models", "pp_llm.engine", "pp_llm.db",
                  "pp_llm.config", "pp_llm.memory", "pp_llm.modelfile",
                  "pp_llm.quantize", "pp_llm.engine_embed", "pp_llm.engine_vlm"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from pp_llm.cli import app

runner = CliRunner()


def test_version():
    """--version returns 0.1.0 and exits 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help():
    """--help exits 0 and mentions pp-llm."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pp-llm" in result.output


def test_aliases_command():
    """aliases command renders a table with alias/repo columns."""
    mock_defaults = {
        "llama3": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    }
    mock_user = {}
    sys.modules["pp_llm.models"].DEFAULT_ALIASES = mock_defaults
    sys.modules["pp_llm.models"].load_user_aliases = MagicMock(return_value=mock_user)

    result = runner.invoke(app, ["aliases"])
    assert result.exit_code == 0
    assert "llama3" in result.output
    assert "mistral" in result.output
    assert "built-in" in result.output


def test_list_command_empty():
    """list command shows 'No models' message when no models are downloaded."""
    sys.modules["pp_llm.models"].list_local_models = MagicMock(return_value=[])

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models" in result.output


def test_list_command_with_model():
    """list command renders a table when models are present."""
    mock_models = [
        {
            "name": "Meta-Llama-3-8B-Instruct-4bit",
            "alias": "llama3",
            "repo_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "size_gb": 4.5,
            "path": "/Users/test/.pp-llm/models/llama3",
        }
    ]
    sys.modules["pp_llm.models"].list_local_models = MagicMock(return_value=mock_models)

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Meta-Llama-3-8B-Instruct-4bit" in result.output or "llama3" in result.output


def test_pull_command():
    """pull command calls download_model with the correct model name."""
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["pp_llm.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["pp_llm.models"].resolve_alias = MagicMock(return_value="mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    sys.modules["pp_llm.models"].download_model = MagicMock(return_value="/tmp/mistral")
    sys.modules["pp_llm.memory"].check_memory_warning = MagicMock(return_value=None)
    sys.modules["pp_llm.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

    result = runner.invoke(app, ["pull", "mistral"])
    assert result.exit_code == 0
    sys.modules["pp_llm.models"].download_model.assert_called_once()
    call_args = sys.modules["pp_llm.models"].download_model.call_args
    assert call_args[0][0] == "mistral" or call_args[1].get("model") == "mistral" or "mistral" in str(call_args)


def test_pull_unknown_model():
    """pull command exits with code 1 when model is not found."""
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["pp_llm.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["pp_llm.models"].resolve_alias = MagicMock(side_effect=ModelNotFoundError("not found"))

    result = runner.invoke(app, ["pull", "nonexistent-model-xyz"])
    assert result.exit_code == 1


def test_rm_with_force():
    """rm --force skips confirmation and calls remove_model."""
    sys.modules["pp_llm.models"].resolve_alias = MagicMock(return_value="mlx-community/some-model")
    sys.modules["pp_llm.models"].get_model_path = MagicMock(return_value="/tmp/some-model")
    sys.modules["pp_llm.models"].remove_model = MagicMock(return_value=True)

    result = runner.invoke(app, ["rm", "some-model", "--force"])
    assert result.exit_code == 0
    sys.modules["pp_llm.models"].remove_model.assert_called_once_with("some-model")


def test_add_alias():
    """alias command calls save_user_alias with correct (name, repo) args."""
    sys.modules["pp_llm.models"].save_user_alias = MagicMock()

    result = runner.invoke(app, ["alias", "my-model", "org/my-hf-model"])
    assert result.exit_code == 0
    sys.modules["pp_llm.models"].save_user_alias.assert_called_once_with("my-model", "org/my-hf-model")


def test_logs_command():
    """logs command shows 'No log entries' when database is empty."""
    mock_db = MagicMock()
    mock_db.query_requests.return_value = []
    sys.modules["pp_llm.db"].get_db = MagicMock(return_value=mock_db)

    result = runner.invoke(app, ["logs"])
    assert result.exit_code == 0
    assert "No log entries" in result.output


def test_serve_help():
    """serve --help shows available options."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output or "host" in result.output
    assert "--port" in result.output or "port" in result.output
