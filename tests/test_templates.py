"""Tests for the prompt template engine and CLI commands."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ppmlx.templates import (
    PromptTemplate,
    VariableSpec,
    TemplateNotFoundError,
    TemplateMissingVariableError,
    TemplateValidationError,
    load_template_from_yaml,
    render_template,
    list_templates,
    get_template,
)

# Mock modules before importing cli
for mod_name in [
    "ppmlx.models", "ppmlx.engine", "ppmlx.db",
    "ppmlx.config", "ppmlx.memory", "ppmlx.modelfile",
    "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
    "ppmlx.registry",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.cli import app

runner = CliRunner()


# ── Template parsing tests ────────────────────────────────────────────


class TestLoadTemplateFromYaml:
    def test_basic_template(self):
        yaml_text = """
name: test
description: A test template
system: You are helpful.
prompt: "Answer: {{input}}"
variables:
  input:
    required: true
    description: User input
parameters:
  temperature: 0.5
"""
        t = load_template_from_yaml(yaml_text)
        assert t.name == "test"
        assert t.description == "A test template"
        assert t.system == "You are helpful."
        assert "{{input}}" in t.prompt
        assert "input" in t.variables
        assert t.variables["input"].required is True
        assert t.parameters["temperature"] == 0.5

    def test_variable_with_default(self):
        yaml_text = """
name: test
prompt: "{{input}} in {{style}}"
variables:
  input:
    required: true
    description: Main input
  style:
    default: bullet points
    description: Output style
"""
        t = load_template_from_yaml(yaml_text)
        assert t.variables["style"].required is False
        assert t.variables["style"].default == "bullet points"
        assert t.variables["input"].required is True

    def test_missing_name_raises(self):
        with pytest.raises(TemplateValidationError, match="name"):
            load_template_from_yaml("description: no name here")

    def test_invalid_yaml_raises(self):
        with pytest.raises(TemplateValidationError, match="Invalid YAML"):
            load_template_from_yaml("name: [unbalanced")

    def test_non_dict_raises(self):
        with pytest.raises(TemplateValidationError, match="mapping"):
            load_template_from_yaml("- item1\n- item2")

    def test_default_model(self):
        yaml_text = """
name: test
prompt: "Hello {{input}}"
default_model: llama3
variables:
  input:
    required: true
"""
        t = load_template_from_yaml(yaml_text)
        assert t.default_model == "llama3"

    def test_no_variables_section(self):
        yaml_text = """
name: test
prompt: "Just a static prompt"
"""
        t = load_template_from_yaml(yaml_text)
        assert t.variables == {}


# ── Rendering tests ───────────────────────────────────────────────────


class TestRenderTemplate:
    def test_basic_render(self):
        t = PromptTemplate(
            name="test",
            system="You are a {{role}}.",
            prompt="Help with: {{input}}",
            variables={
                "input": VariableSpec(required=True),
                "role": VariableSpec(required=False, default="helper"),
            },
        )
        system, prompt = render_template(t, {"input": "my question"})
        assert system == "You are a helper."
        assert prompt == "Help with: my question"

    def test_override_default(self):
        t = PromptTemplate(
            name="test",
            prompt="Style: {{style}} Content: {{input}}",
            variables={
                "input": VariableSpec(required=True),
                "style": VariableSpec(required=False, default="formal"),
            },
        )
        _, prompt = render_template(t, {"input": "hello", "style": "casual"})
        assert "casual" in prompt
        assert "formal" not in prompt

    def test_missing_required_raises(self):
        t = PromptTemplate(
            name="test",
            prompt="{{input}}",
            variables={"input": VariableSpec(required=True)},
        )
        with pytest.raises(TemplateMissingVariableError, match="input"):
            render_template(t, {})

    def test_extra_variables_allowed(self):
        t = PromptTemplate(
            name="test",
            prompt="{{input}} and {{extra}}",
            variables={"input": VariableSpec(required=True)},
        )
        _, prompt = render_template(t, {"input": "hello", "extra": "world"})
        assert prompt == "hello and world"

    def test_unreferenced_variable_in_template(self):
        """Variables in the template string that are not in variables dict stay as-is."""
        t = PromptTemplate(
            name="test",
            prompt="{{input}} and {{unknown}}",
            variables={"input": VariableSpec(required=True)},
        )
        _, prompt = render_template(t, {"input": "hello"})
        assert prompt == "hello and {{unknown}}"


# ── Discovery tests ──────────────────────────────────────────────────


class TestListAndGetTemplates:
    def test_builtin_templates_exist(self):
        """Verify that built-in templates can be discovered."""
        templates = list_templates()
        names = [t.name for t in templates]
        assert "summarize" in names
        assert "translate" in names
        assert "code_review" in names
        assert "explain_code" in names
        assert "rewrite" in names
        assert "brainstorm" in names
        assert "email_draft" in names
        assert "debug_error" in names

    def test_get_existing_template(self):
        t = get_template("summarize")
        assert t.name == "summarize"
        assert t.description
        assert "input" in t.variables

    def test_get_nonexistent_template(self):
        with pytest.raises(TemplateNotFoundError):
            get_template("nonexistent_template_xyz")

    def test_user_template_overrides_builtin(self, tmp_path, monkeypatch):
        """User templates in ~/.ppmlx/templates/ override built-in ones."""
        user_templates = tmp_path / ".ppmlx" / "templates"
        user_templates.mkdir(parents=True)

        custom_yaml = user_templates / "summarize.yaml"
        custom_yaml.write_text(
            "name: summarize\n"
            "description: My custom summarizer\n"
            "prompt: 'Custom: {{input}}'\n"
            "variables:\n"
            "  input:\n"
            "    required: true\n"
        )

        monkeypatch.setenv("HOME", str(tmp_path))

        t = get_template("summarize")
        assert t.source == "user"
        assert t.description == "My custom summarizer"

    def test_user_template_create_and_load(self, tmp_path, monkeypatch):
        """User templates can be created and loaded."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from ppmlx.templates import _user_dir
        user_dir = _user_dir(create=True)

        custom_yaml = user_dir / "my_custom.yaml"
        custom_yaml.write_text(
            "name: my_custom\n"
            "description: A user template\n"
            "prompt: '{{input}}'\n"
            "variables:\n"
            "  input:\n"
            "    required: true\n"
        )

        t = get_template("my_custom")
        assert t.name == "my_custom"
        assert t.source == "user"


# ── CLI command tests ─────────────────────────────────────────────────


class TestTemplateCLI:
    def test_template_list(self):
        """template list shows built-in templates."""
        result = runner.invoke(app, ["template", "list"])
        assert result.exit_code == 0
        assert "summarize" in result.output
        assert "translate" in result.output

    def test_template_show(self):
        """template show displays template details."""
        result = runner.invoke(app, ["template", "show", "summarize"])
        assert result.exit_code == 0
        assert "summarize" in result.output
        assert "input" in result.output

    def test_template_show_nonexistent(self):
        """template show exits with error for unknown template."""
        result = runner.invoke(app, ["template", "show", "nonexistent_xyz"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_template_run_missing_var(self):
        """template run exits with error when required variable is missing."""
        result = runner.invoke(app, ["template", "run", "summarize"])
        assert result.exit_code == 1
        assert "input" in result.output

    def test_template_run_nonexistent(self):
        """template run exits with error for unknown template."""
        result = runner.invoke(app, ["template", "run", "nonexistent_xyz"])
        assert result.exit_code == 1

    def test_template_run_bad_var_format(self):
        """template run exits with error on invalid --var format."""
        result = runner.invoke(app, ["template", "run", "summarize", "--var", "no_equals_sign"])
        assert result.exit_code == 1
        assert "key=value" in result.output

    def test_template_create(self, tmp_path, monkeypatch):
        """template create saves a YAML file to user templates dir."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = runner.invoke(app, [
            "template", "create", "my_test",
            "--description", "Test template",
            "--system", "Be helpful",
            "--prompt", "Do this: {{input}}",
        ])
        assert result.exit_code == 0
        assert "Template created" in result.output

        # Verify the file was created
        created = tmp_path / ".ppmlx" / "templates" / "my_test.yaml"
        assert created.exists()
        content = created.read_text()
        assert "my_test" in content
        assert "input" in content

    def test_template_create_no_prompt(self):
        """template create fails without --prompt."""
        result = runner.invoke(app, [
            "template", "create", "my_test",
        ])
        assert result.exit_code == 1
        assert "--prompt" in result.output

    def test_template_help(self):
        """template --help shows subcommands."""
        result = runner.invoke(app, ["template", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "run" in result.output
        assert "create" in result.output
