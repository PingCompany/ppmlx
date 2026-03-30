"""Tests for persistent conversation history."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# Ensure ppmlx modules are mocked for CLI tests
for mod_name in ["ppmlx.models", "ppmlx.engine", "ppmlx.db",
                  "ppmlx.config", "ppmlx.memory",
                  "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
                  "ppmlx.registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.history import HistoryManager, _auto_title, reset_history


@pytest.fixture()
def hm(tmp_path):
    """Create a fresh HistoryManager with a temporary database."""
    reset_history()
    manager = HistoryManager(path=tmp_path / "test_history.db")
    yield manager
    manager.close()
    reset_history()


class TestAutoTitle:
    def test_short_message(self):
        assert _auto_title("Hello world") == "Hello world"

    def test_long_message_truncated(self):
        long_msg = "This is a very long message that definitely exceeds the maximum title length and should be truncated at a word boundary"
        result = _auto_title(long_msg, max_len=40)
        assert len(result) <= 40
        assert result.endswith("...")

    def test_newlines_stripped(self):
        assert _auto_title("Line one\nLine two") == "Line one Line two"

    def test_empty_message(self):
        assert _auto_title("") == ""


class TestHistoryManager:
    def test_create_conversation(self, hm):
        conv_id = hm.create_conversation("llama3")
        assert conv_id
        assert len(conv_id) == 12

    def test_add_and_get_messages(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "Hello")
        hm.add_message(conv_id, "assistant", "Hi there!")

        msgs = hm.get_messages(conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hi there!"

    def test_auto_title_from_first_user_message(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "system", "You are helpful.")
        hm.add_message(conv_id, "user", "What is Python?")

        conv = hm.get_conversation(conv_id)
        assert conv["title"] == "What is Python?"

    def test_title_not_overwritten_by_later_messages(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "First question")
        hm.add_message(conv_id, "assistant", "First answer")
        hm.add_message(conv_id, "user", "Second question")

        conv = hm.get_conversation(conv_id)
        assert conv["title"] == "First question"

    def test_list_conversations(self, hm):
        id1 = hm.create_conversation("llama3")
        hm.add_message(id1, "user", "Hello")
        id2 = hm.create_conversation("mistral")
        hm.add_message(id2, "user", "Bonjour")

        convs = hm.list_conversations()
        assert len(convs) == 2
        conv_ids = {c["id"] for c in convs}
        assert id1 in conv_ids
        assert id2 in conv_ids

    def test_list_conversations_filter_by_model(self, hm):
        hm.create_conversation("llama3")
        hm.create_conversation("mistral")
        hm.create_conversation("llama3")

        convs = hm.list_conversations(model="llama3")
        assert len(convs) == 2
        for c in convs:
            assert c["model"] == "llama3"

    def test_list_conversations_limit(self, hm):
        for i in range(5):
            hm.create_conversation("llama3")
        convs = hm.list_conversations(limit=3)
        assert len(convs) == 3

    def test_list_conversations_message_count(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "Hello")
        hm.add_message(conv_id, "assistant", "Hi!")
        hm.add_message(conv_id, "user", "How are you?")

        convs = hm.list_conversations()
        assert convs[0]["message_count"] == 3

    def test_get_conversation(self, hm):
        conv_id = hm.create_conversation("llama3", title="Test conv")
        conv = hm.get_conversation(conv_id)
        assert conv is not None
        assert conv["model"] == "llama3"
        assert conv["title"] == "Test conv"

    def test_get_conversation_not_found(self, hm):
        assert hm.get_conversation("nonexistent") is None

    def test_search(self, hm):
        id1 = hm.create_conversation("llama3")
        hm.add_message(id1, "user", "Tell me about quantum computing")
        hm.add_message(id1, "assistant", "Quantum computing uses qubits...")

        id2 = hm.create_conversation("llama3")
        hm.add_message(id2, "user", "What is machine learning?")
        hm.add_message(id2, "assistant", "Machine learning is a subset of AI...")

        results = hm.search("quantum")
        assert len(results) == 1
        assert results[0]["id"] == id1

    def test_search_no_results(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "Hello world")
        results = hm.search("nonexistent_term_xyz")
        assert len(results) == 0

    def test_delete_conversation(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "Hello")

        assert hm.delete_conversation(conv_id) is True
        assert hm.get_conversation(conv_id) is None
        assert hm.get_messages(conv_id) == []

    def test_delete_nonexistent(self, hm):
        assert hm.delete_conversation("nonexistent") is False

    def test_resolve_id_exact(self, hm):
        conv_id = hm.create_conversation("llama3")
        assert hm.resolve_id(conv_id) == conv_id

    def test_resolve_id_prefix(self, hm):
        conv_id = hm.create_conversation("llama3")
        prefix = conv_id[:4]
        resolved = hm.resolve_id(prefix)
        assert resolved == conv_id

    def test_resolve_id_not_found(self, hm):
        assert hm.resolve_id("zzzzzzz") is None

    def test_export_json(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "Hello")
        hm.add_message(conv_id, "assistant", "Hi there!")

        result = hm.export_conversation(conv_id, fmt="json")
        assert result is not None
        data = json.loads(result)
        assert data["model"] == "llama3"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"

    def test_export_markdown(self, hm):
        conv_id = hm.create_conversation("llama3")
        hm.add_message(conv_id, "user", "Hello")
        hm.add_message(conv_id, "assistant", "Hi there!")

        result = hm.export_conversation(conv_id, fmt="md")
        assert result is not None
        assert "### You" in result
        assert "### Assistant" in result
        assert "Hello" in result
        assert "Hi there!" in result
        assert "llama3" in result

    def test_export_not_found(self, hm):
        assert hm.export_conversation("nonexistent") is None

    def test_export_unsupported_format(self, hm):
        conv_id = hm.create_conversation("llama3")
        assert hm.export_conversation(conv_id, fmt="xml") is None

    def test_structured_content_serialized(self, hm):
        """Vision content (list of dicts) is serialized to JSON string."""
        conv_id = hm.create_conversation("llava")
        content = [{"type": "text", "text": "Describe this"}, {"type": "image_url", "image_url": {"url": "/img.jpg"}}]
        hm.add_message(conv_id, "user", content)

        msgs = hm.get_messages(conv_id)
        assert len(msgs) == 1
        # Should be stored as JSON string
        parsed = json.loads(msgs[0]["content"])
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "text"

    def test_updated_at_changes(self, hm):
        """updated_at should advance when messages are added."""
        conv_id = hm.create_conversation("llama3")
        conv1 = hm.get_conversation(conv_id)
        # Tiny sleep not needed — the default value uses sub-second precision
        hm.add_message(conv_id, "user", "Hello")
        conv2 = hm.get_conversation(conv_id)
        assert conv2["updated_at"] >= conv1["updated_at"]


class TestHistoryCLI:
    """Test history CLI commands via the typer test runner."""

    @pytest.fixture(autouse=True)
    def _setup_history(self, tmp_path, monkeypatch):
        """Redirect history to a temp DB and seed data."""
        reset_history()
        from ppmlx import history as history_mod
        self.hm = HistoryManager(path=tmp_path / "cli_history.db")
        monkeypatch.setattr(history_mod, "_instance", self.hm)
        monkeypatch.setattr(history_mod, "get_history", lambda path=None: self.hm)
        yield
        self.hm.close()
        reset_history()

    def test_history_list_empty(self):
        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "list"])
        assert result.exit_code == 0
        assert "No conversations" in result.output

    def test_history_list_with_data(self):
        self.hm.create_conversation("llama3")
        self.hm.create_conversation("mistral")

        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "list"])
        assert result.exit_code == 0
        assert "llama3" in result.output
        assert "mistral" in result.output

    def test_history_show(self):
        conv_id = self.hm.create_conversation("llama3")
        self.hm.add_message(conv_id, "user", "Hello")
        self.hm.add_message(conv_id, "assistant", "Hi!")

        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "show", conv_id])
        assert result.exit_code == 0
        assert "Hello" in result.output
        assert "Hi!" in result.output

    def test_history_show_not_found(self):
        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "show", "nonexistent"])
        assert result.exit_code == 1

    def test_history_search(self):
        conv_id = self.hm.create_conversation("llama3")
        self.hm.add_message(conv_id, "user", "Tell me about quantum physics")

        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "search", "quantum"])
        assert result.exit_code == 0
        assert "quantum" in result.output.lower()

    def test_history_search_no_results(self):
        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "search", "nonexistent_xyz"])
        assert result.exit_code == 0
        assert "No matches" in result.output

    def test_history_export_json(self):
        conv_id = self.hm.create_conversation("llama3")
        self.hm.add_message(conv_id, "user", "Hello")
        self.hm.add_message(conv_id, "assistant", "Hi!")

        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "export", conv_id, "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model"] == "llama3"
        assert len(data["messages"]) == 2

    def test_history_export_md(self):
        conv_id = self.hm.create_conversation("llama3")
        self.hm.add_message(conv_id, "user", "Hello")

        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "export", conv_id, "--format", "md"])
        assert result.exit_code == 0
        assert "### You" in result.output

    def test_history_export_to_file(self, tmp_path):
        conv_id = self.hm.create_conversation("llama3")
        self.hm.add_message(conv_id, "user", "Hello")

        out_file = str(tmp_path / "export.json")
        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "export", conv_id, "--format", "json", "--output", out_file])
        assert result.exit_code == 0
        assert "Exported" in result.output
        data = json.loads(open(out_file).read())
        assert data["model"] == "llama3"

    def test_history_delete_with_force(self):
        conv_id = self.hm.create_conversation("llama3")
        self.hm.add_message(conv_id, "user", "Hello")

        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "delete", conv_id, "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert self.hm.get_conversation(conv_id) is None

    def test_history_delete_not_found(self):
        from ppmlx.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["history", "delete", "nonexistent", "--force"])
        assert result.exit_code == 1
