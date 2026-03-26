"""Tests for ppmlx.arena — Model Comparison Arena."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock ppmlx modules before importing arena
for mod_name in ["ppmlx.config", "ppmlx.models", "ppmlx.engine",
                  "ppmlx.db", "ppmlx.memory", "ppmlx.engine_embed",
                  "ppmlx.engine_vlm", "ppmlx.registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.arena import (
    ArenaDB,
    ArenaMatch,
    ArenaResult,
    DEFAULT_ELO,
    K_FACTOR,
    ARENA_HTML,
    compute_elo_update,
)


# ── ELO calculation tests ────────────────────────────────────────────


def test_elo_update_a_wins():
    """When A wins against an equal-rated opponent, A gains and B loses equally."""
    new_a, new_b = compute_elo_update(1500, 1500, "a")
    assert new_a > 1500
    assert new_b < 1500
    # With equal ratings, expected is 0.5, so change = K * 0.5 = 16
    assert new_a == 1516.0
    assert new_b == 1484.0


def test_elo_update_b_wins():
    """When B wins, B gains and A loses."""
    new_a, new_b = compute_elo_update(1500, 1500, "b")
    assert new_a < 1500
    assert new_b > 1500
    assert new_a == 1484.0
    assert new_b == 1516.0


def test_elo_update_draw():
    """Draw between equal opponents results in no change."""
    new_a, new_b = compute_elo_update(1500, 1500, "draw")
    assert new_a == 1500.0
    assert new_b == 1500.0


def test_elo_update_upset():
    """Lower-rated player beating higher-rated gains more."""
    # B (1400) beats A (1600) — upset
    new_a, new_b = compute_elo_update(1600, 1400, "b")
    change_a = 1600 - new_a
    # A loses more than 16 because A was expected to win
    assert change_a > 16


def test_elo_update_expected_win():
    """Higher-rated player winning gains less than K/2."""
    # A (1600) beats B (1400) — expected
    new_a, new_b = compute_elo_update(1600, 1400, "a")
    change_a = new_a - 1600
    # A gains less than 16 because A was expected to win
    assert 0 < change_a < 16


def test_elo_update_symmetry():
    """Total ELO is conserved (zero-sum)."""
    for outcome in ("a", "b", "draw"):
        new_a, new_b = compute_elo_update(1500, 1500, outcome)
        assert abs((new_a + new_b) - 3000.0) < 0.01


# ── ArenaDB tests ────────────────────────────────────────────────────


@pytest.fixture
def arena_db(tmp_path):
    """Create an ArenaDB with a temporary database."""
    db_path = tmp_path / "test_arena.db"
    return ArenaDB(path=db_path)


@pytest.fixture
def sample_match():
    """Create a sample ArenaMatch for testing."""
    return ArenaMatch(
        match_id="test-match-001",
        prompt="What is 2+2?",
        result_a=ArenaResult(
            model="model-alpha",
            response="2+2 is 4",
            elapsed_ms=150.0,
        ),
        result_b=ArenaResult(
            model="model-beta",
            response="The answer is 4.",
            elapsed_ms=200.0,
        ),
    )


def test_db_default_elo(arena_db):
    """New models should have the default ELO score."""
    assert arena_db.get_elo("unknown-model") == DEFAULT_ELO


def test_db_record_match_a_wins(arena_db, sample_match):
    """Recording a match where A wins updates ELO correctly."""
    new_a, new_b = arena_db.record_match(sample_match, "a")
    assert new_a > DEFAULT_ELO
    assert new_b < DEFAULT_ELO
    # Verify persisted
    assert arena_db.get_elo("model-alpha") == new_a
    assert arena_db.get_elo("model-beta") == new_b


def test_db_record_match_b_wins(arena_db, sample_match):
    """Recording a match where B wins updates ELO correctly."""
    new_a, new_b = arena_db.record_match(sample_match, "b")
    assert new_a < DEFAULT_ELO
    assert new_b > DEFAULT_ELO


def test_db_record_match_draw(arena_db, sample_match):
    """Recording a draw between equal-rated models leaves ELO unchanged."""
    new_a, new_b = arena_db.record_match(sample_match, "draw")
    assert new_a == DEFAULT_ELO
    assert new_b == DEFAULT_ELO


def test_db_win_loss_counters(arena_db, sample_match):
    """Win/loss/draw counters increment correctly."""
    arena_db.record_match(sample_match, "a")
    entries = arena_db.get_all_elos()
    by_model = {e.model: e for e in entries}

    assert by_model["model-alpha"].wins == 1
    assert by_model["model-alpha"].losses == 0
    assert by_model["model-beta"].wins == 0
    assert by_model["model-beta"].losses == 1


def test_db_draw_counters(arena_db, sample_match):
    """Draw counter increments for both models."""
    arena_db.record_match(sample_match, "draw")
    entries = arena_db.get_all_elos()
    by_model = {e.model: e for e in entries}

    assert by_model["model-alpha"].draws == 1
    assert by_model["model-beta"].draws == 1


def test_db_multiple_matches(arena_db):
    """Multiple matches accumulate correctly."""
    match1 = ArenaMatch(
        match_id="m1", prompt="test",
        result_a=ArenaResult(model="A", response="r1", elapsed_ms=100),
        result_b=ArenaResult(model="B", response="r2", elapsed_ms=100),
    )
    match2 = ArenaMatch(
        match_id="m2", prompt="test2",
        result_a=ArenaResult(model="A", response="r3", elapsed_ms=100),
        result_b=ArenaResult(model="B", response="r4", elapsed_ms=100),
    )
    arena_db.record_match(match1, "a")
    arena_db.record_match(match2, "b")

    entries = arena_db.get_all_elos()
    by_model = {e.model: e for e in entries}
    assert by_model["A"].wins == 1
    assert by_model["A"].losses == 1
    assert by_model["B"].wins == 1
    assert by_model["B"].losses == 1


def test_db_get_all_elos_sorted(arena_db):
    """Leaderboard is sorted by score descending."""
    match = ArenaMatch(
        match_id="m1", prompt="test",
        result_a=ArenaResult(model="winner", response="r1", elapsed_ms=100),
        result_b=ArenaResult(model="loser", response="r2", elapsed_ms=100),
    )
    arena_db.record_match(match, "a")

    entries = arena_db.get_all_elos()
    assert len(entries) == 2
    assert entries[0].model == "winner"
    assert entries[0].score > entries[1].score


def test_db_get_recent_matches(arena_db, sample_match):
    """Recent matches are returned in reverse chronological order."""
    arena_db.record_match(sample_match, "a")
    matches = arena_db.get_recent_matches(limit=5)
    assert len(matches) == 1
    assert matches[0]["match_id"] == "test-match-001"
    assert matches[0]["winner"] == "a"
    assert matches[0]["prompt"] == "What is 2+2?"


def test_db_match_stores_responses(arena_db, sample_match):
    """Match record stores the full responses and timings."""
    arena_db.record_match(sample_match, "a")
    matches = arena_db.get_recent_matches()
    m = matches[0]
    assert m["response_a"] == "2+2 is 4"
    assert m["response_b"] == "The answer is 4."
    assert m["time_a_ms"] == 150.0
    assert m["time_b_ms"] == 200.0


def test_db_match_stores_elo_history(arena_db, sample_match):
    """Match record stores before/after ELO scores."""
    arena_db.record_match(sample_match, "a")
    matches = arena_db.get_recent_matches()
    m = matches[0]
    assert m["elo_a_before"] == DEFAULT_ELO
    assert m["elo_b_before"] == DEFAULT_ELO
    assert m["elo_a_after"] > DEFAULT_ELO
    assert m["elo_b_after"] < DEFAULT_ELO


# ── ArenaResult tests ────────────────────────────────────────────────


def test_arena_result_with_error():
    """ArenaResult can capture errors."""
    result = ArenaResult(
        model="test", response="", elapsed_ms=50.0,
        error="Connection refused",
    )
    assert result.error == "Connection refused"
    assert result.response == ""


# ── HTML template tests ──────────────────────────────────────────────


def test_arena_html_contains_key_elements():
    """The Arena HTML template contains essential UI elements."""
    assert "ppmlx Arena" in ARENA_HTML
    assert "model-a" in ARENA_HTML
    assert "model-b" in ARENA_HTML
    assert "/arena/compare" in ARENA_HTML
    assert "/arena/vote" in ARENA_HTML
    assert "/arena/leaderboard" in ARENA_HTML
    assert "Leaderboard" in ARENA_HTML


# ── Server route tests ───────────────────────────────────────────────


# Set up mock engine for server tests
mock_engine = MagicMock()
mock_engine.generate.return_value = ("Hello!", None, 10, 5)
mock_engine.stream_generate.return_value = iter(["Hello"])
mock_engine.list_loaded.return_value = []
sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

mock_embed_engine = MagicMock()
mock_embed_engine.encode.return_value = [[0.1, 0.2, 0.3]]
sys.modules["ppmlx.engine_embed"].get_embed_engine = MagicMock(return_value=mock_embed_engine)

sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=lambda x: x)
sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=[])
sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value={})
sys.modules["ppmlx.models"].is_vision_model = MagicMock(return_value=False)
sys.modules["ppmlx.models"].is_embed_model = MagicMock(return_value=False)

mock_db = MagicMock()
mock_db.get_stats.return_value = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}
sys.modules["ppmlx.db"].get_db = MagicMock(return_value=mock_db)

sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

mock_config = MagicMock()
mock_config.logging.snapshot_interval_seconds = 60
sys.modules["ppmlx.config"].load_config = MagicMock(return_value=mock_config)

from fastapi.testclient import TestClient
from ppmlx.server import app as server_app


@pytest.fixture
def client():
    with TestClient(server_app) as c:
        yield c


def test_arena_page_returns_html(client):
    """GET /arena returns the Arena HTML page."""
    response = client.get("/arena")
    assert response.status_code == 200
    assert "ppmlx Arena" in response.text


def test_arena_leaderboard_empty(client):
    """GET /arena/leaderboard returns empty list when no matches."""
    mock_db = MagicMock()
    mock_db.get_all_elos.return_value = []
    with patch("ppmlx.server._get_arena_db", return_value=mock_db):
        response = client.get("/arena/leaderboard")
        assert response.status_code == 200
        assert response.json() == []


def test_arena_compare_missing_fields(client):
    """POST /arena/compare with missing fields returns 400."""
    response = client.post("/arena/compare", json={"model_a": "a"})
    assert response.status_code == 400


def test_arena_vote_invalid_winner(client):
    """POST /arena/vote with invalid winner returns 400."""
    response = client.post("/arena/vote", json={
        "match_id": "test", "winner": "invalid",
    })
    assert response.status_code == 400


def test_arena_vote_match_not_found(client):
    """POST /arena/vote with unknown match_id returns 404."""
    response = client.post("/arena/vote", json={
        "match_id": "nonexistent", "winner": "a",
    })
    assert response.status_code == 404


def test_arena_matches_endpoint(client):
    """GET /arena/matches returns recent matches."""
    mock_db = MagicMock()
    mock_db.get_recent_matches.return_value = []
    with patch("ppmlx.server._get_arena_db", return_value=mock_db):
        response = client.get("/arena/matches")
        assert response.status_code == 200
        assert response.json() == []


# ── CLI arena command tests ──────────────────────────────────────────

from typer.testing import CliRunner
from ppmlx.cli import app as cli_app

runner = CliRunner()


def test_arena_command_exists():
    """The arena command is registered in the CLI."""
    result = runner.invoke(cli_app, ["arena", "--help"])
    assert result.exit_code == 0
    assert "arena" in result.output.lower()
    assert "model_a" in result.output.lower() or "MODEL_A" in result.output


def test_arena_command_requires_models():
    """The arena command requires two model arguments."""
    result = runner.invoke(cli_app, ["arena"])
    assert result.exit_code != 0
