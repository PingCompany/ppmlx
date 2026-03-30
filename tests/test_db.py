"""Tests for ppmlx.db — SQLite logging layer."""
from __future__ import annotations
import sqlite3
from pathlib import Path

import pytest

from ppmlx.db import Database, get_db, reset_db


def make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "ppmlx.db")
    db.init()
    return db


def test_init_creates_tables(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.flush()
    conn = sqlite3.connect(str(tmp_path / "ppmlx.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    db.close()
    assert "requests" in tables
    assert "model_events" in tables
    assert "system_snapshots" in tables


def test_log_request_and_query(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request("req-1", "/v1/chat", "qwen", "org/qwen-repo", status="ok", total_duration_ms=123.4)
    db.flush()
    rows = db.query_requests()
    db.close()
    assert len(rows) == 1
    assert rows[0]["request_id"] == "req-1"
    assert rows[0]["endpoint"] == "/v1/chat"
    assert rows[0]["model_alias"] == "qwen"
    assert rows[0]["total_duration_ms"] == pytest.approx(123.4)


def test_query_filter_by_model(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request("req-a", "/v1/chat", "model-a", "repo-a")
    db.log_request("req-b", "/v1/chat", "model-b", "repo-b")
    db.flush()
    rows = db.query_requests(model="model-a")
    db.close()
    assert len(rows) == 1
    assert rows[0]["model_alias"] == "model-a"


def test_query_errors_only(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request("req-ok", "/v1/chat", "qwen", "repo", status="ok")
    db.log_request("req-err", "/v1/chat", "qwen", "repo", status="error", error_message="oops")
    db.flush()
    rows = db.query_requests(errors_only=True)
    db.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["request_id"] == "req-err"


def test_log_model_event(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_model_event("load", "org/model", model_alias="qwen", duration_ms=500.0, details={"extra": "info"})
    db.flush()
    conn = sqlite3.connect(str(tmp_path / "ppmlx.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM model_events").fetchall()
    conn.close()
    db.close()
    assert len(rows) == 1
    assert rows[0]["event"] == "load"
    assert rows[0]["model_repo"] == "org/model"


def test_log_system_snapshot(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_system_snapshot(
        memory_total_gb=16.0,
        memory_used_gb=8.5,
        loaded_models=["qwen", "llama"],
        uptime_seconds=3600,
    )
    db.flush()
    conn = sqlite3.connect(str(tmp_path / "ppmlx.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM system_snapshots").fetchall()
    conn.close()
    db.close()
    assert len(rows) == 1
    assert rows[0]["memory_total_gb"] == 16.0
    assert rows[0]["memory_used_gb"] == 8.5


def test_get_stats_empty(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.flush()
    stats = db.get_stats()
    db.close()
    assert stats["total_requests"] == 0
    assert stats["by_model"] == []


def test_get_stats_counts(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request("r1", "/v1/chat", "alpha", "repo-a", tokens_per_second=50.0)
    db.log_request("r2", "/v1/chat", "alpha", "repo-a", tokens_per_second=60.0)
    db.log_request("r3", "/v1/chat", "beta", "repo-b", tokens_per_second=30.0)
    db.flush()
    stats = db.get_stats(since_hours=1)
    db.close()
    assert stats["total_requests"] == 3
    by_model = {m["model"]: m for m in stats["by_model"]}
    assert by_model["alpha"]["count"] == 2
    assert by_model["beta"]["count"] == 1


def test_never_raises_on_bad_path():
    db = Database(Path("/nonexistent/path/x.db"))
    db.init()
    db.log_request("r1", "/v1/chat", "qwen", "repo")
    db.flush()
    db.close()


def test_singleton_reset(tmp_home, tmp_path):
    reset_db()
    db1 = get_db(tmp_path / "ppmlx.db")
    db2_before_reset = get_db()
    assert db1 is db2_before_reset  # same instance

    reset_db()
    db2 = get_db(tmp_path / "ppmlx2.db")
    assert db1 is not db2  # new instance after reset
    db2.close()


def test_query_limit(tmp_home, tmp_path):
    db = make_db(tmp_path)
    for i in range(5):
        db.log_request(f"req-{i}", "/v1/chat", "qwen", "repo")
    db.flush()
    rows = db.query_requests(limit=2)
    db.close()
    assert len(rows) == 2


def test_migration_adds_columns(tmp_home, tmp_path):
    """Create a DB with old schema (no thinking columns), re-init, verify columns added."""
    db_path = tmp_path / "ppmlx.db"
    # Create DB without thinking columns using a trimmed schema
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            request_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            model_alias TEXT NOT NULL,
            model_repo TEXT NOT NULL,
            stream INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ok',
            error_message TEXT,
            prompt_tokens INTEGER,
            messages_count INTEGER,
            system_prompt TEXT,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            time_to_first_token_ms REAL,
            total_duration_ms REAL,
            tokens_per_second REAL,
            temperature REAL,
            top_p REAL,
            max_tokens INTEGER,
            repetition_penalty REAL,
            client_ip TEXT,
            user_agent TEXT
        );
        CREATE TABLE IF NOT EXISTS model_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            event TEXT NOT NULL,
            model_repo TEXT NOT NULL,
            model_alias TEXT,
            duration_ms REAL,
            details TEXT
        );
        CREATE TABLE IF NOT EXISTS system_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            memory_total_gb REAL,
            memory_used_gb REAL,
            loaded_models TEXT,
            uptime_seconds INTEGER
        );
    """)
    conn.commit()
    # Verify thinking columns are absent
    existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
    assert "reasoning_tokens" not in existing
    conn.close()

    # Now init via Database — migration should add the columns
    db = Database(db_path)
    db.init()
    db.flush()

    conn = sqlite3.connect(str(db_path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
    conn.close()
    db.close()

    for col in ("reasoning_tokens", "thinking_duration_ms", "answer_duration_ms",
                "thinking_enabled", "reasoning_budget"):
        assert col in cols, f"Column {col} missing after migration"


def test_log_request_with_thinking_fields(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request(
        "req-think", "/v1/chat", "qwen", "repo",
        reasoning_tokens=500,
        thinking_duration_ms=3000.0,
        answer_duration_ms=1200.0,
        thinking_enabled=True,
        reasoning_budget=1024,
    )
    db.flush()
    rows = db.query_requests()
    db.close()
    assert len(rows) == 1
    r = rows[0]
    assert r["reasoning_tokens"] == 500
    assert r["thinking_duration_ms"] == pytest.approx(3000.0)
    assert r["answer_duration_ms"] == pytest.approx(1200.0)
    assert r["thinking_enabled"] == 1
    assert r["reasoning_budget"] == 1024


def test_log_request_without_thinking_fields(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request("req-plain", "/v1/chat", "qwen", "repo")
    db.flush()
    rows = db.query_requests()
    db.close()
    assert len(rows) == 1
    r = rows[0]
    assert r["reasoning_tokens"] is None
    assert r["thinking_duration_ms"] is None
    assert r["answer_duration_ms"] is None
    assert r["thinking_enabled"] is None
    assert r["reasoning_budget"] is None


def test_query_thinking_stats_empty(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.flush()
    stats = db.query_thinking_stats()
    db.close()
    assert stats["total_thinking_requests"] == 0
    assert stats["avg_reasoning_tokens"] is None
    assert stats["avg_thinking_duration_ms"] is None
    assert stats["avg_answer_duration_ms"] is None
    assert stats["thinking_percentage"] == 0.0
    assert stats["by_model"] == []


def test_query_thinking_stats_with_data(tmp_home, tmp_path):
    db = make_db(tmp_path)
    # Two thinking requests for model-a
    db.log_request("r1", "/v1/chat", "model-a", "repo-a",
                   reasoning_tokens=400, thinking_duration_ms=2000.0,
                   answer_duration_ms=1000.0, thinking_enabled=True, reasoning_budget=800)
    db.log_request("r2", "/v1/chat", "model-a", "repo-a",
                   reasoning_tokens=600, thinking_duration_ms=4000.0,
                   answer_duration_ms=2000.0, thinking_enabled=True, reasoning_budget=800)
    # One thinking request for model-b
    db.log_request("r3", "/v1/chat", "model-b", "repo-b",
                   reasoning_tokens=100, thinking_duration_ms=500.0,
                   answer_duration_ms=300.0, thinking_enabled=True)
    # One non-thinking request
    db.log_request("r4", "/v1/chat", "model-a", "repo-a")
    db.flush()

    stats = db.query_thinking_stats(since_hours=1)
    db.close()

    assert stats["total_thinking_requests"] == 3
    assert stats["avg_reasoning_tokens"] == pytest.approx((400 + 600 + 100) / 3)
    assert stats["avg_thinking_duration_ms"] == pytest.approx((2000 + 4000 + 500) / 3)
    assert stats["avg_answer_duration_ms"] == pytest.approx((1000 + 2000 + 300) / 3)
    assert stats["thinking_percentage"] == pytest.approx(75.0)  # 3 out of 4

    by_model = {m["model"]: m for m in stats["by_model"]}
    assert by_model["model-a"]["count"] == 2
    assert by_model["model-a"]["avg_reasoning_tokens"] == pytest.approx(500.0)
    assert by_model["model-b"]["count"] == 1


def test_get_stats_includes_thinking(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.log_request("r1", "/v1/chat", "qwen", "repo",
                   reasoning_tokens=200, thinking_enabled=True)
    db.log_request("r2", "/v1/chat", "qwen", "repo")
    db.flush()
    stats = db.get_stats(since_hours=1)
    db.close()
    assert "thinking" in stats
    assert stats["thinking"]["thinking_request_count"] == 1
    assert stats["thinking"]["avg_reasoning_tokens"] == pytest.approx(200.0)
