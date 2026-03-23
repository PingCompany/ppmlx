"""Tests for pp_llm.db — SQLite logging layer."""
from __future__ import annotations
import sqlite3
from pathlib import Path

import pytest

from pp_llm.db import Database, get_db, reset_db


def make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "pp-llm.db")
    db.init()
    return db


def test_init_creates_tables(tmp_home, tmp_path):
    db = make_db(tmp_path)
    db.flush()
    conn = sqlite3.connect(str(tmp_path / "pp-llm.db"))
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
    conn = sqlite3.connect(str(tmp_path / "pp-llm.db"))
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
    conn = sqlite3.connect(str(tmp_path / "pp-llm.db"))
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
    db1 = get_db(tmp_path / "pp-llm.db")
    db2_before_reset = get_db()
    assert db1 is db2_before_reset  # same instance

    reset_db()
    db2 = get_db(tmp_path / "pp-llm2.db")
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
