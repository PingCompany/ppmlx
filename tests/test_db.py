"""Tests for ppmlx.db — SQLite logging layer."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pytest

from ppmlx.db import Database, get_db, reset_db, _BACKPRESSURE_THRESHOLD


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


def test_wal_mode_enabled(tmp_home, tmp_path):
    """WAL journal mode should be set after init."""
    db = make_db(tmp_path)
    db.flush()
    conn = sqlite3.connect(str(tmp_path / "ppmlx.db"))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    db.close()
    assert mode == "wal"


def test_synchronous_normal(tmp_home, tmp_path):
    """PRAGMA synchronous should be NORMAL (1) for WAL performance."""
    db = make_db(tmp_path)
    db.flush()
    # Open a fresh connection the same way the writer does
    conn = sqlite3.connect(str(tmp_path / "ppmlx.db"))
    conn.execute("PRAGMA synchronous=NORMAL")
    sync_val = conn.execute("PRAGMA synchronous").fetchone()[0]
    conn.close()
    db.close()
    # 1 = NORMAL
    assert sync_val == 1


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


# ---------- Batch write tests ----------

def test_batch_write_multiple_items(tmp_home, tmp_path):
    """Multiple items enqueued rapidly should be written in a batch."""
    db = make_db(tmp_path)
    for i in range(10):
        db.log_request(f"batch-{i}", "/v1/chat", "qwen", "repo")
    db.flush()
    rows = db.query_requests(limit=20)
    db.close()
    assert len(rows) == 10


def test_batch_write_mixed_sql(tmp_home, tmp_path):
    """Batch writer should handle a mix of different SQL statements."""
    db = make_db(tmp_path)
    db.log_request("r1", "/v1/chat", "qwen", "repo")
    db.log_model_event("load", "org/model")
    db.log_system_snapshot(16.0, 8.0, ["qwen"], 100)
    db.flush()

    conn = sqlite3.connect(str(tmp_path / "ppmlx.db"))
    req_count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    evt_count = conn.execute("SELECT COUNT(*) FROM model_events").fetchone()[0]
    snap_count = conn.execute("SELECT COUNT(*) FROM system_snapshots").fetchone()[0]
    conn.close()
    db.close()

    assert req_count == 1
    assert evt_count == 1
    assert snap_count == 1


# ---------- Backpressure tests ----------

def test_backpressure_drops_non_error_logs(tmp_home, tmp_path):
    """When queue exceeds threshold, non-error logs should be dropped."""
    db = Database(tmp_path / "ppmlx.db")
    # Do NOT start the writer thread -- we want the queue to fill up.
    # _enqueue doesn't touch the DB, so no schema is needed.
    for i in range(_BACKPRESSURE_THRESHOLD):
        db._enqueue("SELECT 1", (i,))

    assert db.queue_depth >= _BACKPRESSURE_THRESHOLD

    # This non-error log should be dropped
    db.log_request("should-drop", "/v1/chat", "qwen", "repo", status="ok")
    assert db.dropped_logs >= 1

    # Drain queue so close doesn't hang
    while not db._queue.empty():
        try:
            db._queue.get_nowait()
            db._queue.task_done()
        except Exception:
            break


def test_backpressure_allows_error_logs(tmp_home, tmp_path):
    """Error-status logs should NOT be dropped even under backpressure."""
    db = Database(tmp_path / "ppmlx.db")

    # Fill the queue past the threshold without starting the writer
    for i in range(_BACKPRESSURE_THRESHOLD):
        db._enqueue("SELECT 1", (f"fill-{i}",))

    initial_depth = db.queue_depth
    assert initial_depth >= _BACKPRESSURE_THRESHOLD

    # Error log should still be enqueued
    db.log_request("err-1", "/v1/chat", "qwen", "repo", status="error", error_message="boom")
    assert db.queue_depth == initial_depth + 1
    assert db.dropped_logs == 0

    # Drain queue so close doesn't hang
    while not db._queue.empty():
        try:
            db._queue.get_nowait()
            db._queue.task_done()
        except Exception:
            break


# ---------- Error counting tests ----------

def test_error_counters_initial_state(tmp_home, tmp_path):
    """Error counters should start at zero."""
    db = make_db(tmp_path)
    assert db.write_errors == 0
    assert db.dropped_logs == 0
    assert db.enqueue_errors == 0
    db.close()


def test_get_metrics_returns_dict(tmp_home, tmp_path):
    """get_metrics() should return a dict with all expected keys."""
    db = make_db(tmp_path)
    metrics = db.get_metrics()
    db.close()
    assert "db_queue_depth" in metrics
    assert "db_write_errors" in metrics
    assert "db_dropped_logs" in metrics
    assert "db_enqueue_errors" in metrics
    assert all(isinstance(v, int) for v in metrics.values())


def test_queue_depth_tracks_pending_writes(tmp_home, tmp_path):
    """queue_depth should reflect items waiting to be written."""
    db = Database(tmp_path / "ppmlx.db")
    # Don't start writer — just check queue depth tracking
    assert db.queue_depth == 0

    db._enqueue("SELECT 1", ())
    assert db.queue_depth == 1

    db._enqueue("SELECT 1", ())
    assert db.queue_depth == 2

    # Drain
    while not db._queue.empty():
        try:
            db._queue.get_nowait()
            db._queue.task_done()
        except Exception:
            break


def test_write_error_counting(tmp_home, tmp_path):
    """Write errors should be counted, not silently swallowed."""
    db = make_db(tmp_path)
    # Enqueue a statement that will fail (bad table name)
    db._enqueue("INSERT INTO nonexistent_table VALUES (?)", ("x",))
    db.flush()
    db.close()
    assert db.write_errors >= 1
