from __future__ import annotations
import json
import queue
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

# Import get_pp_llm_dir lazily to avoid circular imports at module level
# but also provide a fallback for isolated testing
def _get_db_path() -> Path:
    try:
        from pp_llm.config import get_pp_llm_dir
        return get_pp_llm_dir() / "pp-llm.db"
    except ImportError:
        return Path.home() / ".pp-llm" / "pp-llm.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    request_id              TEXT NOT NULL,
    endpoint                TEXT NOT NULL,
    model_alias             TEXT NOT NULL,
    model_repo              TEXT NOT NULL,
    stream                  INTEGER NOT NULL DEFAULT 0,
    status                  TEXT NOT NULL DEFAULT 'ok',
    error_message           TEXT,
    prompt_tokens           INTEGER,
    messages_count          INTEGER,
    system_prompt           TEXT,
    completion_tokens       INTEGER,
    total_tokens            INTEGER,
    time_to_first_token_ms  REAL,
    total_duration_ms       REAL,
    tokens_per_second       REAL,
    temperature             REAL,
    top_p                   REAL,
    max_tokens              INTEGER,
    repetition_penalty      REAL,
    client_ip               TEXT,
    user_agent              TEXT
);

CREATE TABLE IF NOT EXISTS model_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    event       TEXT NOT NULL,
    model_repo  TEXT NOT NULL,
    model_alias TEXT,
    duration_ms REAL,
    details     TEXT
);

CREATE TABLE IF NOT EXISTS system_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    memory_total_gb REAL,
    memory_used_gb  REAL,
    loaded_models   TEXT,
    uptime_seconds  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model_alias);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_model_events_timestamp ON model_events(timestamp);
"""


class Database:
    """Thread-safe SQLite database for pp-llm logging."""

    def __init__(self, path: Path | None = None):
        self._path = path or _get_db_path()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def init(self) -> None:
        """Initialize the database schema and start background writer thread."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path))
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[pp-llm db] Warning: failed to init database: {e}", file=sys.stderr)

        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="pp-llm-db-writer"
            )
            self._thread.start()

    def _writer_loop(self) -> None:
        """Background thread: drain the queue and write to SQLite."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.5)
                    if item is None:  # sentinel
                        break
                    sql, params = item
                    try:
                        conn.execute(sql, params)
                        conn.commit()
                    except Exception as e:
                        print(f"[pp-llm db] Write error: {e}", file=sys.stderr)
                    finally:
                        self._queue.task_done()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"[pp-llm db] Writer thread error: {e}", file=sys.stderr)
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _enqueue(self, sql: str, params: tuple) -> None:
        """Enqueue a write operation; never blocks, never raises."""
        try:
            self._queue.put_nowait((sql, params))
        except Exception:
            pass

    def log_request(
        self,
        request_id: str,
        endpoint: str,
        model_alias: str,
        model_repo: str,
        stream: bool = False,
        status: str = "ok",
        error_message: str | None = None,
        prompt_tokens: int | None = None,
        messages_count: int | None = None,
        system_prompt: str | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        time_to_first_token_ms: float | None = None,
        total_duration_ms: float | None = None,
        tokens_per_second: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        repetition_penalty: float | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        sql = """INSERT INTO requests (
            request_id, endpoint, model_alias, model_repo, stream, status, error_message,
            prompt_tokens, messages_count, system_prompt, completion_tokens, total_tokens,
            time_to_first_token_ms, total_duration_ms, tokens_per_second,
            temperature, top_p, max_tokens, repetition_penalty, client_ip, user_agent
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
        params = (
            request_id, endpoint, model_alias, model_repo, int(stream), status, error_message,
            prompt_tokens, messages_count, system_prompt, completion_tokens, total_tokens,
            time_to_first_token_ms, total_duration_ms, tokens_per_second,
            temperature, top_p, max_tokens, repetition_penalty, client_ip, user_agent,
        )
        self._enqueue(sql, params)

    def log_model_event(
        self,
        event: str,
        model_repo: str,
        model_alias: str | None = None,
        duration_ms: float | None = None,
        details: dict | None = None,
    ) -> None:
        sql = """INSERT INTO model_events (event, model_repo, model_alias, duration_ms, details)
                 VALUES (?,?,?,?,?)"""
        self._enqueue(sql, (event, model_repo, model_alias, duration_ms,
                            json.dumps(details) if details else None))

    def log_system_snapshot(
        self,
        memory_total_gb: float,
        memory_used_gb: float,
        loaded_models: list[str],
        uptime_seconds: int,
    ) -> None:
        sql = """INSERT INTO system_snapshots (memory_total_gb, memory_used_gb, loaded_models, uptime_seconds)
                 VALUES (?,?,?,?)"""
        self._enqueue(sql, (memory_total_gb, memory_used_gb, json.dumps(loaded_models), uptime_seconds))

    def query_requests(
        self,
        limit: int = 20,
        model: str | None = None,
        since_hours: float | None = None,
        errors_only: bool = False,
        min_duration_ms: float | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous query — safe to call from CLI."""
        try:
            conditions: list[str] = []
            params: list[Any] = []
            if model:
                conditions.append("model_alias = ?")
                params.append(model)
            if since_hours:
                conditions.append("timestamp >= datetime('now', ?)")
                params.append(f"-{since_hours} hours")
            if errors_only:
                conditions.append("status = 'error'")
            if min_duration_ms is not None:
                conditions.append("total_duration_ms >= ?")
                params.append(min_duration_ms)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            params.append(limit)
            with sqlite3.connect(self._path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM requests{where} ORDER BY timestamp DESC LIMIT ?", params
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[pp-llm db] Query error: {e}", file=sys.stderr)
            return []

    def get_stats(self, since_hours: float = 24) -> dict[str, Any]:
        """Return aggregate statistics for the past N hours."""
        try:
            since_param = f"-{since_hours} hours"
            with sqlite3.connect(self._path) as conn:
                by_model = conn.execute(
                    """SELECT model_alias,
                              COUNT(*) as count,
                              AVG(tokens_per_second) as avg_tps,
                              AVG(time_to_first_token_ms) as avg_ttft,
                              SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors
                       FROM requests
                       WHERE timestamp >= datetime('now', ?)
                       GROUP BY model_alias
                       ORDER BY count DESC""",
                    (since_param,)
                ).fetchall()
                avg_duration = conn.execute(
                    "SELECT AVG(total_duration_ms) FROM requests WHERE timestamp >= datetime('now', ?)",
                    (since_param,)
                ).fetchone()[0]
            model_rows = [
                {"model": r[0], "count": r[1], "avg_tps": r[2], "avg_ttft": r[3], "errors": r[4]}
                for r in by_model
            ]
            return {
                "total_requests": sum(m["count"] for m in model_rows),
                "avg_duration_ms": avg_duration,
                "by_model": model_rows,
            }
        except Exception as e:
            print(f"[pp-llm db] Stats error: {e}", file=sys.stderr)
            return {"total_requests": 0, "avg_duration_ms": None, "by_model": []}

    def flush(self) -> None:
        """Wait for the write queue to drain."""
        try:
            self._queue.join()
        except Exception:
            pass

    def close(self) -> None:
        """Flush and stop the background writer."""
        self.flush()
        self._stop_event.set()
        self._queue.put_nowait(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)


_db_instance: Database | None = None
_db_lock = threading.Lock()


def get_db(path: Path | None = None) -> Database:
    """Return the singleton Database, initializing on first call."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = Database(path)
                _db_instance.init()
    return _db_instance


def reset_db() -> None:
    """Reset singleton (for testing)."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
    _db_instance = None
