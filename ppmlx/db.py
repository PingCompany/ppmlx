from __future__ import annotations
import json
import logging
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("ppmlx.db")

# Import get_ppmlx_dir lazily to avoid circular imports at module level
# but also provide a fallback for isolated testing
def _get_db_path() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        return get_ppmlx_dir() / "ppmlx.db"
    except ImportError:
        return Path.home() / ".ppmlx" / "ppmlx.db"


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

# Backpressure threshold: if the queue exceeds this size, non-error logs are
# dropped to avoid unbounded memory growth.
_BACKPRESSURE_THRESHOLD = 1000

# Maximum number of items to drain from the queue in a single batch write.
_BATCH_SIZE = 50


class Database:
    """Thread-safe SQLite database for ppmlx logging."""

    def __init__(self, path: Path | None = None):
        self._path = path or _get_db_path()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Metrics counters (thread-safe via the GIL for simple increments)
        self._write_errors: int = 0
        self._dropped_logs: int = 0
        self._enqueue_errors: int = 0

    def init(self) -> None:
        """Initialize the database schema and start background writer thread."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("Failed to init database: %s", e)

        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="ppmlx-db-writer"
            )
            self._thread.start()

    def _writer_loop(self) -> None:
        """Background thread: drain the queue in batches and write to SQLite."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            while not self._stop_event.is_set():
                # Block waiting for at least one item
                try:
                    first = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if first is None:  # sentinel
                    self._queue.task_done()
                    break

                # Drain up to _BATCH_SIZE items (including the first)
                batch: list[tuple[str, tuple]] = [first]
                while len(batch) < _BATCH_SIZE:
                    try:
                        item = self._queue.get_nowait()
                        if item is None:  # sentinel in batch
                            self._queue.task_done()
                            # Process what we have, then exit
                            self._write_batch(conn, batch)
                            return
                        batch.append(item)
                    except queue.Empty:
                        break

                self._write_batch(conn, batch)

        except Exception as e:
            self._write_errors += 1
            log.error("Writer thread fatal error: %s", e)
        finally:
            # Drain remaining items so any flush() calls don't block forever
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _write_batch(self, conn: sqlite3.Connection, batch: list[tuple[str, tuple]]) -> None:
        """Write a batch of (sql, params) items in a single transaction."""
        # Group items by SQL statement for executemany where possible
        groups: dict[str, list[tuple]] = {}
        for sql, params in batch:
            groups.setdefault(sql, []).append(params)

        try:
            for sql, params_list in groups.items():
                if len(params_list) == 1:
                    conn.execute(sql, params_list[0])
                else:
                    conn.executemany(sql, params_list)
            conn.commit()
        except Exception as e:
            self._write_errors += 1
            log.error("Batch write error (%d items): %s", len(batch), e)
        finally:
            for _ in batch:
                self._queue.task_done()

    def _enqueue(self, sql: str, params: tuple, *, is_error: bool = False) -> None:
        """Enqueue a write operation with backpressure.

        If the queue exceeds the backpressure threshold, non-error logs are
        dropped to prevent unbounded memory growth.
        """
        try:
            qsize = self._queue.qsize()
            if qsize >= _BACKPRESSURE_THRESHOLD and not is_error:
                self._dropped_logs += 1
                if self._dropped_logs % 100 == 1:
                    log.warning(
                        "DB queue backpressure: dropped %d non-error log(s) "
                        "(queue size %d)",
                        self._dropped_logs,
                        qsize,
                    )
                return
            self._queue.put_nowait((sql, params))
        except Exception:
            self._enqueue_errors += 1

    @property
    def queue_depth(self) -> int:
        """Current number of pending writes in the queue."""
        return self._queue.qsize()

    @property
    def write_errors(self) -> int:
        """Total number of write errors encountered by the writer thread."""
        return self._write_errors

    @property
    def dropped_logs(self) -> int:
        """Total number of logs dropped due to backpressure."""
        return self._dropped_logs

    @property
    def enqueue_errors(self) -> int:
        """Total number of enqueue failures."""
        return self._enqueue_errors

    def get_metrics(self) -> dict[str, int]:
        """Return db-layer metrics for the /metrics endpoint."""
        return {
            "db_queue_depth": self.queue_depth,
            "db_write_errors": self._write_errors,
            "db_dropped_logs": self._dropped_logs,
            "db_enqueue_errors": self._enqueue_errors,
        }

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
        self._enqueue(sql, params, is_error=(status == "error"))

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
        """Synchronous query -- safe to call from CLI."""
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
            log.error("Query error: %s", e)
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
            log.error("Stats error: %s", e)
            return {"total_requests": 0, "avg_duration_ms": None, "by_model": []}

    def flush(self) -> None:
        """Wait for the write queue to drain."""
        try:
            # If the writer thread has already exited, drain the queue ourselves
            # to prevent queue.join() from blocking indefinitely.
            if self._thread is not None and not self._thread.is_alive():
                while True:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        break
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
