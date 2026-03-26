"""Tests for ppmlx.dashboard — real-time monitoring dashboard."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

# Mock all ppmlx modules that server.py tries to import lazily
for mod in [
    "ppmlx.engine", "ppmlx.engine_vlm", "ppmlx.engine_embed",
    "ppmlx.models", "ppmlx.db", "ppmlx.config", "ppmlx.memory",
    "ppmlx.schema",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Set up mock engine
mock_engine = MagicMock()
mock_engine.generate.return_value = ("Hello!", None, 10, 5)
mock_engine.stream_generate.return_value = iter(["Hello", " ", "world"])
mock_engine.list_loaded.return_value = ["mlx-community/Llama-3-8B"]
sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

# Set up mock db
mock_db = MagicMock()
mock_db.get_stats.return_value = {
    "total_requests": 42,
    "avg_duration_ms": 1234.5,
    "by_model": [
        {"model": "llama3", "count": 30, "avg_tps": 45.2, "avg_ttft": 120.0, "errors": 1},
        {"model": "qwen3", "count": 12, "avg_tps": 38.1, "avg_ttft": 150.0, "errors": 0},
    ],
}
mock_db.query_requests.return_value = [
    {
        "id": 1,
        "timestamp": "2026-03-26T10:00:00.000",
        "request_id": "chatcmpl-abc123",
        "endpoint": "/v1/chat/completions",
        "model_alias": "llama3",
        "model_repo": "mlx-community/Llama-3-8B",
        "stream": 0,
        "status": "ok",
        "error_message": None,
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "total_tokens": 150,
        "time_to_first_token_ms": 120.0,
        "total_duration_ms": 2500.0,
        "tokens_per_second": 40.0,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 2048,
        "repetition_penalty": None,
        "client_ip": "127.0.0.1",
        "user_agent": "test",
        "messages_count": 1,
        "system_prompt": None,
    }
]
mock_db._path = "/tmp/test_ppmlx.db"
sys.modules["ppmlx.db"].get_db = MagicMock(return_value=mock_db)

# Set up mock memory
sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=36.0)

# Set up mock config
mock_config = MagicMock()
mock_config.logging.snapshot_interval_seconds = 60
sys.modules["ppmlx.config"].load_config = MagicMock(return_value=mock_config)

# Set up mock models
sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=lambda x: x)
sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=[])
sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value=[])
sys.modules["ppmlx.models"].is_vision_model = MagicMock(return_value=False)
sys.modules["ppmlx.models"].is_embed_model = MagicMock(return_value=False)

import pytest
from fastapi.testclient import TestClient
from ppmlx.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── HTML Dashboard ────────────────────────────────────────────────────

class TestDashboardHTML:
    def test_dashboard_returns_200(self, client):
        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_dashboard_returns_html(self, client):
        response = client.get("/dashboard")
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_contains_title(self, client):
        response = client.get("/dashboard")
        assert "ppmlx Dashboard" in response.text

    def test_dashboard_has_key_sections(self, client):
        response = client.get("/dashboard")
        html = response.text
        assert "Total Requests" in html
        assert "Avg Latency" in html
        assert "System Memory" in html
        assert "Loaded Models" in html
        assert "Request Rate" in html
        assert "Recent Requests" in html

    def test_dashboard_has_sse_connection(self, client):
        response = client.get("/dashboard")
        assert "EventSource" in response.text
        assert "/api/dashboard/stream" in response.text

    def test_dashboard_has_charts(self, client):
        response = client.get("/dashboard")
        assert "rate-chart" in response.text
        assert "latency-chart" in response.text
        assert "canvas" in response.text

    def test_dashboard_has_filter_controls(self, client):
        response = client.get("/dashboard")
        assert "filter-status" in response.text
        assert "filter-model" in response.text

    def test_dashboard_has_dark_theme(self, client):
        response = client.get("/dashboard")
        assert "--bg:#0d1117" in response.text


# ── System API ────────────────────────────────────────────────────────

class TestSystemAPI:
    def test_system_returns_200(self, client):
        response = client.get("/api/dashboard/system")
        assert response.status_code == 200

    def test_system_has_required_fields(self, client):
        response = client.get("/api/dashboard/system")
        data = response.json()
        assert "memory_total_gb" in data
        assert "loaded_models" in data
        assert "uptime_seconds" in data

    def test_system_memory_is_numeric(self, client):
        response = client.get("/api/dashboard/system")
        data = response.json()
        assert isinstance(data["memory_total_gb"], (int, float))

    def test_system_loaded_models_is_list(self, client):
        response = client.get("/api/dashboard/system")
        data = response.json()
        assert isinstance(data["loaded_models"], list)

    def test_system_uptime_is_nonnegative(self, client):
        response = client.get("/api/dashboard/system")
        data = response.json()
        assert data["uptime_seconds"] >= 0


# ── Metrics API ───────────────────────────────────────────────────────

class TestMetricsAPI:
    def test_metrics_returns_200(self, client):
        response = client.get("/api/dashboard/metrics")
        assert response.status_code == 200

    def test_metrics_has_stats(self, client):
        response = client.get("/api/dashboard/metrics")
        data = response.json()
        assert "stats" in data
        assert "time_series" in data

    def test_metrics_stats_structure(self, client):
        response = client.get("/api/dashboard/metrics")
        data = response.json()
        stats = data["stats"]
        assert "total_requests" in stats
        assert "by_model" in stats

    def test_metrics_time_series_is_list(self, client):
        response = client.get("/api/dashboard/metrics")
        data = response.json()
        assert isinstance(data["time_series"], list)


# ── Requests API ──────────────────────────────────────────────────────

class TestRequestsAPI:
    def test_requests_returns_200(self, client):
        response = client.get("/api/dashboard/requests")
        assert response.status_code == 200

    def test_requests_returns_list(self, client):
        response = client.get("/api/dashboard/requests")
        data = response.json()
        assert isinstance(data, list)

    def test_requests_has_expected_fields(self, client):
        response = client.get("/api/dashboard/requests")
        data = response.json()
        if data:
            row = data[0]
            assert "timestamp" in row
            assert "model_alias" in row
            assert "status" in row
            assert "total_duration_ms" in row

    def test_requests_limit_parameter(self, client):
        response = client.get("/api/dashboard/requests?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# ── SSE Stream ────────────────────────────────────────────────────────

class TestSSEStream:
    def test_stream_endpoint_exists(self, client):
        """The /api/dashboard/stream route should be registered and not 404."""
        from fastapi.routing import APIRoute
        from ppmlx.server import app as _app
        paths = [r.path for r in _app.routes if isinstance(r, APIRoute)]
        assert "/api/dashboard/stream" in paths

    def test_stream_generator_produces_valid_json(self):
        """The SSE event generator should produce valid JSON payloads."""
        from ppmlx.dashboard import _get_system_info, _get_db_stats, _get_recent_requests
        # Test that the helper functions return serializable data
        sys_info = _get_system_info()
        stats = _get_db_stats()
        requests = _get_recent_requests(limit=5)

        payload = {
            "system": sys_info,
            "stats": stats,
            "time_series": [],
            "recent_requests": requests,
        }
        # Should be JSON-serializable
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        assert "system" in parsed
        assert "stats" in parsed
        assert "time_series" in parsed
        assert "recent_requests" in parsed

    def test_stream_payload_system_fields(self):
        """The system portion of the SSE payload should have expected keys."""
        from ppmlx.dashboard import _get_system_info
        info = _get_system_info()
        assert "memory_total_gb" in info
        assert "loaded_models" in info
        assert "uptime_seconds" in info


# ── Integration: dashboard wired into server ──────────────────────────

class TestDashboardIntegration:
    def test_dashboard_route_registered(self, client):
        """Dashboard route should be accessible on the main app."""
        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_api_routes_registered(self, client):
        """All dashboard API routes should be accessible."""
        for path in [
            "/api/dashboard/system",
            "/api/dashboard/metrics",
            "/api/dashboard/requests",
        ]:
            response = client.get(path)
            assert response.status_code == 200, f"Failed for {path}"

    def test_existing_endpoints_still_work(self, client):
        """Adding the dashboard should not break existing endpoints."""
        for path in ["/health", "/metrics", "/v1/models"]:
            response = client.get(path)
            assert response.status_code == 200, f"Failed for {path}"
