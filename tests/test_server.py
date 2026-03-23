"""Tests for pp_llm.server — FastAPI OpenAI-compatible API."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock

# Mock all pp_llm modules that server.py tries to import lazily
for mod in [
    "pp_llm.engine", "pp_llm.engine_vlm", "pp_llm.engine_embed",
    "pp_llm.models", "pp_llm.db", "pp_llm.config", "pp_llm.memory",
    "pp_llm.schema",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Set up mock engine
mock_engine = MagicMock()
mock_engine.generate.return_value = ("Hello!", None, 10, 5)
mock_engine.stream_generate.return_value = iter(["Hello", " ", "world"])
mock_engine.list_loaded.return_value = []
sys.modules["pp_llm.engine"].get_engine = MagicMock(return_value=mock_engine)

# Set up mock embed engine
mock_embed_engine = MagicMock()
mock_embed_engine.encode.return_value = [[0.1, 0.2, 0.3]]
sys.modules["pp_llm.engine_embed"].get_embed_engine = MagicMock(return_value=mock_embed_engine)

# Set up mock models
sys.modules["pp_llm.models"].resolve_alias = MagicMock(side_effect=lambda x: x)
sys.modules["pp_llm.models"].list_local_models = MagicMock(return_value=[])
sys.modules["pp_llm.models"].all_aliases = MagicMock(return_value=[])
sys.modules["pp_llm.models"].is_vision_model = MagicMock(return_value=False)
sys.modules["pp_llm.models"].is_embed_model = MagicMock(return_value=False)

# Set up mock db
mock_db = MagicMock()
mock_db.get_stats.return_value = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}
sys.modules["pp_llm.db"].get_db = MagicMock(return_value=mock_db)

# Set up mock memory
sys.modules["pp_llm.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

# Set up mock config
mock_config = MagicMock()
mock_config.logging.snapshot_interval_seconds = 60
sys.modules["pp_llm.config"].load_config = MagicMock(return_value=mock_config)

import pytest
from fastapi.testclient import TestClient
from pp_llm.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_has_required_fields(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "loaded_models" in data
    assert "uptime_seconds" in data
    assert isinstance(data["loaded_models"], list)
    assert isinstance(data["uptime_seconds"], int)


def test_metrics_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data or "loaded_models" in data


def test_list_models_returns_200(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_chat_completion_nonstreaming(client):
    # Reset engine mock to return fresh values
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    sys.modules["pp_llm.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert content  # non-empty


def test_chat_completion_streaming_format(client):
    # Reset stream_generate mock
    def fresh_stream(*args, **kwargs):
        return iter(["Hello", " ", "world"])
    mock_engine.stream_generate = fresh_stream
    sys.modules["pp_llm.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    body = response.text
    assert "data:" in body
    assert "data: [DONE]" in body


def test_completions_endpoint(client):
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    sys.modules["pp_llm.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/completions", json={
        "model": "test-model",
        "prompt": "Once upon a time",
        "max_tokens": 50,
    })
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    text = data["choices"][0]["text"]
    assert text  # non-empty


def test_embeddings_endpoint(client):
    mock_embed_engine.encode.return_value = [[0.1, 0.2, 0.3]]
    sys.modules["pp_llm.engine_embed"].get_embed_engine = MagicMock(return_value=mock_embed_engine)

    response = client.post("/v1/embeddings", json={
        "model": "embed-model",
        "input": "Hello world",
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, list)


def test_unknown_model_uses_name_directly(client):
    """Model not in aliases falls back to raw name; engine still called."""
    sys.modules["pp_llm.models"].resolve_alias = MagicMock(side_effect=Exception("not found"))
    mock_engine.generate.return_value = ("Response!", None, 5, 3)
    sys.modules["pp_llm.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/chat/completions", json={
        "model": "unknown-model-xyz",
        "messages": [{"role": "user", "content": "test"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "unknown-model-xyz"

    # Restore
    sys.modules["pp_llm.models"].resolve_alias = MagicMock(side_effect=lambda x: x)


def test_cors_headers_present(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS middleware sets Access-Control-Allow-Origin on actual requests too
    response2 = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in response2.headers
