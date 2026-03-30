# ppmlx

CLI for running LLMs on Apple Silicon via MLX with an OpenAI-compatible API.

## Dev Setup

```bash
uv sync --python 3.11
uv run pytest tests/ -v
```

## Starting the Server

```bash
# Basic start (default port 6767)
uv run ppmlx serve

# Pre-load a model on startup
uv run ppmlx serve --model llama3

# Bind to all interfaces (e.g. for LAN access)
uv run ppmlx serve --host 0.0.0.0 --port 6767

# Interactive model selection
uv run ppmlx serve --interactive
```

Server listens on `http://127.0.0.1:6767` by default.

## Connecting (OpenAI-compatible)

**Python SDK:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6767/v1", api_key="local")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello"}],
)
```

**curl:**
```bash
# List models
curl http://localhost:6767/v1/models

# Chat
curl http://localhost:6767/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings
curl http://localhost:6767/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed", "input": "Hello world"}'
```

**Any OpenAI-compatible tool** (LangChain, LlamaIndex, Open WebUI, etc.):
- `base_url`: `http://localhost:6767/v1`
- `api_key`: any non-empty string (e.g. `"local"`)

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat (streaming supported) |
| `POST /v1/responses` | Responses API (Codex, newer OpenAI tools) |
| `POST /v1/completions` | Text completion |
| `POST /v1/embeddings` | Embeddings |
| `GET /v1/models` | List loaded/available models |
| `GET /health` | Health check |
| `GET /metrics` | Usage metrics |

## Key CLI Commands

```bash
ppmlx launch          # Interactive TUI launcher (action + model picker)
ppmlx pull <model>    # Download a model
ppmlx run <model>     # Interactive chat REPL
ppmlx serve           # Start API server
ppmlx list            # List local models
ppmlx ps              # Show loaded models + memory usage
ppmlx rm <model>      # Remove a model
ppmlx config          # View/set configuration
ppmlx quantize        # Convert & quantize a model to MLX format
```

## Config

`~/.ppmlx/config.toml` — all optional. Key fields:
```toml
[server]
host = "127.0.0.1"
port = 6767

[generation]
temperature = 0.7
max_tokens = 2048
```

## Project Structure

```
ppmlx/
  cli.py          # Typer CLI (entry point)
  server.py       # FastAPI app (OpenAI-compatible routes)
  engine.py       # MLX LLM inference
  engine_embed.py # MLX embedding inference
  models.py       # Model registry + HuggingFace download
  config.py       # Config loading (~/.ppmlx/config.toml)
  db.py           # SQLite request logging
  schema.py       # Pydantic request/response schemas
tests/
  conftest.py     # MLX stubs for CI (no GPU needed)
```
