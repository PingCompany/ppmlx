# ppmlx

**CLI for running LLMs on Apple Silicon via MLX** — OpenAI-compatible API on port 6767.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Install

> **Requires:** macOS on Apple Silicon (M1/M2/M3/M4), Python 3.11+

### uv (recommended)

```bash
uv tool install ppmlx
```

### pipx

```bash
pipx install ppmlx
```

### curl | sh (one-liner)

```bash
curl -fsSL https://raw.githubusercontent.com/PingCompany/ppmlx/main/scripts/install.sh | sh
```

### From source

```bash
git clone https://github.com/PingCompany/ppmlx
cd ppmlx
uv tool install .
```

### Homebrew

Homebrew tap coming soon. For now, use `uv tool install ppmlx`.

---

## Quick Start

```bash
# 1. Download a model
ppmlx pull llama3

# 2. Interactive chat REPL
ppmlx run llama3

# 3. Start OpenAI-compatible API server on :6767
ppmlx serve
```

---

## Benchmarks

Measured on a MacBook Pro M4 Pro (48 GB unified memory, macOS 15.x). Each scenario was run 3 times with `temperature=0` and `max_tokens=8192`; values below are averages.

### GLM-4.7-Flash (4-bit, ~5 GB)

| Scenario | Metric | ppmlx | Ollama | Delta |
|---|---|---|---|---|
| **Simple** (short prompt, short answer) | tok/s | 63.1 | 40.5 | **+56%** |
| | TTFT | 374 ms | 832 ms | **-55%** |
| **Complex** (short prompt, long answer) | tok/s | 55.6 | 38.8 | **+43%** |
| | TTFT | 496 ms | 412 ms | +20% |
| **Long context** (~4 K token prompt) | tok/s | 42.1 | 27.5 | **+53%** |
| | TTFT | 6,792 ms | 8,401 ms | **-19%** |

### Qwen 3.5 9B (4-bit, ~6 GB)

| Scenario | Metric | ppmlx | Ollama | Delta |
|---|---|---|---|---|
| **Simple** | tok/s | 48.2 | 22.7 | **+112%** |
| | TTFT | 537 ms | 324 ms | +66% |
| **Complex** | tok/s | 47.2 | 23.0 | **+106%** |
| | TTFT | 567 ms | 455 ms | +25% |
| **Long context** | tok/s | 43.2 | 23.7 | **+82%** |
| | TTFT | 9,212 ms | 11,461 ms | **-20%** |

> **tok/s** = tokens per second (higher is better). **TTFT** = time to first token (lower is better). Delta is relative to Ollama.

**Methodology.** Streaming chat completions over the OpenAI-compatible API; TTFT measured from request start to first SSE content chunk. See [`scripts/bench_common.sh`](scripts/bench_common.sh) and the per-model scripts in `scripts/` for the full, reproducible setup.

---

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6767/v1", api_key="local")

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## curl Example

```bash
# List available models
curl http://localhost:6767/v1/models

# Chat completion
curl http://localhost:6767/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "What is Apple Silicon?"}],
    "stream": false
  }'

# Embeddings
curl http://localhost:6767/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed", "input": "Hello world"}'
```

---

## Model Aliases

### Llama Family

| Alias        | HuggingFace Repo                                              |
|--------------|---------------------------------------------------------------|
| `llama3`     | mlx-community/Meta-Llama-3-8B-Instruct-4bit                   |
| `llama3-70b` | mlx-community/Meta-Llama-3-70B-Instruct-4bit                  |
| `llama3.2`   | mlx-community/Llama-3.2-3B-Instruct-4bit                      |
| `llama3.1`   | mlx-community/Meta-Llama-3.1-8B-Instruct-4bit                 |

### Mistral / Mixtral Family

| Alias           | HuggingFace Repo                                              |
|-----------------|---------------------------------------------------------------|
| `mistral`       | mlx-community/Mistral-7B-Instruct-v0.3-4bit                   |
| `mixtral`       | mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit                 |
| `mistral-nemo`  | mlx-community/Mistral-Nemo-Instruct-2407-4bit                  |

### Qwen Family

| Alias        | HuggingFace Repo                                              |
|--------------|---------------------------------------------------------------|
| `qwen2.5`    | mlx-community/Qwen2.5-7B-Instruct-4bit                        |
| `qwen2.5-14b`| mlx-community/Qwen2.5-14B-Instruct-4bit                       |
| `qwen2.5-72b`| mlx-community/Qwen2.5-72B-Instruct-4bit                       |

### Phi / Gemma Family

| Alias        | HuggingFace Repo                                              |
|--------------|---------------------------------------------------------------|
| `phi4`       | mlx-community/phi-4-4bit                                      |
| `phi3.5`     | mlx-community/Phi-3.5-mini-instruct-4bit                      |
| `gemma2`     | mlx-community/gemma-2-9b-it-4bit                              |
| `gemma2-27b` | mlx-community/gemma-2-27b-it-4bit                             |

### Code Models

| Alias          | HuggingFace Repo                                              |
|----------------|---------------------------------------------------------------|
| `codellama`    | mlx-community/CodeLlama-13b-Instruct-hf-4bit                  |
| `deepseek-coder`| mlx-community/deepseek-coder-6.7b-instruct-4bit              |

### Embedding Models

| Alias          | HuggingFace Repo                                              |
|----------------|---------------------------------------------------------------|
| `nomic-embed`  | mlx-community/nomic-embed-text-v1.5                           |
| `bge-small`    | mlx-community/bge-small-en-v1.5                               |

---

## RAM Requirements

| Model Size  | Min RAM | Recommended RAM | Notes                           |
|-------------|---------|------------------|---------------------------------|
| 1-3B params | 4 GB    | 8 GB             | Llama 3.2 3B, Phi 3.5 mini      |
| 7-8B params | 8 GB    | 16 GB            | Llama 3 8B, Mistral 7B          |
| 13-14B      | 16 GB   | 24 GB            | CodeLlama 13B, Qwen 2.5 14B     |
| 27-34B      | 24 GB   | 36 GB            | Gemma 2 27B                     |
| 70-72B      | 48 GB   | 64 GB            | Llama 3 70B, Qwen 2.5 72B       |

> All values are for 4-bit quantized models. Unquantized models require 2-4x more RAM.

---

## CLI Commands

| Command              | Description                                          |
|----------------------|------------------------------------------------------|
| `ppmlx pull <model>`| Download a model from HuggingFace Hub                |
| `ppmlx run <model>` | Start interactive chat REPL                          |
| `ppmlx serve`       | Start OpenAI-compatible API server on :6767          |
| `ppmlx list`        | List locally downloaded models                       |
| `ppmlx rm <model>`  | Remove a downloaded model                            |
| `ppmlx alias <n> <repo>` | Add a custom model alias                       |
| `ppmlx aliases`     | Show all model aliases (built-in + custom)           |
| `ppmlx ps`          | Show currently loaded models and memory usage        |
| `ppmlx quantize`    | Convert and quantize a model to MLX format           |
| `ppmlx create`      | Create a custom model from a Modelfile               |
| `ppmlx logs`        | Query the request log database                       |
| `ppmlx info <model>`| Show detailed model information                      |
| `ppmlx estimate <m>`| Estimate RAM requirements before downloading         |

---

## Modelfile Example

Create a `Modelfile` to define a custom model with a system prompt:

```
FROM llama3

SYSTEM """
You are a helpful coding assistant. You write clean, well-documented code
and explain your reasoning step by step.
"""

PARAMETER temperature 0.2
PARAMETER max_tokens 4096
PARAMETER top_p 0.9
```

Then build it:

```bash
ppmlx create coding-assistant -f Modelfile
ppmlx run coding-assistant
```

---

## Configuration

ppmlx reads configuration from `~/.ppmlx/config.toml`. All values are optional.

```toml
[server]
host = "127.0.0.1"      # Bind address (default: 127.0.0.1)
port = 6767             # Port (default: 6767)
cors = true             # Enable CORS (default: true)
cors_origins = ["*"]    # Allowed CORS origins

[models]
dir = "~/.ppmlx/models"   # Model storage directory
default_alias = "llama3"    # Default model for bare requests

[generation]
temperature = 0.7       # Default sampling temperature
max_tokens = 2048       # Default max output tokens
top_p = 0.9             # Default top-p
repetition_penalty = 1.1

[logging]
db_path = "~/.ppmlx/ppmlx.db"   # SQLite log database
log_requests = true                 # Log all requests
log_level = "info"                  # Server log level
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ppmlx CLI                       │
│  (typer + rich)                                     │
│  pull / run / serve / list / rm / quantize / ...    │
└───────────────────┬─────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   FastAPI Server    │
         │   port :6767        │
         │                     │
         │  /v1/chat/completions│
         │  /v1/completions    │
         │  /v1/embeddings     │
         │  /v1/models         │
         │  /health /metrics   │
         └──────┬──────┬───────┘
                │      │
      ┌─────────▼──┐ ┌─▼──────────────┐
      │  LLM Engine│ │  Embed Engine  │
      │  (mlx-lm)  │ │(mlx-embeddings)│
      └─────────┬──┘ └─┬──────────────┘
                │       │
      ┌─────────▼───────▼──────────────┐
      │       MLX / Metal GPU          │
      │   Apple Silicon Unified Memory │
      └────────────────────────────────┘
                    │
      ┌─────────────▼──────────────────┐
      │   SQLite Request Log           │
      │   ~/.ppmlx/ppmlx.db          │
      └────────────────────────────────┘
```

---

## Uninstall

### uv

```bash
uv tool uninstall ppmlx
```

### pipx

```bash
pipx uninstall ppmlx
```

### Manual cleanup (all methods)

```bash
# Remove downloaded models and config
rm -rf ~/.ppmlx
```

---

## Contributing

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Install dev dependencies: `uv sync --python 3.11`
4. Run tests: `uv run pytest tests/ -v`
5. Submit a pull request.

### Development Setup

```bash
git clone https://github.com/PingCompany/ppmlx
cd ppmlx
uv sync --python 3.11
uv run ppmlx --version
uv run pytest tests/ -v
```

### Project Structure

```
ppmlx/
  __init__.py       # version
  cli.py            # Typer CLI entry point
  server.py         # FastAPI application
  engine.py         # MLX LLM inference engine
  engine_embed.py   # MLX embedding engine
  engine_vlm.py     # MLX vision-language engine
  models.py         # model registry, aliases, download
  config.py         # configuration loading
  db.py             # SQLite request logging
  memory.py         # RAM estimation utilities
  modelfile.py      # Modelfile parser
  quantize.py       # MLX quantization helpers
tests/
  conftest.py       # MLX stubs for CI
  test_cli.py       # CLI tests
scripts/
  install.sh        # One-liner installer
homebrew/
  Formula/ppmlx.rb # Homebrew formula
.github/workflows/
  tests.yml         # CI tests
  release.yml       # PyPI release
  homebrew-update.yml
```

---

## License

MIT — see [LICENSE](LICENSE).
