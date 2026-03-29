# ppmlx

**Run LLMs on your Mac.** OpenAI-compatible API powered by Apple Silicon.

[![CI](https://github.com/the-focus-company/ppmlx/actions/workflows/tests.yml/badge.svg)](https://github.com/the-focus-company/ppmlx/actions)
[![PyPI](https://img.shields.io/pypi/v/ppmlx)](https://pypi.org/project/ppmlx/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Install

```bash
pip install ppmlx
```

> Requires macOS on Apple Silicon (M1+) and Python 3.11+
>
> Privacy note: `ppmlx` never sends prompts, responses, file contents, paths, or tokens anywhere. Optional anonymous usage analytics can be disabled with `ppmlx config --no-analytics`.

## Get Started

```bash
ppmlx pull qwen3.5:9b      # download a model
ppmlx run qwen3.5:9b       # chat in the terminal
ppmlx serve                 # start API server on :6767
```

That's it. Any OpenAI-compatible tool works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6767/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Commands

| Command | Description | Key Options |
|---|---|---|
| `ppmlx launch` | Interactive launcher (pick action + model) | `-m model`, `--host`, `--port`, `--flush` |
| `ppmlx serve` | Start API server on :6767 | `-m model`, `--embed-model`, `-i`, `--no-cors` |
| `ppmlx run <model>` | Interactive chat REPL | `-s system`, `-t temp`, `--max-tokens` |
| `ppmlx pull [model]` | Download model (multiselect if no arg) | `--token` |
| `ppmlx list` | Show downloaded models | `-a` all (incl. registry), `--path` |
| `ppmlx rm <model>` | Remove a model | `-f` skip confirmation |
| `ppmlx ps` | Show loaded models & memory | |
| `ppmlx quantize <model>` | Convert & quantize HF model to MLX | `-b bits`, `--group-size`, `-o output` |
| `ppmlx config` | View/set configuration | `--hf-token` |

## Connect Your Tools

Point any OpenAI-compatible client at `http://localhost:6767/v1` with any API key:

- **Cursor** — Settings > AI > OpenAI-compatible
- **Continue** — config.json: provider `openai`, apiBase above
- **LangChain / LlamaIndex** — set `base_url` and `api_key="local"`

## Config

Optional. `~/.ppmlx/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 6767

[defaults]
temperature = 0.7
max_tokens = 2048

[analytics]
enabled = true
provider = "posthog"
respect_do_not_track = true
```

## Anonymous Usage Analytics

`ppmlx` supports privacy-preserving anonymous product analytics, disabled by default — you are asked to opt in on first run.

What is sent:
- command and API event names such as `serve_started`, `model_pulled`, `api_chat_completions`
- app version, Python minor version, OS family, CPU architecture
- coarse booleans/counters such as `stream=true`, `tools=true`, `batch_size=4`

What is never sent:
- prompts, responses, tool arguments, file contents, file paths
- HuggingFace tokens, API keys, repo IDs, model prompts, request bodies

When events are sent:
- when a CLI command starts
- when OpenAI-compatible API endpoints are hit

Why:
- understand which workflows matter most
- prioritize compatibility work across commands and API surfaces
- measure adoption without collecting user content

Opt out:

```bash
ppmlx config --no-analytics
```

or:

```toml
[analytics]
enabled = false
```

For maintainer-operated analytics, the recommended sink is self-hosted PostHog. Configure it with:

```bash
export PPMLX_ANALYTICS_HOST="https://analytics.example.com"
export PPMLX_ANALYTICS_PROJECT_API_KEY="your-posthog-project-api-key"
```

If you prefer, you can also set the same values in `~/.ppmlx/config.toml`.

## API Documentation

When the server is running, interactive API docs are available at:

- **Swagger UI**: [http://localhost:6767/docs](http://localhost:6767/docs)
- **ReDoc**: [http://localhost:6767/redoc](http://localhost:6767/redoc)

## Requirements

- macOS on Apple Silicon (M1 or later)
- Python 3.11+
- At least 8 GB unified memory (16 GB+ recommended for larger models)

## ppmlx vs Ollama

| | ppmlx | Ollama |
|---|---|---|
| Runtime | MLX (Apple-native) | llama.cpp (cross-platform) |
| Platform | macOS Apple Silicon only | macOS, Linux, Windows |
| GPU backend | Metal (unified memory) | Metal / CUDA / ROCm |
| API | OpenAI-compatible | Ollama + OpenAI-compatible |
| Language | Python | Go + C++ |
| Quantization | MLX format | GGUF format |

Choose **ppmlx** if you want maximum Apple Silicon performance with a pure-Python, MLX-native stack. Choose **Ollama** if you need cross-platform support or GGUF models.

## License

MIT
