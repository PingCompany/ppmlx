# r/LocalLLaMA Launch Post

**Title:** I built an MLX-native alternative to Ollama for Apple Silicon -- 43-112% faster in benchmarks

Hey r/LocalLLaMA,

I've been running local LLMs on my MacBook Pro for a while using Ollama, and I kept wondering: why does my M4 Pro feel slower than it should? The answer turned out to be architectural. Ollama converts models to GGUF format and runs them through llama.cpp. That's great for cross-platform support, but on Apple Silicon it means you're going through a translation layer instead of using Apple's own MLX framework, which is built specifically for the Metal GPU and unified memory.

So I built **ppmlx** -- an MLX-native LLM server that skips the GGUF conversion entirely and runs models directly through MLX.

## Benchmarks

Tested on MacBook Pro M4 Pro, 48 GB. All models 4-bit quantized. 3 runs averaged.

**Qwen3.5:9B:**

| Scenario | ppmlx | Ollama | Speedup |
|---|---|---|---|
| Simple (short prompt, short answer) | 48.2 tok/s | 22.7 tok/s | +112% |
| Complex (short prompt, long answer) | 47.2 tok/s | 23.0 tok/s | +105% |
| Long Context (~4K token prompt) | 43.2 tok/s | 23.7 tok/s | +82% |

**GLM-4.7-Flash (58B sparse):**

| Scenario | ppmlx | Ollama | Speedup |
|---|---|---|---|
| Simple | 63.1 tok/s | 40.5 tok/s | +56% |
| Complex | 55.6 tok/s | 38.8 tok/s | +43% |
| Long Context | 42.1 tok/s | 27.5 tok/s | +53% |

Benchmark scripts are in the repo if you want to reproduce these on your own hardware: [scripts/](https://github.com/PingCompany/ppmlx/tree/main/scripts)

## Quick Start

```bash
# Install
uv tool install ppmlx      # or: pip install ppmlx

# Pull a model and start chatting
ppmlx pull llama3
ppmlx run llama3

# Start the API server
ppmlx serve
```

The API server runs on `localhost:6767` and is fully OpenAI-compatible. Point any client at `http://localhost:6767/v1` with any API key.

## Features

- **OpenAI-compatible API** -- chat completions, streaming, tool calling, Responses API
- **Vision** -- image understanding via mlx-vlm
- **Embeddings** -- vector generation for RAG (nomic-embed, bge-small, etc.)
- **168+ model aliases** -- Llama 3, Qwen 2.5/3.5, Mistral, Phi 4, Gemma 2, DeepSeek, GLM, and more, all mapped to mlx-community HuggingFace repos
- **Works with everything** -- Claude Code, Cursor, Codex, Open WebUI, LangChain, LlamaIndex, any OpenAI SDK
- **Interactive TUI** -- `ppmlx launch` for a nice model picker
- **Config & logging** -- `~/.ppmlx/config.toml`, every request logged to SQLite

## How It Works

ppmlx loads models from the HuggingFace `mlx-community` namespace -- these are models already converted to MLX format (mostly 4-bit quantized). When you run `ppmlx pull llama3`, it downloads `mlx-community/Meta-Llama-3-8B-Instruct-4bit` from HuggingFace Hub. The inference engine uses `mlx-lm` directly on the Metal GPU with unified memory -- no GGUF, no llama.cpp.

The server is FastAPI + Uvicorn, the CLI is Typer + Rich. The whole thing is Python.

## What This Isn't

This only works on macOS with Apple Silicon. If you need cross-platform support, Ollama is still the better choice. ppmlx is specifically for people who have an M-series Mac and want to squeeze maximum performance out of it.

## Links

- **GitHub:** https://github.com/PingCompany/ppmlx
- **Website:** https://ppmlx.dev
- **PyPI:** https://pypi.org/project/ppmlx/

This is a solo project and I'm actively developing it. I'd really appreciate feedback -- especially on:
- Models you'd want to see in the registry
- Features you need for your workflow
- Any issues you hit during setup

MIT licensed. PRs welcome.
