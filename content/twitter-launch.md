# X/Twitter Launch Content

## Standalone Tweets (3 variants)

### Tweet 1 -- Performance Hook

ppmlx: MLX-native LLM server for Apple Silicon.

48 tok/s vs Ollama's 23 tok/s on Qwen3.5:9B. Same model, same Mac, +112% faster.

OpenAI-compatible API. Drop-in replacement.

uv tool install ppmlx

https://github.com/PingCompany/ppmlx

#LocalLLM #AppleSilicon #MLX #AI

### Tweet 2 -- Developer Workflow Hook

Your M-series Mac is faster at local LLM inference than you think.

ppmlx runs models natively through Apple's MLX framework -- no GGUF conversion. 43-112% faster than Ollama. OpenAI API on localhost:6767.

uv tool install ppmlx
ppmlx serve

https://ppmlx.dev

#LocalLLM #AppleSilicon #MLX #AI

### Tweet 3 -- Simplicity Hook

Two commands to run LLMs 43-112% faster on your Mac:

uv tool install ppmlx
ppmlx serve

MLX-native. OpenAI-compatible. 168+ models. Works with Claude Code, Cursor, Open WebUI, LangChain.

https://github.com/PingCompany/ppmlx

#LocalLLM #AppleSilicon #MLX #AI

---

## Launch Thread (7 tweets)

### 1/7 -- Announcement

I built ppmlx -- an MLX-native LLM inference server for Apple Silicon.

It's 43-112% faster than Ollama on the same hardware because it skips the GGUF/llama.cpp translation layer and runs models directly through Apple's MLX framework.

Open source, MIT licensed.

https://github.com/PingCompany/ppmlx

### 2/7 -- The Problem

Ollama is great software, but its architecture wasn't designed for Apple Silicon.

It converts models to GGUF and runs them through llama.cpp. That's cross-platform, but it can't fully exploit the Metal GPU and unified memory that make M-series chips special.

ppmlx uses MLX natively. No conversion step.

### 3/7 -- Benchmarks

Real numbers (MacBook Pro M4 Pro, 48 GB, 4-bit quant):

Qwen3.5:9B:
  ppmlx: 48.2 tok/s
  Ollama: 22.7 tok/s
  +112% faster

GLM-4.7-Flash:
  ppmlx: 63.1 tok/s
  Ollama: 40.5 tok/s
  +56% faster

[VISUAL: Bar chart -- see guidance below]

### 4/7 -- Drop-in Replacement

ppmlx serves an OpenAI-compatible API on localhost:6767.

Change your base_url, keep everything else. It works with:
- Claude Code
- Cursor / Codex
- Open WebUI
- LangChain / LlamaIndex
- Any OpenAI SDK (Python, Node, Go, Rust)

### 5/7 -- Quick Start

```
uv tool install ppmlx
ppmlx pull llama3
ppmlx serve
```

That's it. 168+ models with curated aliases: Llama 3, Qwen 3.5, Mistral, Phi 4, Gemma 2, DeepSeek, and more.

All mapped to mlx-community HuggingFace repos.

### 6/7 -- Features

What's inside:
- Streaming chat completions
- Tool/function calling
- Vision (images via mlx-vlm)
- Embeddings for RAG
- Responses API (Codex support)
- Interactive TUI launcher
- Request logging to SQLite
- Config via ~/.ppmlx/config.toml

### 7/7 -- Call to Action

ppmlx is a solo project and I'm building it in public.

If you have an M-series Mac, I'd love for you to try it:

https://ppmlx.dev

Star the repo, file issues, send PRs. Feedback on model support and missing features is especially welcome.

#LocalLLM #AppleSilicon #MLX #AI

---

## Benchmark Visual Guidance

Create two bar chart images for the thread:

### Chart 1: Token Generation Speed (tok/s)

- **Title:** "Token Generation: ppmlx (MLX) vs Ollama (GGUF)"
- **Subtitle:** "MacBook Pro M4 Pro, 48 GB -- 4-bit quantized, 3 runs averaged"
- **Layout:** Horizontal grouped bar chart, two models
- **Data:**
  - GLM-4.7-Flash Simple: ppmlx 63.1 tok/s, Ollama 40.5 tok/s (+56%)
  - GLM-4.7-Flash Complex: ppmlx 55.6 tok/s, Ollama 38.8 tok/s (+43%)
  - Qwen3.5:9B Simple: ppmlx 48.2 tok/s, Ollama 22.7 tok/s (+112%)
  - Qwen3.5:9B Complex: ppmlx 47.2 tok/s, Ollama 23.0 tok/s (+105%)
- **Colors:** ppmlx bars in warm amber/orange (#e8a04c), Ollama bars in muted gray (#545a66)
- **Annotations:** Show percentage delta next to each pair
- **Style:** Dark background (#0a0a0f), clean, minimal, monospace labels

### Chart 2: Time to First Token (TTFT)

- **Title:** "Time to First Token: ppmlx vs Ollama"
- **Subtitle:** same hardware subtitle
- **Layout:** Horizontal bar chart, lower is better
- **Data:**
  - GLM-4.7-Flash Simple: ppmlx 374ms, Ollama 832ms
  - GLM-4.7-Flash Long Context: ppmlx 6.8s, Ollama 8.4s
  - Qwen3.5:9B Simple: ppmlx 537ms, Ollama 324ms (note: Ollama wins here)
  - Qwen3.5:9B Long Context: ppmlx 9.2s, Ollama 11.5s
- **Colors:** Same scheme as Chart 1
- **Note:** Be transparent -- Ollama has faster TTFT on Qwen simple prompt. Honesty builds trust.
- **Style:** Match Chart 1

### Design Notes

- Use 16:9 aspect ratio for Twitter card display
- Include the ppmlx.dev URL in the bottom right corner
- Error bars showing standard deviation are optional but add credibility
- Export as PNG at 2x resolution for sharp display
