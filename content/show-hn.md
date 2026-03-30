# Show HN Drafts

## Draft 1 — Technical Focus

**Title:** Show HN: ppmlx -- MLX-native LLM server for Apple Silicon (Ollama alternative)

I built ppmlx because I was frustrated with Ollama's performance on my Mac. Ollama converts models to GGUF and runs them through llama.cpp -- a great approach for cross-platform support, but it leaves performance on the table on Apple Silicon. ppmlx uses Apple's MLX framework directly, running models in their native MLX format on the Metal GPU with unified memory.

The result: 43-112% faster token generation in our benchmarks (M4 Pro, 48 GB, 4-bit quantized models). On Qwen3.5:9B, ppmlx hits 48 tok/s vs Ollama's 23 tok/s. On GLM-4.7-Flash, it's 63 tok/s vs 40 tok/s.

The API is OpenAI-compatible, so anything that talks to OpenAI works out of the box -- Claude Code, Cursor, Open WebUI, LangChain, any OpenAI SDK. Just point your base_url to localhost:6767.

```
uv tool install ppmlx
ppmlx serve
```

That's it. 168+ models available through curated aliases (llama3, qwen2.5, mistral, phi4, etc.) that map to HuggingFace MLX-community repos.

Features: streaming, tool calling, vision, embeddings, Responses API. Built with Python, MLX, FastAPI.

GitHub: https://github.com/PingCompany/ppmlx
Website: https://ppmlx.dev

---

## Draft 2 — Developer Story

**Title:** Show HN: ppmlx -- MLX-native LLM server for Apple Silicon (Ollama alternative)

I'm an Apple Silicon developer who got tired of watching my M4 Pro underperform. Ollama is great software, but its architecture -- converting HuggingFace models to GGUF and running them through llama.cpp -- means it can't fully exploit the Metal GPU and unified memory that make Apple Silicon special.

So I built ppmlx: an LLM inference server that runs models natively through Apple's MLX framework. No format conversion. No translation layer. Direct Metal GPU access.

Benchmarks on M4 Pro (48 GB, 4-bit quant):
- Qwen3.5:9B: 48.2 tok/s (ppmlx) vs 22.7 tok/s (Ollama) -- +112%
- GLM-4.7-Flash: 63.1 tok/s (ppmlx) vs 40.5 tok/s (Ollama) -- +56%

It's a drop-in replacement. OpenAI-compatible API on localhost:6767. Works with Claude Code, Cursor, Open WebUI, LangChain -- anything that speaks the OpenAI protocol.

```
uv tool install ppmlx    # or: pip install ppmlx
ppmlx pull llama3
ppmlx serve
```

168+ models. Streaming, tool calling, vision, embeddings. MIT licensed.

https://github.com/PingCompany/ppmlx

Happy to answer questions about MLX internals, the benchmark methodology, or anything else.

---

## Draft 3 — Problem/Solution

**Title:** Show HN: ppmlx -- MLX-native LLM server for Apple Silicon (Ollama alternative)

Problem: Ollama runs LLMs through llama.cpp with GGUF format conversion. This works everywhere, but on Apple Silicon it means you're not using MLX -- Apple's own ML framework optimized for the Metal GPU and unified memory architecture.

Solution: ppmlx is an OpenAI-compatible LLM server that runs models natively through MLX. Same convenience as Ollama (model aliases, one-command install, familiar API), but 43-112% faster on Apple Silicon.

Real benchmarks (M4 Pro, 48 GB, 3 runs averaged):

| Model | ppmlx | Ollama | Delta |
|---|---|---|---|
| GLM-4.7-Flash (simple) | 63.1 tok/s | 40.5 tok/s | +56% |
| Qwen3.5:9B (simple) | 48.2 tok/s | 22.7 tok/s | +112% |

Install and run:
```
uv tool install ppmlx
ppmlx serve --model llama3
```

Then point any OpenAI-compatible client to http://localhost:6767/v1.

Stack: Python + MLX + FastAPI + Typer. 168+ curated model aliases mapping to mlx-community HuggingFace repos. Supports chat completions, streaming, tool calling, vision, embeddings, and the Responses API.

GitHub: https://github.com/PingCompany/ppmlx
Website: https://ppmlx.dev
MIT License.
