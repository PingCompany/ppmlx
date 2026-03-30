# 30-Day "Building ppmlx in Public" Comms Plan

Target platform: X/Twitter. Post daily, ideally between 9-11am PT (peak dev audience).

---

## Day 1 -- Launch Day

**Topic:** Product launch announcement
**Post:**
I just open-sourced ppmlx -- an MLX-native LLM server for Apple Silicon.

43-112% faster than Ollama on the same Mac. OpenAI-compatible API. 168+ models.

Two commands to try it:
uv tool install ppmlx
ppmlx serve

https://github.com/PingCompany/ppmlx

#LocalLLM #AppleSilicon #MLX
**Media:** Benchmark bar chart (tok/s comparison ppmlx vs Ollama)

---

## Day 2 -- Why MLX

**Topic:** Technical deep dive on why MLX beats GGUF on Apple Silicon
**Post:**
Why is ppmlx faster than Ollama on Apple Silicon?

Ollama: HuggingFace -> GGUF -> llama.cpp -> Metal
ppmlx: HuggingFace -> MLX -> Metal

Skipping the GGUF conversion means models run in their native format on the Metal GPU with direct unified memory access. No translation overhead.

That's 43-112% more tok/s on the same hardware.
**Media:** Diagram showing the two inference paths side by side (simple flowchart)

---

## Day 3 -- Benchmark Transparency

**Topic:** Sharing raw benchmark data and methodology
**Post:**
I'm publishing all ppmlx benchmark data and scripts.

3 scenarios per model, 3 runs each, standard deviation reported. Temperature 0, streaming enabled.

No cherry-picking. Ollama actually has faster TTFT on Qwen3.5:9B for short prompts (324ms vs 537ms). I show that too.

Scripts: https://github.com/PingCompany/ppmlx/tree/main/scripts
**Media:** Screenshot of terminal running bench_compare.sh output

---

## Day 4 -- Claude Code Integration

**Topic:** Using ppmlx as a local backend for Claude Code
**Post:**
You can run Claude Code against a local LLM through ppmlx.

1. ppmlx serve
2. Set OPENAI_BASE_URL=http://localhost:6767/v1
3. Use any model -- Qwen3.5, Llama 3, DeepSeek

Local inference, zero API costs, your code never leaves your machine.
**Media:** Screenshot of Claude Code running with ppmlx as backend

---

## Day 5 -- Model Registry

**Topic:** How the 168+ model alias system works
**Post:**
ppmlx ships with 168+ curated model aliases.

`ppmlx pull llama3` downloads mlx-community/Meta-Llama-3-8B-Instruct-4bit from HuggingFace.

No manual URL hunting. No format conversion. The mlx-community maintains optimized MLX versions of every major model family.

`ppmlx list` to see what's available.
**Media:** Screenshot of `ppmlx list` output showing model aliases

---

## Day 6 -- Weekend Project: Open WebUI

**Topic:** Integration tutorial with Open WebUI
**Post:**
Weekend project: local ChatGPT-style UI in 2 minutes.

1. ppmlx serve
2. docker run -p 3000:8080 ghcr.io/open-webui/open-webui
3. Set OpenAI URL to http://host.docker.internal:6767/v1

Full chat interface, conversation history, model switching. All local.

https://ppmlx.dev
**Media:** Screenshot of Open WebUI connected to ppmlx

---

## Day 7 -- Week 1 Recap

**Topic:** First week metrics and community feedback
**Post:**
Week 1 of ppmlx in public:

- [X] GitHub stars
- [X] PyPI downloads
- [X] issues filed
- [X] PRs merged

Biggest feedback so far: [summarize top request]

Thank you to everyone who tried it. Keep the issues coming.

https://github.com/PingCompany/ppmlx
**Media:** None (text-only for variety)

---

## Day 8 -- RAM Guide

**Topic:** Practical guide to model sizes vs Mac RAM
**Post:**
"What models can I run on my Mac?"

8 GB RAM: 3B models (Llama 3.2 3B, Phi 3.5 mini)
16 GB: 7-8B models (Llama 3 8B, Mistral 7B, Qwen 2.5 7B)
24 GB: 13-14B models (Qwen 2.5 14B)
36 GB: 27B models (Gemma 2 27B)
48-64 GB: 70B+ models (Llama 3 70B, Qwen 2.5 72B)

All 4-bit quantized. ppmlx estimate <model> tells you before downloading.
**Media:** Table graphic with RAM tiers

---

## Day 9 -- Embeddings for RAG

**Topic:** Using ppmlx for local embeddings
**Post:**
ppmlx isn't just chat -- it serves embeddings too.

```
curl http://localhost:6767/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed", "input": "Hello world"}'
```

Works with LangChain, LlamaIndex, any RAG pipeline. Local vectors, zero API calls.

`ppmlx pull nomic-embed && ppmlx serve`
**Media:** Code snippet screenshot (dark theme)

---

## Day 10 -- Behind the Scenes: FastAPI

**Topic:** Architecture deep dive on the server
**Post:**
ppmlx's server is ~800 lines of FastAPI.

Routes:
- /v1/chat/completions (streaming via SSE)
- /v1/embeddings
- /v1/models
- /v1/responses (Codex support)
- /health, /metrics

No framework magic. No middleware stack. Uvicorn serves it. Every request logged to SQLite.

https://github.com/PingCompany/ppmlx/blob/main/ppmlx/server.py
**Media:** Architecture diagram from README

---

## Day 11 -- Vision Support

**Topic:** Image understanding with local models
**Post:**
ppmlx supports vision models through mlx-vlm.

Send an image in a chat message, get a description back. Same OpenAI API format, same localhost:6767.

Useful for:
- Automated image captioning
- Document analysis
- Screenshot understanding
- Local multimodal agents

All running on your Mac's GPU.
**Media:** Example of sending an image and getting a response

---

## Day 12 -- Tool Calling

**Topic:** Function calling support
**Post:**
ppmlx supports OpenAI-style tool/function calling.

Define tools in your request, the model decides when to call them, you execute locally and feed results back.

This is what powers coding agents like Codex. ppmlx speaks the same protocol, so they work out of the box.

No cloud. No API keys. Your tools, your machine.
**Media:** Code example showing tool definition and response

---

## Day 13 -- Cursor Integration

**Topic:** Using ppmlx with Cursor IDE
**Post:**
Using Cursor with a local LLM through ppmlx:

1. ppmlx serve --model qwen2.5-14b
2. Cursor settings -> Models -> Custom -> base URL: http://localhost:6767/v1
3. Code with a 14B model running entirely on your Mac

No token costs. No data leaving your machine. Same autocomplete, same chat, same apply.
**Media:** Screenshot of Cursor settings pointing to ppmlx

---

## Day 14 -- Two Week Milestone

**Topic:** What I've learned building in public
**Post:**
2 weeks of building ppmlx in public.

Things I've learned:
1. Apple Silicon developers are underserved by existing tools
2. People care more about tok/s than feature lists
3. OpenAI API compatibility is table stakes
4. Benchmark transparency builds trust faster than marketing

What should I build next? Reply with your top feature request.
**Media:** None (engagement post)

---

## Day 15 -- LangChain Tutorial

**Topic:** Integration tutorial
**Post:**
ppmlx + LangChain in 5 lines:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:6767/v1",
    api_key="local",
    model="llama3"
)
```

That's it. Chains, agents, RAG -- all running on local inference.

https://ppmlx.dev
**Media:** Code screenshot

---

## Day 16 -- Performance Tip: Quantization

**Topic:** Technical explainer on 4-bit quantization in MLX
**Post:**
Why ppmlx models default to 4-bit quantization:

- 70% less memory than full precision
- 5-10% quality loss (usually negligible for chat)
- Fits larger models in your Mac's RAM
- MLX handles dequantization on-the-fly during inference

ppmlx quantize lets you convert any HuggingFace model to MLX format yourself.
**Media:** Before/after RAM usage comparison

---

## Day 17 -- Config Deep Dive

**Topic:** Customizing ppmlx
**Post:**
ppmlx is zero-config by default, but everything is tunable.

~/.ppmlx/config.toml:
- Server: host, port, CORS
- Generation: temperature, max_tokens, top_p
- Models: storage directory, default alias
- Logging: SQLite path, log level

One file, no env vars, no YAML.

https://github.com/PingCompany/ppmlx#configuration
**Media:** Screenshot of example config.toml

---

## Day 18 -- Benchmark: New Model

**Topic:** Benchmarking a new model addition
**Post:**
New benchmark: [MODEL NAME] on ppmlx vs Ollama.

ppmlx: XX tok/s
Ollama: XX tok/s
Delta: +XX%

[Brief observation about the results]

Full data: https://github.com/PingCompany/ppmlx/tree/main/benchmark_results

Want me to benchmark a specific model? Reply with the name.
**Media:** Bar chart for the new model

---

## Day 19 -- How ppmlx Loads Models

**Topic:** Behind the scenes on model loading
**Post:**
How ppmlx loads models:

1. You type `ppmlx pull llama3`
2. Alias resolves to mlx-community/Meta-Llama-3-8B-Instruct-4bit
3. huggingface_hub downloads to ~/.ppmlx/models/
4. On `ppmlx serve`, first request triggers lazy loading
5. Model loads directly into unified memory via MLX
6. LRU cache manages multiple models

No daemon. No background process. Models load on demand.
**Media:** Flowchart of the loading process

---

## Day 20 -- Community Highlight

**Topic:** Highlighting community usage or contribution
**Post:**
[Highlight a community member's use case, issue, or PR]

This is why I build in public. Real users finding real issues and building real things.

If you're using ppmlx for something interesting, DM me -- I'd love to feature your project.

https://github.com/PingCompany/ppmlx
**Media:** Screenshot of the highlighted contribution/usage

---

## Day 21 -- Three Week Recap

**Topic:** Progress update and roadmap
**Post:**
3 weeks of ppmlx:

Shipped:
- [list recent features/fixes]

Coming next:
- Model Garden (visual model browser)
- ppmlx bench (built-in benchmarking)
- MCP Server support
- Speculative decoding

Follow progress: https://github.com/PingCompany/ppmlx/issues
**Media:** Roadmap graphic

---

## Day 22 -- LlamaIndex Integration

**Topic:** RAG pipeline tutorial
**Post:**
Local RAG with ppmlx + LlamaIndex:

1. ppmlx serve (chat + embeddings on one server)
2. LlamaIndex connects to localhost:6767 for both LLM and embedding calls
3. Index your documents, query locally

No API keys. No rate limits. No data leaving your network.

Code example: [link to gist or blog]
**Media:** Code screenshot of LlamaIndex setup

---

## Day 23 -- Unified Memory Explainer

**Topic:** Why Apple Silicon's unified memory matters for LLMs
**Post:**
Why Apple Silicon is quietly great for LLMs:

Traditional GPU: RAM + VRAM are separate. Model must fit in VRAM.
Apple Silicon: CPU and GPU share the same memory pool.

A 48 GB MacBook Pro can load models that would need a $1,500+ GPU on a PC. ppmlx exploits this through MLX -- direct access to the full memory pool.
**Media:** Diagram comparing discrete GPU vs unified memory architecture

---

## Day 24 -- Streaming Explained

**Topic:** How SSE streaming works in ppmlx
**Post:**
ppmlx streams tokens via Server-Sent Events -- same protocol as OpenAI.

Each token arrives as a JSON chunk the moment it's generated. No buffering. Your UI can display text as fast as the model produces it.

Average time to first token on Qwen3.5:9B: 537ms. Then ~48 tokens/second streaming.
**Media:** GIF showing streaming response in a terminal

---

## Day 25 -- Python SDK Example

**Topic:** Practical code example
**Post:**
Simplest ppmlx + Python setup:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6767/v1",
    api_key="local"
)

for chunk in client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Explain MLX in one paragraph"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

5 lines. Local inference. Full streaming.
**Media:** Code screenshot with terminal output

---

## Day 26 -- Security Advantage

**Topic:** Privacy benefits of local inference
**Post:**
When you run LLMs through ppmlx, your data never leaves your machine.

No API calls to external servers. No telemetry. No logging to someone else's database (just your own local SQLite).

For proprietary code, sensitive documents, or compliance-restricted environments: local inference isn't a nice-to-have, it's a requirement.
**Media:** None (text-only)

---

## Day 27 -- Interactive TUI

**Topic:** Showcasing the terminal UI
**Post:**
ppmlx launch gives you an interactive TUI for picking models and actions.

Arrow keys to navigate, enter to select. Choose between serve, run (REPL), pull, or info for any model.

Built with questionary + Rich. Terminal UIs deserve to feel good.
**Media:** GIF recording of the TUI in action

---

## Day 28 -- Performance Comparison: Long Context

**Topic:** Benchmark deep dive on long context performance
**Post:**
Long context is where ppmlx really pulls ahead.

With a ~4K token prompt on Qwen3.5:9B:
ppmlx: 43.2 tok/s, TTFT 9.2s
Ollama: 23.7 tok/s, TTFT 11.5s

+82% faster generation. 20% faster prefill.

MLX's prompt processing is significantly faster because it operates directly on the Metal GPU compute pipeline without GGUF decode overhead.
**Media:** Long context benchmark chart

---

## Day 29 -- What's Next

**Topic:** Upcoming features and roadmap
**Post:**
What's coming to ppmlx:

1. ppmlx bench -- built-in benchmarking, compare models on your hardware
2. Model Garden -- visual browser for the 168+ model registry
3. MCP Server -- use ppmlx as a tool provider for AI agents
4. Speculative decoding -- even faster generation with draft models

Which one do you want first? Reply to vote.
**Media:** None (engagement post)

---

## Day 30 -- Month Recap

**Topic:** Reflecting on 30 days of building in public
**Post:**
30 days of building ppmlx in public.

By the numbers:
- [X] GitHub stars
- [X] PyPI downloads
- [X] contributors
- [X] models in registry
- [X] issues closed

Biggest takeaway: shipping daily and sharing honestly builds a community faster than polished marketing.

Thank you to everyone following along. Month 2 starts tomorrow.

https://github.com/PingCompany/ppmlx
**Media:** Collage of highlights from the 30 days

---

## Content Mix Summary

| Type | Count |
|---|---|
| Product announcements | 4 (days 1, 18, 29, 30) |
| Technical deep dives | 7 (days 2, 10, 16, 19, 23, 24, 28) |
| Integration tutorials | 5 (days 4, 6, 13, 15, 22) |
| Benchmark content | 4 (days 1, 3, 18, 28) |
| Feature showcases | 4 (days 5, 11, 12, 27) |
| Behind the scenes | 3 (days 10, 17, 19) |
| Community engagement | 5 (days 7, 14, 20, 29, 30) |
| Practical tips | 3 (days 8, 25, 26) |
| Roadmap/progress | 2 (days 21, 29) |

## Hashtag Strategy

Primary (use on every post): #LocalLLM #AppleSilicon
Secondary (rotate): #MLX #AI #OpenSource #BuildInPublic
Contextual: #LangChain #RAG #FastAPI (when relevant to the topic)

## Posting Schedule

- Weekdays: Technical content, tutorials, benchmarks (higher engagement)
- Weekends: Integration tutorials, weekend projects (casual tone)
- Monday: Recap or fresh announcement to start the week
- Friday: Community engagement or Q&A prompt
