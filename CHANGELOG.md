# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.10.0] - 2026-04-07

### Added
- **TurboQuant KV-cache compression**: 3-4 bit KV-cache quantization via PolarQuant + QJL (Quantized Johnson-Lindenstrauss), reducing memory usage ~6x with zero accuracy loss. Enable with `--kv-quant turboquant` on `ppmlx serve` or `[kv_cache] quantize = "turboquant"` in config.toml. Based on Google Research paper (ICLR 2026, arxiv 2504.19874).
- **`[kv_cache]` config section**: `quantize`, `bits`, `qjl`, `qjl_dim` settings for KV-cache compression

## [0.9.1] - 2026-04-07

### Added
- **`max_output_chars`**: global cap (default 20k) on all tool output to protect model context — configurable in TUI and `[agent]` config

### Fixed
- **Download progress bar**: rewritten from filesystem polling to direct tqdm byte-counter hook — speed, ETA, and downloaded amount now update in real time

### Changed
- Download progress uses custom `_ProgressTqdm` class that captures exact bytes from HuggingFace HTTP chunks instead of scanning disk

## [0.9.0] - 2026-04-07

### Added
- **Agent tools**: `search_files` (grep across files), `find_files` (recursive glob), `patch_file` (edit fragment without rewriting whole file)
- **Delegate tool**: `delegate` sends tasks to Claude Code CLI or Codex CLI asynchronously; `check_task` retrieves results later
- **Permission levels**: `readonly`, `write`, `execute`, `full` — controls which tools the agent can use, with confirmation prompts on dangerous operations
- **Agent config in TUI**: `ppmlx config` now has an Agent section — max read lines, max iterations, temperature, sandbox, permission level
- **`read_file` line ranges**: `start_line`/`end_line` parameters + automatic truncation at configurable `max_read_lines` (default 200)
- **Voice system prompt**: voice mode auto-injects a conversational system prompt (short answers, progressive disclosure, no markdown)
- **`[agent]` config section**: `max_read_lines`, `max_iterations`, `temperature`, `sandbox`, `permission_level` in `~/.ppmlx/config.toml`

### Fixed
- **TTS boundary glitches**: single continuous `OutputStream` replaces per-sentence `sd.play()`; 10ms fade-in/fade-out on each segment
- **TTS not interruptible**: audio written in 100ms chunks so Ctrl+C fires between them; `_stop` event halts generation thread
- **STT cutting off speech**: adaptive silence threshold (2× noise floor), grace period (1.0s after silence), increased default `silence_duration` to 2.0s, 300ms trailing buffer in trim
- **TTS noise/echo**: noise gate (zeros samples <1% of peak), gain capped at 6× to prevent noise amplification
- **TTS drops inline code**: backtick content now kept as spoken text instead of being stripped
- **Rich markup crash**: tool output with `[brackets]` escaped before printing
- **TUI save crash**: `tts_voice = None` no longer written to TOML (was `NoneType is not TOML serializable`)
- **Ctrl+C everywhere**: works during recording, TTS playback, agent execution, and PTT key wait; double Ctrl+C exits

### Changed
- Default STT silence_duration: 1.5s → 2.0s
- Noise floor multiplier: 3× → 2× (less aggressive threshold calibration)
- TTS speech segments: strategic pauses — 0.25s between sentences, 0.40s between list items, 0.60s between paragraphs

## [0.8.3] - 2026-04-07

### Added
- **Overlapped TTS playback**: generate sentence N+1 while playing sentence N for ~2x faster voice responses
- **Sentence-level TTS streaming**: speak sentence by sentence instead of waiting for full response
- **Voice config in TUI**: `ppmlx config` now has a Voice section — STT/TTS model, voice, speed, volume, PTT mode/key, silence duration
- **TTS speed control**: `tts_speed` setting (0.8x–1.5x) adjusts playback rate without re-generation
- **TTS volume normalization**: `tts_volume` setting (default 1.10) with DC offset removal for cleaner audio
- **Auto-download voice models**: installing the voice addon now pre-downloads default STT and TTS models
- **HF token propagation**: `hf_token` from ppmlx config is now passed to huggingface_hub in voice mode (fixes unauthenticated download warnings)
- **Text cleaning for TTS**: strips markdown, code blocks, links, and formatting before speech generation

### Fixed
- **Voice exit handling**: auto-silence mode now shows a text prompt — Enter for voice, or type `/exit` to quit; double Ctrl+C also exits
- **Recording stops on silence before speech**: fixed premature recording stop by waiting for speech before silence detection
- **Hidden HF progress bars**: suppressed noisy `tqdm`/`huggingface_hub` output during model loading with clean Rich spinners instead
- **STT model compatibility**: switched default from `whisper-large-v3-turbo-4bit` (incompatible `model.safetensors`) to `whisper-large-v3-turbo-q4` (`weights.npz`)
- **`NameError: log`** crash on STT preload failure in agent CLI

### Changed
- Default STT model: `whisper-large-v3-turbo` → `whisper-large-v3-turbo-q4` (~400 MB, 4x smaller)

## [0.8.2] - 2026-04-07

### Added
- **Push-to-talk** voice input: hold a key to record, release to send — configurable via `[voice] ptt_mode` and `ptt_key` in `~/.ppmlx/config.toml`
- `[voice]` config section: `stt_model`, `tts_model`, `tts_voice`, `ptt_mode`, `ptt_key`, `silence_threshold`, `silence_duration`
- `pynput` added to `ppmlx[voice]` extras for PTT key detection

### Fixed
- **"Thank you" hallucination**: pass `condition_on_previous_text=False` and `no_speech_threshold=0.6` to Whisper; skip clips < 0.3 s
- **ffmpeg not required for live recording**: pass numpy array directly to mlx_whisper instead of writing a temp WAV (ffmpeg only needed for `transcribe_file`)
- **Tokenizer regex warning silenced**: suppress the "incorrect regex pattern" log on Qwen/Mistral-derived models when transformers has the `_patch_mistral_regex` bug
- **`AttributeError` on model load**: catch `_patch_mistral_regex` crash in newer transformers and fall back gracefully
- Installer now shows a spinner + captures subprocess output during addon install/uninstall

## [0.8.1] - 2026-04-07

### Fixed
- `agent.py` crash: `too many values to unpack` — now uses `result.text` from `GenerateResult`
- Mistral tokenizer regex warning: pass `fix_mistral_regex=True` when loading model

### Added
- Generation stats after each response (`/set verbose` in REPL, or `[ui] show_stats = true` in config)
- Markdown rendering for assistant output (`/set markdown` in REPL, or `[ui] markdown = true` in config)
- Option+letter key bindings for Polish diacritical characters in REPL (ą ę ó ź ż ć ń ś ł)
- `ppmlx/render.py` — shared `stream_and_collect()` and `print_response()` helpers

### Changed
- **Slimmer base install**: removed `mlx-vlm`, `posthog`, `sse-starlette`, `setproctitle` from core deps
  - `mlx-vlm` → `pip install ppmlx[vision]`
  - `posthog` → `pip install ppmlx[analytics]`
  - `sse-starlette` removed (was unused)
  - `setproctitle` removed (had safe fallback already)
- Added `vision` and `analytics` optional groups + `all` meta-group
- `ppmlx install` now also shows `vision` and `analytics` components

## [0.8.0] - 2026-04-07

### Added
- **Agent mode** (`ppmlx agent`): interactive agent REPL with tool execution
  - Built-in tools: `bash`, `read_file`, `write_file`, `list_files`
  - `--sandbox` flag for read-only mode (disables write/destructive commands)
  - Live tool execution display with color-coded results
  - Multi-turn conversations with full context
  - Configurable max iterations, temperature, model
- **Voice mode** (`ppmlx agent --voice`): voice-powered agent
  - Push-to-talk with automatic silence detection
  - Speech-to-text via Whisper on MLX (`mlx-whisper`)
  - Text-to-speech via Voxtral TTS on MLX (`mlx-audio`)
  - All processing runs locally on Apple Silicon — fully offline
  - Configurable STT/TTS models and voices
  - `--stt-model`, `--tts-model`, `--tts-voice` options
- `ppmlx/voice.py` — `VoiceInput` (STT) and `VoiceOutput` (TTS) classes
- `[voice]` optional dependency group: `pip install ppmlx[voice]`

## [0.7.1] - 2026-04-07

### Fixed
- **GPU crash**: retry path in `generate()` now acquires `_generate_lock` (prevented Metal crash on concurrent requests)
- **Corrupted output**: retry no longer reuses stale `prompt_cache` from the original prompt
- **3 NameError bugs**: `except Exception:` → `except Exception as exc:` in completions, responses, and anthropic error handlers
- **Router score inflation**: image detection added +2 per message instead of +2 total
- **Router false positives**: removed `=>`, `.map(`, `import` from code patterns (matched natural language)
- **Silent config failures**: TOML parse errors now log a warning instead of silently reverting to defaults
- **`_parse_bool`**: empty string and `"off"` now correctly evaluate to `False`
- **Benchmark stddev**: uses Bessel's correction (n-1) for accurate sample standard deviation

### Changed
- Config loaded once per process (`_get_config()` cache) — was 3× disk reads per request
- Extracted `_build_mlx_kwargs()` — eliminates 20 lines of duplicated kwargs construction
- Extracted `_resolve_thinking()` — identical 5-line block appeared 4× in server.py
- Extracted `_print_comparison_table()` — shared by `print_comparison` and `print_speculative_comparison`
- `PromptCacheStore.is_enabled` property replaces private `_max_entries` access
- `_skip_copy` flag on `PromptCacheStore.store()` eliminates one deep-copy (~400 MB for 7B models)
- `_apply_prompt_cache()` returns prompt token count (avoids redundant re-tokenization)
- `PromptCacheStore.stats()` includes estimated memory usage per entry
- Prompt cache key uses SHA-1 instead of `hash()` (prevents silent hash collisions)
- `all_aliases()` hoisted out of `list_local_models` loop (was N disk reads)
- `hashlib` import moved to module level
- `_ensure_chat_template` simplified and adds `log.debug` on failures
- Router logger corrected from `ppmlx.engine` to `ppmlx.router`

### Removed
- `_get_or_load()` trivial wrapper (5 call sites inlined to `self.load()`)
- `_get_text_stream()` and `_text_generate()` dead code (48 lines, zero callers)
- Duplicate `RouterConfig` from `router.py` (single source in `config.py`)
- Dead `ScenarioStats.scenario` field
- 3 duplicate comment blocks

## [0.7.0] - 2026-04-07

### Added
- **Smart model router**: send `model: "auto"` and ppmlx picks the right model
  - Complexity analysis: scores requests 1-10 based on prompt length, code presence,
    reasoning keywords, tool count, multi-turn depth, images, and output length
  - Simple requests (greeting, short questions) → small fast model
  - Complex requests (architecture, debugging, long code) → large capable model
  - Configurable via `[router]` in config.toml (threshold, small/large model)
  - Environment variables: `PPMLX_ROUTER_ENABLED`, `PPMLX_ROUTER_THRESHOLD`,
    `PPMLX_ROUTER_SMALL_MODEL`, `PPMLX_ROUTER_LARGE_MODEL`
- `ppmlx/router.py` — `analyze_complexity()` and `route()` functions

## [0.6.0] - 2026-04-07

### Added
- **Prompt caching**: KV-cache reuse across requests with shared prefixes
  - Automatic prefix matching — system prompts prefilled once, reused for every request
  - Conversation continuation — each turn reuses the full prefix from the previous turn
  - LRU eviction with configurable `prompt_cache_limit` (default: 4 entries)
  - Transparent to API — no client changes needed, works with all endpoints
  - Disabled automatically when speculative decoding is active
- `prompt_cache_limit` config option (`[defaults]`, env `PPMLX_PROMPT_CACHE_LIMIT`)
- `ppmlx/prompt_cache.py` — thread-safe `PromptCacheStore` with prefix matching
- Prompt cache cleared on model unload (prevents stale KV states)

## [0.5.0] - 2026-04-07

### Added
- Auto draft model pairing for speculative decoding (`DRAFT_PAIRS` registry)
- `get_draft_model()` auto-detects best draft model for a target (e.g. qwen3.5:9b -> 0.8b)
- `auto_speculative` config option (`[defaults]`, env `PPMLX_AUTO_SPECULATIVE`)
- User-defined draft pairs via `~/.ppmlx/draft_pairs.json`
- Server auto-speculative: automatically pairs draft models when `auto_speculative=true`
- `ppmlx config bench --speculative` flag for normal vs speculative side-by-side comparison
- `ppmlx config bench --draft-model` and `--speculative-tokens` options
- `print_speculative_comparison()` with speedup percentages and multipliers
- Gemma-4 channel token normalization (`_Gemma4Normalizer` state machine)
- KV-cache patch for `RotatingKVCache` crash when prompt exceeds sliding window
- Thinking model detection for Gemma-4-style `<|think|>` markers
- Registry fetch module
- TUI config refactor

## [0.4.1] - 2026-03-31

### Changed
- Deduplicate `_resolve_model_path` across engine modules into `models.py`
- Extract shared think-tag stream processor, eliminating ~100 lines of duplication
- Remove `setproctitle` dependency

### Fixed
- Incorrect `reasoning_text` assignment in streaming responses
- `_flush_port` now verifies PID belongs to ppmlx before killing (H3)
- Vision engine rejects `file://` URLs and bare paths from API requests (C3)

### Security
- CORS defaults to localhost-only; configurable via `cors_origins` in config.toml (C2)
- Request body size limit middleware (default 10 MB, configurable) (H1)
- Server-side `max_tokens` cap (default 32768, configurable) (H2)
- Embedding input limited to 256 texts per request (H4)
- WebSocket message size limit (10 MB) (H5)
- Removed debug JSONL logging to `/tmp/` (C1)

### Added
- `SECURITY_AUDIT.md` documenting all findings and fixes
- Homebrew formula with `arch: :arm64` constraint and auto-update workflow

## [0.4.0] - 2026-03-30

### Added
- Thinking/reasoning model support: `think` and `reasoning_budget` API parameters
- `reasoning_effort` mapping (low/medium/high) to reasoning budget tokens
- Thinking metrics tracking in SQLite DB with migration
- Streaming thinking/reasoning delta support in chat completions
- Empty-answer retry logic for thinking models in engine
- `ppmlx logs` and `ppmlx stats` CLI commands for log analysis
- `ppmlx config --thinking`, `--reasoning-budget`, `--effort-base`, `--max-tools-tokens` flags
- `[thinking]` section in config (`enabled`, `default_reasoning_budget`, `effort_base`)
- Thinking configuration panel in TUI

## [0.3.0] - 2026-03-28

### Added
- First-run analytics opt-in prompt (analytics disabled by default)
- Configurable CORS origins via `PPMLX_CORS_ORIGINS` env var
- Pydantic validation on all API request bodies (bounds checking, batch limits)
- Interactive Swagger docs at `/docs` and ReDoc at `/redoc`
- Network binding warning when server exposed on non-localhost
- Version sync test (pyproject.toml vs __init__.py)
- ruff linter and mypy type checker in CI pipeline
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- GitHub issue templates (bug report, feature request) and PR template
- "Requirements" and "ppmlx vs Ollama" sections in README

### Changed
- Analytics default changed from opt-out to opt-in
- API error responses now return generic messages (no internal details leaked)
- Removed `allow_credentials=True` from CORS middleware
- `uv.lock` now tracked in git (removed from .gitignore)

### Fixed
- Unused variables and imports flagged by ruff

## [0.2.0] - 2026-03-27

### Added
- Analytics module with privacy-first design (opt-in, data sanitization, DNT support)
- First-run prompt asking users to opt in to anonymous analytics
- Curses-based TUI model picker with search/filter
- Open WebUI launcher support
- Responses API endpoint (`/v1/responses`) for Codex compatibility
- Anthropic Messages API endpoint (`/v1/messages`)
- Vision model support via mlx-vlm
- Model quantization command (`ppmlx quantize`)
- SQLite request logging and metrics (`/metrics` endpoint)
- Tool calling support with awareness injection
- Configurable tool awareness prompts
- Interactive model selection for serve/run/rm commands
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md

### Changed
- Expanded core tool list with case-insensitive matching
- Improved streaming with thinking model support (`<think>` blocks)
- Generic error messages in API responses (no internal details leaked)
- Pydantic validation on all API request bodies

### Removed
- Debug request logging to `/tmp`

## [0.1.0] - 2026-03-20

### Added
- Initial release
- CLI with serve, pull, run, list, ps, rm, config commands
- OpenAI-compatible API server (chat completions, completions, embeddings)
- Model registry with 168+ pre-configured models
- Homebrew formula
- Astro marketing website
