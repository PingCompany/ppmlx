# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
