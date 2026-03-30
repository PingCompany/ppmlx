# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
