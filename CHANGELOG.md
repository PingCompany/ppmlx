# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-03-29

### Added
- Unified interactive TUI tables across all commands (list, ps, pull, rm, serve, run, launch)
- Read-only TUI browser with search and keyboard navigation for `list` and `ps`
- First-run analytics opt-in prompt (analytics disabled by default)
- Configurable CORS origins via `PPMLX_CORS_ORIGINS` env var
- Pydantic validation on all API request bodies (bounds checking, batch limits)
- Interactive Swagger docs at `/docs` and ReDoc at `/redoc`
- Version sync test (pyproject.toml vs __init__.py)
- ruff linter and mypy type checker in dev tooling
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- GitHub issue templates (bug report, feature request) and PR template
- "Requirements" and "ppmlx vs Ollama" sections in README

### Changed
- All model views use consistent table layout with column headers, status flags, and aligned columns
- Analytics default changed from opt-out to opt-in
- Removed `allow_credentials=True` from CORS middleware
- `uv.lock` now tracked in git (removed from .gitignore)
- Dropped `questionary` dependency (replaced by `prompt-toolkit`)

### Removed
- Rich static `_model_table` (replaced by TUI browser)

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
