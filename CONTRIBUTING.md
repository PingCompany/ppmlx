# Contributing to ppmlx

Thanks for your interest in contributing to ppmlx! This guide covers everything you need to get started.

## Dev Environment Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/ppmlx.git
cd ppmlx

# Install dependencies (requires uv)
uv sync --python 3.11
```

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests use MLX stubs so they run on any machine (no Apple Silicon / GPU required).

## Coding Conventions

- **Type hints** on all function signatures.
- **Pydantic** for request/response validation (not raw dicts).
- **No `print()` in library code** -- use `logging` or Typer's `echo`/`rich` output. `print()` is fine in CLI-only paths (`cli.py`).
- **Format with ruff** before committing.
- Keep modules focused: inference logic in `engine.py`, HTTP routes in `server.py`, CLI in `cli.py`.

## Pull Request Process

1. **Fork** the repo and create a feature branch from `main`.
2. **Make your changes** -- keep PRs focused on a single concern.
3. **Add or update tests** for any new functionality.
4. **Run the test suite** to make sure everything passes.
5. **Open a PR** against `main` with a clear description of what and why.

A maintainer will review your PR. Small, well-scoped PRs are reviewed faster.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add vision model support
fix: correct streaming for thinking models
docs: update API endpoint table
chore: bump mlx-lm dependency
test: add coverage for quantize command
```

Keep the subject line under 72 characters. Use the body for additional context if needed.

## Questions?

Open a discussion or issue on GitHub, or email rafal@ppmlx.dev.
