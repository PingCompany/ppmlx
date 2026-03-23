#!/usr/bin/env bash
# Uninstall and reinstall ppmlx from local source via uv tool
# Usage: ./scripts/reinstall.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uninstalling ppmlx..."
uv tool uninstall ppmlx 2>/dev/null || true

echo "Installing from ${REPO_ROOT}..."
uv tool install "${REPO_ROOT}" --python 3.11

echo "Done. $(ppmlx --version)"
