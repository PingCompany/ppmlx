#!/usr/bin/env bash
# Benchmark: ollama + glm-4.7-flash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "glm-4.7-flash" "ollama" \
    "http://localhost:11434/v1/chat/completions" \
    "ollama/glm-4.7-flash"
