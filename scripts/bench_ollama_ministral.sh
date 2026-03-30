#!/usr/bin/env bash
# Benchmark: ollama + ministral-3:14b
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "ministral-3:14b" "ollama" \
    "http://localhost:11434/v1/chat/completions" \
    "ollama:ministral-3:14b"
