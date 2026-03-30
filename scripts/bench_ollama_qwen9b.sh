#!/usr/bin/env bash
# Benchmark: ollama + qwen3.5:9b
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "qwen3.5:9b" "ollama" \
    "http://localhost:11434/v1/chat/completions" \
    "ollama:qwen3.5:9b"
