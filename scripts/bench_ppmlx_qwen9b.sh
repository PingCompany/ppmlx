#!/usr/bin/env bash
# Benchmark: ppmlx + qwen3.5:9b
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "qwen3.5:9b" "ppmlx" \
    "http://localhost:6767/v1/chat/completions" \
    "ppmlx:qwen3.5:9b"
