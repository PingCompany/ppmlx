#!/usr/bin/env bash
# Benchmark: ppmlx + glm-4.7-flash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "glm-4.7-flash" "ppmlx" \
    "http://localhost:6767/v1/chat/completions" \
    "ppmlx/glm-4.7-flash"
