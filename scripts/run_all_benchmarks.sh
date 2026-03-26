#!/usr/bin/env bash
# run_all_benchmarks.sh — Orchestrate all ppmlx vs ollama benchmark pairs
#
# For each model: start ppmlx -> run ppmlx bench -> kill ppmlx ->
#                 start ollama -> run ollama bench -> kill ollama
#
# Usage: ./scripts/run_all_benchmarks.sh [--models glm,qwen] [--skip-ollama] [--skip-ppmlx]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colours ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Config ───────────────────────────────────────────────────────────────
PPMLX_PORT=6767
OLLAMA_PORT=11434
HEALTH_TIMEOUT=60          # seconds to wait for server health
HEALTH_INTERVAL=2          # seconds between health checks
COOLDOWN=5                 # seconds between benchmarks

# All model benchmark pairs (model_key maps to script suffix)
ALL_MODELS=("glm" "qwen" "gptoss")

# Model display names for logging
declare -A MODEL_NAMES=(
    [glm]="glm-4.7-flash"
    [qwen]="qwen3.5:27b"
    [gptoss]="gpt-oss:20b"
)

# ── Argument parsing ────────────────────────────────────────────────────
SKIP_PPMLX=false
SKIP_OLLAMA=false
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            IFS=',' read -ra SELECTED_MODELS <<< "$2"
            shift 2
            ;;
        --skip-ppmlx)
            SKIP_PPMLX=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --help|-h)
            printf "Usage: %s [--models glm,qwen,gptoss] [--skip-ppmlx] [--skip-ollama]\n" "$0"
            printf "\nOptions:\n"
            printf "  --models LIST    Comma-separated model keys: glm, qwen, gptoss\n"
            printf "  --skip-ppmlx     Skip ppmlx benchmarks\n"
            printf "  --skip-ollama    Skip ollama benchmarks\n"
            exit 0
            ;;
        *)
            printf "${RED}Unknown option: %s${RESET}\n" "$1"
            exit 1
            ;;
    esac
done

# Default to all models if none specified
if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

# ── Helpers ──────────────────────────────────────────────────────────────

log_step() { printf "\n${BOLD}${CYAN}>>> %s${RESET}\n" "$*"; }
log_ok()   { printf "${GREEN}    %s${RESET}\n" "$*"; }
log_warn() { printf "${YELLOW}    %s${RESET}\n" "$*"; }
log_err()  { printf "${RED}    %s${RESET}\n" "$*"; }

# Check if a port is in use
port_in_use() {
    local port="$1"
    lsof -i :"$port" -sTCP:LISTEN &>/dev/null
}

# Wait for a server to become healthy
wait_for_health() {
    local url="$1"
    local name="$2"
    local elapsed=0

    printf "${DIM}    Waiting for %s to be ready...${RESET}" "$name"
    while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
        if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
            printf "\r${GREEN}    %s is ready (%ds)${RESET}              \n" "$name" "$elapsed"
            return 0
        fi
        sleep "$HEALTH_INTERVAL"
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done
    printf "\r${RED}    %s did not become healthy within %ds${RESET}    \n" "$name" "$HEALTH_TIMEOUT"
    return 1
}

# Kill process on a given port
kill_server_on_port() {
    local port="$1"
    local name="$2"

    if port_in_use "$port"; then
        printf "${DIM}    Stopping %s on port %d...${RESET}\n" "$name" "$port"
        # Find and kill processes listening on the port
        local pids
        pids=$(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if port_in_use "$port"; then
                echo "$pids" | xargs kill -9 2>/dev/null || true
                sleep 1
            fi
        fi
        if port_in_use "$port"; then
            log_err "Failed to free port $port"
            return 1
        fi
        log_ok "Stopped $name"
    fi
    return 0
}

# Start ppmlx server in background
start_ppmlx() {
    local model="$1"
    log_step "Starting ppmlx server (model: $model)"

    if port_in_use "$PPMLX_PORT"; then
        log_warn "Port $PPMLX_PORT already in use, stopping existing server"
        kill_server_on_port "$PPMLX_PORT" "ppmlx" || return 1
    fi

    cd "$PROJECT_DIR"
    uv run ppmlx serve --model "$model" --port "$PPMLX_PORT" &>/dev/null &
    local ppmlx_pid=$!
    echo "$ppmlx_pid" > /tmp/bench_ppmlx.pid

    # Wait for health
    if ! wait_for_health "http://localhost:${PPMLX_PORT}/health" "ppmlx"; then
        kill "$ppmlx_pid" 2>/dev/null || true
        return 1
    fi

    # Warm up: send a quick request to load the model into memory
    printf "${DIM}    Warming up model...${RESET}"
    curl -sf --max-time 60 "http://localhost:${PPMLX_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1}" \
        >/dev/null 2>&1 || true
    printf "\r${GREEN}    Model warmed up${RESET}              \n"

    return 0
}

# Stop ppmlx server
stop_ppmlx() {
    log_step "Stopping ppmlx server"
    kill_server_on_port "$PPMLX_PORT" "ppmlx"
    rm -f /tmp/bench_ppmlx.pid
}

# Start ollama server
start_ollama() {
    local model="$1"
    log_step "Starting ollama server (model: $model)"

    if ! command -v ollama &>/dev/null; then
        log_err "ollama not found in PATH"
        return 1
    fi

    if ! port_in_use "$OLLAMA_PORT"; then
        ollama serve &>/dev/null &
        echo $! > /tmp/bench_ollama.pid
    else
        log_ok "ollama already running on port $OLLAMA_PORT"
    fi

    if ! wait_for_health "http://localhost:${OLLAMA_PORT}/api/tags" "ollama"; then
        return 1
    fi

    # Pre-load the model
    printf "${DIM}    Pre-loading model %s...${RESET}" "$model"
    ollama run "$model" "Hi" --nowordwrap >/dev/null 2>&1 || true
    printf "\r${GREEN}    Model %s loaded${RESET}              \n" "$model"

    return 0
}

# Stop ollama server
stop_ollama() {
    log_step "Stopping ollama server"
    if [[ -f /tmp/bench_ollama.pid ]]; then
        kill_server_on_port "$OLLAMA_PORT" "ollama"
        rm -f /tmp/bench_ollama.pid
    else
        # We didn't start it, so don't kill it
        log_ok "ollama was already running, leaving it alone"
    fi
}

# Verify a benchmark script exists
check_script() {
    local script="$1"
    if [[ ! -f "$script" ]]; then
        log_err "Script not found: $script"
        return 1
    fi
    if [[ ! -x "$script" ]]; then
        chmod +x "$script"
    fi
    return 0
}

# ── Pre-flight checks ───────────────────────────────────────────────────

log_step "Pre-flight checks"

# Check required tools
for tool in curl jq python3 lsof; do
    if ! command -v "$tool" &>/dev/null; then
        log_err "Required tool not found: $tool"
        exit 1
    fi
done
log_ok "All required tools available (curl, jq, python3, lsof)"

# Check benchmark scripts exist
MISSING=false
for model_key in "${SELECTED_MODELS[@]}"; do
    if [[ "$SKIP_PPMLX" != "true" ]]; then
        if ! check_script "$SCRIPT_DIR/bench_ppmlx_${model_key}.sh"; then
            MISSING=true
        fi
    fi
    if [[ "$SKIP_OLLAMA" != "true" ]]; then
        if ! check_script "$SCRIPT_DIR/bench_ollama_${model_key}.sh"; then
            MISSING=true
        fi
    fi
done
if [[ "$MISSING" == "true" ]]; then
    log_err "Some benchmark scripts are missing. Aborting."
    exit 1
fi
log_ok "All benchmark scripts found"

# Ensure results directory exists
mkdir -p "$PROJECT_DIR/benchmark_results"

# ── Main loop ────────────────────────────────────────────────────────────

TOTAL=${#SELECTED_MODELS[@]}
CURRENT=0
FAILED=()

printf "\n${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}\n"
printf "${BOLD}║          ppmlx vs Ollama — Full Benchmark Suite             ║${RESET}\n"
printf "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}\n"
printf "${DIM}Date:   %s${RESET}\n" "$(date)"
printf "${DIM}Models: %s${RESET}\n" "${SELECTED_MODELS[*]}"
printf "${DIM}Skip ppmlx: %s  |  Skip ollama: %s${RESET}\n" "$SKIP_PPMLX" "$SKIP_OLLAMA"

for model_key in "${SELECTED_MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    model_name="${MODEL_NAMES[$model_key]:-$model_key}"

    printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "${BOLD}  Model %d/%d: %s${RESET}\n" "$CURRENT" "$TOTAL" "$model_name"
    printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

    # ── ppmlx benchmark ──────────────────────────────────────────────
    if [[ "$SKIP_PPMLX" != "true" ]]; then
        if start_ppmlx "$model_name"; then
            log_step "Running ppmlx benchmark for $model_name"
            if ! bash "$SCRIPT_DIR/bench_ppmlx_${model_key}.sh"; then
                log_err "ppmlx benchmark failed for $model_name"
                FAILED+=("ppmlx:$model_name")
            fi
            stop_ppmlx
        else
            log_err "Could not start ppmlx for $model_name"
            FAILED+=("ppmlx:$model_name")
        fi

        # Cooldown between backends
        if [[ "$SKIP_OLLAMA" != "true" ]]; then
            printf "${DIM}    Cooling down for %ds...${RESET}\n" "$COOLDOWN"
            sleep "$COOLDOWN"
        fi
    fi

    # ── ollama benchmark ─────────────────────────────────────────────
    if [[ "$SKIP_OLLAMA" != "true" ]]; then
        if start_ollama "$model_name"; then
            log_step "Running ollama benchmark for $model_name"
            if ! bash "$SCRIPT_DIR/bench_ollama_${model_key}.sh"; then
                log_err "ollama benchmark failed for $model_name"
                FAILED+=("ollama:$model_name")
            fi
            stop_ollama
        else
            log_err "Could not start ollama for $model_name"
            FAILED+=("ollama:$model_name")
        fi
    fi

    # Cooldown between models
    if [[ $CURRENT -lt $TOTAL ]]; then
        printf "${DIM}    Cooling down for %ds before next model...${RESET}\n" "$COOLDOWN"
        sleep "$COOLDOWN"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────

printf "\n${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}\n"
printf "${BOLD}║                    Benchmark Complete                        ║${RESET}\n"
printf "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}\n"

if [[ ${#FAILED[@]} -eq 0 ]]; then
    printf "${GREEN}All benchmarks completed successfully.${RESET}\n"
else
    printf "${RED}Failed benchmarks:${RESET}\n"
    for f in "${FAILED[@]}"; do
        printf "${RED}  - %s${RESET}\n" "$f"
    done
fi

printf "\n${DIM}Results in: %s${RESET}\n" "$PROJECT_DIR/benchmark_results"
printf "${DIM}Run 'python3 scripts/parse_results.py' to generate a comparison table.${RESET}\n\n"
