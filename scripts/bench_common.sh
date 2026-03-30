#!/usr/bin/env bash
# bench_common.sh — shared functions for ppmlx vs ollama benchmarks
# Source this file, don't run it directly.
#
# Scenarios:
#   S1 Simple      — short prompt → short answer (TTFT, baseline tok/s)
#   S2 Complex     — short prompt → long answer (sustained generation throughput)
#   S3 LongContext — large prompt → medium answer (prefill/prompt processing speed)
#   S4 Agentic     — real multi-turn with tools via pi (wall-time, tool use)
#
# Run order: for each run R in 1..RUNS → S1, S2, S3, S4

RUNS=3
MAX_TOKENS=8192
RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/benchmark_results"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$RESULTS_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Prompts ───────────────────────────────────────────────────────────────

SIMPLE_PROMPT="Explain the difference between TCP and UDP in exactly 3 bullet points. Be precise and technical."

COMPLEX_PROMPT="Design a rate limiter for an API gateway. Describe:
1. The algorithm you'd choose and why (token bucket vs sliding window vs fixed window)
2. Data structures needed
3. How to handle distributed deployment (multiple gateway instances)
4. Edge cases (clock skew, burst traffic, graceful degradation)
Provide pseudocode for the core logic."

# S3: Long context — embed real source code (~4K tokens) and ask for analysis.
# Built dynamically from engine.py so the prompt is always realistic.
_build_long_context_prompt() {
    local engine_src
    engine_src=$(cat "$PROJECT_DIR/ppmlx/engine.py" 2>/dev/null || echo "# file not found")
    printf 'Analyze the following Python module. Provide:\n1. A summary of its architecture and main classes\n2. Thread-safety analysis — are there race conditions?\n3. Performance bottlenecks and optimization suggestions\n4. Any bugs or edge cases that could cause failures\n\n```python\n%s\n```' "$engine_src"
}
LONG_CONTEXT_PROMPT="$(_build_long_context_prompt)"

# S4: Agentic — prompt that requires pi to use tools (read files, explore).
AGENTIC_PROMPT="Review the ppmlx project in the current directory for security vulnerabilities. Focus on the server endpoints and how user input is handled. Read the relevant source files (especially server.py and engine.py) and provide a detailed security audit report with specific line references."

# ── API call helper (streaming — measures TTFT) ───────────────────────────
#
# Returns JSON: {ms, tokens, tok_s, ttft_ms, error}

call_api() {
    local url="$1"
    local model="$2"
    local prompt="$3"
    local label="$4"

    local body
    body=$(jq -n \
        --arg model "$model" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        '{
            model: $model,
            messages: [{"role": "user", "content": $prompt}],
            max_tokens: $max_tokens,
            temperature: 0.0,
            stream: true
        }')

    python3 -c "
import sys, json, time, urllib.request

url  = sys.argv[1]
data = sys.argv[2].encode()

req = urllib.request.Request(url, data=data, headers={
    'Content-Type': 'application/json',
    'Authorization': 'Bearer local',
})

start_ns  = time.time_ns()
ttft_ns   = None
chunks    = 0
full_text = []
usage     = {}  # from final chunk with usage info

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b''
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b'\n\n' in buf:
                line, buf = buf.split(b'\n\n', 1)
                line = line.decode('utf-8', errors='replace').strip()
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload == '[DONE]':
                    break
                try:
                    obj     = json.loads(payload)
                    # Capture usage from any chunk that has it
                    if 'usage' in obj and obj['usage']:
                        usage = obj['usage']
                    delta   = obj.get('choices', [{}])[0].get('delta', {})
                    # Some backends (ollama+GLM) put thinking in delta.reasoning
                    # and only surface final answer in delta.content.
                    # Count both for TTFT/throughput; prefer content, fallback reasoning.
                    content = delta.get('content', '') or delta.get('reasoning', '')
                    if content:
                        if ttft_ns is None:
                            ttft_ns = time.time_ns()
                        chunks += 1
                        full_text.append(content)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
except Exception as e:
    print(json.dumps({'ms':0,'tokens':0,'tok_s':0,'ttft_ms':0,'error':True}))
    sys.exit(0)

end_ns     = time.time_ns()
elapsed_ms = round((end_ns - start_ns) / 1_000_000)
ttft_ms    = round((ttft_ns - start_ns) / 1_000_000) if ttft_ns else 0
# Prefer real token count from usage; fall back to chunk count
completion_tokens = usage.get('completion_tokens', 0) or chunks
tok_s      = round(completion_tokens / (elapsed_ms / 1000), 2) if elapsed_ms > 0 and completion_tokens > 0 else 0

print(json.dumps({'ms': elapsed_ms, 'tokens': completion_tokens, 'tok_s': tok_s,
                  'ttft_ms': ttft_ms, 'error': False}))
" "$url" "$body" 2>/dev/null
}

# ── Agentic call helper (direct API, multi-turn with tools) ────────────────
#
# Makes non-streaming API calls with tools defined.  When the model responds
# with finish_reason=tool_calls, provides a simulated tool result and
# continues.  Measures: ms, turns, tool_calls, answer_chars.
#
# Returns JSON: {ms, tokens, turns, tool_calls, tool_names, answer_chars, error}

call_agentic() {
    local url="$1"
    local model="$2"
    local prompt="$3"

    _BENCH_AG_URL="$url" _BENCH_AG_MODEL="$model" _BENCH_AG_PROMPT="$prompt" \
    python3 - 2>/dev/null << 'PYEOF'
import os, sys, json, time, urllib.request

EMPTY = {'ms':0,'tokens':0,'turns':0,
         'tool_calls':0,'tool_names':[],'answer_chars':0,'error':True}

url    = os.environ.get('_BENCH_AG_URL', '')
model  = os.environ.get('_BENCH_AG_MODEL', '')
prompt = os.environ.get('_BENCH_AG_PROMPT', '')
if not url or not model or not prompt:
    print(json.dumps(EMPTY)); sys.exit(0)

MAX_TURNS = 5

tools = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]

# Simulated tool results (realistic enough for the model to continue)
TOOL_RESULTS = {
    "bash": '{"stdout": "total 48\\ndrwxr-xr-x  12 user staff  384 Mar 25 10:00 .\\n-rw-r--r--   1 user staff 1234 Mar 25 09:55 server.py\\n-rw-r--r--   1 user staff  890 Mar 25 09:50 engine.py\\n-rw-r--r--   1 user staff  456 Mar 25 09:45 config.py", "exit_code": 0}',
    "read_file": '{"content": "from fastapi import FastAPI, Request\\nimport json\\n\\napp = FastAPI()\\n\\n@app.post(\\"/v1/chat/completions\\")\\nasync def chat(request: Request):\\n    body = await request.json()\\n    # TODO: validate input\\n    model = body.get(\\"model\\")\\n    messages = body.get(\\"messages\\", [])\\n    return {\\"choices\\": [{\\"message\\": {\\"content\\": \\"hello\\"}}]}\\n"}',
}

messages = [{"role": "user", "content": prompt}]
total_tokens = 0
all_tool_calls = []
answer_text = ""

start_ns = time.time_ns()
try:
    for turn in range(MAX_TURNS):
        body = json.dumps({
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": 2048,
            "temperature": 0.0,
            "stream": False,
        }).encode()

        req = urllib.request.Request(url, data=body, headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer local',
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})
        total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        # Append assistant message to history
        messages.append(msg)

        finish = choice.get("finish_reason", "stop")
        if finish == "tool_calls" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                all_tool_calls.append(name)
                result = TOOL_RESULTS.get(name, '{"result": "ok"}')
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })
        else:
            # Model finished — extract answer
            answer_text = msg.get("content", "") or ""
            break

except Exception as e:
    print(json.dumps(EMPTY)); sys.exit(0)

end_ns = time.time_ns()
elapsed_ms = round((end_ns - start_ns) / 1_000_000)
turns_count = len([m for m in messages if m.get("role") == "assistant"])

print(json.dumps({
    'ms':           elapsed_ms,
    'tokens':       total_tokens,
    'turns':        turns_count,
    'tool_calls':   len(all_tool_calls),
    'tool_names':   all_tool_calls,
    'answer_chars': len(answer_text),
    'error':        turns_count == 0,
}))
PYEOF
}

# ── Statistics helper ─────────────────────────────────────────────────────
#
# Usage: _stats "v1 v2 v3" → emits: mean stddev min max cv_pct

_stats() {
    python3 - "$1" <<'PYEOF'
import sys, math
vals = [float(x) for x in sys.argv[1].split() if x]
if not vals:
    print("0 0 0 0 0"); sys.exit()
n    = len(vals)
mean = sum(vals) / n
sd   = math.sqrt(sum((v - mean)**2 for v in vals) / n)
cv   = sd / mean * 100 if mean else 0
print(f"{mean:.1f} {sd:.1f} {min(vals):.1f} {max(vals):.1f} {cv:.1f}")
PYEOF
}

# ── Peak memory helper ────────────────────────────────────────────────────

get_memory_mb() {
    local backend="$1"
    if [[ "$backend" == "ppmlx" ]]; then
        local mem_pressure
        mem_pressure=$(memory_pressure 2>/dev/null \
            | grep "System-wide memory free percentage:" \
            | awk '{print $NF}' | tr -d '%') || mem_pressure=""
        echo "${mem_pressure:-0}"
    elif [[ "$backend" == "ollama" ]]; then
        local ps
        ps=$(curl -s --max-time 5 "http://localhost:11434/api/ps" 2>/dev/null) || ps="{}"
        local vram
        vram=$(echo "$ps" | jq -r '.models[0].size_vram // 0' 2>/dev/null) || vram=0
        python3 -c "print(round($vram / 1048576))"
    else
        echo "0"
    fi
}

# ── Full benchmark ────────────────────────────────────────────────────────
#
# Outer loop: runs (R=1..RUNS)
# Inner order: S1 → S2 → S3 → S4
# S1-S3: direct API calls (fair tok/s comparison)
# S4: multi-turn agentic via direct API with tools (non-streaming)
# Prints one-line result per scenario immediately after it completes.
# Final block shows per-scenario stats across runs.

run_full_benchmark() {
    local model="$1"
    local backend="$2"
    local api_url="$3"
    # $4 is unused (was pi_model)

    local outfile="$RESULTS_DIR/${backend}_${model//[:\/]/_}.json"
    local timestamp
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    printf "\n${BOLD}════════════════════════════════════════${RESET}\n"
    printf "${BOLD}  %s — %s${RESET}\n" "$backend" "$model"
    printf "${BOLD}════════════════════════════════════════${RESET}\n"
    printf "Runs: %d | Max tokens (S1-S3): %d\n\n" "$RUNS" "$MAX_TOKENS"

    # ── Warmup (load model, fill caches — result discarded) ──────────
    printf "  ${DIM}Warmup...${RESET}"
    call_api "$api_url" "$model" "Say hello." "warmup" >/dev/null 2>&1 || true
    printf "\r  ${GREEN}Warmup done${RESET}\n\n"

    # Raw data arrays (space-separated strings)
    local s1_ms_v="" s1_tps_v="" s1_tok_v="" s1_ttft_v=""
    local s2_ms_v="" s2_tps_v="" s2_tok_v="" s2_ttft_v=""
    local s3_ms_v="" s3_tps_v="" s3_tok_v="" s3_ttft_v=""
    local s4_ms_v="" s4_ttft_v="" s4_tok_v="" s4_tps_v=""
    local s4_chars_v="" s4_think_v="" s4_ans_v="" s4_turns_v="" s4_tools_v=""

    local mem_after=0

    for run in $(seq 1 "$RUNS"); do
        printf "${BOLD}── Run %d/%d ──────────────────────────────${RESET}\n" "$run" "$RUNS"

        # ── Scenario 1: Simple ──────────────────────────────────────────
        printf "  ${DIM}S1 Simple...${RESET}"
        local r1
        r1=$(call_api "$api_url" "$model" "$SIMPLE_PROMPT" "$backend")
        if [[ "$(echo "$r1" | jq -r '.error')" == "false" ]]; then
            local r1_ms r1_tps r1_tok r1_ttft
            r1_ms=$(echo   "$r1" | jq -r '.ms')
            r1_tps=$(echo  "$r1" | jq -r '.tok_s')
            r1_tok=$(echo  "$r1" | jq -r '.tokens')
            r1_ttft=$(echo "$r1" | jq -r '.ttft_ms')
            s1_ms_v="$s1_ms_v $r1_ms"; s1_tps_v="$s1_tps_v $r1_tps"
            s1_tok_v="$s1_tok_v $r1_tok"; s1_ttft_v="$s1_ttft_v $r1_ttft"
            printf "\r  ${GREEN}S1 Simple     %6dms | TTFT %4dms | %5.1f tok/s | %4d tok${RESET}\n" \
                "$r1_ms" "$r1_ttft" "$r1_tps" "$r1_tok"
        else
            printf "\r  ${RED}S1 Simple     FAILED${RESET}\n"
        fi

        # ── Scenario 2: Complex ─────────────────────────────────────────
        printf "  ${DIM}S2 Complex...${RESET}"
        local r2
        r2=$(call_api "$api_url" "$model" "$COMPLEX_PROMPT" "$backend")
        if [[ "$(echo "$r2" | jq -r '.error')" == "false" ]]; then
            local r2_ms r2_tps r2_tok r2_ttft
            r2_ms=$(echo   "$r2" | jq -r '.ms')
            r2_tps=$(echo  "$r2" | jq -r '.tok_s')
            r2_tok=$(echo  "$r2" | jq -r '.tokens')
            r2_ttft=$(echo "$r2" | jq -r '.ttft_ms')
            s2_ms_v="$s2_ms_v $r2_ms"; s2_tps_v="$s2_tps_v $r2_tps"
            s2_tok_v="$s2_tok_v $r2_tok"; s2_ttft_v="$s2_ttft_v $r2_ttft"
            printf "\r  ${GREEN}S2 Complex    %6dms | TTFT %4dms | %5.1f tok/s | %4d tok${RESET}\n" \
                "$r2_ms" "$r2_ttft" "$r2_tps" "$r2_tok"
        else
            printf "\r  ${RED}S2 Complex    FAILED${RESET}\n"
        fi

        # Memory sample after first run (model is loaded)
        if [[ "$run" -eq 1 ]]; then
            mem_after=$(get_memory_mb "$backend")
        fi

        # ── Scenario 3: Long Context (direct API — prefill speed) ──────
        printf "  ${DIM}S3 LongCtx...${RESET}"
        local r3
        r3=$(call_api "$api_url" "$model" "$LONG_CONTEXT_PROMPT" "$backend")
        if [[ "$(echo "$r3" | jq -r '.error')" == "false" ]]; then
            local r3_ms r3_tps r3_tok r3_ttft
            r3_ms=$(echo   "$r3" | jq -r '.ms')
            r3_tps=$(echo  "$r3" | jq -r '.tok_s')
            r3_tok=$(echo  "$r3" | jq -r '.tokens')
            r3_ttft=$(echo "$r3" | jq -r '.ttft_ms')
            s3_ms_v="$s3_ms_v $r3_ms"; s3_tps_v="$s3_tps_v $r3_tps"
            s3_tok_v="$s3_tok_v $r3_tok"; s3_ttft_v="$s3_ttft_v $r3_ttft"
            printf "\r  ${GREEN}S3 LongCtx    %6dms | TTFT %4dms | %5.1f tok/s | %4d tok${RESET}\n" \
                "$r3_ms" "$r3_ttft" "$r3_tps" "$r3_tok"
        else
            printf "\r  ${RED}S3 LongCtx    FAILED${RESET}\n"
        fi

        # ── Scenario 4: Agentic (direct API, multi-turn with tools) ──
        printf "  ${DIM}S4 Agentic...${RESET}"
        local r4
        r4=$(call_agentic "$api_url" "$model" "$AGENTIC_PROMPT") || true
        if [[ "$(echo "$r4" | jq -r '.error')" == "false" ]]; then
            local r4_ms r4_tok r4_ans r4_turns r4_tools
            r4_ms=$(echo    "$r4" | jq -r '.ms')
            r4_tok=$(echo   "$r4" | jq -r '.tokens')
            r4_ans=$(echo   "$r4" | jq -r '.answer_chars')
            r4_turns=$(echo "$r4" | jq -r '.turns')
            r4_tools=$(echo "$r4" | jq -r '.tool_calls')
            s4_ms_v="$s4_ms_v $r4_ms";     s4_ttft_v="$s4_ttft_v 0"
            s4_tok_v="$s4_tok_v $r4_tok";   s4_tps_v="$s4_tps_v 0"
            s4_chars_v="$s4_chars_v 0"
            s4_think_v="$s4_think_v 0"; s4_ans_v="$s4_ans_v $r4_ans"
            s4_turns_v="$s4_turns_v $r4_turns"; s4_tools_v="$s4_tools_v $r4_tools"
            printf "\r  ${GREEN}S4 Agentic    %6dms | %4d tok | turns %d | tools %d | ans %d chars${RESET}\n" \
                "$r4_ms" "$r4_tok" "$r4_turns" "$r4_tools" "$r4_ans"
        else
            printf "\r  ${RED}S4 Agentic    FAILED${RESET}\n"
            s4_ms_v="$s4_ms_v 0"; s4_ttft_v="$s4_ttft_v 0"; s4_tok_v="$s4_tok_v 0"
            s4_tps_v="$s4_tps_v 0"; s4_chars_v="$s4_chars_v 0"
            s4_think_v="$s4_think_v 0"; s4_ans_v="$s4_ans_v 0"
            s4_turns_v="$s4_turns_v 0"; s4_tools_v="$s4_tools_v 0"
        fi

        printf "\n"
    done

    # ── Summary stats ─────────────────────────────────────────────────────
    printf "${BOLD}── Summary (avg ± sd  |  CV%%) ─────────────────${RESET}\n"

    _print_stats_row() {
        local label="$1" vals_ms="$2" vals_tps="$3" vals_tok="$4" vals_ttft="$5"
        read -r a_ms  sd_ms  _ _ cv_ms  <<< "$(_stats "$vals_ms")"
        read -r a_tps sd_tps _ _ cv_tps <<< "$(_stats "$vals_tps")"
        read -r a_tok sd_tok _ _ cv_tok <<< "$(_stats "$vals_tok")"
        read -r a_ttft sd_ttft _ _ cv_ttft <<< "$(_stats "$vals_ttft")"
        printf "  ${BOLD}%-14s${RESET}  %6.0fms ±%4.0f (CV %4.1f%%)  |  TTFT %4.0fms ±%3.0f (CV %4.1f%%)  |  %5.1f tok/s ±%4.1f  |  %4.0f tok ±%3.0f\n" \
            "$label" \
            "$a_ms" "$sd_ms" "$cv_ms" \
            "$a_ttft" "$sd_ttft" "$cv_ttft" \
            "$a_tps" "$sd_tps" \
            "$a_tok" "$sd_tok"
    }

    _print_stats_row "S1 Simple"     "$s1_ms_v" "$s1_tps_v" "$s1_tok_v" "$s1_ttft_v"
    _print_stats_row "S2 Complex"    "$s2_ms_v" "$s2_tps_v" "$s2_tok_v" "$s2_ttft_v"
    _print_stats_row "S3 LongCtx"    "$s3_ms_v" "$s3_tps_v" "$s3_tok_v" "$s3_ttft_v"

    # S4 special summary (wall-time focus — tok/s not comparable between backends)
    read -r a4_ms   sd4_ms   _ _ cv4_ms   <<< "$(_stats "$s4_ms_v")"
    read -r a4_ttft sd4_ttft _ _ cv4_ttft <<< "$(_stats "$s4_ttft_v")"
    read -r a4_tok  sd4_tok  _ _ cv4_tok  <<< "$(_stats "$s4_tok_v")"
    read -r a4_chars  sd4_chars  _ _ cv4_chars  <<< "$(_stats "$s4_chars_v")"
    read -r a4_think  sd4_think  _ _ cv4_think  <<< "$(_stats "$s4_think_v")"
    read -r a4_ans    sd4_ans    _ _ cv4_ans    <<< "$(_stats "$s4_ans_v")"
    read -r a4_turns  sd4_turns  _ _ cv4_turns  <<< "$(_stats "$s4_turns_v")"
    read -r a4_tools  sd4_tools  _ _ cv4_tools  <<< "$(_stats "$s4_tools_v")"
    printf "  ${BOLD}%-14s${RESET}  %6.0fms ±%4.0f (CV %4.1f%%)  |  TTFT %4.0fms ±%3.0f (CV %4.1f%%)  |  %4.0f tok ±%3.0f  |  turns %.0f  |  tools %.0f  |  ans %4.0f±%3.0f chars\n" \
        "S4 Agentic" \
        "$a4_ms" "$sd4_ms" "$cv4_ms" \
        "$a4_ttft" "$sd4_ttft" "$cv4_ttft" \
        "$a4_tok" "$sd4_tok" \
        "$a4_turns" \
        "$a4_tools" \
        "$a4_ans" "$sd4_ans"

    printf "\n"

    # ── Write JSON ────────────────────────────────────────────────────────
    python3 - "$outfile" "$timestamp" "$RUNS" "$MAX_TOKENS" "$mem_after" \
        "$s1_ms_v" "$s1_tps_v" "$s1_tok_v" "$s1_ttft_v" \
        "$s2_ms_v" "$s2_tps_v" "$s2_tok_v" "$s2_ttft_v" \
        "$s3_ms_v" "$s3_tps_v" "$s3_tok_v" "$s3_ttft_v" \
        "$s4_ms_v" "$s4_ttft_v" "$s4_tok_v" "$s4_tps_v" \
        "$s4_chars_v" "$s4_think_v" "$s4_ans_v" "$s4_turns_v" "$s4_tools_v" \
        "$model" "$backend" <<'PYEOF'
import sys, json, math

def arr(s):
    return [float(x) for x in s.split() if x]

def stats(vals):
    if not vals: return {"avg":0,"sd":0,"min":0,"max":0,"cv":0,"runs":[]}
    n = len(vals); m = sum(vals)/n
    sd = math.sqrt(sum((v-m)**2 for v in vals)/n)
    return {"avg":round(m,2),"sd":round(sd,2),"min":min(vals),"max":max(vals),
            "cv":round(sd/m*100,1) if m else 0,"runs":vals}

(outfile, ts, runs, max_tok, mem,
 s1_ms, s1_tps, s1_tok, s1_ttft,
 s2_ms, s2_tps, s2_tok, s2_ttft,
 s3_ms, s3_tps, s3_tok, s3_ttft,
 s4_ms, s4_ttft, s4_tok, s4_tps,
 s4_chars, s4_think, s4_ans, s4_turns, s4_tools,
 model, backend) = sys.argv[1:]

doc = {
  "backend": backend, "model": model,
  "timestamp": ts, "runs": int(runs),
  "max_tokens_s1s3": int(max_tok), "memory_mb": int(mem),
  "results": {
    "simple":      {"ms": stats(arr(s1_ms)),  "tok_s": stats(arr(s1_tps)),
                    "tokens": stats(arr(s1_tok)), "ttft_ms": stats(arr(s1_ttft))},
    "complex":     {"ms": stats(arr(s2_ms)),  "tok_s": stats(arr(s2_tps)),
                    "tokens": stats(arr(s2_tok)), "ttft_ms": stats(arr(s2_ttft))},
    "long_context": {"ms": stats(arr(s3_ms)),  "tok_s": stats(arr(s3_tps)),
                     "tokens": stats(arr(s3_tok)), "ttft_ms": stats(arr(s3_ttft))},
    "agentic":     {"ms":         stats(arr(s4_ms)),
                    "ttft_ms":    stats(arr(s4_ttft)),
                    "tokens":     stats(arr(s4_tok)),
                    "tok_s":      stats(arr(s4_tps)),
                    "chars_total":stats(arr(s4_chars)),
                    "think_chars":stats(arr(s4_think)),
                    "answer_chars":stats(arr(s4_ans)),
                    "turns":      stats(arr(s4_turns)),
                    "tool_calls": stats(arr(s4_tools))},
  }
}
with open(outfile, "w") as f:
    json.dump(doc, f, indent=2)
print(outfile)
PYEOF

    local saved_to
    saved_to=$(cat /tmp/_bench_outfile 2>/dev/null || echo "$outfile")
    printf "${GREEN}Results saved → %s${RESET}\n" "$outfile"

    # ── Cleanup: free GPU/memory after benchmark ──────────────────────
    printf "\n${DIM}Cleaning up %s...${RESET}" "$backend"
    if [[ "$backend" == "ppmlx" ]]; then
        pkill -f "ppmlx serve" 2>/dev/null || true
    elif [[ "$backend" == "ollama" ]]; then
        ollama stop "$model" 2>/dev/null || true
    fi
    sleep 2
    printf " ${GREEN}done${RESET}\n"
}
