#!/usr/bin/env bash
# bench_compare.sh — Compare ppmlx vs ollama results from benchmark_results/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../benchmark_results"

BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; RESET='\033[0m'

if [[ ! -d "$RESULTS_DIR" ]] || [[ -z "$(ls "$RESULTS_DIR"/*.json 2>/dev/null)" ]]; then
    printf "${RED}No results found in %s${RESET}\n" "$RESULTS_DIR"
    printf "Run the bench_ppmlx_*.sh and bench_ollama_*.sh scripts first.\n"
    exit 1
fi

printf "\n${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}\n"
printf "${BOLD}║          ppmlx (MLX) vs Ollama — Benchmark Results          ║${RESET}\n"
printf "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}\n"
printf "${DIM}Date: %s${RESET}\n" "$(date)"
printf "${DIM}Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')${RESET}\n"
printf "${DIM}RAM: $(sysctl -n hw.memsize 2>/dev/null | python3 -c 'import sys; print(f"{int(sys.stdin.read())//1073741824}GB")' 2>/dev/null || echo 'unknown')${RESET}\n"

MODELS=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | xargs -I{} jq -r '.model' {} | sort -u)

# Collect all data for final summary
ALL_DATA=""

for model in $MODELS; do
    ppmlx_file="$RESULTS_DIR/ppmlx_${model//[:\/]/_}.json"
    ollama_file="$RESULTS_DIR/ollama_${model//[:\/]/_}.json"

    printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "${BOLD}  %s${RESET}\n" "$model"
    printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

    if [[ ! -f "$ppmlx_file" ]]; then
        printf "  ${YELLOW}ppmlx results missing${RESET}\n"
        continue
    fi
    if [[ ! -f "$ollama_file" ]]; then
        printf "  ${YELLOW}ollama results missing${RESET}\n"
        continue
    fi

    # Memory
    p_mem=$(jq -r '.memory_mb' "$ppmlx_file")
    o_mem=$(jq -r '.memory_mb' "$ollama_file")

    # Header
    printf "\n  ${DIM}%-10s %8s %8s %8s %8s  │  %8s %8s %8s %8s  │  %8s${RESET}\n" \
        "" "tok/s" "TTFT" "total" "tokens" "tok/s" "TTFT" "total" "tokens" "Δ tok/s"
    printf "  ${DIM}%-10s %8s %8s %8s %8s  │  %8s %8s %8s %8s  │  %8s${RESET}\n" \
        "" "ppmlx" "ppmlx" "ppmlx" "ppmlx" "ollama" "ollama" "ollama" "ollama" ""

    for scenario in simple complex long_context; do
        p_tps=$(jq -r ".results.${scenario}.tok_s.avg // 0" "$ppmlx_file")
        o_tps=$(jq -r ".results.${scenario}.tok_s.avg // 0" "$ollama_file")
        p_ms=$(jq -r ".results.${scenario}.ms.avg // 0" "$ppmlx_file")
        o_ms=$(jq -r ".results.${scenario}.ms.avg // 0" "$ollama_file")
        p_ttft=$(jq -r ".results.${scenario}.ttft_ms.avg // 0" "$ppmlx_file")
        o_ttft=$(jq -r ".results.${scenario}.ttft_ms.avg // 0" "$ollama_file")
        p_tok=$(jq -r ".results.${scenario}.tokens.avg // 0" "$ppmlx_file")
        o_tok=$(jq -r ".results.${scenario}.tokens.avg // 0" "$ollama_file")

        speedup=$(python3 -c "
p, o = $p_tps, $o_tps
if o > 0: print(f'{((p-o)/o)*100:+.1f}%')
else: print('N/A')
")
        ttft_diff=$(python3 -c "
p, o = $p_ttft, $o_ttft
if o > 0: print(f'{((o-p)/o)*100:+.1f}%')
else: print('N/A')
")

        label=$(printf "%-10s" "$scenario")
        printf "  %s ${GREEN}%7.1f${RESET} %7dms %7dms %7d  │  ${CYAN}%7.1f${RESET} %7dms %7dms %7d  │  ${BOLD}%7s${RESET}\n" \
            "$label" "$p_tps" "$p_ttft" "$p_ms" "$p_tok" "$o_tps" "$o_ttft" "$o_ms" "$o_tok" "$speedup"

        ALL_DATA="$ALL_DATA $p_tps $o_tps $p_ttft $o_ttft"
    done

    # Agentic row (pi CLI — wall clock only)
    p_ms=$(jq -r ".results.agentic.ms.avg // 0" "$ppmlx_file")
    o_ms=$(jq -r ".results.agentic.ms.avg // 0" "$ollama_file")
    p_chars=$(jq -r ".results.agentic.answer_chars.avg // 0" "$ppmlx_file")
    o_chars=$(jq -r ".results.agentic.answer_chars.avg // 0" "$ollama_file")

    if [[ "$p_ms" -gt 0 && "$o_ms" -gt 0 ]]; then
        speedup=$(python3 -c "
p, o = $p_ms, $o_ms
if p > 0: print(f'{((o-p)/o)*100:+.1f}%')
else: print('N/A')
")
        printf "  ${DIM}%-10s${RESET} ${GREEN}%7s${RESET} %8s %7dms %6dch  │  ${CYAN}%7s${RESET} %8s %7dms %6dch  │  ${BOLD}%7s${RESET}\n" \
            "agentic" "—" "—" "$p_ms" "$p_chars" "—" "—" "$o_ms" "$o_chars" "$speedup"
    fi

    # Memory footnote
    if [[ "$p_mem" != "0" || "$o_mem" != "0" ]]; then
        printf "\n  ${DIM}Memory: ppmlx %s  |  ollama %s${RESET}\n" \
            "$([ "$p_mem" != "0" ] && echo "${p_mem}MB" || echo "N/A")" \
            "$([ "$o_mem" != "0" ] && echo "${o_mem}MB" || echo "N/A")"
    fi
done

# ── Overall Summary ──────────────────────────────────────────────────────

printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "${BOLD}  Overall${RESET}\n"
printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

python3 -c "
data = '$ALL_DATA'.split()
# data is groups of 4: p_tps o_tps p_ttft o_ttft
i = 0
p_tps_all, o_tps_all = [], []
p_ttft_all, o_ttft_all = [], []
while i + 3 < len(data):
    pt, ot, pf, of_ = float(data[i]), float(data[i+1]), float(data[i+2]), float(data[i+3])
    if pt > 0: p_tps_all.append(pt)
    if ot > 0: o_tps_all.append(ot)
    if pf > 0: p_ttft_all.append(pf)
    if of_ > 0: o_ttft_all.append(of_)
    i += 4

if p_tps_all and o_tps_all:
    p_avg = sum(p_tps_all) / len(p_tps_all)
    o_avg = sum(o_tps_all) / len(o_tps_all)
    diff = ((p_avg - o_avg) / o_avg) * 100
    print(f'  Avg throughput:  ppmlx {p_avg:.1f} tok/s  |  ollama {o_avg:.1f} tok/s  |  {diff:+.1f}%')

if p_ttft_all and o_ttft_all:
    p_avg = sum(p_ttft_all) / len(p_ttft_all)
    o_avg = sum(o_ttft_all) / len(o_ttft_all)
    diff = ((o_avg - p_avg) / o_avg) * 100
    print(f'  Avg TTFT:        ppmlx {p_avg:.0f}ms       |  ollama {o_avg:.0f}ms       |  {diff:+.1f}% faster')
"

printf "\n${DIM}Results in: %s${RESET}\n\n" "$RESULTS_DIR"
