#!/usr/bin/env python3
"""Parse benchmark results from JSON files and produce a markdown summary table.

Reads all JSON results from benchmark_results/ and produces:
  Model | Scenario | ppmlx tok/s | Ollama tok/s | delta%

Usage:
    python3 scripts/parse_results.py                  # print to stdout
    python3 scripts/parse_results.py -o results.md    # write to file
    python3 scripts/parse_results.py --json            # output raw JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"

SCENARIOS = ["simple", "complex", "long_context", "agentic"]
SCENARIO_LABELS = {
    "simple": "Simple",
    "complex": "Complex",
    "long_context": "Long Context",
    "agentic": "Agentic",
}


def load_results(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Load all JSON results, grouped by model then backend.

    Returns: {model: {backend: data}}
    """
    grouped: dict[str, dict[str, dict]] = {}

    if not results_dir.is_dir():
        return grouped

    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue

        model = data.get("model", "unknown")
        backend = data.get("backend", "unknown")
        grouped.setdefault(model, {})[backend] = data

    return grouped


def safe_get(data: dict, scenario: str, metric: str) -> float | None:
    """Safely extract a metric avg from nested results.

    Handles the structure: results.<scenario>.<metric>.avg
    """
    try:
        return data["results"][scenario][metric]["avg"]
    except (KeyError, TypeError):
        return None


def pct_delta(ppmlx_val: float | None, ollama_val: float | None) -> str:
    """Compute percentage difference: (ppmlx - ollama) / ollama * 100."""
    if ppmlx_val is None or ollama_val is None:
        return "N/A"
    if ollama_val == 0:
        return "N/A" if ppmlx_val == 0 else "+inf"
    delta = ((ppmlx_val - ollama_val) / ollama_val) * 100
    return f"{delta:+.1f}%"


def fmt_val(val: float | None, decimals: int = 1) -> str:
    """Format a value for display, or '-' if missing."""
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def generate_markdown(grouped: dict[str, dict[str, dict]]) -> str:
    """Generate a markdown comparison table."""
    lines: list[str] = []

    lines.append("# ppmlx vs Ollama Benchmark Results\n")

    if not grouped:
        lines.append("No benchmark results found.\n")
        return "\n".join(lines)

    # ── Throughput comparison table ──────────────────────────────────
    lines.append("## Throughput (tok/s)\n")
    lines.append(
        "| Model | Scenario | ppmlx tok/s | Ollama tok/s | Delta |"
    )
    lines.append(
        "|-------|----------|-------------|--------------|-------|"
    )

    for model in sorted(grouped):
        ppmlx_data = grouped[model].get("ppmlx")
        ollama_data = grouped[model].get("ollama")

        for scenario in SCENARIOS:
            label = SCENARIO_LABELS[scenario]
            p_tps = safe_get(ppmlx_data, scenario, "tok_s") if ppmlx_data else None
            o_tps = safe_get(ollama_data, scenario, "tok_s") if ollama_data else None
            delta = pct_delta(p_tps, o_tps)

            lines.append(
                f"| {model} | {label} | {fmt_val(p_tps, 1)} | {fmt_val(o_tps, 1)} | {delta} |"
            )

    # ── TTFT comparison table ───────────────────────────────────────
    lines.append("")
    lines.append("## Time to First Token (ms)\n")
    lines.append(
        "| Model | Scenario | ppmlx TTFT | Ollama TTFT | Delta |"
    )
    lines.append(
        "|-------|----------|------------|-------------|-------|"
    )

    for model in sorted(grouped):
        ppmlx_data = grouped[model].get("ppmlx")
        ollama_data = grouped[model].get("ollama")

        for scenario in SCENARIOS:
            label = SCENARIO_LABELS[scenario]
            p_ttft = safe_get(ppmlx_data, scenario, "ttft_ms") if ppmlx_data else None
            o_ttft = safe_get(ollama_data, scenario, "ttft_ms") if ollama_data else None
            # For TTFT, lower is better, so invert the delta direction
            ttft_delta = pct_delta(o_ttft, p_ttft) if p_ttft and o_ttft else "N/A"

            lines.append(
                f"| {model} | {label} | {fmt_val(p_ttft, 0)} | {fmt_val(o_ttft, 0)} | {ttft_delta} |"
            )

    # ── Per-model details ───────────────────────────────────────────
    lines.append("")
    lines.append("## Detailed Results\n")

    for model in sorted(grouped):
        lines.append(f"### {model}\n")

        for backend_name in ["ppmlx", "ollama"]:
            data = grouped[model].get(backend_name)
            if not data:
                lines.append(f"**{backend_name}**: No results\n")
                continue

            ts = data.get("timestamp", "unknown")
            runs = data.get("runs", "?")
            mem = data.get("memory_mb", 0)

            lines.append(f"**{backend_name}** (runs: {runs}, timestamp: {ts}, memory: {mem}MB)\n")
            lines.append("| Scenario | Duration (ms) | tok/s | Tokens | TTFT (ms) |")
            lines.append("|----------|---------------|-------|--------|-----------|")

            for scenario in SCENARIOS:
                label = SCENARIO_LABELS[scenario]
                ms = safe_get(data, scenario, "ms")
                tps = safe_get(data, scenario, "tok_s")
                tok = safe_get(data, scenario, "tokens")
                ttft = safe_get(data, scenario, "ttft_ms")

                lines.append(
                    f"| {label} | {fmt_val(ms, 0)} | {fmt_val(tps, 1)} | {fmt_val(tok, 0)} | {fmt_val(ttft, 0)} |"
                )

            lines.append("")

    return "\n".join(lines)


def generate_json_summary(grouped: dict[str, dict[str, dict]]) -> str:
    """Generate a JSON summary of all results."""
    summary: list[dict] = []

    for model in sorted(grouped):
        ppmlx_data = grouped[model].get("ppmlx")
        ollama_data = grouped[model].get("ollama")

        for scenario in SCENARIOS:
            entry = {
                "model": model,
                "scenario": scenario,
                "ppmlx_tok_s": safe_get(ppmlx_data, scenario, "tok_s") if ppmlx_data else None,
                "ollama_tok_s": safe_get(ollama_data, scenario, "tok_s") if ollama_data else None,
                "ppmlx_ttft_ms": safe_get(ppmlx_data, scenario, "ttft_ms") if ppmlx_data else None,
                "ollama_ttft_ms": safe_get(ollama_data, scenario, "ttft_ms") if ollama_data else None,
                "ppmlx_tokens": safe_get(ppmlx_data, scenario, "tokens") if ppmlx_data else None,
                "ollama_tokens": safe_get(ollama_data, scenario, "tokens") if ollama_data else None,
            }
            p = entry["ppmlx_tok_s"]
            o = entry["ollama_tok_s"]
            delta_str = pct_delta(p, o)
            entry["delta_pct"] = float(delta_str.rstrip("%")) if delta_str not in ("N/A", "+inf") else None
            summary.append(entry)

    return json.dumps(summary, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ppmlx benchmark results into a comparison table."
    )
    parser.add_argument(
        "-o", "--output",
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown",
    )
    parser.add_argument(
        "--dir",
        default=str(RESULTS_DIR),
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    results_dir = Path(args.dir)
    grouped = load_results(results_dir)

    if not grouped:
        print(f"No benchmark results found in {results_dir}", file=sys.stderr)
        print("Run benchmark scripts first, or specify --dir.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = generate_json_summary(grouped)
    else:
        output = generate_markdown(grouped)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
