export interface ModelResult {
  ppmlx: number;
  psd: number;
  ollama: number;
  osd: number;
  delta: string;
  ttft_ppmlx: number | string;
  ttft_ollama: number | string;
}

export interface BenchScenario {
  id: string;
  name: string;
  desc: string;
  /** "toks" = tok/s higher-is-better, "time" = seconds lower-is-better */
  metric: "toks" | "time";
  unit: string;
  maxVal: number;
  models: Record<string, ModelResult>;
}

export interface BenchModelMeta {
  id: string;
  label: string;
  shortLabel: string;
}

export const models: BenchModelMeta[] = [
  { id: "glm", label: "GLM-4.7-Flash \u00b7 58B \u00b7 4-bit", shortLabel: "GLM-4.7-Flash" },
  { id: "qwen9b", label: "Qwen3.5:9B \u00b7 9B \u00b7 4-bit", shortLabel: "Qwen3.5:9B" },
];

/**
 * Benchmark data from real runs on MacBook Pro M4 Pro, 48 GB.
 * Source: benchmark_results/*.json (2026-03-25)
 * Each scenario: 3 runs averaged, all models 4-bit quantized.
 */
export const scenarios: BenchScenario[] = [
  {
    id: "simple",
    name: "Simple",
    desc: "Short prompt, short answer. Measures baseline throughput and time to first token.",
    metric: "toks",
    unit: "tok/s",
    maxVal: 70,
    models: {
      glm: { ppmlx: 63.1, psd: 0.3, ollama: 40.5, osd: 1.7, delta: "+56%", ttft_ppmlx: 374, ttft_ollama: 832 },
      qwen9b: { ppmlx: 48.2, psd: 0.7, ollama: 22.7, osd: 0.1, delta: "+112%", ttft_ppmlx: 537, ttft_ollama: 324 },
    },
  },
  {
    id: "complex",
    name: "Complex",
    desc: "Short prompt, long answer. Tests sustained generation throughput over thousands of tokens.",
    metric: "toks",
    unit: "tok/s",
    maxVal: 70,
    models: {
      glm: { ppmlx: 55.6, psd: 2.3, ollama: 38.8, osd: 0.4, delta: "+43%", ttft_ppmlx: 496, ttft_ollama: 412 },
      qwen9b: { ppmlx: 47.2, psd: 1.3, ollama: 23.0, osd: 0.1, delta: "+105%", ttft_ppmlx: 567, ttft_ollama: 455 },
    },
  },
  {
    id: "longctx",
    name: "Long Context",
    desc: "Large prompt (~4K tokens), medium answer. Measures prompt processing (prefill) speed.",
    metric: "toks",
    unit: "tok/s",
    maxVal: 70,
    models: {
      glm: { ppmlx: 42.1, psd: 1.9, ollama: 27.5, osd: 1.1, delta: "+53%", ttft_ppmlx: "6.8s", ttft_ollama: "8.4s" },
      qwen9b: { ppmlx: 43.2, psd: 0.5, ollama: 23.7, osd: 0.3, delta: "+82%", ttft_ppmlx: "9.2s", ttft_ollama: "11.5s" },
    },
  },
  {
    id: "agentic",
    name: "Agentic",
    desc: "Multi-turn tool-calling workflow (3 turns, 2\u20135 tool calls). Total wall-clock time, lower is better.",
    metric: "time",
    unit: "s",
    maxVal: 100,
    models: {
      glm: { ppmlx: 11.7, psd: 0.1, ollama: 55.1, osd: 0.1, delta: "4.7\u00d7", ttft_ppmlx: "11.7s total", ttft_ollama: "55.1s total" },
      qwen9b: { ppmlx: 8.1, psd: 0.0, ollama: 87.6, osd: 0.7, delta: "10.8\u00d7", ttft_ppmlx: "8.1s total", ttft_ollama: "87.6s total" },
    },
  },
];
