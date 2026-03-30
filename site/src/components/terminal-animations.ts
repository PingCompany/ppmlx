/** Terminal demo animations — extracted from original inline script. */

const TITLES = ["ppmlx launch", "ppmlx serve", "ppmlx pull", "ppmlx run", "ppmlx quantize"];
const DESCS = [
  "TUI launcher \u2014 pick action & model, launch coding agents in one step",
  "API server \u2014 OpenAI-compatible on localhost:6767, zero config",
  "Multi-select download \u2014 Space to toggle, Enter to confirm",
  "Interactive chat \u2014 select model, auto-download, streaming REPL",
  "Convert any HuggingFace model to MLX format with quantization",
];
const CMDS = ["ppmlx launch", "ppmlx serve", "ppmlx pull", "ppmlx run", "ppmlx quantize custom-7b"];

let bEl: HTMLElement;
let tEl: HTMLElement | null;
let dEl: HTMLElement | null;
let gen = 0;
let cur = 0;

function L(h: string) {
  return '<div class="term-line">' + h + "</div>";
}

function typeCmd(cmd: string, g: number, onDone: () => void) {
  bEl.innerHTML = L('<span class="t-prompt">$ </span><span class="t-typed"></span><span class="cursor"></span>');
  const el = bEl.querySelector(".t-typed")!;
  let i = 0;
  (function tick() {
    if (g !== gen) return;
    if (i < cmd.length) {
      el.textContent += cmd.charAt(i);
      i++;
      setTimeout(tick, 33);
    } else {
      setTimeout(() => {
        if (g !== gen) return;
        bEl.querySelector(".cursor")?.remove();
        onDone();
      }, 400);
    }
  })();
}

function advance(g: number, delay: number) {
  setTimeout(() => {
    if (g !== gen) return;
    switchTo((cur + 1) % 5);
  }, delay);
}

type StepPair = [number, () => void];

function runSteps(g: number, steps: StepPair[], onDone?: () => void) {
  let i = 0;
  (function next() {
    if (g !== gen) return;
    if (i >= steps.length) { onDone?.(); return; }
    const pair = steps[i++];
    setTimeout(() => {
      if (g !== gen) return;
      pair[1]();
      next();
    }, pair[0]);
  })();
}

/* ── LAUNCH ── */
function animLaunch(g: number) {
  const items = ["Run a model", "Launch Claude Code", "Launch Codex", "Launch Opencode", "Launch Pi"];
  const descs = ["Start an interactive chat with a model", "Agentic coding across large codebases", "OpenAI's open-source coding agent", "Anomaly's open-source coding agent", "Minimal AI agent toolkit"];
  const pm = [
    { h: "\u2605 Favorites" }, { a: "\u2605 qwen3.5:9b", s: "5.6 GB" }, { a: "\u2605 glm-4.7-flash", s: "15.7 GB" },
    { h: "Downloaded" }, { a: "glm4", s: "15.7 GB" }, { a: "qwen3.5:122b", s: "64.8 GB" },
    { h: "Available" }, { a: "gemma2", s: "5.2 GB" }, { a: "deepseek-coder", s: "3.8 GB" },
  ] as Array<{ h?: string; a?: string; s?: string }>;

  function menu(c: number, model: string | null) {
    let h = L('<span class="t-bold">ppmlx v0.2.0</span>') + L("&nbsp;");
    for (let i = 0; i < items.length; i++) {
      const sel = i === c;
      const tag = i > 0 && model ? '<span class="t-dim" style="float:right">(' + model + ")</span>" : "";
      h += L((sel ? '<span class="t-val">\u25b6 </span>' : "  ") + '<span class="' + (sel ? "t-bold" : "t-cmd") + '">' + items[i] + "</span>" + tag);
      h += L('  <span class="t-dim">' + descs[i] + "</span>");
    }
    h += L("&nbsp;") + L('<span class="t-dim">\u2191/\u2193 navigate \u00b7 enter launch \u00b7 \u2192 change model \u00b7 esc quit</span>');
    bEl.innerHTML = h;
  }

  function picker(c: number) {
    let h = L('<span class="t-bold">Select model:</span> <span class="cursor"></span>') + L("&nbsp;");
    let ri = 0;
    for (let i = 0; i < pm.length; i++) {
      const m = pm[i];
      if (m.h) { h += L('  <span class="t-dim t-bold">' + m.h + "</span>"); }
      else {
        const sel = ri === c;
        h += L((sel ? '<span class="t-val">\u25b6 </span>' : "  ") + '<span class="' + (sel ? "t-bold" : "t-cmd") + '">' + m.a + '</span><span style="float:right" class="t-dim">' + m.s + "</span>");
        ri++;
      }
    }
    h += L("&nbsp;") + L('<span class="t-dim">\u2191/\u2193 navigate \u00b7 enter select \u00b7 \u2190 back</span>');
    bEl.innerHTML = h;
  }

  function launching(model: string) {
    bEl.innerHTML =
      L('<span class="t-dim">Starting ppmlx server on 127.0.0.1:6767...</span>') +
      L('<span class="t-ok">Server ready.</span> <span class="t-dim">Pre-loading ' + model + "...</span>") +
      L('<span class="t-dim">Model loaded  </span><span class="t-val">61.6 tok/s</span><span class="t-dim">  TTFT </span><span class="t-ok">319ms</span>') +
      L("&nbsp;") +
      L('<span class="t-ok">Launching Claude Code...</span>') +
      L('<span class="t-prompt">$ </span><span class="t-cmd">claude <span class="t-flag">--model</span> <span class="t-arg">' + model + "</span></span>") +
      L("&nbsp;") +
      L('<span class="t-dim">ANTHROPIC_BASE_URL=http://127.0.0.1:6767</span>') +
      L('<span class="t-dim">ANTHROPIC_API_KEY=local</span>');
  }

  menu(0, null);
  runSteps(g, [
    [1200, () => menu(1, null)],
    [1000, () => picker(0)],
    [800, () => picker(1)],
    [900, () => picker(2)],
    [600, () => picker(1)],
    [1000, () => menu(1, "glm-4.7-flash")],
    [1500, () => launching("glm-4.7-flash")],
  ], () => advance(g, 4000));
}

/* ── SERVE ── */
function animServe(g: number) {
  const cmd = L('<span class="t-prompt">$ </span><span class="t-cmd">ppmlx serve</span>');

  setTimeout(() => {
    if (g !== gen) return;
    bEl.innerHTML = cmd + L("&nbsp;") +
      L('<span class="t-ok t-bold">ppmlx server v0.1.0</span>') +
      L('  Listening on <span class="t-blue">http://127.0.0.1:6767</span>') +
      L("&nbsp;") + L('  <span class="t-dim">Endpoints:</span>') +
      L('    <span class="t-blue">POST</span> /v1/chat/completions') +
      L('    <span class="t-blue">POST</span> /v1/responses') +
      L('    <span class="t-blue">POST</span> /v1/messages') +
      L('    <span class="t-blue">POST</span> /v1/embeddings') +
      L('    <span class="t-dim">GET</span>  /v1/models') +
      L('    <span class="t-dim">GET</span>  /health');
  }, 200);

  setTimeout(() => {
    if (g !== gen) return;
    bEl.innerHTML += L("&nbsp;") +
      L('<span class="t-dim">INFO:</span>     Uvicorn running on <span class="t-blue">http://127.0.0.1:6767</span>') +
      L('<span class="t-dim">INFO:</span>     Application startup complete');
  }, 1800);

  setTimeout(() => {
    if (g !== gen) return;
    bEl.innerHTML += L("&nbsp;") +
      L('<span class="t-ok">\u2190</span> <span class="t-dim">POST /v1/chat/completions</span> <span class="t-val">200</span> <span class="t-dim">1247ms \u00b7 42.3 tok/s</span>');
  }, 3500);

  setTimeout(() => {
    if (g !== gen) return;
    bEl.innerHTML += L('<span class="t-ok">\u2190</span> <span class="t-dim">POST /v1/messages</span> <span class="t-val">200</span> <span class="t-dim">892ms \u00b7 58.1 tok/s</span>');
  }, 4500);

  advance(g, 7000);
}

/* ── PULL ── */
function animPull(g: number) {
  const cmd = L('<span class="t-prompt">$ </span><span class="t-cmd">ppmlx pull</span>');
  const models = [
    { n: "qwen3.5:9b", r: "mlx-community/Qwen3.5-9B-4bit", sel: false },
    { n: "glm-4.7-flash", r: "mlx-community/glm-4-9b-chat-4bit", sel: false },
    { n: "mistral", r: "mlx-community/Mistral-7B-v0.3-4bit", sel: false },
    { n: "phi4", r: "mlx-community/phi-4-4bit", sel: false },
    { n: "gemma2", r: "mlx-community/gemma-2-9b-it-4bit", sel: false },
  ];

  function drawCB(cursor: number) {
    let h = cmd + L("&nbsp;") + L('<span class="t-dim">Select models to download  (Space=toggle, Enter=confirm):</span>');
    for (let i = 0; i < models.length; i++) {
      const m = models[i];
      const c = i === cursor;
      const ck = m.sel ? '<span class="t-ok">\u25cf</span>' : '<span class="t-dim">\u25cb</span>';
      h += L((c ? '<span class="t-val">\u25b6 </span>' : "  ") + ck + ' <span class="' + (c ? "t-bold" : "t-cmd") + '">' + m.n + '</span><span style="float:right" class="t-dim">' + m.r + "</span>");
    }
    bEl.innerHTML = h;
  }

  function drawProgress() {
    bEl.innerHTML = cmd + L("&nbsp;") +
      L('<span class="t-blue">Pulling mistral</span> <span class="t-dim">(mlx-community/Mistral-7B-v0.3-4bit)</span>') +
      '<div class="term-line"><div class="term-progress"><div class="term-progress-bar"><div class="term-progress-fill" id="pb1"></div></div><span class="t-dim" id="pp1">0%</span></div></div>';

    setTimeout(() => {
      if (g !== gen) return;
      const bar = document.getElementById("pb1");
      if (bar) bar.style.width = "100%";
      let p = 0;
      const iv = setInterval(() => {
        if (g !== gen) { clearInterval(iv); return; }
        p = Math.min(p + 3, 100);
        const el = document.getElementById("pp1");
        if (el) el.textContent = (p * 0.041).toFixed(1) + "/4.1 GB  \u2193 45 MB/s";
        if (p >= 100) { clearInterval(iv); drawPullDone(); }
      }, 30);
    }, 100);
  }

  function drawPullDone() {
    setTimeout(() => {
      if (g !== gen) return;
      bEl.innerHTML = cmd + L("&nbsp;") +
        L('<span class="t-ok">\u2713 mistral</span> <span class="t-dim">\u2192 ~/.ppmlx/models/</span>') +
        L("&nbsp;") +
        L('<span class="t-blue">Pulling phi4</span> <span class="t-dim">(mlx-community/phi-4-4bit)</span>') +
        '<div class="term-line"><div class="term-progress"><div class="term-progress-bar"><div class="term-progress-fill" id="pb2"></div></div><span class="t-dim" id="pp2">0%</span></div></div>';

      setTimeout(() => {
        if (g !== gen) return;
        const bar = document.getElementById("pb2");
        if (bar) bar.style.width = "100%";
        let p = 0;
        const iv = setInterval(() => {
          if (g !== gen) { clearInterval(iv); return; }
          p = Math.min(p + 2, 100);
          const el = document.getElementById("pp2");
          if (el) el.textContent = (p * 0.084).toFixed(1) + "/8.4 GB  \u2193 52 MB/s";
          if (p >= 100) {
            clearInterval(iv);
            setTimeout(() => {
              if (g !== gen) return;
              bEl.innerHTML += L('<span class="t-ok">\u2713 phi4</span> <span class="t-dim">\u2192 ~/.ppmlx/models/</span>') +
                L("&nbsp;") + L('<span class="t-ok t-bold">2 models downloaded.</span>');
              advance(g, 3000);
            }, 300);
          }
        }, 25);
      }, 100);
    }, 500);
  }

  drawCB(0);
  runSteps(g, [
    [1000, () => drawCB(1)],
    [800, () => drawCB(2)],
    [600, () => { models[2].sel = true; drawCB(2); }],
    [800, () => drawCB(3)],
    [600, () => { models[3].sel = true; drawCB(3); }],
    [1000, () => drawProgress()],
  ]);
}

/* ── RUN ── */
function animRun(g: number) {
  const cmd = L('<span class="t-prompt">$ </span><span class="t-cmd">ppmlx run</span>');
  const ml = [
    { n: "\u2605 qwen3.5:9b", s: "5.6 GB", dl: true },
    { n: "\u2605 glm-4.7-flash", s: "15.7 GB", dl: true },
    { n: "  mistral", s: "4.1 GB", dl: false },
    { n: "  phi4", s: "8.4 GB", dl: false },
    { n: "  gemma2", s: "5.2 GB", dl: false },
  ];

  function drawSel(c: number) {
    let h = cmd + L("&nbsp;") + L('<span class="t-dim">Select a model  (\u2191/\u2193 navigate, Enter=confirm):</span>');
    for (let i = 0; i < ml.length; i++) {
      const m = ml[i];
      const sel = i === c;
      const mark = m.dl ? '  <span class="t-ok">\u2713</span>' : "";
      h += L((sel ? '<span class="t-val"> \u25b6 </span>' : "   ") + '<span class="' + (sel ? "t-bold" : "t-cmd") + '">' + m.n + '</span><span style="float:right" class="t-dim">' + m.s + mark + "</span>");
    }
    bEl.innerHTML = h;
  }

  function drawChat() {
    setTimeout(() => {
      if (g !== gen) return;
      bEl.innerHTML =
        L('<span class="t-ok">Chatting with <span class="t-bold">mistral</span>. Type /help or /bye to exit.</span>') +
        L("&nbsp;") +
        L('<span class="t-blue"><b>You:</b></span> What is MLX?') +
        L("&nbsp;") +
        '<div class="term-line"><span class="t-green"><b>Assistant:</b></span> <span id="sout"></span><span class="cursor"></span></div>';

      const out = document.getElementById("sout");
      const text = "MLX is Apple\u2019s open-source ML framework for Apple Silicon. It uses unified memory so GPU and CPU share the same pool \u2014 no copies. This makes loading and running large models significantly faster than frameworks designed for CUDA.";
      let ci = 0;
      const iv = setInterval(() => {
        if (g !== gen) { clearInterval(iv); return; }
        if (ci < text.length && out) { out.textContent += text.substring(ci, ci + 3); ci += 3; }
        else {
          clearInterval(iv);
          setTimeout(() => {
            if (g !== gen) return;
            bEl.querySelector(".cursor")?.remove();
            bEl.innerHTML += L("&nbsp;") +
              L('<span class="t-dim">prompt 8 tokens \u00b7 completion 52 tokens \u00b7 58.3 tok/s \u00b7 0.89s</span>');
            advance(g, 3000);
          }, 300);
        }
      }, 25);
    }, 500);
  }

  function drawDL() {
    bEl.innerHTML = cmd + L("&nbsp;") +
      L('<span class="t-dim">Model not found locally. Downloading mistral...</span>') +
      '<div class="term-line"><div class="term-progress"><div class="term-progress-bar"><div class="term-progress-fill" id="rpb"></div></div><span class="t-dim" id="rpp">0%</span></div></div>';

    setTimeout(() => {
      if (g !== gen) return;
      const bar = document.getElementById("rpb");
      if (bar) bar.style.width = "100%";
      let p = 0;
      const iv = setInterval(() => {
        if (g !== gen) { clearInterval(iv); return; }
        p = Math.min(p + 3, 100);
        const el = document.getElementById("rpp");
        if (el) el.textContent = (p * 0.041).toFixed(1) + "/4.1 GB";
        if (p >= 100) { clearInterval(iv); drawChat(); }
      }, 30);
    }, 100);
  }

  drawSel(0);
  runSteps(g, [
    [1000, () => drawSel(1)],
    [800, () => drawSel(2)],
    [1200, () => drawDL()],
  ]);
}

/* ── QUANTIZE ── */
function animQuantize(g: number) {
  const cmd = L('<span class="t-prompt">$ </span><span class="t-cmd">ppmlx quantize <span class="t-arg">custom-7b</span></span>');
  const steps: StepPair[] = [
    [300, () => { bEl.innerHTML += L("&nbsp;") + L('<span class="t-blue">Downloading custom-7b from HuggingFace...</span>'); }],
    [1500, () => { bEl.innerHTML += L('<span class="t-dim">Converting safetensors \u2192 MLX format...</span>'); }],
    [1800, () => { bEl.innerHTML += L('<span class="t-dim">Quantizing to 4-bit (group_size=64)...</span>'); }],
    [2000, () => {
      bEl.innerHTML += L("&nbsp;") +
        L('<span class="t-ok">\u2713 Quantized model saved to: ~/.ppmlx/models/custom-7b-4bit/</span>') +
        L("&nbsp;") +
        L('<span class="t-dim">  Original:  </span><span class="t-cmd">13.2 GB</span>') +
        L('<span class="t-dim">  Quantized: </span><span class="t-val">4.1 GB</span> <span class="t-ok">(69% smaller)</span>') +
        L("&nbsp;") +
        L('<span class="t-dim">Run it: </span><span class="t-cmd">ppmlx run <span class="t-arg">custom-7b-4bit</span></span>');
    }],
  ];
  bEl.innerHTML = cmd;
  runSteps(g, steps, () => advance(g, 4000));
}

const ANIMS = [animLaunch, animServe, animPull, animRun, animQuantize];

function switchTo(idx: number) {
  gen++;
  cur = idx;
  const g = gen;
  document.querySelectorAll(".demo-tab").forEach((t, i) => t.classList.toggle("active", i === idx));
  if (tEl) tEl.textContent = TITLES[idx];
  if (dEl) dEl.textContent = DESCS[idx];
  typeCmd(CMDS[idx], g, () => ANIMS[idx](g));
}

export function initTerminalDemo() {
  const body = document.getElementById("demo-body");
  if (!body) return;
  bEl = body;
  tEl = document.getElementById("demo-title");
  dEl = document.getElementById("demo-desc");
  const term = document.getElementById("demo-term");

  document.querySelectorAll(".demo-tab").forEach((t) => {
    t.addEventListener("click", () => switchTo(parseInt((t as HTMLElement).dataset.idx || "0")));
  });

  let started = false;
  if (term) {
    new IntersectionObserver(
      (ee) => {
        if (ee[0].isIntersecting && !started) {
          started = true;
          setTimeout(() => switchTo(0), 600);
        }
      },
      { threshold: 0.3 }
    ).observe(term);
  }
}
