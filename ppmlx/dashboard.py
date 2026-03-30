"""Real-time monitoring dashboard for ppmlx.

Provides:
- GET /dashboard — single-page HTML dashboard (dark theme, auto-refresh)
- GET /api/dashboard/metrics — time-series metrics (request rate, latency)
- GET /api/dashboard/requests — recent request log
- GET /api/dashboard/system — CPU/memory/GPU stats + loaded models
- GET /api/dashboard/stream — SSE stream pushing new data every 2 seconds
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

router = APIRouter()

# ── Helpers ──────────────────────────────────────────────────────────────

def _get_start_time() -> float:
    """Return server start time from ppmlx.server module."""
    try:
        from ppmlx.server import _start_time
        return _start_time
    except Exception:
        return time.time()


def _get_system_info() -> dict[str, Any]:
    """Gather system RAM, loaded models, uptime."""
    ram_gb = 0.0
    try:
        from ppmlx.memory import get_system_ram_gb
        ram_gb = get_system_ram_gb()
    except Exception:
        pass

    loaded: list[str] = []
    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        pass

    uptime = int(time.time() - _get_start_time())

    return {
        "memory_total_gb": round(ram_gb, 1),
        "loaded_models": loaded,
        "uptime_seconds": uptime,
    }


def _get_db_stats(since_hours: float = 24) -> dict[str, Any]:
    """Return aggregate stats from the database."""
    try:
        from ppmlx.db import get_db
        return get_db().get_stats(since_hours=since_hours)
    except Exception:
        return {"total_requests": 0, "avg_duration_ms": None, "by_model": []}


def _get_recent_requests(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent request rows from the database."""
    try:
        from ppmlx.db import get_db
        return get_db().query_requests(limit=limit)
    except Exception:
        return []


def _get_time_series(since_hours: float = 1) -> list[dict[str, Any]]:
    """Return per-minute request counts and avg latency for the given window."""
    try:
        from ppmlx.db import get_db
        return get_db().query_time_series(since_hours=since_hours)
    except Exception:
        return []


# ── API Endpoints ────────────────────────────────────────────────────────

@router.get("/api/dashboard/system")
async def dashboard_system():
    """System info: memory, loaded models, uptime."""
    return JSONResponse(_get_system_info())


@router.get("/api/dashboard/metrics")
async def dashboard_metrics():
    """Aggregate stats + time-series data."""
    stats = _get_db_stats()
    ts = _get_time_series()
    return JSONResponse({"stats": stats, "time_series": ts})


@router.get("/api/dashboard/requests")
async def dashboard_requests(limit: int = 50):
    """Recent request log."""
    rows = _get_recent_requests(limit=limit)
    return JSONResponse(rows)


@router.get("/api/dashboard/stream")
async def dashboard_stream(request: Request):
    """SSE stream pushing dashboard data every 2 seconds."""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            payload = {
                "system": _get_system_info(),
                "stats": _get_db_stats(),
                "time_series": _get_time_series(),
                "recent_requests": _get_recent_requests(limit=20),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── HTML Dashboard ───────────────────────────────────────────────────────

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ppmlx Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;
      --accent:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;--purple:#bc8cff}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     background:var(--bg);color:var(--text);line-height:1.5}
.header{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;
        display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:18px;font-weight:600}
.header h1 span{color:var(--accent)}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;
            background:var(--green);margin-right:8px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.header-right{font-size:13px;color:var(--dim)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
      gap:16px;padding:20px 24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;
      padding:16px;overflow:hidden}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.5px;
         color:var(--dim);margin-bottom:12px}
.stat{font-size:32px;font-weight:700;color:var(--accent)}
.stat-label{font-size:12px;color:var(--dim);margin-top:2px}
.model-list{list-style:none}
.model-list li{padding:6px 0;border-bottom:1px solid var(--border);font-size:14px;
               display:flex;justify-content:space-between}
.model-list li:last-child{border-bottom:none}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;
       font-weight:600;text-transform:uppercase}
.badge-ok{background:rgba(63,185,80,.15);color:var(--green)}
.badge-error{background:rgba(248,81,73,.15);color:var(--red)}
.full-width{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{text-align:left;padding:8px;border-bottom:2px solid var(--border);
         color:var(--dim);font-weight:600;cursor:pointer;user-select:none;
         white-space:nowrap}
thead th:hover{color:var(--accent)}
tbody td{padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
tbody tr:hover{background:rgba(88,166,255,.06)}
.chart-container{position:relative;width:100%;height:160px}
canvas{width:100%!important;height:100%!important}
.empty{text-align:center;padding:32px;color:var(--dim);font-size:14px}
.filter-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filter-bar select,.filter-bar input{background:var(--bg);color:var(--text);
    border:1px solid var(--border);border-radius:4px;padding:4px 8px;font-size:12px}
@media(max-width:600px){.grid{padding:12px}.header{padding:12px}}
</style>
</head>
<body>
<div class="header">
  <h1><span class="status-dot"></span><span>ppmlx</span> Dashboard</h1>
  <div class="header-right">
    <span id="uptime">--</span> uptime &middot; v<span id="version">--</span>
  </div>
</div>

<div class="grid">
  <!-- Row 1: Key stats -->
  <div class="card">
    <h2>Total Requests (24h)</h2>
    <div class="stat" id="total-requests">0</div>
  </div>
  <div class="card">
    <h2>Avg Latency</h2>
    <div class="stat" id="avg-latency">--</div>
    <div class="stat-label">milliseconds</div>
  </div>
  <div class="card">
    <h2>System Memory</h2>
    <div class="stat" id="memory-total">--</div>
    <div class="stat-label">GB total</div>
  </div>
  <div class="card">
    <h2>Loaded Models</h2>
    <div id="loaded-models-count" class="stat">0</div>
    <ul class="model-list" id="loaded-models"></ul>
  </div>

  <!-- Row 2: Charts -->
  <div class="card" style="grid-column:span 2">
    <h2>Request Rate (last hour)</h2>
    <div class="chart-container"><canvas id="rate-chart"></canvas></div>
  </div>
  <div class="card" style="grid-column:span 2">
    <h2>Avg Latency (last hour)</h2>
    <div class="chart-container"><canvas id="latency-chart"></canvas></div>
  </div>

  <!-- Row 3: Model breakdown -->
  <div class="card full-width">
    <h2>Model Usage (24h)</h2>
    <div id="model-usage-container">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Requests</th><th>Avg TPS</th><th>Avg TTFT</th><th>Errors</th>
          </tr>
        </thead>
        <tbody id="model-usage-body"></tbody>
      </table>
    </div>
  </div>

  <!-- Row 4: Request log -->
  <div class="card full-width">
    <h2>Recent Requests</h2>
    <div class="filter-bar">
      <select id="filter-status">
        <option value="">All Status</option>
        <option value="ok">OK</option>
        <option value="error">Error</option>
      </select>
      <input type="text" id="filter-model" placeholder="Filter by model..." />
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th data-sort="timestamp">Timestamp</th>
            <th data-sort="model_alias">Model</th>
            <th data-sort="endpoint">Endpoint</th>
            <th data-sort="status">Status</th>
            <th data-sort="total_duration_ms">Duration (ms)</th>
            <th data-sort="tokens_per_second">TPS</th>
            <th data-sort="prompt_tokens">Prompt Tok</th>
            <th data-sort="completion_tokens">Compl Tok</th>
          </tr>
        </thead>
        <tbody id="request-log-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
(function(){
"use strict";

// ── State ────────────────────────────────────────────────────────────
let allRequests = [];
let sortKey = "timestamp";
let sortAsc = false;
let filterStatus = "";
let filterModel = "";

// ── DOM refs ─────────────────────────────────────────────────────────
const $=id=>document.getElementById(id);

// ── Formatters ───────────────────────────────────────────────────────
function fmtDuration(s){
  if(s<60) return s+"s";
  if(s<3600) return Math.floor(s/60)+"m "+s%60+"s";
  const h=Math.floor(s/3600);
  return h+"h "+Math.floor((s%3600)/60)+"m";
}
function fmtNum(n,d){
  if(n==null) return "--";
  return typeof d==="number"?n.toFixed(d):n.toLocaleString();
}
function fmtTime(ts){
  if(!ts)return "--";
  try{const d=new Date(ts);return d.toLocaleTimeString()}catch(e){return ts}
}

// ── Chart drawing (pure Canvas, no deps) ─────────────────────────────
function drawLineChart(canvasId,labels,data,color,unit){
  const canvas=$(canvasId);
  if(!canvas)return;
  const ctx=canvas.getContext("2d");
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.parentElement.getBoundingClientRect();
  canvas.width=rect.width*dpr;
  canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const W=rect.width,H=rect.height;
  const pad={t:10,r:10,b:28,l:48};
  const gW=W-pad.l-pad.r,gH=H-pad.t-pad.b;

  ctx.clearRect(0,0,W,H);

  if(!data.length||data.every(v=>v==null)){
    ctx.fillStyle="#8b949e";ctx.font="13px sans-serif";ctx.textAlign="center";
    ctx.fillText("No data yet",W/2,H/2);return;
  }

  const vals=data.map(v=>v??0);
  const maxV=Math.max(...vals,1);

  // Grid lines
  ctx.strokeStyle="#30363d";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.t+gH*(1-i/4);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+gW,y);ctx.stroke();
    ctx.fillStyle="#8b949e";ctx.font="10px sans-serif";ctx.textAlign="right";
    ctx.fillText(fmtNum(maxV*i/4,0),pad.l-4,y+3);
  }

  // Line
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.lineJoin="round";ctx.beginPath();
  for(let i=0;i<vals.length;i++){
    const x=pad.l+(i/(Math.max(vals.length-1,1)))*gW;
    const y=pad.t+gH*(1-vals[i]/maxV);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();

  // Fill
  ctx.globalAlpha=.1;ctx.fillStyle=color;
  ctx.lineTo(pad.l+gW,pad.t+gH);ctx.lineTo(pad.l,pad.t+gH);ctx.closePath();ctx.fill();
  ctx.globalAlpha=1;

  // X labels (show ~5)
  if(labels.length){
    ctx.fillStyle="#8b949e";ctx.font="10px sans-serif";ctx.textAlign="center";
    const step=Math.max(1,Math.floor(labels.length/5));
    for(let i=0;i<labels.length;i+=step){
      const x=pad.l+(i/(Math.max(labels.length-1,1)))*gW;
      const lbl=labels[i];
      const short=lbl?lbl.substring(11,16):"";
      ctx.fillText(short,x,H-4);
    }
  }
}

// ── Render functions ─────────────────────────────────────────────────
function renderSystem(sys){
  $("uptime").textContent=fmtDuration(sys.uptime_seconds||0);
  $("memory-total").textContent=fmtNum(sys.memory_total_gb,1);
  const models=sys.loaded_models||[];
  $("loaded-models-count").textContent=models.length;
  const ul=$("loaded-models");
  ul.innerHTML=models.length?models.map(m=>"<li>"+m+"</li>").join(""):"<li class='empty'>None loaded</li>";
}

function renderStats(stats){
  $("total-requests").textContent=fmtNum(stats.total_requests);
  $("avg-latency").textContent=stats.avg_duration_ms!=null?fmtNum(stats.avg_duration_ms,0):"--";
  const tbody=$("model-usage-body");
  const models=stats.by_model||[];
  if(!models.length){
    tbody.innerHTML="<tr><td colspan='5' class='empty'>No requests yet</td></tr>";return;
  }
  tbody.innerHTML=models.map(m=>"<tr>"
    +"<td>"+m.model+"</td>"
    +"<td>"+m.count+"</td>"
    +"<td>"+(m.avg_tps!=null?fmtNum(m.avg_tps,1):"--")+"</td>"
    +"<td>"+(m.avg_ttft!=null?fmtNum(m.avg_ttft,0)+" ms":"--")+"</td>"
    +"<td>"+(m.errors?"<span class='badge badge-error'>"+m.errors+"</span>":"<span class='badge badge-ok'>0</span>")+"</td>"
    +"</tr>").join("");
}

function renderTimeSeries(ts){
  const labels=ts.map(p=>p.minute);
  const counts=ts.map(p=>p.count);
  const latencies=ts.map(p=>p.avg_latency_ms);
  drawLineChart("rate-chart",labels,counts,"#58a6ff","req");
  drawLineChart("latency-chart",labels,latencies,"#d29922","ms");
}

function renderRequests(rows){
  allRequests=rows;
  applyFiltersAndRender();
}

function applyFiltersAndRender(){
  let filtered=allRequests;
  if(filterStatus){
    filtered=filtered.filter(r=>r.status===filterStatus);
  }
  if(filterModel){
    const q=filterModel.toLowerCase();
    filtered=filtered.filter(r=>(r.model_alias||"").toLowerCase().includes(q));
  }
  // Sort
  filtered.sort((a,b)=>{
    let va=a[sortKey],vb=b[sortKey];
    if(va==null)va="";if(vb==null)vb="";
    if(typeof va==="number"&&typeof vb==="number") return sortAsc?va-vb:vb-va;
    return sortAsc?String(va).localeCompare(String(vb)):String(vb).localeCompare(String(va));
  });
  const tbody=$("request-log-body");
  if(!filtered.length){
    tbody.innerHTML="<tr><td colspan='8' class='empty'>No requests recorded</td></tr>";return;
  }
  tbody.innerHTML=filtered.map(r=>"<tr>"
    +"<td>"+fmtTime(r.timestamp)+"</td>"
    +"<td>"+(r.model_alias||"--")+"</td>"
    +"<td>"+(r.endpoint||"--")+"</td>"
    +"<td><span class='badge "+(r.status==="error"?"badge-error":"badge-ok")+"'>"+r.status+"</span></td>"
    +"<td>"+(r.total_duration_ms!=null?fmtNum(r.total_duration_ms,0):"--")+"</td>"
    +"<td>"+(r.tokens_per_second!=null?fmtNum(r.tokens_per_second,1):"--")+"</td>"
    +"<td>"+(r.prompt_tokens!=null?r.prompt_tokens:"--")+"</td>"
    +"<td>"+(r.completion_tokens!=null?r.completion_tokens:"--")+"</td>"
    +"</tr>").join("");
}

// ── Column sort ──────────────────────────────────────────────────────
document.querySelectorAll("th[data-sort]").forEach(th=>{
  th.addEventListener("click",()=>{
    const key=th.dataset.sort;
    if(sortKey===key)sortAsc=!sortAsc;else{sortKey=key;sortAsc=false}
    applyFiltersAndRender();
  });
});

// ── Filters ──────────────────────────────────────────────────────────
$("filter-status").addEventListener("change",e=>{filterStatus=e.target.value;applyFiltersAndRender()});
$("filter-model").addEventListener("input",e=>{filterModel=e.target.value;applyFiltersAndRender()});

// ── SSE real-time updates ────────────────────────────────────────────
function connectSSE(){
  const es=new EventSource("/api/dashboard/stream");
  es.onmessage=function(evt){
    try{
      const d=JSON.parse(evt.data);
      if(d.system)renderSystem(d.system);
      if(d.stats)renderStats(d.stats);
      if(d.time_series)renderTimeSeries(d.time_series);
      if(d.recent_requests)renderRequests(d.recent_requests);
    }catch(e){console.error("SSE parse error",e)}
  };
  es.onerror=function(){
    es.close();
    setTimeout(connectSSE,5000);
  };
}

// ── Initial fetch (don't wait for first SSE tick) ────────────────────
async function initialFetch(){
  try{
    const [sys,met,req]=await Promise.all([
      fetch("/api/dashboard/system").then(r=>r.json()),
      fetch("/api/dashboard/metrics").then(r=>r.json()),
      fetch("/api/dashboard/requests?limit=50").then(r=>r.json()),
    ]);
    renderSystem(sys);
    if(met.stats)renderStats(met.stats);
    if(met.time_series)renderTimeSeries(met.time_series);
    renderRequests(req);
  }catch(e){console.error("Initial fetch error",e)}
}

// ── Version ──────────────────────────────────────────────────────────
fetch("/health").then(r=>r.json()).then(d=>{
  if(d.version)$("version").textContent=d.version;
}).catch(()=>{});

initialFetch();
connectSSE();

// Redraw charts on resize (debounced)
let resizeTimer;
window.addEventListener("resize",()=>{
  clearTimeout(resizeTimer);
  resizeTimer=setTimeout(()=>{
    fetch("/api/dashboard/metrics").then(r=>r.json()).then(d=>{
      if(d.time_series)renderTimeSeries(d.time_series);
    }).catch(()=>{});
  },250);
});

})();
</script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard."""
    return HTMLResponse(_DASHBOARD_HTML)
