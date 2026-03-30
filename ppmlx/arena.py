"""Model Comparison Arena — side-by-side model comparison with ELO scoring.

Send the same prompt to two models concurrently, display results, vote,
and track ELO scores in a local SQLite database.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── ELO constants ─────────────────────────────────────────────────────

DEFAULT_ELO = 1500
K_FACTOR = 32


# ── Database path ─────────────────────────────────────────────────────

def _get_arena_db_path() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        base = get_ppmlx_dir()
    except ImportError:
        base = Path.home() / ".ppmlx"
    base.mkdir(parents=True, exist_ok=True)
    return base / "arena.db"


# ── Schema ────────────────────────────────────────────────────────────

_ARENA_SCHEMA = """
CREATE TABLE IF NOT EXISTS elo_scores (
    model       TEXT PRIMARY KEY,
    score       REAL NOT NULL DEFAULT 1500,
    wins        INTEGER NOT NULL DEFAULT 0,
    losses      INTEGER NOT NULL DEFAULT 0,
    draws       INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS arena_matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    match_id    TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    model_a     TEXT NOT NULL,
    model_b     TEXT NOT NULL,
    response_a  TEXT,
    response_b  TEXT,
    time_a_ms   REAL,
    time_b_ms   REAL,
    winner      TEXT,
    elo_a_before REAL,
    elo_b_before REAL,
    elo_a_after  REAL,
    elo_b_after  REAL
);

CREATE INDEX IF NOT EXISTS idx_matches_timestamp ON arena_matches(timestamp);
CREATE INDEX IF NOT EXISTS idx_matches_models ON arena_matches(model_a, model_b);
"""


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class ArenaResult:
    """Result of a single model's response in an arena match."""
    model: str
    response: str
    elapsed_ms: float
    error: str | None = None


@dataclass
class ArenaMatch:
    """A completed arena match between two models."""
    match_id: str
    prompt: str
    result_a: ArenaResult
    result_b: ArenaResult


@dataclass
class EloEntry:
    """ELO score entry for a model."""
    model: str
    score: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    draws: int = 0


# ── ELO calculation ──────────────────────────────────────────────────

def compute_elo_update(
    score_a: float, score_b: float, outcome: str,
) -> tuple[float, float]:
    """Compute new ELO scores after a match.

    outcome: "a" (model A wins), "b" (model B wins), "draw"
    Returns (new_score_a, new_score_b).
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((score_b - score_a) / 400.0))
    expected_b = 1.0 - expected_a

    if outcome == "a":
        actual_a, actual_b = 1.0, 0.0
    elif outcome == "b":
        actual_a, actual_b = 0.0, 1.0
    else:  # draw
        actual_a, actual_b = 0.5, 0.5

    new_a = score_a + K_FACTOR * (actual_a - expected_a)
    new_b = score_b + K_FACTOR * (actual_b - expected_b)
    return round(new_a, 1), round(new_b, 1)


# ── ArenaDB ──────────────────────────────────────────────────────────

class ArenaDB:
    """SQLite persistence for arena matches and ELO scores."""

    def __init__(self, path: Path | None = None):
        self._path = path or _get_arena_db_path()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._path)) as conn:
            conn.executescript(_ARENA_SCHEMA)

    def get_elo(self, model: str) -> float:
        with sqlite3.connect(str(self._path)) as conn:
            row = conn.execute(
                "SELECT score FROM elo_scores WHERE model = ?", (model,)
            ).fetchone()
        return row[0] if row else DEFAULT_ELO

    def get_all_elos(self) -> list[EloEntry]:
        with sqlite3.connect(str(self._path)) as conn:
            rows = conn.execute(
                "SELECT model, score, wins, losses, draws FROM elo_scores "
                "ORDER BY score DESC"
            ).fetchall()
        return [
            EloEntry(model=r[0], score=r[1], wins=r[2], losses=r[3], draws=r[4])
            for r in rows
        ]

    def _upsert_elo(self, conn: sqlite3.Connection, model: str, score: float,
                    win: int = 0, loss: int = 0, draw: int = 0) -> None:
        conn.execute(
            """INSERT INTO elo_scores (model, score, wins, losses, draws)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(model) DO UPDATE SET
                   score = ?,
                   wins = wins + ?,
                   losses = losses + ?,
                   draws = draws + ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
            (model, score, win, loss, draw, score, win, loss, draw),
        )

    def record_match(self, match: ArenaMatch, winner: str) -> tuple[float, float]:
        """Record a match result and update ELO scores.

        winner: "a", "b", or "draw"
        Returns (new_elo_a, new_elo_b).
        """
        elo_a = self.get_elo(match.result_a.model)
        elo_b = self.get_elo(match.result_b.model)
        new_a, new_b = compute_elo_update(elo_a, elo_b, winner)

        win_a = int(winner == "a")
        win_b = int(winner == "b")
        is_draw = int(winner == "draw")

        with sqlite3.connect(str(self._path)) as conn:
            conn.execute(
                """INSERT INTO arena_matches
                   (match_id, prompt, model_a, model_b, response_a, response_b,
                    time_a_ms, time_b_ms, winner,
                    elo_a_before, elo_b_before, elo_a_after, elo_b_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (match.match_id, match.prompt,
                 match.result_a.model, match.result_b.model,
                 match.result_a.response, match.result_b.response,
                 match.result_a.elapsed_ms, match.result_b.elapsed_ms,
                 winner, elo_a, elo_b, new_a, new_b),
            )
            self._upsert_elo(conn, match.result_a.model, new_a,
                             win=win_a, loss=win_b, draw=is_draw)
            self._upsert_elo(conn, match.result_b.model, new_b,
                             win=win_b, loss=win_a, draw=is_draw)

        return new_a, new_b

    def get_recent_matches(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(str(self._path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM arena_matches ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ── Arena runner ─────────────────────────────────────────────────────

async def _query_model_async(
    base_url: str, model: str, prompt: str, temperature: float = 0.7,
) -> ArenaResult:
    """Async version — send a chat completion request to the local ppmlx API."""
    import httpx

    start = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            elapsed = (time.monotonic() - start) * 1000
            return ArenaResult(model=model, response=text, elapsed_ms=elapsed)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return ArenaResult(
            model=model, response="", elapsed_ms=elapsed, error=str(e),
        )


def run_arena_match(
    model_a: str, model_b: str, prompt: str,
    base_url: str = "http://127.0.0.1:6767",
    temperature: float = 0.7,
) -> ArenaMatch:
    """Run a single arena match: query both models concurrently."""
    match_id = "arena-" + uuid.uuid4().hex[:12]

    async def _run() -> tuple[ArenaResult, ArenaResult]:
        a, b = await asyncio.gather(
            _query_model_async(base_url, model_a, prompt, temperature),
            _query_model_async(base_url, model_b, prompt, temperature),
        )
        return a, b

    result_a, result_b = asyncio.run(_run())
    return ArenaMatch(
        match_id=match_id, prompt=prompt,
        result_a=result_a, result_b=result_b,
    )


# ── Web UI HTML ──────────────────────────────────────────────────────

ARENA_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ppmlx Arena</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
          --dim: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149;
          --orange: #d29922; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  .subtitle { color: var(--dim); margin-bottom: 20px; font-size: 0.9rem; }

  /* Prompt input */
  .prompt-area { display: flex; gap: 10px; margin-bottom: 20px; }
  .prompt-area textarea { flex: 1; background: var(--card); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px; padding: 12px; font-size: 0.95rem;
    resize: vertical; min-height: 60px; font-family: inherit; }
  .prompt-area textarea:focus { outline: none; border-color: var(--accent); }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 8px;
    padding: 10px 20px; font-size: 0.95rem; cursor: pointer; font-weight: 600;
    white-space: nowrap; }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-secondary { background: var(--card); border: 1px solid var(--border); color: var(--text); }
  .btn-secondary:hover { border-color: var(--accent); }

  /* Model selectors */
  .model-selectors { display: flex; gap: 20px; margin-bottom: 20px; }
  .model-select { flex: 1; }
  .model-select label { display: block; color: var(--dim); font-size: 0.85rem; margin-bottom: 4px; }
  .model-select select { width: 100%; background: var(--card); border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 8px; font-size: 0.9rem; }

  /* Arena display */
  .arena { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .response-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; min-height: 200px; }
  .response-card.winner { border-color: var(--green); }
  .card-header { display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  .model-name { font-weight: 600; color: var(--accent); }
  .response-time { color: var(--dim); font-size: 0.85rem; }
  .response-text { white-space: pre-wrap; line-height: 1.6; font-size: 0.9rem; }
  .response-text.empty { color: var(--dim); font-style: italic; }
  .error-text { color: var(--red); }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Voting */
  .voting { display: flex; gap: 12px; justify-content: center; margin-bottom: 30px; }
  .vote-btn { padding: 10px 24px; font-size: 1rem; border-radius: 8px; }
  .vote-btn.a { background: var(--accent); }
  .vote-btn.b { background: var(--green); }
  .vote-btn.draw { background: var(--orange); color: #000; }

  /* Leaderboard */
  .leaderboard { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; }
  .leaderboard h2 { font-size: 1.1rem; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--dim); font-size: 0.85rem; padding: 8px 12px;
    border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  tr:last-child td { border-bottom: none; }
  .elo-score { font-weight: 700; color: var(--accent); }

  @media (max-width: 768px) {
    .arena { grid-template-columns: 1fr; }
    .model-selectors { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>ppmlx Arena</h1>
  <p class="subtitle">Side-by-side model comparison with ELO scoring</p>

  <div class="model-selectors">
    <div class="model-select">
      <label>Model A</label>
      <select id="model-a"></select>
    </div>
    <div class="model-select">
      <label>Model B</label>
      <select id="model-b"></select>
    </div>
  </div>

  <div class="prompt-area">
    <textarea id="prompt" placeholder="Enter your prompt..." rows="2"></textarea>
    <button class="btn" id="run-btn" onclick="runArena()">Compare</button>
  </div>

  <div class="arena" id="arena" style="display:none;">
    <div class="response-card" id="card-a">
      <div class="card-header">
        <span class="model-name" id="name-a"></span>
        <span class="response-time" id="time-a"></span>
      </div>
      <div class="response-text" id="text-a"></div>
    </div>
    <div class="response-card" id="card-b">
      <div class="card-header">
        <span class="model-name" id="name-b"></span>
        <span class="response-time" id="time-b"></span>
      </div>
      <div class="response-text" id="text-b"></div>
    </div>
  </div>

  <div class="voting" id="voting" style="display:none;">
    <button class="btn vote-btn a" onclick="vote('a')">A is better</button>
    <button class="btn vote-btn draw" onclick="vote('draw')">Draw</button>
    <button class="btn vote-btn b" onclick="vote('b')">B is better</button>
  </div>

  <div class="leaderboard" id="leaderboard">
    <h2>Leaderboard</h2>
    <table>
      <thead><tr><th>#</th><th>Model</th><th>ELO</th><th>W</th><th>L</th><th>D</th></tr></thead>
      <tbody id="lb-body"><tr><td colspan="6" style="color:var(--dim)">No matches yet</td></tr></tbody>
    </table>
  </div>
</div>

<script>
let currentMatch = null;

async function loadModels() {
  try {
    const resp = await fetch('/v1/models');
    const data = await resp.json();
    const models = data.data || [];
    const selA = document.getElementById('model-a');
    const selB = document.getElementById('model-b');
    selA.innerHTML = '';
    selB.innerHTML = '';
    models.forEach((m, i) => {
      const optA = new Option(m.id, m.id);
      const optB = new Option(m.id, m.id);
      selA.add(optA);
      selB.add(optB);
    });
    // Default to different models if possible
    if (models.length > 1) selB.selectedIndex = 1;
  } catch(e) {
    console.error('Failed to load models:', e);
  }
}

async function loadLeaderboard() {
  try {
    const resp = await fetch('/arena/leaderboard');
    const data = await resp.json();
    const tbody = document.getElementById('lb-body');
    if (!data.length) {
      tbody.innerHTML = '<tr><td colspan="6" style="color:var(--dim)">No matches yet</td></tr>';
      return;
    }
    tbody.innerHTML = data.map((e, i) =>
      `<tr><td>${i+1}</td><td>${e.model}</td><td class="elo-score">${e.score.toFixed(1)}</td>` +
      `<td>${e.wins}</td><td>${e.losses}</td><td>${e.draws}</td></tr>`
    ).join('');
  } catch(e) {
    console.error('Failed to load leaderboard:', e);
  }
}

async function runArena() {
  const modelA = document.getElementById('model-a').value;
  const modelB = document.getElementById('model-b').value;
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  if (modelA === modelB) { alert('Please select two different models'); return; }

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Running...';

  document.getElementById('arena').style.display = 'grid';
  document.getElementById('voting').style.display = 'none';
  document.getElementById('name-a').textContent = modelA;
  document.getElementById('name-b').textContent = modelB;
  document.getElementById('text-a').innerHTML = '<div class="spinner"></div> Generating...';
  document.getElementById('text-b').innerHTML = '<div class="spinner"></div> Generating...';
  document.getElementById('time-a').textContent = '';
  document.getElementById('time-b').textContent = '';
  document.getElementById('card-a').classList.remove('winner');
  document.getElementById('card-b').classList.remove('winner');

  try {
    const resp = await fetch('/arena/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_a: modelA, model_b: modelB, prompt: prompt}),
    });
    const data = await resp.json();
    currentMatch = data;

    const ta = document.getElementById('text-a');
    const tb = document.getElementById('text-b');
    if (data.result_a.error) {
      ta.innerHTML = `<span class="error-text">Error: ${data.result_a.error}</span>`;
    } else {
      ta.textContent = data.result_a.response;
      ta.classList.remove('empty');
    }
    if (data.result_b.error) {
      tb.innerHTML = `<span class="error-text">Error: ${data.result_b.error}</span>`;
    } else {
      tb.textContent = data.result_b.response;
      tb.classList.remove('empty');
    }
    document.getElementById('time-a').textContent = `${(data.result_a.elapsed_ms/1000).toFixed(1)}s`;
    document.getElementById('time-b').textContent = `${(data.result_b.elapsed_ms/1000).toFixed(1)}s`;
    document.getElementById('voting').style.display = 'flex';
  } catch(e) {
    document.getElementById('text-a').innerHTML = `<span class="error-text">${e}</span>`;
    document.getElementById('text-b').innerHTML = '';
  }
  btn.disabled = false;
  btn.textContent = 'Compare';
}

async function vote(winner) {
  if (!currentMatch) return;
  try {
    const resp = await fetch('/arena/vote', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({match_id: currentMatch.match_id, winner: winner}),
    });
    const data = await resp.json();
    document.getElementById('voting').style.display = 'none';
    if (winner === 'a') document.getElementById('card-a').classList.add('winner');
    else if (winner === 'b') document.getElementById('card-b').classList.add('winner');
    await loadLeaderboard();
  } catch(e) {
    console.error('Vote failed:', e);
  }
}

// Handle Enter key in prompt
document.getElementById('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runArena(); }
});

loadModels();
loadLeaderboard();
</script>
</body>
</html>"""
