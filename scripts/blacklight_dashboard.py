#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PHASE_LABELS = {
    "initializing": "Initializing",
    "collecting": "Collecting teacher data",
    "syncing": "Training model",
    "stopping": "Stopping",
    "finished": "Finished",
}


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(num_bytes, 0))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_duration(seconds: int) -> str:
    if seconds < 0:
        return "n/a"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_var_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / "var" / raw_path


def path_exists_and_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_model_stats(model_path: Path) -> tuple[int, int]:
    if not model_path.exists():
        return (0, 0)
    with model_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == 2:
                parts = line.strip().split()
                if len(parts) >= 3:
                    return (safe_int(parts[1]), safe_int(parts[2]))
                break
    return (0, 0)


def read_recent_metrics(metrics_path: Path, limit: int = 180) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    if not metrics_path.exists():
        return ([], None)

    rows: deque[dict[str, str]] = deque(maxlen=limit)
    last_row: dict[str, str] | None = None
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            last_row = row
    return (list(rows), last_row)


def read_recent_events(events_path: Path, limit: int = 12) -> list[str]:
    if not events_path.exists():
        return []
    lines = [line.strip() for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-limit:]


def latest_checkpoint(run_dir: Path) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return ""
    checkpoints = sorted(checkpoint_dir.glob("blacklight_ep*.txt"))
    if not checkpoints:
        return ""
    return str(checkpoints[-1])


def source_bytes(repo_root: Path, source_list_path: Path) -> int:
    if not source_list_path.exists():
        return 0

    total = 0
    for raw_line in source_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        source_path = resolve_var_path(repo_root, line)
        if source_path.exists():
            total += source_path.stat().st_size
    return total


def build_commands(repo_root: Path, manifest: dict[str, str]) -> dict[str, str]:
    repo = str(repo_root)
    run_name = manifest.get("RUN_NAME", "")

    commands = {
        "collect": f"cd {repo}\n./scripts/blacklight.sh collect",
        "train": f"cd {repo}\n./scripts/blacklight.sh train",
        "dashboard": f"cd {repo}\n./scripts/blacklight.sh dashboard",
    }

    if run_name:
        commands["benchmark"] = (
            f"cd {repo}\n"
            f"./scripts/blacklight.sh bench --suite classic_primary --candidate {run_name}"
        )

    return commands


def resolve_manifest(repo_root: Path, run_name: str | None) -> dict[str, str]:
    if run_name:
        manifest_path = repo_root / "var" / "blacklight_runs" / run_name / "run_manifest.env"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for {run_name}")
        return parse_env_file(manifest_path)

    latest_manifest_path = repo_root / "var" / "blacklight_runs" / "latest_run.env"
    if not latest_manifest_path.exists():
        raise FileNotFoundError("No latest Blacklight run found yet.")
    return parse_env_file(latest_manifest_path)


def build_payload(repo_root: Path, run_name: str | None = None) -> dict[str, Any]:
    manifest = resolve_manifest(repo_root, run_name)
    run_dir = Path(manifest.get("RUN_DIR_ABS", ""))
    model_path = Path(manifest.get("MODEL_ABS", ""))
    metrics_path = Path(manifest.get("METRICS_ABS", ""))
    progress_path = Path(manifest.get("PROGRESS_ABS", ""))
    events_path = Path(manifest.get("EVENTS_LOG_ABS", ""))
    source_list_path = Path(manifest.get("SOURCE_LIST_ABS", ""))
    progress = parse_env_file(progress_path)
    recent_metrics, last_metric = read_recent_metrics(metrics_path)
    events = read_recent_events(events_path)

    model_episodes, model_updates = read_model_stats(model_path)
    trainer_pid = safe_int(progress.get("TRAINER_PID", "0"))
    trainer_running = path_exists_and_running(trainer_pid)
    phase = progress.get("PHASE") or "unknown"
    phase_label = PHASE_LABELS.get(phase, "Unknown")
    started_at = safe_int(progress.get("TRAINING_STARTED_AT", "0"))
    elapsed_seconds = max(int(time.time()) - started_at, 0) if started_at else 0
    data_bytes = source_bytes(repo_root, source_list_path)

    latest_policy_loss = safe_float(last_metric.get("policy_loss") if last_metric else 0.0)
    latest_entropy = safe_float(last_metric.get("entropy") if last_metric else 0.0)
    latest_steps = safe_float(last_metric.get("steps") if last_metric else 0.0)
    latest_episode = safe_int(last_metric.get("episode") if last_metric else 0)

    chart_rows = []
    for row in recent_metrics[-120:]:
        chart_rows.append(
            {
                "episode": safe_int(row.get("episode")),
                "policy_loss": safe_float(row.get("policy_loss")),
                "entropy": safe_float(row.get("entropy")),
                "steps": safe_float(row.get("steps")),
                "updates": safe_int(row.get("policy_updates")),
            }
        )

    return {
        "generated_at": int(time.time()),
        "phase": phase,
        "phase_label": phase_label,
        "trainer_running": trainer_running,
        "training_mode": manifest.get("TRAINING_MODE", "teacher"),
        "replay_only": manifest.get("REPLAY_ONLY", "0") == "1",
        "run": {
            "name": manifest.get("RUN_NAME", ""),
            "dir": str(run_dir),
            "model_file": str(model_path),
            "metrics_file": str(metrics_path),
            "events_file": str(events_path),
            "trainer_log": manifest.get("TRAINER_LOG_ABS", ""),
            "latest_checkpoint": latest_checkpoint(run_dir),
            "parent_model": manifest.get("PARENT_MODEL", ""),
            "source_list": str(source_list_path),
        },
        "summary": {
            "model_episodes": model_episodes,
            "model_updates": model_updates,
            "metrics_episode": latest_episode,
            "policy_loss": latest_policy_loss,
            "entropy": latest_entropy,
            "steps": latest_steps,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_display": format_duration(elapsed_seconds),
            "parallel_workers": safe_int(manifest.get("PARALLEL_WORKERS", "0")),
            "sync_seconds": safe_int(manifest.get("SYNC_SECONDS", "0")),
            "data_bytes": data_bytes,
            "data_display": format_bytes(data_bytes),
            "trainer_pid": trainer_pid,
        },
        "charts": chart_rows,
        "events": events,
        "commands": build_commands(repo_root, manifest),
    }


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blacklight Dashboard</title>
  <style>
    :root {
      --bg-1: #f2eadf;
      --bg-2: #d8e5dc;
      --card: rgba(255, 252, 246, 0.82);
      --line: rgba(27, 43, 36, 0.12);
      --ink: #14211b;
      --muted: #50635a;
      --accent: #0e7c66;
      --accent-soft: rgba(14, 124, 102, 0.14);
      --warm: #bc5e39;
      --warm-soft: rgba(188, 94, 57, 0.14);
      --shadow: 0 24px 60px rgba(35, 45, 35, 0.12);
      --radius-xl: 26px;
      --radius-lg: 18px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "SF Pro Text", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.78), transparent 32%),
        radial-gradient(circle at 85% 10%, rgba(14,124,102,0.16), transparent 24%),
        linear-gradient(140deg, var(--bg-1), var(--bg-2));
    }

    .page {
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 22px 42px;
    }

    .hero {
      background: linear-gradient(135deg, rgba(250,247,241,0.9), rgba(234,244,239,0.82));
      border: 1px solid rgba(255,255,255,0.65);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      padding: 30px 32px;
      display: grid;
      gap: 18px;
    }

    .eyebrow {
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 0.72rem;
      color: var(--muted);
      font-weight: 700;
    }

    h1 {
      margin: 0;
      font-family: "Iowan Old Style", "Georgia", serif;
      font-size: clamp(2.3rem, 4vw, 4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .hero p {
      margin: 0;
      max-width: 60rem;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.6;
    }

    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(20, 33, 27, 0.08);
      font-size: 0.92rem;
      color: var(--muted);
    }

    .pill-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warm);
      box-shadow: 0 0 0 5px rgba(188, 94, 57, 0.12);
    }

    .pill-dot.live {
      background: var(--accent);
      box-shadow: 0 0 0 5px rgba(14, 124, 102, 0.16);
    }

    .grid {
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }

    .card {
      grid-column: span 3;
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.64);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      padding: 20px;
      backdrop-filter: blur(18px);
    }

    .card.wide { grid-column: span 6; }
    .card.full { grid-column: 1 / -1; }

    .label {
      font-size: 0.76rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 10px;
    }

    .metric {
      font-size: clamp(1.8rem, 3.4vw, 3rem);
      font-weight: 700;
      line-height: 1;
      letter-spacing: -0.04em;
      margin: 0;
    }

    .subtext {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.96rem;
      line-height: 1.5;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 16px;
    }

    .section-title {
      margin: 0;
      font-size: 1.1rem;
      font-weight: 700;
    }

    .section-kicker {
      color: var(--muted);
      font-size: 0.92rem;
    }

    .charts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .chart-shell {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px 12px;
      background: rgba(255,255,255,0.55);
    }

    .chart-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 10px;
    }

    .chart-title strong {
      font-size: 1rem;
    }

    .chart-title span {
      color: var(--muted);
      font-size: 0.86rem;
    }

    canvas {
      width: 100%;
      height: 220px;
      display: block;
    }

    .events {
      display: grid;
      gap: 10px;
    }

    .event {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.5);
      font-size: 0.93rem;
      line-height: 1.5;
      color: var(--muted);
    }

    .event strong {
      color: var(--ink);
    }

    .details {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 18px;
    }

    .detail-row {
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }

    .detail-row .detail-label {
      font-size: 0.76rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 700;
    }

    .detail-row .detail-value {
      word-break: break-word;
      font-size: 0.95rem;
      line-height: 1.45;
    }

    .commands {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .command-card {
      border-radius: 18px;
      padding: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.78), rgba(250,246,238,0.72));
      border: 1px solid rgba(20, 33, 27, 0.08);
    }

    .command-card.warm {
      background:
        linear-gradient(180deg, rgba(255,246,241,0.92), rgba(255,238,230,0.82));
    }

    .command-card.teal {
      background:
        linear-gradient(180deg, rgba(240,250,246,0.92), rgba(227,244,238,0.82));
    }

    .command-card h3 {
      margin: 0 0 8px;
      font-size: 1rem;
    }

    .command-card p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }

    pre {
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: rgba(19, 28, 24, 0.95);
      color: #f6f1e8;
      overflow-x: auto;
      font-size: 0.86rem;
      line-height: 1.45;
      font-family: "SFMono-Regular", "Menlo", monospace;
    }

    .copy-row {
      display: flex;
      justify-content: flex-end;
      margin-top: 12px;
    }

    button {
      appearance: none;
      border: none;
      cursor: pointer;
      padding: 10px 14px;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      font-size: 0.9rem;
      box-shadow: 0 8px 22px rgba(14, 124, 102, 0.24);
    }

    button.secondary {
      background: var(--warm);
      box-shadow: 0 8px 22px rgba(188, 94, 57, 0.24);
    }

    .footer-note {
      margin-top: 24px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.6;
      text-align: center;
    }

    @media (max-width: 1080px) {
      .card, .card.wide { grid-column: span 6; }
      .commands, .charts, .details { grid-template-columns: 1fr; }
    }

    @media (max-width: 720px) {
      .page { padding: 18px 14px 28px; }
      .hero { padding: 22px 20px; }
      .card, .card.wide { grid-column: 1 / -1; }
      h1 { font-size: 2.2rem; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Blacklight Command Deck</div>
      <h1>Neural Training Without the Terminal Sludge</h1>
      <p>The one-screen view for Blacklight collection, training, and benchmarking. The bot data stays game-native. The UI just stops making you babysit raw shell output.</p>
      <div class="hero-meta">
        <div class="pill"><span class="pill-dot" id="liveDot"></span><span id="runName">Loading run...</span></div>
        <div class="pill"><span id="phaseText">Checking state...</span></div>
        <div class="pill"><span id="refreshText">Waiting for first refresh...</span></div>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <div class="label">Updates</div>
        <p class="metric" id="updatesMetric">0</p>
        <div class="subtext">Current saved model updates in the CNN checkpoint.</div>
      </article>
      <article class="card">
        <div class="label">Policy Loss</div>
        <p class="metric" id="policyMetric">0.000000</p>
        <div class="subtext">Lower is better. This is how closely the net is matching the teacher actions.</div>
      </article>
      <article class="card">
        <div class="label">Entropy</div>
        <p class="metric" id="entropyMetric">0.000000</p>
        <div class="subtext">Higher means more spread in choices. Lower means more confident policy behavior.</div>
      </article>
      <article class="card">
        <div class="label">Teacher Data</div>
        <p class="metric" id="dataMetric">0 B</p>
        <div class="subtext">Teacher gameplay data available to train the current CNN.</div>
      </article>

      <article class="card wide">
        <div class="section-header">
          <h2 class="section-title">Learning Curve</h2>
          <div class="section-kicker">Recent training trend from the live metrics CSV</div>
        </div>
        <div class="charts">
          <div class="chart-shell">
            <div class="chart-title"><strong>Policy Loss</strong><span id="policyTrendText">Waiting for data...</span></div>
            <canvas id="policyChart" width="800" height="220"></canvas>
          </div>
          <div class="chart-shell">
            <div class="chart-title"><strong>Entropy</strong><span id="entropyTrendText">Waiting for data...</span></div>
            <canvas id="entropyChart" width="800" height="220"></canvas>
          </div>
        </div>
      </article>

      <article class="card wide">
        <div class="section-header">
          <h2 class="section-title">Run Details</h2>
          <div class="section-kicker">The files and state that actually matter</div>
        </div>
        <div class="details">
          <div class="detail-row">
            <div class="detail-label">Run Directory</div>
            <div class="detail-value" id="runDir"></div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Model File</div>
            <div class="detail-value" id="modelFile"></div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Metrics File</div>
            <div class="detail-value" id="metricsFile"></div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Latest Checkpoint</div>
            <div class="detail-value" id="checkpointFile"></div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Parent Model</div>
            <div class="detail-value" id="parentModel"></div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Elapsed</div>
            <div class="detail-value" id="elapsedText"></div>
          </div>
        </div>
      </article>

      <article class="card full">
        <div class="section-header">
          <h2 class="section-title">What You Actually Run</h2>
          <div class="section-kicker">One command to collect data, one command to train, plus benchmark and dashboard</div>
        </div>
        <div class="commands">
          <div class="command-card teal">
            <h3>Collect Teacher Data</h3>
            <p>This generates fresh teacher gameplay data from the built-in bots using the current best CNN as the baseline player.</p>
            <pre id="collectCommand"></pre>
            <div class="copy-row"><button data-copy="collectCommand">Copy Collect</button></div>
          </div>
          <div class="command-card warm">
            <h3>Train The CNN</h3>
            <p>This trains the CNN from the latest collected teacher data and automatically resumes the best known model.</p>
            <pre id="trainCommand"></pre>
            <div class="copy-row"><button data-copy="trainCommand">Copy Train</button></div>
          </div>
          <div class="command-card teal">
            <h3>Benchmark The Current Model</h3>
            <p>This is the one you use after training to see whether a run is actually worth keeping.</p>
            <pre id="benchCommand"></pre>
            <div class="copy-row"><button data-copy="benchCommand">Copy Benchmark</button></div>
          </div>
          <div class="command-card warm">
            <h3>Open The Dashboard</h3>
            <p>This opens the browser UI so you can watch the run without living in terminal output.</p>
            <pre id="dashboardCommand"></pre>
            <div class="copy-row"><button class="secondary" data-copy="dashboardCommand">Copy Dashboard</button></div>
          </div>
        </div>
      </article>

      <article class="card full">
        <div class="section-header">
          <h2 class="section-title">Recent Events</h2>
          <div class="section-kicker">The last lines from the run event log</div>
        </div>
        <div class="events" id="eventsList"></div>
      </article>
    </section>

    <div class="footer-note">
      This dashboard auto-refreshes every 3 seconds. The CLI still exists. You just do not have to stare at it anymore.
    </div>
  </div>

  <script>
    const query = new URLSearchParams(window.location.search);
    const runParam = query.get("run");
    const apiPath = runParam ? `/api/dashboard?run=${encodeURIComponent(runParam)}` : "/api/dashboard";

    function numberish(value, digits = 6) {
      const num = Number(value);
      if (!Number.isFinite(num)) return "n/a";
      return num.toFixed(digits);
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value || "n/a";
    }

    function drawChart(canvasId, points, key, color, fillColor) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      if (!points.length) {
        ctx.fillStyle = "rgba(80, 99, 90, 0.8)";
        ctx.font = "16px Avenir Next";
        ctx.fillText("No data yet", 24, 32);
        return;
      }

      const values = points.map((point) => Number(point[key])).filter((value) => Number.isFinite(value));
      if (!values.length) {
        return;
      }

      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(max - min, 0.000001);
      const left = 24;
      const top = 16;
      const bottom = height - 26;
      const right = width - 16;

      ctx.strokeStyle = "rgba(20, 33, 27, 0.10)";
      ctx.lineWidth = 1;
      for (let index = 0; index < 4; index += 1) {
        const y = top + ((bottom - top) * index) / 3;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
      }

      ctx.beginPath();
      points.forEach((point, index) => {
        const x = left + ((right - left) * index) / Math.max(points.length - 1, 1);
        const value = Number(point[key]);
        const y = bottom - ((value - min) / span) * (bottom - top);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.stroke();

      ctx.lineTo(right, bottom);
      ctx.lineTo(left, bottom);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    function renderEvents(events) {
      const root = document.getElementById("eventsList");
      root.innerHTML = "";
      if (!events.length) {
        const node = document.createElement("div");
        node.className = "event";
        node.textContent = "No events yet.";
        root.appendChild(node);
        return;
      }
      for (const line of events) {
        const item = document.createElement("div");
        item.className = "event";
        item.textContent = line;
        root.appendChild(item);
      }
    }

    function bindCopyButtons() {
      document.querySelectorAll("button[data-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          const target = document.getElementById(button.dataset.copy);
          if (!target) return;
          await navigator.clipboard.writeText(target.textContent);
          const previous = button.textContent;
          button.textContent = "Copied";
          setTimeout(() => { button.textContent = previous; }, 1200);
        });
      });
    }

    async function refresh() {
      const response = await fetch(apiPath, { cache: "no-store" });
      const payload = await response.json();

      setText("runName", payload.run.name || "No active run");
      setText("phaseText", payload.phase_label + (payload.trainer_running ? " • live" : " • stopped"));
      setText("refreshText", `Refreshed ${new Date(payload.generated_at * 1000).toLocaleTimeString()}`);
      document.getElementById("liveDot").className = payload.trainer_running ? "pill-dot live" : "pill-dot";

      setText("updatesMetric", String(payload.summary.model_updates || 0));
      setText("policyMetric", numberish(payload.summary.policy_loss));
      setText("entropyMetric", numberish(payload.summary.entropy));
      setText("dataMetric", payload.summary.data_display);
      setText("elapsedText", payload.summary.elapsed_display);

      setText("runDir", payload.run.dir);
      setText("modelFile", payload.run.model_file);
      setText("metricsFile", payload.run.metrics_file);
      setText("checkpointFile", payload.run.latest_checkpoint || "No checkpoint yet");
      setText("parentModel", payload.run.parent_model || "No parent model");

      setText("collectCommand", payload.commands.collect || "n/a");
      setText("trainCommand", payload.commands.train || "n/a");
      setText("benchCommand", payload.commands.benchmark || "n/a");
      setText("dashboardCommand", payload.commands.dashboard || "n/a");

      const charts = payload.charts || [];
      const policyValues = charts.map((point) => Number(point.policy_loss)).filter((value) => Number.isFinite(value));
      const entropyValues = charts.map((point) => Number(point.entropy)).filter((value) => Number.isFinite(value));
      const lastPolicy = policyValues.length ? policyValues[policyValues.length - 1] : 0;
      const firstPolicy = policyValues.length ? policyValues[0] : 0;
      const lastEntropy = entropyValues.length ? entropyValues[entropyValues.length - 1] : 0;
      const firstEntropy = entropyValues.length ? entropyValues[0] : 0;
      setText("policyTrendText", `${numberish(firstPolicy)} → ${numberish(lastPolicy)}`);
      setText("entropyTrendText", `${numberish(firstEntropy)} → ${numberish(lastEntropy)}`);

      drawChart("policyChart", charts, "policy_loss", "#0e7c66", "rgba(14, 124, 102, 0.16)");
      drawChart("entropyChart", charts, "entropy", "#bc5e39", "rgba(188, 94, 57, 0.16)");
      renderEvents(payload.events || []);
    }

    bindCopyButtons();
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    repo_root: Path
    default_run_name: str | None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_html(HTML_PAGE.encode("utf-8"))
            return

        if parsed.path == "/api/dashboard":
            params = parse_qs(parsed.query)
            run_name = params.get("run", [self.default_run_name])[0]
            try:
                payload = build_payload(self.repo_root, run_name)
            except FileNotFoundError as exc:
                self.respond_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self.respond_json(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def respond_html(self, content: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def respond_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def make_handler(repo_root: Path, run_name: str | None):
    class _Handler(DashboardHandler):
        pass

    _Handler.repo_root = repo_root
    _Handler.default_run_name = run_name
    return _Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the Blacklight browser dashboard.")
    parser.add_argument("--repo-root", required=True, help="Absolute repo root path")
    parser.add_argument("--run", help="Specific run name to inspect")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser")
    parser.add_argument("--dump-json", action="store_true", help="Print one JSON snapshot and exit")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.dump_json:
        payload = build_payload(repo_root, args.run)
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return 0

    handler_cls = make_handler(repo_root, args.run)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    url = f"http://{args.host}:{args.port}/"

    def stop_server(_signum: int, _frame: Any) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    print(f"Blacklight dashboard running at {url}")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
