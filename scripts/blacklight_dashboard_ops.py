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


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "Open-ended"
    if seconds < 0:
        return "n/a"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def clamp_percent(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 100:
        return 100.0
    return value


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

    rows: list[dict[str, str]] = []
    last_row: dict[str, str] | None = None
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            last_row = row

    if len(rows) > limit:
        rows = rows[-limit:]
    return (rows, last_row)


def read_recent_lines(path: Path, limit: int = 12) -> list[str]:
    if not path.exists():
        return []
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-limit:]


def latest_checkpoint(run_dir: Path) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return ""
    checkpoints = sorted(checkpoint_dir.glob("blacklight_ep*.txt"))
    if not checkpoints:
        return ""
    return str(checkpoints[-1])


def read_sync_average(events_path: Path) -> str:
    if not events_path.exists():
        return "n/a"

    durations: list[float] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if "sync complete" not in line or "duration=" not in line:
            continue
        for token in line.split():
            if token.startswith("duration="):
                durations.append(safe_float(token.split("=", 1)[1], 0.0))
                break

    if not durations:
        return "n/a"
    return f"{sum(durations) / len(durations):.2f}s"


def short_path(repo_root: Path, raw_path: str) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def source_label(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts
    if "workers" in parts:
        index = parts.index("workers")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.name or rel_path


def parse_source_progress(repo_root: Path, source_list_path: Path, state_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    offsets: dict[str, int] = {}
    episodes: dict[str, int] = {}

    if state_path.exists():
        for raw_line in state_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            key, path, value = parts[0], parts[1], parts[2]
            if key == "source":
                offsets[path] = safe_int(value)
            elif key == "source_episodes":
                episodes[path] = safe_int(value)

    rows: list[dict[str, Any]] = []
    total_bytes = 0
    processed_bytes = 0
    processed_count = 0
    total_count = 0

    if source_list_path.exists():
        for raw_line in source_list_path.read_text(encoding="utf-8").splitlines():
            rel_path = raw_line.strip()
            if not rel_path:
                continue

            total_count += 1
            abs_path = resolve_var_path(repo_root, rel_path)
            size = abs_path.stat().st_size if abs_path.exists() else 0
            offset = max(offsets.get(rel_path, 0), 0)
            if size > 0:
                offset = min(offset, size)
            progress_percent = clamp_percent((offset / size) * 100.0) if size > 0 else 0.0
            done = size > 0 and offset >= size
            if done:
                processed_count += 1
            total_bytes += size
            processed_bytes += offset

            rows.append(
                {
                    "label": source_label(rel_path),
                    "path": rel_path,
                    "size_bytes": size,
                    "size_display": format_bytes(size),
                    "offset_bytes": offset,
                    "offset_display": format_bytes(offset),
                    "episodes": episodes.get(rel_path, 0),
                    "progress_percent": round(progress_percent, 1),
                    "done": done,
                }
            )

    summary = {
        "count": total_count,
        "processed_count": processed_count,
        "total_bytes": total_bytes,
        "processed_bytes": processed_bytes,
        "total_display": format_bytes(total_bytes),
        "processed_display": format_bytes(processed_bytes),
        "processed_percent": round(clamp_percent((processed_bytes / total_bytes) * 100.0), 1) if total_bytes > 0 else 0.0,
        "state_ready": state_path.exists(),
    }
    return (rows, summary)


def build_commands(repo_root: Path, manifest: dict[str, str]) -> dict[str, str]:
    repo = str(repo_root)
    run_name = manifest.get("RUN_NAME", "")

    commands = {
        "collect": f"cd {repo}\n./scripts/blacklight.sh collect",
        "train": f"cd {repo}\n./scripts/blacklight.sh train",
        "monitor": f"cd {repo}\n./scripts/blacklight.sh monitor",
        "status": f"cd {repo}\n./scripts/blacklight.sh status",
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
    progress = parse_env_file(Path(manifest.get("PROGRESS_ABS", "")))
    run_dir = Path(manifest.get("RUN_DIR_ABS", ""))
    model_path = Path(manifest.get("MODEL_ABS", ""))
    metrics_path = Path(manifest.get("METRICS_ABS", ""))
    metrics_summary_path = Path(manifest.get("METRICS_SUMMARY_ABS", ""))
    events_path = Path(manifest.get("EVENTS_LOG_ABS", ""))
    trainer_log_path = Path(manifest.get("TRAINER_LOG_ABS", ""))
    source_list_path = Path(manifest.get("SOURCE_LIST_ABS", ""))
    state_path = Path(manifest.get("STATE_FILE_ABS", ""))

    recent_metrics, last_metric = read_recent_metrics(metrics_path)
    events = read_recent_lines(events_path, limit=10)
    trainer_tail = read_recent_lines(trainer_log_path, limit=14)
    model_episodes, model_updates = read_model_stats(model_path)
    parent_model = manifest.get("PARENT_MODEL", "")
    parent_episodes, parent_updates = read_model_stats(Path(parent_model)) if parent_model else (0, 0)

    trainer_pid = safe_int(progress.get("TRAINER_PID", "0"))
    trainer_running = path_exists_and_running(trainer_pid)
    phase = progress.get("PHASE") or "unknown"
    phase_label = PHASE_LABELS.get(phase, "Unknown")
    replay_only = manifest.get("REPLAY_ONLY", "0") == "1"
    collect_only = manifest.get("COLLECT_ONLY", "0") == "1"
    training_mode = manifest.get("TRAINING_MODE", "teacher")

    started_at = safe_int(progress.get("TRAINING_STARTED_AT", "0"))
    end_at = safe_int(progress.get("TRAINING_END_AT", "0"))
    cycle_deadline = safe_int(progress.get("CYCLE_DEADLINE", "0"))
    sync_started_at = safe_int(progress.get("SYNC_STARTED_AT", "0"))
    current_cycle = safe_int(progress.get("CURRENT_CYCLE", "0"))
    workers_running = safe_int(progress.get("WORKERS_RUNNING", "0"))
    parallel_workers = safe_int(manifest.get("PARALLEL_WORKERS", "0"))
    sync_seconds = safe_int(manifest.get("SYNC_SECONDS", "0"))
    last_sync_duration = safe_int(progress.get("LAST_SYNC_DURATION", "0"))

    now = int(time.time())
    elapsed_seconds = max(now - started_at, 0) if started_at else 0
    remaining_seconds: int | None = None
    if end_at > started_at > 0:
        remaining_seconds = max(end_at - now, 0)

    source_rows, source_summary = parse_source_progress(repo_root, source_list_path, state_path)

    metrics_summary = parse_env_file(metrics_summary_path)
    latest_policy_loss = safe_float(
        metrics_summary.get("average_policy_loss")
        or (last_metric.get("policy_loss") if last_metric else 0.0)
    )
    latest_entropy = safe_float(
        metrics_summary.get("average_entropy")
        or (last_metric.get("entropy") if last_metric else 0.0)
    )
    latest_steps = safe_float(
        metrics_summary.get("average_steps")
        or (last_metric.get("steps") if last_metric else 0.0)
    )
    latest_episode = safe_int(
        metrics_summary.get("episodes")
        or (last_metric.get("episode") if last_metric else 0)
    )
    learned_episodes = safe_int(metrics_summary.get("learned_episodes"))

    overall_progress: dict[str, Any] | None = None
    secondary_progress: dict[str, Any] | None = None
    workers_progress: dict[str, Any] | None = None

    if replay_only and source_summary["total_bytes"] > 0:
        if source_summary["state_ready"]:
            overall_progress = {
                "label": "Replay read",
                "percent": source_summary["processed_percent"],
                "display": f"{source_summary['processed_display']} of {source_summary['total_display']}",
            }
        else:
            overall_progress = {
                "label": "Replay offsets",
                "percent": 0.0,
                "display": "Trainer is live • waiting for offset snapshots",
            }
        secondary_progress = {
            "label": "Trainer process",
            "percent": 100.0 if trainer_running else 0.0,
            "display": f"PID {trainer_pid}" if trainer_running else "Stopped",
        }
    elif end_at > started_at > 0:
        run_percent = clamp_percent(((now - started_at) / max(end_at - started_at, 1)) * 100.0)
        overall_progress = {
            "label": "Run time",
            "percent": round(run_percent, 1),
            "display": f"{format_duration(elapsed_seconds)} elapsed • {format_duration(remaining_seconds)} left",
        }
        if cycle_deadline > now and sync_seconds > 0:
            cycle_remaining = max(cycle_deadline - now, 0)
            cycle_elapsed = max(sync_seconds - cycle_remaining, 0)
            cycle_percent = clamp_percent((cycle_elapsed / max(sync_seconds, 1)) * 100.0)
            secondary_progress = {
                "label": "Current window",
                "percent": round(cycle_percent, 1),
                "display": f"{format_duration(cycle_remaining)} until rollover",
            }

    if parallel_workers > 0:
        workers_percent = clamp_percent((workers_running / parallel_workers) * 100.0)
        workers_progress = {
            "label": "Workers active",
            "percent": round(workers_percent, 1),
            "display": f"{workers_running} of {parallel_workers}",
        }

    if collect_only:
        intent = "Collecting teacher examples only. Weights stay frozen until you run train."
    elif replay_only:
        intent = "Replaying teacher logs into the CNN. No worker games are running in this phase."
    else:
        intent = "Alternating between worker collection windows and sync passes."

    if replay_only:
        phase_detail = f"Training from {source_summary['count']} teacher logs"
    elif collect_only:
        phase_detail = f"Capturing teacher data from {parallel_workers} workers"
    else:
        phase_detail = "Hybrid collect and sync run"

    chart_rows: list[dict[str, Any]] = []
    for row in recent_metrics[-140:]:
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
        "generated_at": now,
        "status": {
            "phase": phase,
            "phase_label": phase_label,
            "trainer_running": trainer_running,
            "training_mode": training_mode,
            "replay_only": replay_only,
            "collect_only": collect_only,
            "intent": intent,
            "phase_detail": phase_detail,
        },
        "run": {
            "name": manifest.get("RUN_NAME", ""),
            "dir": str(run_dir),
            "dir_short": short_path(repo_root, str(run_dir)),
            "model_file": str(model_path),
            "model_file_short": short_path(repo_root, str(model_path)),
            "metrics_file": str(metrics_path),
            "metrics_file_short": short_path(repo_root, str(metrics_path)),
            "summary_file": str(metrics_summary_path),
            "summary_file_short": short_path(repo_root, str(metrics_summary_path)),
            "events_file": str(events_path),
            "events_file_short": short_path(repo_root, str(events_path)),
            "trainer_log": str(trainer_log_path),
            "trainer_log_short": short_path(repo_root, str(trainer_log_path)),
            "latest_checkpoint": latest_checkpoint(run_dir),
            "latest_checkpoint_short": short_path(repo_root, latest_checkpoint(run_dir)),
            "parent_model": parent_model,
            "parent_model_short": short_path(repo_root, parent_model) if parent_model else "",
            "source_list": str(source_list_path),
            "source_list_short": short_path(repo_root, str(source_list_path)),
        },
        "progress": {
            "elapsed_display": format_duration(elapsed_seconds),
            "remaining_display": format_duration(remaining_seconds) if remaining_seconds is not None else "Open-ended",
            "current_cycle": current_cycle,
            "parallel_workers": parallel_workers,
            "workers_running": workers_running,
            "trainer_pid": trainer_pid,
            "sync_seconds": sync_seconds,
            "last_sync_display": format_duration(last_sync_duration) if last_sync_duration else "n/a",
            "avg_sync_display": read_sync_average(events_path),
            "overall": overall_progress,
            "secondary": secondary_progress,
            "workers": workers_progress,
            "sync_elapsed_display": format_duration(max(now - sync_started_at, 0)) if sync_started_at > 0 else "n/a",
        },
        "signals": {
            "model_episodes": model_episodes,
            "model_updates": model_updates,
            "parent_episodes": parent_episodes,
            "parent_updates": parent_updates,
            "delta_updates": max(model_updates - parent_updates, 0),
            "metrics_episode": latest_episode,
            "learned_episodes": learned_episodes,
            "policy_loss": latest_policy_loss,
            "entropy": latest_entropy,
            "avg_steps": latest_steps,
            "data_display": source_summary["total_display"],
            "model_size_display": format_bytes(model_path.stat().st_size if model_path.exists() else 0),
            "metrics_size_display": format_bytes(metrics_path.stat().st_size if metrics_path.exists() else 0),
        },
        "sources": {
            "rows": source_rows,
            "summary": source_summary,
        },
        "charts": chart_rows,
        "logs": {
            "events": events,
            "trainer_tail": trainer_tail,
        },
        "commands": build_commands(repo_root, manifest),
    }


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blacklight Training Deck</title>
  <style>
    :root {
      --paper: #f3efe7;
      --mist: #dde7df;
      --panel: rgba(251, 249, 244, 0.84);
      --panel-strong: rgba(252, 250, 246, 0.92);
      --line: rgba(25, 38, 31, 0.12);
      --ink: #162118;
      --muted: #5a6a61;
      --accent: #0c7b68;
      --accent-soft: rgba(12, 123, 104, 0.15);
      --warn: #b95d36;
      --warn-soft: rgba(185, 93, 54, 0.14);
      --shadow: 0 26px 70px rgba(24, 37, 28, 0.12);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 14px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "SF Pro Text", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 0 0, rgba(255,255,255,0.84), transparent 26%),
        radial-gradient(circle at 100% 0, rgba(12,123,104,0.12), transparent 22%),
        linear-gradient(140deg, var(--paper), var(--mist));
    }

    .page {
      max-width: 1440px;
      margin: 0 auto;
      padding: 22px 20px 36px;
    }

    .masthead {
      border: 1px solid rgba(255,255,255,0.7);
      border-radius: var(--radius-xl);
      background: linear-gradient(180deg, rgba(252, 251, 247, 0.92), rgba(245, 248, 243, 0.82));
      box-shadow: var(--shadow);
      padding: 24px 26px;
    }

    .eyebrow {
      margin: 0 0 10px;
      font-size: 0.72rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }

    .masthead-row {
      display: flex;
      gap: 18px;
      justify-content: space-between;
      align-items: flex-start;
    }

    .masthead h1 {
      margin: 0;
      font-family: "Iowan Old Style", "Georgia", serif;
      font-size: clamp(2rem, 4vw, 3.7rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }

    .masthead p {
      margin: 10px 0 0;
      max-width: 58rem;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
    }

    .status-rail {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
      min-width: 280px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(22, 33, 24, 0.08);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .pill-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 5px rgba(185, 93, 54, 0.14);
    }

    .pill-dot.live {
      background: var(--accent);
      box-shadow: 0 0 0 5px rgba(12, 123, 104, 0.16);
    }

    .top-grid,
    .main-grid {
      display: grid;
      gap: 16px;
      margin-top: 18px;
    }

    .top-grid {
      grid-template-columns: repeat(6, minmax(0, 1fr));
    }

    .main-grid {
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }

    .tile,
    .panel {
      border-radius: var(--radius-lg);
      border: 1px solid rgba(255,255,255,0.68);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }

    .tile {
      padding: 18px;
      min-height: 148px;
    }

    .panel {
      padding: 20px;
    }

    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: 1 / -1; }

    .kicker {
      font-size: 0.74rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 10px;
    }

    .big-number {
      margin: 0;
      font-size: clamp(1.9rem, 3vw, 2.8rem);
      line-height: 1;
      letter-spacing: -0.04em;
      font-weight: 700;
    }

    .support {
      margin-top: 10px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 0.95rem;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: baseline;
      margin-bottom: 16px;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 1.08rem;
    }

    .panel-head p {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .bar-stack {
      display: grid;
      gap: 16px;
    }

    .bar-block {
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.52);
      padding: 14px 16px;
    }

    .bar-top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 10px;
    }

    .bar-top strong {
      font-size: 0.96rem;
    }

    .bar-top span {
      color: var(--muted);
      font-size: 0.88rem;
      text-align: right;
    }

    .bar-track {
      height: 12px;
      border-radius: 999px;
      background: rgba(22, 33, 24, 0.08);
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #54b69d);
      transition: width 240ms ease;
    }

    .bar-fill.warn {
      background: linear-gradient(90deg, var(--warn), #e28b66);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 18px;
      margin-top: 16px;
    }

    .stat-row {
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }

    .stat-row .label {
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 6px;
      font-weight: 700;
    }

    .stat-row .value {
      font-size: 0.96rem;
      line-height: 1.45;
      word-break: break-word;
    }

    .charts {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }

    .chart-shell {
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.52);
      padding: 14px 16px 12px;
    }

    .chart-meta {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 8px;
    }

    .chart-meta strong {
      font-size: 0.96rem;
    }

    .chart-meta span {
      color: var(--muted);
      font-size: 0.84rem;
    }

    canvas {
      width: 100%;
      height: 210px;
      display: block;
    }

    .source-list {
      display: grid;
      gap: 10px;
    }

    .source-row {
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.5);
      padding: 12px 14px;
    }

    .source-top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 8px;
    }

    .source-top strong {
      font-size: 0.94rem;
    }

    .source-top span {
      color: var(--muted);
      font-size: 0.84rem;
    }

    .source-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      color: var(--muted);
      font-size: 0.86rem;
      margin-top: 8px;
    }

    .detail-list {
      display: grid;
      gap: 12px;
    }

    .detail-block {
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }

    .detail-block .detail-label {
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 6px;
      font-weight: 700;
    }

    .detail-block .detail-value {
      font-size: 0.95rem;
      line-height: 1.45;
      word-break: break-word;
    }

    .log-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .log-box {
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(18, 28, 24, 0.95);
      color: #f4efe7;
      min-height: 280px;
      overflow: hidden;
    }

    .log-box header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      font-size: 0.86rem;
      color: rgba(255,255,255,0.78);
    }

    .log-box pre {
      margin: 0;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.84rem;
      line-height: 1.5;
      font-family: "SFMono-Regular", "Menlo", monospace;
      max-height: 320px;
      overflow: auto;
    }

    .commands {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
    }

    .command {
      border-radius: 18px;
      padding: 16px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(248,244,236,0.7));
    }

    .command h3 {
      margin: 0 0 6px;
      font-size: 0.98rem;
    }

    .command p {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }

    .command pre {
      margin: 0;
      padding: 12px;
      border-radius: 14px;
      background: rgba(18, 28, 24, 0.95);
      color: #f4efe7;
      font-size: 0.82rem;
      line-height: 1.45;
      overflow: auto;
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
      font-size: 0.88rem;
      box-shadow: 0 8px 22px rgba(12, 123, 104, 0.24);
    }

    .footnote {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.6;
      text-align: center;
    }

    @media (max-width: 1240px) {
      .top-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .commands { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 980px) {
      .span-4, .span-5, .span-6, .span-7, .span-8 { grid-column: 1 / -1; }
      .log-grid { grid-template-columns: 1fr; }
      .masthead-row { flex-direction: column; }
      .status-rail { justify-content: flex-start; }
    }

    @media (max-width: 720px) {
      .page { padding: 16px 12px 28px; }
      .top-grid { grid-template-columns: 1fr; }
      .commands, .stats-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="masthead">
      <div class="eyebrow">Blacklight Training Deck</div>
      <div class="masthead-row">
        <div>
          <h1 id="runName">Loading run...</h1>
          <p id="intentText">Resolving the active Blacklight run and building an operator view.</p>
        </div>
        <div class="status-rail">
          <div class="pill"><span class="pill-dot" id="liveDot"></span><span id="phaseText">Checking state...</span></div>
          <div class="pill"><span id="modeText">Mode</span></div>
          <div class="pill"><span id="refreshText">Waiting for first refresh...</span></div>
        </div>
      </div>
    </section>

    <section class="top-grid">
      <article class="tile">
        <div class="kicker">Current Action</div>
        <p class="big-number" id="phaseHeadline">...</p>
        <div class="support" id="phaseDetail">Checking the trainer process and current phase.</div>
      </article>
      <article class="tile">
        <div class="kicker">Model Updates</div>
        <p class="big-number" id="updatesMetric">0</p>
        <div class="support" id="updatesSupport">No parent comparison yet.</div>
      </article>
      <article class="tile">
        <div class="kicker">Policy Loss</div>
        <p class="big-number" id="policyMetric">n/a</p>
        <div class="support">Latest average policy loss from the replay run.</div>
      </article>
      <article class="tile">
        <div class="kicker">Entropy</div>
        <p class="big-number" id="entropyMetric">n/a</p>
        <div class="support">Latest entropy signal from the metrics summary.</div>
      </article>
      <article class="tile">
        <div class="kicker">Teacher Data</div>
        <p class="big-number" id="dataMetric">0 B</p>
        <div class="support" id="dataSupport">No source files resolved yet.</div>
      </article>
      <article class="tile">
        <div class="kicker">Elapsed</div>
        <p class="big-number" id="elapsedMetric">00:00:00</p>
        <div class="support" id="elapsedSupport">Waiting for the first progress refresh.</div>
      </article>
    </section>

    <section class="main-grid">
      <article class="panel span-7">
        <div class="panel-head">
          <h2>Run Track</h2>
          <p>The glanceable progress view: what phase the trainer is in, what data it has read, and what still needs attention.</p>
        </div>
        <div class="bar-stack">
          <div class="bar-block" id="overallBlock">
            <div class="bar-top"><strong id="overallLabel">Overall</strong><span id="overallMeta">Waiting for data...</span></div>
            <div class="bar-track"><div class="bar-fill" id="overallFill"></div></div>
          </div>
          <div class="bar-block" id="secondaryBlock">
            <div class="bar-top"><strong id="secondaryLabel">Phase window</strong><span id="secondaryMeta">Waiting for data...</span></div>
            <div class="bar-track"><div class="bar-fill warn" id="secondaryFill"></div></div>
          </div>
          <div class="bar-block" id="workersBlock">
            <div class="bar-top"><strong id="workersLabel">Workers</strong><span id="workersMeta">Waiting for data...</span></div>
            <div class="bar-track"><div class="bar-fill" id="workersFill"></div></div>
          </div>
        </div>
        <div class="stats-grid">
          <div class="stat-row"><div class="label">Current Cycle</div><div class="value" id="cycleValue">0</div></div>
          <div class="stat-row"><div class="label">Trainer PID</div><div class="value" id="trainerPidValue">n/a</div></div>
          <div class="stat-row"><div class="label">Last Sync</div><div class="value" id="lastSyncValue">n/a</div></div>
          <div class="stat-row"><div class="label">Average Sync</div><div class="value" id="avgSyncValue">n/a</div></div>
          <div class="stat-row"><div class="label">Source Files</div><div class="value" id="sourceCountValue">0</div></div>
          <div class="stat-row"><div class="label">Source Progress</div><div class="value" id="sourceProgressValue">n/a</div></div>
        </div>
      </article>

      <article class="panel span-5">
        <div class="panel-head">
          <h2>Signals</h2>
          <p>The core training numbers that matter while the run is live.</p>
        </div>
        <div class="stats-grid">
          <div class="stat-row"><div class="label">Model Episodes</div><div class="value" id="modelEpisodesValue">0</div></div>
          <div class="stat-row"><div class="label">Metrics Episode</div><div class="value" id="metricsEpisodeValue">0</div></div>
          <div class="stat-row"><div class="label">Learned Episodes</div><div class="value" id="learnedEpisodesValue">0</div></div>
          <div class="stat-row"><div class="label">Average Steps</div><div class="value" id="avgStepsValue">n/a</div></div>
          <div class="stat-row"><div class="label">Model Size</div><div class="value" id="modelSizeValue">n/a</div></div>
          <div class="stat-row"><div class="label">Metrics File Size</div><div class="value" id="metricsSizeValue">n/a</div></div>
        </div>
      </article>

      <article class="panel span-6">
        <div class="panel-head">
          <h2>Learning Curves</h2>
          <p>Recent training trend from the live metrics CSV rather than terminal spam.</p>
        </div>
        <div class="charts">
          <div class="chart-shell">
            <div class="chart-meta"><strong>Policy Loss</strong><span id="policyTrendText">Waiting for data...</span></div>
            <canvas id="policyChart" width="800" height="220"></canvas>
          </div>
          <div class="chart-shell">
            <div class="chart-meta"><strong>Entropy</strong><span id="entropyTrendText">Waiting for data...</span></div>
            <canvas id="entropyChart" width="800" height="220"></canvas>
          </div>
        </div>
      </article>

      <article class="panel span-6">
        <div class="panel-head">
          <h2>Source Coverage</h2>
          <p>Exactly which teacher logs are being consumed, how much of each has been read, and whether replay is actually moving.</p>
        </div>
        <div class="source-list" id="sourceList"></div>
      </article>

      <article class="panel span-6">
        <div class="panel-head">
          <h2>Files That Matter</h2>
          <p>The actual run artifacts you would inspect, diff, archive, or benchmark next.</p>
        </div>
        <div class="detail-list">
          <div class="detail-block"><div class="detail-label">Run Directory</div><div class="detail-value" id="runDir"></div></div>
          <div class="detail-block"><div class="detail-label">Model File</div><div class="detail-value" id="modelFile"></div></div>
          <div class="detail-block"><div class="detail-label">Metrics File</div><div class="detail-value" id="metricsFile"></div></div>
          <div class="detail-block"><div class="detail-label">Latest Checkpoint</div><div class="detail-value" id="checkpointFile"></div></div>
          <div class="detail-block"><div class="detail-label">Parent Model</div><div class="detail-value" id="parentModel"></div></div>
          <div class="detail-block"><div class="detail-label">Source List</div><div class="detail-value" id="sourceListPath"></div></div>
          <div class="detail-block"><div class="detail-label">Trainer Log</div><div class="detail-value" id="trainerLogPath"></div></div>
          <div class="detail-block"><div class="detail-label">Events Log</div><div class="detail-value" id="eventsLogPath"></div></div>
        </div>
      </article>

      <article class="panel span-6">
        <div class="panel-head">
          <h2>Live Output</h2>
          <p>The two places you actually check when a run looks suspicious: the trainer tail and the event trail.</p>
        </div>
        <div class="log-grid">
          <div class="log-box">
            <header><strong>Trainer Tail</strong><span id="trainerTailMeta">Last 14 lines</span></header>
            <pre id="trainerTail"></pre>
          </div>
          <div class="log-box">
            <header><strong>Recent Events</strong><span id="eventsMeta">Last 10 lines</span></header>
            <pre id="eventsTail"></pre>
          </div>
        </div>
      </article>

      <article class="panel span-12">
        <div class="panel-head">
          <h2>Actions</h2>
          <p>The actual commands you run next. No filler, no guessing which shell incantation matters.</p>
        </div>
        <div class="commands">
          <div class="command">
            <h3>Collect</h3>
            <p>Gather more teacher data from the built-in bots.</p>
            <pre id="collectCommand"></pre>
            <div class="copy-row"><button data-copy="collectCommand">Copy</button></div>
          </div>
          <div class="command">
            <h3>Train</h3>
            <p>Replay the teacher logs into the current best model.</p>
            <pre id="trainCommand"></pre>
            <div class="copy-row"><button data-copy="trainCommand">Copy</button></div>
          </div>
          <div class="command">
            <h3>Monitor</h3>
            <p>Open the improved terminal view for this same run.</p>
            <pre id="monitorCommand"></pre>
            <div class="copy-row"><button data-copy="monitorCommand">Copy</button></div>
          </div>
          <div class="command">
            <h3>Benchmark</h3>
            <p>Measure whether the run is actually worth keeping.</p>
            <pre id="benchCommand"></pre>
            <div class="copy-row"><button data-copy="benchCommand">Copy</button></div>
          </div>
          <div class="command">
            <h3>Status</h3>
            <p>Dump the current run state in one CLI shot.</p>
            <pre id="statusCommand"></pre>
            <div class="copy-row"><button data-copy="statusCommand">Copy</button></div>
          </div>
        </div>
      </article>
    </section>

    <div class="footnote">
      Auto-refreshes every 3 seconds. This view is meant to answer the operator questions first: what phase the run is in, whether it is healthy, what data it is using, and what command comes next.
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
      const node = document.getElementById(id);
      if (!node) return;
      node.textContent = value || "n/a";
    }

    function setBar(prefix, payload) {
      const block = document.getElementById(`${prefix}Block`);
      if (!block) return;
      if (!payload) {
        block.style.display = "none";
        return;
      }
      block.style.display = "";
      setText(`${prefix}Label`, payload.label);
      setText(`${prefix}Meta`, payload.display);
      const fill = document.getElementById(`${prefix}Fill`);
      fill.style.width = `${Math.max(0, Math.min(100, Number(payload.percent) || 0))}%`;
    }

    function drawChart(canvasId, points, key, color, fillColor) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      if (!points.length) {
        ctx.fillStyle = "rgba(90, 106, 97, 0.9)";
        ctx.font = "15px Avenir Next";
        ctx.fillText("No data yet", 24, 32);
        return;
      }

      const values = points.map((point) => Number(point[key])).filter((value) => Number.isFinite(value));
      if (!values.length) return;

      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(max - min, 0.000001);
      const left = 24;
      const top = 14;
      const bottom = height - 24;
      const right = width - 16;

      ctx.strokeStyle = "rgba(22, 33, 24, 0.10)";
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

    function renderSources(rows) {
      const root = document.getElementById("sourceList");
      root.innerHTML = "";
      if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "source-row";
        empty.textContent = "No source files resolved for this run yet.";
        root.appendChild(empty);
        return;
      }

      rows.forEach((row) => {
        const item = document.createElement("div");
        item.className = "source-row";
        item.innerHTML = `
          <div class="source-top">
            <strong>${row.label}</strong>
            <span>${row.progress_percent.toFixed(1)}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${row.progress_percent}%;"></div></div>
          <div class="source-meta">
            <span>${row.offset_display} of ${row.size_display}</span>
            <span>${row.episodes} episodes</span>
            <span>${row.path}</span>
          </div>
        `;
        root.appendChild(item);
      });
    }

    function renderLogLines(id, lines, emptyMessage) {
      const node = document.getElementById(id);
      if (!node) return;
      node.textContent = lines.length ? lines.join("\\n") : emptyMessage;
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
      setText("intentText", payload.status.intent || "No intent text available.");
      setText("phaseText", `${payload.status.phase_label}${payload.status.trainer_running ? " • live" : " • stopped"}`);
      setText("modeText", `${payload.status.training_mode} mode${payload.status.replay_only ? " • replay" : payload.status.collect_only ? " • collect" : ""}`);
      setText("refreshText", `Refreshed ${new Date(payload.generated_at * 1000).toLocaleTimeString()}`);
      setText("phaseHeadline", payload.status.phase_label);
      setText("phaseDetail", payload.status.phase_detail);

      const liveDot = document.getElementById("liveDot");
      liveDot.className = payload.status.trainer_running ? "pill-dot live" : "pill-dot";

      setText("updatesMetric", String(payload.signals.model_updates || 0));
      setText("updatesSupport", `Parent had ${payload.signals.parent_updates || 0} updates • delta ${payload.signals.delta_updates || 0}`);
      setText("policyMetric", numberish(payload.signals.policy_loss));
      setText("entropyMetric", numberish(payload.signals.entropy));
      setText("dataMetric", payload.signals.data_display);
      setText(
        "dataSupport",
        payload.sources.summary.state_ready
          ? `${payload.sources.summary.processed_display} replayed across ${payload.sources.summary.count || 0} source files`
          : `Offset snapshots pending across ${payload.sources.summary.count || 0} source files`
      );
      setText("elapsedMetric", payload.progress.elapsed_display);
      setText("elapsedSupport", `${payload.progress.remaining_display} remaining • last sync ${payload.progress.last_sync_display}`);

      setBar("overall", payload.progress.overall);
      setBar("secondary", payload.progress.secondary);
      setBar("workers", payload.progress.workers);

      setText("cycleValue", String(payload.progress.current_cycle || 0));
      setText("trainerPidValue", payload.progress.trainer_pid ? String(payload.progress.trainer_pid) : "n/a");
      setText("lastSyncValue", payload.progress.last_sync_display || "n/a");
      setText("avgSyncValue", payload.progress.avg_sync_display || "n/a");
      setText("sourceCountValue", `${payload.sources.summary.processed_count || 0}/${payload.sources.summary.count || 0}`);
      setText(
        "sourceProgressValue",
        payload.sources.summary.state_ready
          ? `${payload.sources.summary.processed_display} of ${payload.sources.summary.total_display} • ${numberish(payload.sources.summary.processed_percent, 1)}%`
          : "Offsets pending"
      );

      setText("modelEpisodesValue", String(payload.signals.model_episodes || 0));
      setText("metricsEpisodeValue", String(payload.signals.metrics_episode || 0));
      setText("learnedEpisodesValue", String(payload.signals.learned_episodes || 0));
      setText("avgStepsValue", numberish(payload.signals.avg_steps, 3));
      setText("modelSizeValue", payload.signals.model_size_display);
      setText("metricsSizeValue", payload.signals.metrics_size_display);

      setText("runDir", payload.run.dir_short || payload.run.dir);
      setText("modelFile", payload.run.model_file_short || payload.run.model_file);
      setText("metricsFile", payload.run.metrics_file_short || payload.run.metrics_file);
      setText("checkpointFile", payload.run.latest_checkpoint_short || "No checkpoint yet");
      setText("parentModel", payload.run.parent_model_short || "No parent model");
      setText("sourceListPath", payload.run.source_list_short || payload.run.source_list || "n/a");
      setText("trainerLogPath", payload.run.trainer_log_short || payload.run.trainer_log || "n/a");
      setText("eventsLogPath", payload.run.events_file_short || payload.run.events_file || "n/a");

      setText("collectCommand", payload.commands.collect || "n/a");
      setText("trainCommand", payload.commands.train || "n/a");
      setText("monitorCommand", payload.commands.monitor || "n/a");
      setText("benchCommand", payload.commands.benchmark || "n/a");
      setText("statusCommand", payload.commands.status || "n/a");

      const charts = payload.charts || [];
      const policyValues = charts.map((point) => Number(point.policy_loss)).filter((value) => Number.isFinite(value));
      const entropyValues = charts.map((point) => Number(point.entropy)).filter((value) => Number.isFinite(value));
      const firstPolicy = policyValues.length ? policyValues[0] : null;
      const lastPolicy = policyValues.length ? policyValues[policyValues.length - 1] : null;
      const firstEntropy = entropyValues.length ? entropyValues[0] : null;
      const lastEntropy = entropyValues.length ? entropyValues[entropyValues.length - 1] : null;
      setText("policyTrendText", firstPolicy === null ? "No chart data yet" : `${numberish(firstPolicy)} → ${numberish(lastPolicy)}`);
      setText("entropyTrendText", firstEntropy === null ? "No chart data yet" : `${numberish(firstEntropy)} → ${numberish(lastEntropy)}`);
      drawChart("policyChart", charts, "policy_loss", "#0c7b68", "rgba(12, 123, 104, 0.16)");
      drawChart("entropyChart", charts, "entropy", "#b95d36", "rgba(185, 93, 54, 0.16)");

      renderSources(payload.sources.rows || []);
      renderLogLines("trainerTail", payload.logs.trainer_tail || [], "No trainer output yet.");
      renderLogLines("eventsTail", payload.logs.events || [], "No events yet.");
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
