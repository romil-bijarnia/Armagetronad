#!/usr/bin/env python3
"""Terminal dashboard for Blacklight training and local machine load."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RESET = "\033[0m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BLOCKS = "▁▂▃▄▅▆▇█"
MISSING_BLOCK = "·"


class Theme:
    cyan = "\033[38;5;80m"
    blue = "\033[38;5;75m"
    purple = "\033[38;5;99m"
    pink = "\033[38;5;205m"
    yellow = "\033[38;5;222m"
    orange = "\033[38;5;214m"
    green = "\033[38;5;119m"
    red = "\033[38;5;203m"
    gray = "\033[38;5;244m"
    dim = "\033[38;5;240m"
    white = "\033[38;5;255m"
    bold = "\033[1m"


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def visible_len(value: str) -> int:
    return len(strip_ansi(value))


def colorize(value: str, color: str, enabled: bool) -> str:
    if not enabled or not color:
        return value
    return f"{color}{value}{RESET}"


def fit(value: str, width: int) -> str:
    plain = strip_ansi(value)
    if len(plain) <= width:
        return value + (" " * (width - len(plain)))
    if width <= 1:
        return plain[:width]
    return plain[: width - 1] + "…"


def run_text(args: list[str], timeout: float = 1.5) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:
        return ""


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            parts = shlex.split(raw_value)
            value = parts[0] if parts else ""
        except ValueError:
            value = raw_value.strip("'\"")
        values[key] = value
    return values


def human_bytes(value: int | float | str) -> str:
    try:
        size = float(value)
    except Exception:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.1f} {units[unit]}"


def fmt_duration(seconds: int | float | str | None) -> str:
    try:
        total = int(float(seconds if seconds is not None else 0))
    except Exception:
        return "n/a"
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def percent_color(value: float | None) -> str:
    if value is None:
        return Theme.gray
    if value >= 90:
        return Theme.red
    if value >= 70:
        return Theme.orange
    if value >= 50:
        return Theme.yellow
    return Theme.green


def clamp_percent(value: float | int | str | None) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return max(0.0, min(100.0, parsed))


def bar(value: float | None, width: int, color: str, use_color: bool) -> str:
    if value is None:
        return colorize(MISSING_BLOCK * width, Theme.dim, use_color)
    value = clamp_percent(value) or 0.0
    filled = round((value / 100.0) * width)
    return colorize("█" * filled, color, use_color) + colorize("░" * (width - filled), Theme.dim, use_color)


def dense_bar(value: float | None, width: int, color: str, use_color: bool) -> str:
    if value is None:
        return colorize("░" * width, Theme.dim, use_color)
    return bar(value, width, color, use_color)


def spark(values: Iterable[float | None], width: int, color: str, use_color: bool) -> str:
    recent = list(values)[-width:]
    if len(recent) < width:
        recent = [None] * (width - len(recent)) + recent
    chars: list[str] = []
    for value in recent:
        if value is None:
            chars.append(MISSING_BLOCK)
            continue
        bounded = clamp_percent(value) or 0.0
        index = min(len(BLOCKS) - 1, int((bounded / 100.0) * len(BLOCKS)))
        chars.append(BLOCKS[index])
    return colorize("".join(chars), color, use_color)


def history_spark(values: Iterable[float | None], width: int, color: str, use_color: bool) -> str:
    width = max(1, width)
    recent = list(values)[-width:]
    known = [value for value in recent if value is not None]
    if not known:
        return colorize("▁" * width, Theme.dim, use_color)
    if len(recent) < width:
        recent = [known[0]] * (width - len(recent)) + recent
    chars: list[str] = []
    for value in recent:
        if value is None:
            chars.append(" ")
            continue
        bounded = clamp_percent(value) or 0.0
        index = min(len(BLOCKS) - 1, int((bounded / 100.0) * (len(BLOCKS) - 1)))
        chars.append(BLOCKS[index])
    return colorize("".join(chars), color, use_color)


def last_nonempty_line(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunk = b""
        while position > 0:
            read_size = min(8192, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size) + chunk
            lines = chunk.splitlines()
            if len(lines) > 1:
                for line in reversed(lines):
                    if line.strip():
                        return line.decode(errors="replace")
        return chunk.decode(errors="replace").strip().splitlines()[-1] if chunk.strip() else ""


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    cpu: float
    mem: float
    rss_kb: int
    command: str


@dataclass
class SystemSample:
    cpu: float | None
    gpu: float | None
    trainer_cpu: float | None
    memory_used: float | None
    memory_total: int
    disk_used: float | None
    disk_total: int
    battery: str
    load_avg: tuple[float, float, float]
    processes: list[ProcessInfo]


@dataclass
class BlacklightState:
    env: dict[str, str]
    model_episodes: int
    model_updates: int
    parent_updates: int
    metrics: dict[str, str]
    source_bytes: int
    source_count: int
    latest_event: str
    latest_checkpoint: str


def model_stats(path_value: str | None) -> tuple[int, int]:
    if not path_value:
        return 0, 0
    path = Path(path_value)
    if not path.exists():
        return 0, 0
    try:
        with path.open(errors="replace") as handle:
            handle.readline()
            handle.readline()
            fields = handle.readline().split()
        return int(float(fields[1])), int(float(fields[2]))
    except Exception:
        return 0, 0


def read_metrics(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    header_line = ""
    try:
        with path.open(errors="replace") as handle:
            header_line = handle.readline().strip()
    except Exception:
        return {}
    last_line = last_nonempty_line(path)
    if not header_line or not last_line or last_line == header_line:
        return {}
    try:
        header = next(csv.reader([header_line]))
        row = next(csv.reader([last_line]))
    except Exception:
        return {}
    return {key: row[index] for index, key in enumerate(header) if index < len(row)}


def source_list_stats(repo_root: Path, path_value: str | None) -> tuple[int, int]:
    if not path_value:
        return 0, 0
    path = Path(path_value)
    if not path.exists():
        return 0, 0
    total = 0
    count = 0
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        count += 1
        source_path = Path(line)
        if not source_path.is_absolute():
            source_path = repo_root / "var" / line
        if source_path.exists():
            total += source_path.stat().st_size
    return total, count


def latest_event(path_value: str | None) -> str:
    if not path_value:
        return ""
    line = last_nonempty_line(Path(path_value))
    return re.sub(r"^\[[^]]+\]\s*", "", line)


def latest_checkpoint(env: dict[str, str], repo_root: Path) -> str:
    prefix = env.get("CHECKPOINT_PREFIX_ABS")
    if not prefix:
        return ""
    pointer = Path(f"{prefix}_latest.txt")
    if not pointer.exists():
        return ""
    value = pointer.read_text(errors="replace").splitlines()[0].strip()
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / "var" / value
    return path.name


def load_blacklight(repo_root: Path) -> BlacklightState:
    env = parse_env_file(repo_root / "var" / "blacklight_runs" / "latest_run.env")
    progress_path = env.get("PROGRESS_ABS")
    if progress_path:
        env.update(parse_env_file(Path(progress_path)))
    model_episodes, model_updates = model_stats(env.get("MODEL_ABS"))
    _parent_episodes, parent_updates = model_stats(env.get("PARENT_MODEL"))
    metrics = read_metrics(env.get("METRICS_ABS"))
    source_bytes, source_count = source_list_stats(repo_root, env.get("SOURCE_LIST_ABS"))
    return BlacklightState(
        env=env,
        model_episodes=model_episodes,
        model_updates=model_updates,
        parent_updates=parent_updates,
        metrics=metrics,
        source_bytes=source_bytes,
        source_count=source_count,
        latest_event=latest_event(env.get("EVENTS_LOG_ABS")),
        latest_checkpoint=latest_checkpoint(env, repo_root),
    )


def cpu_percent_from_processes(processes: list[ProcessInfo]) -> float | None:
    try:
        cores = int(run_text(["sysctl", "-n", "hw.ncpu"], timeout=1.0).strip())
    except Exception:
        cores = os.cpu_count() or 1
    if cores <= 0:
        cores = 1
    total = sum(process.cpu for process in processes)
    return max(0.0, min(100.0, total / cores))


def gpu_percent_from_ioreg() -> float | None:
    output = run_text(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"], timeout=2.0)
    match = re.search(r'"Device Utilization %"=([0-9.]+)', output)
    if not match:
        return None
    return float(match.group(1))


def memory_usage() -> tuple[float | None, int]:
    total_text = run_text(["sysctl", "-n", "hw.memsize"], timeout=1.0).strip()
    try:
        total = int(total_text)
    except Exception:
        total = 0
    output = run_text(["vm_stat"], timeout=1.0)
    page_match = re.search(r"page size of ([0-9]+) bytes", output)
    page_size = int(page_match.group(1)) if page_match else 16384
    pages: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        number = re.sub(r"[^0-9]", "", value)
        if number:
            pages[key.strip()] = int(number)
    free_pages = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
    if total <= 0:
        return None, 0
    free_bytes = free_pages * page_size
    used = max(0, min(total, total - free_bytes))
    return (used / total) * 100.0, total


def disk_usage(repo_root: Path) -> tuple[float | None, int]:
    try:
        usage = shutil.disk_usage(repo_root)
    except Exception:
        return None, 0
    return (usage.used / usage.total) * 100.0, usage.total


def battery_status() -> str:
    output = run_text(["pmset", "-g", "batt"], timeout=1.0)
    percent = re.search(r"([0-9]+)%", output)
    source = "AC" if "AC Power" in output else "Batt"
    if not percent:
        return "n/a"
    return f"{percent.group(1)}% {source}"


def process_table() -> list[ProcessInfo]:
    output = run_text(["ps", "-axo", "pid=,ppid=,%cpu=,%mem=,rss=,comm="], timeout=4.5)
    processes: list[ProcessInfo] = []
    for line in output.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            processes.append(
                ProcessInfo(
                    pid=int(parts[0]),
                    ppid=int(parts[1]),
                    cpu=float(parts[2]),
                    mem=float(parts[3]),
                    rss_kb=int(parts[4]),
                    command=parts[5],
                )
            )
        except Exception:
            continue
    return sorted(processes, key=lambda item: item.cpu, reverse=True)


def sample_system(repo_root: Path, trainer_pid: int) -> SystemSample:
    processes = process_table()
    trainer_cpu: float | None = None
    if trainer_pid > 0:
        child_cpu = sum(process.cpu for process in processes if process.ppid == trainer_pid)
        own_cpu = sum(process.cpu for process in processes if process.pid == trainer_pid)
        trainer_cpu = child_cpu if child_cpu > 0 else own_cpu
    mem_used, mem_total = memory_usage()
    disk_used, disk_total = disk_usage(repo_root)
    try:
        load_avg = os.getloadavg()
    except Exception:
        load_avg = (0.0, 0.0, 0.0)
    return SystemSample(
        cpu=cpu_percent_from_processes(processes),
        gpu=gpu_percent_from_ioreg(),
        trainer_cpu=trainer_cpu,
        memory_used=mem_used,
        memory_total=mem_total,
        disk_used=disk_used,
        disk_total=disk_total,
        battery=battery_status(),
        load_avg=load_avg,
        processes=processes[:10],
    )


def history_path(repo_root: Path, state: BlacklightState) -> Path:
    run_dir = state.env.get("RUN_DIR_ABS")
    if run_dir:
        return Path(run_dir) / "terminal_dashboard_history.tsv"
    return repo_root / "var" / "terminal_dashboard_history.tsv"


def read_history(path: Path, limit: int = 240) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines()[-limit:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        row: dict[str, float | None] = {}
        for key, value in zip(("time", "cpu", "gpu", "trainer", "mem", "disk", "updates", "csv"), parts):
            try:
                parsed = float(value)
                row[key] = None if parsed < 0 else parsed
            except Exception:
                row[key] = None
        rows.append(row)
    return rows


def append_history(path: Path, sample: SystemSample, state: BlacklightState) -> list[dict[str, float | None]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        csv_episode = float(state.metrics.get("episode", -1))
    except Exception:
        csv_episode = -1
    values = [
        time.time(),
        sample.cpu if sample.cpu is not None else -1,
        sample.gpu if sample.gpu is not None else -1,
        sample.trainer_cpu if sample.trainer_cpu is not None else -1,
        sample.memory_used if sample.memory_used is not None else -1,
        sample.disk_used if sample.disk_used is not None else -1,
        state.model_updates,
        csv_episode,
    ]
    with path.open("a") as handle:
        handle.write("\t".join(f"{value:.3f}" for value in values) + "\n")
    rows = read_history(path)
    if len(rows) > 240:
        with path.open("w") as handle:
            for row in rows[-240:]:
                handle.write(
                    "\t".join(
                        f"{(row.get(key) if row.get(key) is not None else -1):.3f}"
                        for key in ("time", "cpu", "gpu", "trainer", "mem", "disk", "updates", "csv")
                    )
                    + "\n"
                )
        rows = rows[-240:]
    return rows


def phase_label(phase: str) -> str:
    return {
        "initializing": "INITIALIZING",
        "collecting": "COLLECTING",
        "syncing": "TRAINING",
        "stopping": "STOPPING",
        "finished": "FINISHED",
    }.get(phase or "", "UNKNOWN")


def box(title: str, lines: list[str], width: int, use_color: bool, color: str = Theme.purple) -> list[str]:
    width = max(24, width)
    top_title = f"─ {title} "
    top = "╭" + top_title + ("─" * max(0, width - 2 - len(strip_ansi(top_title)))) + "╮"
    bottom = "╰" + ("─" * (width - 2)) + "╯"
    rendered = [colorize(top, color, use_color)]
    inner = width - 4
    for line in lines:
        rendered.append(
            colorize("│", Theme.dim, use_color)
            + " "
            + fit(line, inner)
            + " "
            + colorize("│", Theme.dim, use_color)
        )
    rendered.append(colorize(bottom, color, use_color))
    return rendered


def join_columns(columns: list[list[str]], gap: int = 2) -> list[str]:
    heights = [len(column) for column in columns]
    widths = [visible_len(column[0]) if column else 0 for column in columns]
    lines: list[str] = []
    for index in range(max(heights, default=0)):
        parts: list[str] = []
        for column, width in zip(columns, widths):
            if index < len(column):
                parts.append(fit(column[index], width))
            else:
                parts.append(" " * width)
        lines.append((" " * gap).join(parts).rstrip())
    return lines


def load_chart(history: list[dict[str, float | None]], width: int, height: int, use_color: bool) -> list[str]:
    width = max(30, width)
    height = max(6, height)
    points = history[-width:]
    if len(points) < width:
        points = ([{}] * (width - len(points))) + points
    series = [
        ("cpu", Theme.cyan, "●"),
        ("trainer", Theme.yellow, "■"),
        ("gpu", Theme.pink, "◆"),
    ]
    grid: list[list[str]] = [[" " for _ in range(width)] for _ in range(height)]
    occupied: list[list[str]] = [["" for _ in range(width)] for _ in range(height)]
    for x, point in enumerate(points):
        for key, color, marker in series:
            value = point.get(key)
            if value is None:
                continue
            bounded = clamp_percent(value) or 0.0
            y = height - 1 - round((bounded / 100.0) * (height - 1))
            if occupied[y][x]:
                grid[y][x] = colorize("✦", Theme.white, use_color)
            else:
                grid[y][x] = colorize(marker, color, use_color)
                occupied[y][x] = key
    output: list[str] = []
    for row_index, row in enumerate(grid):
        label = round(100 - (row_index * (100 / max(1, height - 1))))
        output.append(f"{label:>3} ┤{''.join(row)}")
    output.append("    " + "└" + ("─" * width))
    legend = (
        colorize("● CPU", Theme.cyan, use_color)
        + "  "
        + colorize("■ Trainer", Theme.yellow, use_color)
        + "  "
        + colorize("◆ GPU", Theme.pink, use_color)
    )
    output.append(f"    {legend}")
    return output


def kv(label: str, value: str, use_color: bool, color: str = Theme.white) -> str:
    return colorize(f"{label:<12}", Theme.gray, use_color) + colorize(value, color, use_color)


def percent_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def resource_status(label: str, value: float | None) -> tuple[str, str]:
    if value is None:
        return "offline", Theme.gray
    if label == "GPU" and value < 5:
        return "idle", Theme.gray
    if label == "Disk" and value < 65:
        return "roomy", Theme.green
    if label == "Memory" and value >= 90:
        return "pressure", Theme.red
    if value >= 90:
        return "saturated", Theme.red
    if value >= 70:
        return "hot", Theme.orange
    if value >= 40:
        return "active", Theme.yellow
    return "cool", Theme.green


def resource_lane(
    label: str,
    value: float | None,
    values: Iterable[float | None],
    width: int,
    color: str,
    use_color: bool,
    note: str = "",
) -> str:
    status, status_color = resource_status(label, value)
    pct = percent_text(value).rjust(4)
    note_text = status if not note else f"{status}  {note}"
    gauge_width = max(12, min(30, width // 5))
    spark_width = max(14, min(88, width - gauge_width - len(label) - len(pct) - len(note_text) - 12))
    label_part = colorize(f"{label:<8}", color, use_color)
    pct_part = colorize(pct, percent_color(value), use_color)
    gauge = dense_bar(value, gauge_width, percent_color(value), use_color)
    wave = history_spark(values, spark_width, color, use_color)
    status_part = colorize(note_text, status_color, use_color)
    return fit(f"{label_part} {pct_part}  {gauge}  {wave}  {status_part}", width)


def compact_float(value: str, fallback: str = "n/a") -> str:
    try:
        parsed = float(value)
    except Exception:
        return fallback if not value else value
    return f"{parsed:.6f}"


def process_rows(sample: SystemSample, width: int, rows: int, use_color: bool) -> list[str]:
    bar_width = max(8, min(18, width // 6))
    command_width = max(16, width - bar_width - 28)
    output = [
        colorize(f"{'':>2} {'PID':>6} {'CPU':>5} {'LOAD':<{bar_width}} {'MEM':>5}  COMMAND", Theme.gray, use_color)
    ]
    for rank, process in enumerate(sample.processes[:rows], start=1):
        command = Path(process.command).name or process.command
        cpu_value = min(100.0, max(0.0, process.cpu))
        cpu_color = percent_color(cpu_value)
        output.append(
            f"{rank:>2} {process.pid:>6} "
            + colorize(f"{process.cpu:>4.0f}%", cpu_color, use_color)
            + " "
            + dense_bar(cpu_value, bar_width, cpu_color, use_color)
            + f" {process.mem:>4.1f}%  "
            + fit(command, command_width)
        )
    return output


def update_rate(history: list[dict[str, float | None]], key: str) -> float | None:
    points = [row for row in history if row.get("time") is not None and row.get(key) is not None]
    if len(points) < 2:
        return None
    start = points[max(0, len(points) - 30)]
    end = points[-1]
    delta_updates = (end[key] or 0) - (start[key] or 0)
    delta_time = (end["time"] or 0) - (start["time"] or 0)
    if delta_time <= 0:
        return None
    return (delta_updates / delta_time) * 60.0


def render_dashboard(repo_root: Path, use_color: bool) -> str:
    state = load_blacklight(repo_root)
    progress = state.env
    trainer_pid = int(progress.get("TRAINER_PID") or 0)
    sample = sample_system(repo_root, trainer_pid)
    history = append_history(history_path(repo_root, state), sample, state)
    cols, rows = shutil.get_terminal_size((150, 44))
    cols = max(90, cols)
    now = int(time.time())
    started = int(float(progress.get("TRAINING_STARTED_AT") or 0))
    elapsed = fmt_duration(now - started) if started else "n/a"
    phase = progress.get("PHASE") or "unknown"
    delta_updates = max(0, state.model_updates - state.parent_updates) if state.parent_updates else 0
    metrics = state.metrics
    backend = "CPU+GPU" if progress.get("GPU_LEARNER") == "1" else "CPU trainer"
    if progress.get("GPU_DEVICE"):
        backend = f"{backend} ({progress.get('GPU_DEVICE')})"
    update_rate_value = update_rate(history, "updates")
    csv_rate_value = update_rate(history, "csv")
    if update_rate_value is not None and update_rate_value >= 1:
        rate_text = f"{update_rate_value:.0f}/min"
    elif csv_rate_value is not None and csv_rate_value >= 1:
        rate_text = f"{csv_rate_value:.0f} ep/min"
    elif update_rate_value is not None or csv_rate_value is not None:
        rate_text = "warming"
    else:
        rate_text = "warming"
    policy = metrics.get("policy_loss", "n/a")
    entropy = metrics.get("entropy", "n/a")
    csv_ep = metrics.get("episode", "0")

    header = colorize("╔" + ("═" * (cols - 2)) + "╗", Theme.cyan, use_color)
    title = f" BLACKLIGHT TRAINING DECK  {time.strftime('%Y-%m-%d %H:%M:%S')} "
    title_line = colorize("║", Theme.cyan, use_color) + fit(colorize(title, Theme.white + Theme.bold, use_color), cols - 2) + colorize("║", Theme.cyan, use_color)
    header_bottom = colorize("╚" + ("═" * (cols - 2)) + "╝", Theme.cyan, use_color)

    if cols >= 132:
        left_w = max(40, min(50, cols // 3))
        mid_w = max(42, min(54, cols // 3))
        right_w = cols - left_w - mid_w - 4
    else:
        left_w = cols // 2 - 1
        mid_w = cols - left_w - 2
        right_w = cols

    phase_color = Theme.green if phase == "finished" else Theme.cyan if phase == "syncing" else Theme.yellow
    blacklight_lines = [
        kv("Run", progress.get("RUN_NAME", "unknown"), use_color),
        kv("Phase", phase_label(phase), use_color, phase_color),
        kv("Mode", progress.get("TRAINING_MODE", "unknown"), use_color),
        kv("Backend", backend, use_color, Theme.cyan if progress.get("GPU_LEARNER") == "1" else Theme.gray),
        kv("Elapsed", elapsed, use_color, Theme.yellow),
        kv("Updates", f"{state.model_updates:,}  (+{delta_updates:,})", use_color, Theme.green),
        kv("Rate", rate_text, use_color, Theme.yellow),
        kv("Checkpoint", state.latest_checkpoint or "pending", use_color),
    ]

    training_lines = [
        kv("CSV Ep", f"{int(float(csv_ep)):,}" if str(csv_ep).replace(".", "", 1).isdigit() else str(csv_ep), use_color),
        kv("Policy", str(policy), use_color, Theme.yellow),
        kv("Entropy", str(entropy), use_color, Theme.pink),
        kv("Teacher", f"{human_bytes(state.source_bytes)} • {state.source_count} logs", use_color, Theme.cyan),
        kv("Trainer PID", str(trainer_pid or "idle"), use_color),
        kv("Event", state.latest_event or "none", use_color, Theme.gray),
    ]

    system_lines = [
        kv("CPU", percent_text(sample.cpu), use_color, percent_color(sample.cpu)),
        kv("Trainer", percent_text(sample.trainer_cpu), use_color, percent_color(sample.trainer_cpu)),
        kv("GPU", percent_text(sample.gpu), use_color, percent_color(sample.gpu)),
        kv("Memory", f"{sample.memory_used:.0f}% of {human_bytes(sample.memory_total)}" if sample.memory_used is not None else "n/a", use_color),
        kv("Disk", f"{sample.disk_used:.0f}% of {human_bytes(sample.disk_total)}" if sample.disk_used is not None else "n/a", use_color),
        kv("Battery", sample.battery, use_color),
        kv("Load", " ".join(f"{value:.2f}" for value in sample.load_avg), use_color),
    ]

    top_panels: list[str]
    if cols >= 132:
        top_panels = join_columns(
            [
                box("Blacklight", blacklight_lines, left_w, use_color, Theme.purple),
                box("Training Stats", training_lines, mid_w, use_color, Theme.blue),
                box("Laptop Load", system_lines, right_w, use_color, Theme.cyan),
            ]
        )
    else:
        top_panels = join_columns(
            [
                box("Blacklight", blacklight_lines, left_w, use_color, Theme.purple),
                box("Training Stats", training_lines + system_lines[:4], mid_w, use_color, Theme.blue),
            ]
        )

    resource_inner = cols - 4
    resource_summary = (
        colorize("now", Theme.gray, use_color)
        + "  "
        + colorize(f"CPU {percent_text(sample.cpu)}", percent_color(sample.cpu), use_color)
        + "  "
        + colorize(f"Trainer {percent_text(sample.trainer_cpu)}", percent_color(sample.trainer_cpu), use_color)
        + "  "
        + colorize(f"GPU {percent_text(sample.gpu)}", percent_color(sample.gpu), use_color)
        + "  "
        + colorize(f"Memory {percent_text(sample.memory_used)}", percent_color(sample.memory_used), use_color)
        + "  "
        + colorize(f"Disk {percent_text(sample.disk_used)}", percent_color(sample.disk_used), use_color)
        + "  "
        + colorize(f"Battery {sample.battery}", Theme.gray, use_color)
    )
    resource_lines = [
        resource_summary,
        resource_lane(
            "CPU",
            sample.cpu,
            [row.get("cpu") for row in history],
            resource_inner,
            Theme.cyan,
            use_color,
            "load " + " ".join(f"{value:.2f}" for value in sample.load_avg),
        ),
        resource_lane(
            "Trainer",
            sample.trainer_cpu,
            [row.get("trainer") for row in history],
            resource_inner,
            Theme.yellow,
            use_color,
            f"pid {trainer_pid}" if trainer_pid else "idle",
        ),
        resource_lane(
            "GPU",
            sample.gpu,
            [row.get("gpu") for row in history],
            resource_inner,
            Theme.pink,
            use_color,
            "Apple AGX",
        ),
        resource_lane(
            "Memory",
            sample.memory_used,
            [row.get("mem") for row in history],
            resource_inner,
            Theme.green,
            use_color,
            f"of {human_bytes(sample.memory_total)}" if sample.memory_total else "",
        ),
        resource_lane(
            "Disk",
            sample.disk_used,
            [row.get("disk") for row in history],
            resource_inner,
            Theme.purple,
            use_color,
            f"of {human_bytes(sample.disk_total)}" if sample.disk_total else "",
        ),
    ]
    resource_panel = box("Performance Console", resource_lines, cols, use_color, Theme.cyan)

    gain_target = 100000.0
    gain_fill = min(100.0, (delta_updates / gain_target) * 100.0) if delta_updates else 0.0
    gain_width = max(12, min(34, cols // 5))
    csv_display = f"{int(float(csv_ep)):,}" if str(csv_ep).replace(".", "", 1).isdigit() else str(csv_ep)
    learning_lines = [
        colorize("Session Gain  ", Theme.gray, use_color)
        + colorize(f"+{delta_updates:,}", Theme.green, use_color)
        + "  "
        + dense_bar(gain_fill, gain_width, Theme.green, use_color)
        + colorize(f"  {state.model_updates:,} total", Theme.gray, use_color),
        colorize("Rate          ", Theme.gray, use_color)
        + colorize(rate_text, Theme.yellow, use_color)
        + colorize("    CSV Ep ", Theme.gray, use_color)
        + colorize(csv_display, Theme.white, use_color)
        + colorize("    Teacher ", Theme.gray, use_color)
        + colorize(f"{human_bytes(state.source_bytes)} / {state.source_count} logs", Theme.cyan, use_color),
        colorize("Policy        ", Theme.gray, use_color)
        + colorize(compact_float(str(policy)), Theme.yellow, use_color)
        + colorize("    Entropy ", Theme.gray, use_color)
        + colorize(compact_float(str(entropy)), Theme.pink, use_color),
        colorize("Checkpoint    ", Theme.gray, use_color) + colorize(state.latest_checkpoint or "pending", Theme.white, use_color),
        colorize("Event         ", Theme.gray, use_color) + colorize(state.latest_event or "none", Theme.gray, use_color),
    ]

    lower: list[str]
    if cols >= 130:
        learn_width = max(64, cols // 2)
        proc_width = cols - learn_width - 2
        process_lines = process_rows(sample, proc_width - 4, min(8, max(4, rows - 34)), use_color)
        lower = join_columns(
            [
                box("Blacklight Learning Stream", learning_lines, learn_width, use_color, Theme.purple),
                box("Hot Processes", process_lines, proc_width, use_color, Theme.yellow),
            ]
        )
    else:
        process_lines = process_rows(sample, cols - 4, min(8, max(4, rows - 34)), use_color)
        lower = box("Blacklight Learning Stream", learning_lines, cols, use_color, Theme.purple)
        lower.extend(box("Hot Processes", process_lines, cols, use_color, Theme.yellow))

    lines = [header, title_line, header_bottom, *top_panels, *resource_panel, *lower]
    if len(lines) > rows - 1:
        lines = lines[: rows - 1]
    return "\n".join(lines)


def run_dashboard(repo_root: Path, interval: int, once: bool, use_color: bool) -> int:
    stop = False

    def handle_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    interactive = sys.stdout.isatty() and not once
    if interactive:
        sys.stdout.write("\033[?1049h\033[?25l")
        sys.stdout.flush()
    try:
        while not stop:
            frame = render_dashboard(repo_root, use_color)
            if interactive:
                sys.stdout.write("\033[H\033[2J" + frame)
                sys.stdout.flush()
            else:
                print(frame)
            if once:
                break
            time.sleep(interval)
    finally:
        if interactive:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Blacklight terminal dashboard")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    force_color = os.environ.get("BLACKLIGHT_MONITOR_FORCE_COLOR") == "1"
    use_color = force_color or (sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR"))
    interval = max(1, args.interval)
    return run_dashboard(Path(args.repo_root).resolve(), interval, args.once, use_color)


if __name__ == "__main__":
    raise SystemExit(main())
