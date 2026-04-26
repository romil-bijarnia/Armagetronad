#!/usr/bin/env python3
"""GPU-backed teacher-imitation learner for Blacklight.

This script trains the same CNN model format used by gTrainedAI.cpp, but runs
the replay learner through PyTorch so Apple Silicon MPS/CUDA can do the heavy
matrix work while the CPU streams teacher examples from the Armagetron logs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator


MAGIC = "ARMAGETRON_TRAINED_AI_CNN_V1"
SCALARS = 6
MAP_SIZE = 25
MAP_CHANNELS = 7
CONV1 = 32
CONV2 = 64
HIDDEN0 = 256
HIDDEN1 = 128
ACTIONS = 3
KERNEL = 3
MAP_CELLS = MAP_SIZE * MAP_SIZE
FEATURES = SCALARS + MAP_CHANNELS * MAP_CELLS
DENSE_INPUT = SCALARS + CONV2 * MAP_CELLS
WEIGHT_DECAY = 0.0002
WEIGHT_CLIP = 5.0


@dataclass
class ModelStats:
    baseline: float
    episodes: int
    updates: int


@dataclass
class TeacherExample:
    action: int
    can_left: bool
    can_right: bool
    scalars: list[float]
    grid: list[float]


@dataclass
class TrainSummary:
    examples: int
    batches: int
    avg_loss: float
    avg_entropy: float
    device: str
    model_path: Path
    run_dir: Path


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def quote_env(value: str | int | float) -> str:
    return shlex.quote(str(value))


def write_env(path: Path, values: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for key, value in values.items():
            handle.write(f"{key}={quote_env(value)}\n")


def append_event(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def source_paths(repo_root: Path, source_list: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_line in source_list.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = repo_root / "var" / path
        if path.exists() and path.is_file():
            paths.append(path)
    if not paths:
        die(f"No readable teacher logs found in source list: {source_list}")
    return paths


def parse_teacher_line(line: str) -> TeacherExample | None:
    parts = line.split()
    if len(parts) != 7 + FEATURES or parts[0] != "teacher_v1":
        return None
    try:
        action = int(parts[3])
        can_left = int(parts[4]) != 0
        can_right = int(parts[5]) != 0
        scalars = [float(value) for value in parts[7 : 7 + SCALARS]]
        grid = [float(value) for value in parts[7 + SCALARS :]]
    except ValueError:
        return None
    if action < 0 or action >= ACTIONS:
        return None
    return TeacherExample(action, can_left, can_right, scalars, grid)


def iter_teacher_examples(paths: Iterable[Path], max_examples: int | None) -> Iterator[TeacherExample]:
    emitted = 0
    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                if max_examples is not None and emitted >= max_examples:
                    return
                if not line.startswith("teacher_v1 "):
                    continue
                example = parse_teacher_line(line)
                if example is None:
                    continue
                emitted += 1
                yield example


def load_model(path: Path, torch: object) -> tuple[ModelStats, dict[str, object]]:
    if not path.exists():
        die(f"Initial model does not exist: {path}")

    with path.open(errors="replace") as handle:
        magic = handle.readline().strip()
        dims = [int(value) for value in handle.readline().split()]
        stats_line = handle.readline().split()
        if magic != MAGIC:
            die(f"Unsupported model magic in {path}: {magic}")
        expected = [SCALARS, MAP_SIZE, MAP_CHANNELS, CONV1, CONV2, HIDDEN0, HIDDEN1, ACTIONS]
        if dims != expected:
            die(f"Unsupported model dimensions in {path}: {dims}")
        if len(stats_line) != 3:
            die(f"Invalid model stats line in {path}")

        stats = ModelStats(float(stats_line[0]), int(float(stats_line[1])), int(float(stats_line[2])))

        def read_floats(count: int) -> list[float]:
            values: list[float] = []
            while len(values) < count:
                line = handle.readline()
                if not line:
                    break
                values.extend(float(value) for value in line.split())
            if len(values) != count:
                die(f"Model file ended early while reading {path}")
            return values

        state: dict[str, object] = {}
        conv1_values = read_floats(CONV1 * (MAP_CHANNELS * KERNEL * KERNEL + 1))
        conv2_values = read_floats(CONV2 * (CONV1 * KERNEL * KERNEL + 1))
        dense0_values = read_floats(HIDDEN0 * (DENSE_INPUT + 1))
        dense1_values = read_floats(HIDDEN1 * (HIDDEN0 + 1))
        policy_values = read_floats(ACTIONS * (HIDDEN1 + 1))
        value_values = read_floats(HIDDEN1 + 1)

    def split_rows(values: list[float], rows: int, width: int) -> tuple[list[float], list[float]]:
        weights: list[float] = []
        biases: list[float] = []
        index = 0
        for _ in range(rows):
            weights.extend(values[index : index + width])
            index += width
            biases.append(values[index])
            index += 1
        return weights, biases

    conv1_w, conv1_b = split_rows(conv1_values, CONV1, MAP_CHANNELS * KERNEL * KERNEL)
    conv2_w, conv2_b = split_rows(conv2_values, CONV2, CONV1 * KERNEL * KERNEL)
    dense0_w, dense0_b = split_rows(dense0_values, HIDDEN0, DENSE_INPUT)
    dense1_w, dense1_b = split_rows(dense1_values, HIDDEN1, HIDDEN0)
    policy_w, policy_b = split_rows(policy_values, ACTIONS, HIDDEN1)
    value_w = value_values[:HIDDEN1]
    value_b = value_values[HIDDEN1]

    state["conv1.weight"] = torch.tensor(conv1_w, dtype=torch.float32).reshape(CONV1, MAP_CHANNELS, KERNEL, KERNEL)
    state["conv1.bias"] = torch.tensor(conv1_b, dtype=torch.float32)
    state["conv2.weight"] = torch.tensor(conv2_w, dtype=torch.float32).reshape(CONV2, CONV1, KERNEL, KERNEL)
    state["conv2.bias"] = torch.tensor(conv2_b, dtype=torch.float32)
    state["dense0.weight"] = torch.tensor(dense0_w, dtype=torch.float32).reshape(HIDDEN0, DENSE_INPUT)
    state["dense0.bias"] = torch.tensor(dense0_b, dtype=torch.float32)
    state["dense1.weight"] = torch.tensor(dense1_w, dtype=torch.float32).reshape(HIDDEN1, HIDDEN0)
    state["dense1.bias"] = torch.tensor(dense1_b, dtype=torch.float32)
    state["policy.weight"] = torch.tensor(policy_w, dtype=torch.float32).reshape(ACTIONS, HIDDEN1)
    state["policy.bias"] = torch.tensor(policy_b, dtype=torch.float32)
    state["value.weight"] = torch.tensor(value_w, dtype=torch.float32).reshape(1, HIDDEN1)
    state["value.bias"] = torch.tensor([value_b], dtype=torch.float32)
    return stats, state


def write_model(path: Path, model: object, stats: ModelStats) -> None:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        out.write(f"{MAGIC}\n")
        out.write(f"{SCALARS} {MAP_SIZE} {MAP_CHANNELS} {CONV1} {CONV2} {HIDDEN0} {HIDDEN1} {ACTIONS}\n")
        out.write(f"{stats.baseline:.9f} {stats.episodes} {stats.updates}\n")

        def row(values: object) -> str:
            return " ".join(f"{float(value):.9f}" for value in values)

        conv1_w = state["conv1.weight"].reshape(CONV1, -1)
        conv1_b = state["conv1.bias"]
        for oc in range(CONV1):
            out.write(row(conv1_w[oc]) + f" {float(conv1_b[oc]):.9f}\n")

        conv2_w = state["conv2.weight"].reshape(CONV2, -1)
        conv2_b = state["conv2.bias"]
        for oc in range(CONV2):
            out.write(row(conv2_w[oc]) + f" {float(conv2_b[oc]):.9f}\n")

        dense0_w = state["dense0.weight"]
        dense0_b = state["dense0.bias"]
        for j in range(HIDDEN0):
            out.write(row(dense0_w[j]) + f" {float(dense0_b[j]):.9f}\n")

        dense1_w = state["dense1.weight"]
        dense1_b = state["dense1.bias"]
        for j in range(HIDDEN1):
            out.write(row(dense1_w[j]) + f" {float(dense1_b[j]):.9f}\n")

        policy_w = state["policy.weight"]
        policy_b = state["policy.bias"]
        for action in range(ACTIONS):
            out.write(row(policy_w[action]) + f" {float(policy_b[action]):.9f}\n")

        out.write(row(state["value.weight"].reshape(-1)) + f" {float(state['value.bias'][0]):.9f}\n")


def choose_device(torch: object, requested: str) -> object:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        die("PyTorch is installed, but MPS is not available on this Python/runtime.")
    if requested == "cuda" and not torch.cuda.is_available():
        die("PyTorch is installed, but CUDA is not available here.")
    return torch.device(requested)


def require_torch() -> object:
    try:
        import torch
        return torch
    except ModuleNotFoundError:
        die(
            "Blacklight GPU training needs PyTorch with Apple MPS support.\n"
            "Set it up with:\n"
            "  ./scripts/blacklight.sh gpu-setup\n"
            "Then run:\n"
            "  ./scripts/blacklight.sh train --resume champion"
        )


def build_model(torch: object) -> object:
    import torch.nn as nn

    class BlacklightCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(MAP_CHANNELS, CONV1, KERNEL, padding=1)
            self.conv2 = nn.Conv2d(CONV1, CONV2, KERNEL, padding=1)
            self.dense0 = nn.Linear(DENSE_INPUT, HIDDEN0)
            self.dense1 = nn.Linear(HIDDEN0, HIDDEN1)
            self.policy = nn.Linear(HIDDEN1, ACTIONS)
            self.value = nn.Linear(HIDDEN1, 1)

        def forward(self, scalars: object, grid: object, can_left: object, can_right: object) -> tuple[object, object]:
            x = torch.tanh(self.conv1(grid))
            x = torch.tanh(self.conv2(x))
            x = torch.cat((scalars, x.reshape(x.shape[0], -1)), dim=1)
            x = torch.tanh(self.dense0(x))
            x = torch.tanh(self.dense1(x))
            logits = self.policy(x)
            blocked = torch.full_like(logits, -1.0e20)
            logits = torch.where(can_left[:, None] | (torch.arange(ACTIONS, device=logits.device)[None, :] != 0), logits, blocked)
            logits = torch.where(can_right[:, None] | (torch.arange(ACTIONS, device=logits.device)[None, :] != 2), logits, blocked)
            return logits, self.value(x).reshape(-1)

    return BlacklightCNN()


def checkpoint_path(prefix: Path, episodes: int) -> Path:
    return prefix.parent / f"{prefix.name}_ep{episodes:08d}.txt"


def maybe_write_checkpoint(model: object, stats: ModelStats, prefix: Path, checkpoint_every: int) -> None:
    if checkpoint_every <= 0 or stats.episodes % checkpoint_every != 0:
        return
    path = checkpoint_path(prefix, stats.episodes)
    write_model(path, model, stats)
    latest = prefix.parent / f"{prefix.name}_latest.txt"
    latest.write_text(f"{path}\n{stats.episodes}\n")


def append_metrics(metrics_path: Path, row: dict[str, float | int]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "episode",
        "survived",
        "reward",
        "distance",
        "average_predicted_value",
        "steps",
        "learned",
        "policy_episodes",
        "policy_updates",
        "cumulative_win_rate",
        "cumulative_average_reward",
        "cumulative_average_distance",
        "cumulative_average_predicted_value",
        "policy_loss",
        "value_loss",
        "entropy",
        "cumulative_average_policy_loss",
        "cumulative_average_value_loss",
        "cumulative_average_entropy",
    ]
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    with metrics_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_metrics_summary(
    metrics_path: Path,
    examples: int,
    loss_total: float,
    entropy_total: float,
    last_loss: float,
    last_entropy: float,
    stats: ModelStats,
) -> None:
    avg_loss = loss_total / examples if examples else 0.0
    avg_entropy = entropy_total / examples if examples else 0.0
    summary = metrics_path.with_suffix(metrics_path.suffix + ".latest")
    with summary.open("w") as out:
        out.write(f"episodes {examples}\n")
        out.write("wins 0\n")
        out.write("win_rate 0.000000\n")
        out.write("average_reward 0.000000\n")
        out.write("average_distance 0.000000\n")
        out.write("average_predicted_value 0.000000\n")
        out.write("average_steps 1.000000\n")
        out.write(f"steps_total {examples}\n")
        out.write(f"learned_episodes {examples}\n")
        out.write(f"average_policy_loss {avg_loss:.6f}\n")
        out.write("average_value_loss 0.000000\n")
        out.write(f"average_entropy {avg_entropy:.6f}\n")
        out.write("last_survived 0\n")
        out.write("last_reward 0.000000\n")
        out.write("last_distance 0.000000\n")
        out.write("last_average_predicted_value 0.000000\n")
        out.write("last_steps 1\n")
        out.write(f"last_policy_loss {last_loss:.6f}\n")
        out.write("last_value_loss 0.000000\n")
        out.write(f"last_entropy {last_entropy:.6f}\n")
        out.write("last_learned 1\n")
        out.write(f"policy_episodes {stats.episodes}\n")
        out.write(f"policy_updates {stats.updates}\n")


def train(args: argparse.Namespace) -> TrainSummary:
    torch = require_torch()
    import torch.nn.functional as functional

    repo_root = Path(args.repo_root).resolve()
    source_list = Path(args.source_list).resolve()
    initial_model = Path(args.initial_model).resolve()
    run_name = args.run_name or f"{args.generation}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = repo_root / "var" / "blacklight_runs" / run_name
    model_path = run_dir / "trained_ai_teacher_cnn_model.txt"
    metrics_path = run_dir / "trained_ai_training_metrics.csv"
    checkpoint_prefix = run_dir / "checkpoints" / "blacklight"
    progress_path = run_dir / "training_progress.env"
    events_path = run_dir / "training_events.log"
    manifest_path = run_dir / "run_manifest.env"
    latest_manifest = repo_root / "var" / "blacklight_runs" / "latest_run.env"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("")

    started = int(time.time())
    paths = source_paths(repo_root, source_list)
    device = choose_device(torch, args.device)
    model = build_model(torch)
    stats, state = load_model(initial_model, torch)
    model.load_state_dict(state)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    examples = 0
    batches = 0
    loss_total = 0.0
    entropy_total = 0.0
    last_loss = 0.0
    last_entropy = 0.0

    manifest_values = {
        "RUN_NAME": run_name,
        "RUN_DIR_REL": f"blacklight_runs/{run_name}",
        "RUN_DIR_ABS": str(run_dir),
        "MODEL_REL": f"blacklight_runs/{run_name}/trained_ai_teacher_cnn_model.txt",
        "MODEL_ABS": str(model_path),
        "METRICS_REL": f"blacklight_runs/{run_name}/trained_ai_training_metrics.csv",
        "METRICS_ABS": str(metrics_path),
        "METRICS_SUMMARY_ABS": str(metrics_path) + ".latest",
        "CHECKPOINT_PREFIX_REL": f"blacklight_runs/{run_name}/checkpoints/blacklight",
        "CHECKPOINT_PREFIX_ABS": str(checkpoint_prefix),
        "GENERATION": args.generation,
        "TRAINING_MODE": "teacher",
        "PROFILE": "teacher-gpu",
        "PARENT_MODEL": str(args.parent_model or initial_model),
        "PARALLEL_WORKERS": 0,
        "SYNC_SECONDS": 0,
        "SOURCE_LIST_ABS": str(source_list),
        "PROGRESS_ABS": str(progress_path),
        "EVENTS_LOG_ABS": str(events_path),
        "DURATION_SECONDS": 0,
        "PROMOTION_STATUS": "unreviewed",
        "GPU_LEARNER": 1,
        "GPU_DEVICE": str(device),
    }
    write_env(manifest_path, manifest_values)
    write_env(latest_manifest, manifest_values)
    append_event(events_path, f"gpu learner started device={device} sources={len(paths)}")

    def update_progress(phase: str) -> None:
        write_env(
            progress_path,
            {
                "PHASE": phase,
                "TRAINER_PID": os.getpid(),
                "TRAINING_STARTED_AT": started,
                "CURRENT_CYCLE": 0,
                "WORKERS_RUNNING": 0,
                "LAST_SYNC_DURATION": 0,
                "GPU_LEARNER": 1,
                "GPU_DEVICE": str(device),
            },
        )

    update_progress("syncing")

    batch_scalars: list[list[float]] = []
    batch_grids: list[list[float]] = []
    batch_actions: list[int] = []
    batch_can_left: list[bool] = []
    batch_can_right: list[bool] = []

    def flush_batch() -> None:
        nonlocal examples, batches, loss_total, entropy_total, last_loss, last_entropy
        if not batch_actions:
            return

        scalars = torch.tensor(batch_scalars, dtype=torch.float32, device=device)
        grids = torch.tensor(batch_grids, dtype=torch.float32, device=device).reshape(-1, MAP_CHANNELS, MAP_SIZE, MAP_SIZE)
        actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
        can_left = torch.tensor(batch_can_left, dtype=torch.bool, device=device)
        can_right = torch.tensor(batch_can_right, dtype=torch.bool, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _value = model(scalars, grids, can_left, can_right)
        loss = functional.cross_entropy(logits, actions)
        loss.backward()
        optimizer.step()

        shrink = 1.0 - args.lr * WEIGHT_DECAY
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith(".weight"):
                    param.mul_(shrink)
                    param.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
            probabilities = torch.softmax(logits, dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1.0e-6))).sum(dim=1).mean()

        batch_size = len(batch_actions)
        examples += batch_size
        batches += 1
        stats.episodes += batch_size
        stats.updates += batch_size
        last_loss = float(loss.detach().cpu())
        last_entropy = float(entropy.detach().cpu())
        loss_total += last_loss * batch_size
        entropy_total += last_entropy * batch_size

        append_metrics(
            metrics_path,
            {
                "episode": examples,
                "survived": 0,
                "reward": "0.000000",
                "distance": "0.000000",
                "average_predicted_value": "0.000000",
                "steps": batch_size,
                "learned": 1,
                "policy_episodes": stats.episodes,
                "policy_updates": stats.updates,
                "cumulative_win_rate": "0.000000",
                "cumulative_average_reward": "0.000000",
                "cumulative_average_distance": "0.000000",
                "cumulative_average_predicted_value": "0.000000",
                "policy_loss": f"{last_loss:.6f}",
                "value_loss": "0.000000",
                "entropy": f"{last_entropy:.6f}",
                "cumulative_average_policy_loss": f"{loss_total / examples:.6f}",
                "cumulative_average_value_loss": "0.000000",
                "cumulative_average_entropy": f"{entropy_total / examples:.6f}",
            },
        )
        write_metrics_summary(metrics_path, examples, loss_total, entropy_total, last_loss, last_entropy, stats)
        if args.save_every > 0 and examples % args.save_every < batch_size:
            write_model(model_path, model, stats)
        maybe_write_checkpoint(model, stats, checkpoint_prefix, args.checkpoint_every)

        batch_scalars.clear()
        batch_grids.clear()
        batch_actions.clear()
        batch_can_left.clear()
        batch_can_right.clear()

    for _epoch in range(args.epochs):
        for example in iter_teacher_examples(paths, args.max_examples):
            batch_scalars.append(example.scalars)
            batch_grids.append(example.grid)
            batch_actions.append(example.action)
            batch_can_left.append(example.can_left)
            batch_can_right.append(example.can_right)
            if len(batch_actions) >= args.batch_size:
                flush_batch()
        flush_batch()

    write_model(model_path, model, stats)
    maybe_write_checkpoint(model, stats, checkpoint_prefix, args.checkpoint_every)
    write_metrics_summary(metrics_path, examples, loss_total, entropy_total, last_loss, last_entropy, stats)
    update_progress("finished")
    append_event(events_path, f"gpu learner finished examples={examples} batches={batches} device={device}")
    return TrainSummary(
        examples=examples,
        batches=batches,
        avg_loss=loss_total / examples if examples else math.nan,
        avg_entropy=entropy_total / examples if examples else math.nan,
        device=str(device),
        model_path=model_path,
        run_dir=run_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Blacklight from teacher logs on GPU when PyTorch MPS/CUDA is available.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-list", required=True)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--parent-model", default="")
    parser.add_argument("--generation", default="blacklight_gpu_train")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--check-backend", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        die("--batch-size must be at least 1")
    if args.epochs < 1:
        die("--epochs must be at least 1")
    if args.lr <= 0:
        die("--lr must be positive")
    if args.max_examples <= 0:
        args.max_examples = None

    if args.check_backend:
        torch = require_torch()
        device = choose_device(torch, args.device)
        print(f"torch={torch.__version__}")
        print(f"mps_available={torch.backends.mps.is_available()}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"selected_device={device}")
        return 0

    summary = train(args)
    print(f"Blacklight GPU learner finished on {summary.device}")
    print(f"examples {summary.examples}")
    print(f"batches {summary.batches}")
    print(f"average_policy_loss {summary.avg_loss:.6f}")
    print(f"average_entropy {summary.avg_entropy:.6f}")
    print(f"model {summary.model_path}")
    print(f"run_dir {summary.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
