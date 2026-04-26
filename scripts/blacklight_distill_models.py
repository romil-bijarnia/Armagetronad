#!/usr/bin/env python3
"""Blend model-source weights into the single Blacklight current model.

This is intentionally conservative. It keeps the current model as the anchor
and moves each parameter a small fraction toward the mean of one or more source
models. It does not pick a new winner and it does not need PyTorch; it works
directly on the text CNN model format that Armagetron loads.
"""

from __future__ import annotations

import argparse
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


MAGIC = "ARMAGETRON_TRAINED_AI_CNN_V1"
EXPECTED_DIMS = "6 25 7 32 64 256 128 3"


@dataclass
class ModelHeader:
    magic: str
    dims: str
    baseline: float
    episodes: int
    updates: int


def die(message: str) -> None:
    raise SystemExit(message)


def parse_stats(raw: str, path: Path) -> tuple[float, int, int]:
    parts = raw.split()
    if len(parts) != 3:
        die(f"Invalid stats line in {path}: {raw.rstrip()}")
    try:
        return float(parts[0]), int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        die(f"Invalid numeric stats in {path}: {raw.rstrip()}")


def read_header(handle: TextIO, path: Path) -> ModelHeader:
    magic = handle.readline().strip()
    dims = handle.readline().strip()
    stats_line = handle.readline()

    if magic != MAGIC:
        die(f"Unsupported model magic in {path}: {magic}")
    if dims != EXPECTED_DIMS:
        die(f"Unsupported model dimensions in {path}: {dims}")
    if not stats_line:
        die(f"Missing stats line in {path}")

    baseline, episodes, updates = parse_stats(stats_line, path)
    return ModelHeader(magic, dims, baseline, episodes, updates)


def parse_payload_line(raw: str, path: Path, line_number: int) -> list[float]:
    try:
        return [float(value) for value in raw.split()]
    except ValueError:
        die(f"Invalid float payload in {path}:{line_number}")


def quote_env(value: str | int | float) -> str:
    return shlex.quote(str(value))


def write_summary(path: Path, values: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for key, value in values.items():
            handle.write(f"{key}={quote_env(value)}\n")


def distill(args: argparse.Namespace) -> dict[str, str | int | float]:
    current_path = Path(args.current).resolve()
    source_paths = [Path(source).resolve() for source in args.source]
    output_path = Path(args.output).resolve()
    alpha = float(args.alpha)

    if not current_path.is_file():
        die(f"Current model does not exist: {current_path}")
    if not source_paths:
        die("At least one --source model is required.")
    if alpha <= 0.0 or alpha > 1.0:
        die("--alpha must be greater than 0 and no more than 1.")

    for source_path in source_paths:
        if not source_path.is_file():
            die(f"Source model does not exist: {source_path}")
        if source_path == current_path:
            die(f"Source model is the current model, so there is nothing to distill: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_handles: list[TextIO] = []
    source_headers: list[ModelHeader] = []
    total_values = 0
    total_abs_delta = 0.0
    max_abs_delta = 0.0
    line_count = 0

    with current_path.open(errors="replace") as current_handle:
        current_header = read_header(current_handle, current_path)

        try:
            for source_path in source_paths:
                handle = source_path.open(errors="replace")
                source_handles.append(handle)
                source_headers.append(read_header(handle, source_path))

            with output_path.open("w") as out:
                out.write(f"{MAGIC}\n")
                out.write(f"{EXPECTED_DIMS}\n")
                out.write(
                    f"{current_header.baseline:.9f} "
                    f"{current_header.episodes} "
                    f"{current_header.updates + 1}\n"
                )

                for line_number, current_raw in enumerate(current_handle, start=4):
                    if not current_raw.strip():
                        continue
                    current_values = parse_payload_line(current_raw, current_path, line_number)
                    source_lines = [handle.readline() for handle in source_handles]
                    if any(not line for line in source_lines):
                        die(f"A source model ended early while reading payload line {line_number}.")

                    source_values = [
                        parse_payload_line(source_raw, source_paths[index], line_number)
                        for index, source_raw in enumerate(source_lines)
                    ]
                    if any(len(values) != len(current_values) for values in source_values):
                        die(f"Payload width mismatch at line {line_number}.")

                    blended: list[float] = []
                    for index, current_value in enumerate(current_values):
                        source_mean = sum(values[index] for values in source_values) / len(source_values)
                        new_value = (1.0 - alpha) * current_value + alpha * source_mean
                        if not math.isfinite(new_value):
                            die(f"Non-finite blended value at line {line_number}.")
                        delta = abs(new_value - current_value)
                        total_abs_delta += delta
                        max_abs_delta = max(max_abs_delta, delta)
                        total_values += 1
                        blended.append(new_value)

                    out.write(" ".join(f"{value:.9f}" for value in blended))
                    out.write("\n")
                    line_count += 1

            for index, handle in enumerate(source_handles):
                extra = handle.readline()
                if extra.strip():
                    die(f"Source model has extra payload after current model ended: {source_paths[index]}")

        finally:
            for handle in source_handles:
                handle.close()

    mean_abs_delta = total_abs_delta / total_values if total_values else 0.0
    source_episode_max = max(header.episodes for header in source_headers)
    source_update_max = max(header.updates for header in source_headers)

    summary: dict[str, str | int | float] = {
        "CURRENT_MODEL_ABS": str(current_path),
        "OUTPUT_MODEL_ABS": str(output_path),
        "SOURCE_COUNT": len(source_paths),
        "ALPHA": f"{alpha:.9f}",
        "PAYLOAD_LINES": line_count,
        "PAYLOAD_VALUES": total_values,
        "MEAN_ABS_DELTA": f"{mean_abs_delta:.12f}",
        "MAX_ABS_DELTA": f"{max_abs_delta:.12f}",
        "CURRENT_EPISODES": current_header.episodes,
        "CURRENT_UPDATES": current_header.updates,
        "SOURCE_MAX_EPISODES": source_episode_max,
        "SOURCE_MAX_UPDATES": source_update_max,
        "OUTPUT_EPISODES": current_header.episodes,
        "OUTPUT_UPDATES": current_header.updates + 1,
    }
    for index, source_path in enumerate(source_paths, start=1):
        summary[f"SOURCE_MODEL_{index}_ABS"] = str(source_path)

    if args.summary:
        write_summary(Path(args.summary).resolve(), summary)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill source Blacklight model weights into the current model.")
    parser.add_argument("--current", required=True, help="Current champion model to keep as the anchor.")
    parser.add_argument("--source", action="append", default=[], help="Model source to absorb. Can be passed more than once.")
    parser.add_argument("--output", required=True, help="Output path for the distilled current model.")
    parser.add_argument("--summary", default="", help="Optional env-style summary output path.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Fraction of source weights to absorb into current.")
    args = parser.parse_args()

    summary = distill(args)
    print("Blacklight current-model distillation complete")
    print(f"current_model {summary['CURRENT_MODEL_ABS']}")
    print(f"output_model {summary['OUTPUT_MODEL_ABS']}")
    print(f"sources {summary['SOURCE_COUNT']}")
    print(f"alpha {summary['ALPHA']}")
    print(f"payload_values {summary['PAYLOAD_VALUES']}")
    print(f"mean_abs_delta {summary['MEAN_ABS_DELTA']}")
    print(f"max_abs_delta {summary['MAX_ABS_DELTA']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
