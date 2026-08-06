#!/usr/bin/env python3
"""Print last scalar values for reward-preset runs under runs/."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "eval/return_mean",
    "eval/collision_rate",
    "eval/final_dist_mean",
    "eval/capture_rate",
    "eval/success_rate",
]


def last_scalar(run_dir: Path, tag: str) -> float | None:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    return float(events[-1].value) if events else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "rw_baseline",
            "rw_dist_up",
            "rw_ship_safe_dn",
            "rw_coll_soft",
            "rw_shape_up",
            "rw_combo",
        ],
    )
    args = parser.parse_args()
    root = Path(args.logdir)
    header = ["run", *TAGS]
    print("\t".join(header))
    for name in args.runs:
        row = [name]
        run_dir = root / name
        for tag in TAGS:
            val = last_scalar(run_dir, tag) if run_dir.is_dir() else None
            row.append(f"{val:.4g}" if val is not None else "NA")
        print("\t".join(row))


if __name__ == "__main__":
    main()
