#!/usr/bin/env python3
"""汇总 TF 规模消融 run 的末期 eval 指标。"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = [
    "eval/final_dist_mean",
    "eval/collision_rate",
    "eval/capture_rate",
    "eval/success_rate",
    "eval/return_mean",
    "loss/explained_variance",
]


def last_scalar(ea: EventAccumulator, tag: str) -> float | None:
    if tag not in ea.Tags().get("scalars", []):
        return None
    ev = ea.Scalars(tag)
    return float(ev[-1].value) if ev else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    root = Path(args.logdir)

    rows = []
    for name in args.runs:
        path = root / name
        if not path.exists():
            rows.append((name, None))
            continue
        ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
        ea.Reload()
        rows.append((name, {k: last_scalar(ea, k) for k in KEYS}))

    headers = ["run"] + KEYS
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for name, metrics in rows:
        if metrics is None:
            print(f"| {name} | " + " | ".join(["missing"] * len(KEYS)) + " |")
            continue
        cells = []
        for k in KEYS:
            v = metrics[k]
            if v is None:
                cells.append("n/a")
            elif "rate" in k:
                cells.append(f"{v*100:.1f}%")
            elif "dist" in k or "return" in k:
                cells.append(f"{v:.1f}")
            else:
                cells.append(f"{v:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
