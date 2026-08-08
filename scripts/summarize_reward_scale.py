#!/usr/bin/env python3
"""汇总 reward scale 消融 run 的 eval 指标，并列出 1M 晋级候选。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl.reward_scale_ablation import select_rsc_promotions  # noqa: E402


def _scalars(ea: EventAccumulator, tag: str) -> list[float]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [float(e.value) for e in ea.Scalars(tag)]


def load_run_eval_summary(run_dir: Path) -> dict[str, float] | None:
    if not run_dir.exists():
        return None
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    capt = _scalars(ea, "eval/capture_rate")
    dist = _scalars(ea, "eval/final_dist_mean")
    coll = _scalars(ea, "eval/collision_rate")
    ret = _scalars(ea, "eval/return_mean")
    if not dist:
        return None
    return {
        "capture_rate": capt[-1] if capt else 0.0,
        "final_dist_mean": dist[-1],
        "collision_rate": coll[-1] if coll else float("nan"),
        "return_mean": ret[-1] if ret else float("nan"),
        "best_capture_rate": max(capt) if capt else 0.0,
        "best_final_dist_mean": min(dist),
        "best_collision_rate": min(coll) if coll else float("nan"),
        "best_return_mean": max(ret) if ret else float("nan"),
    }


def load_rsc_final_metrics(
    logdir: Path, *, phase: str = "1m"
) -> dict[str, dict[str, float]]:
    phase_norm = phase.strip().lower()
    prefix = f"rsc_{phase_norm}_"
    out: dict[str, dict[str, float]] = {}
    if not logdir.exists():
        return out
    for path in sorted(logdir.iterdir()):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        short = path.name[len(prefix) :]
        summary = load_run_eval_summary(path)
        if summary is None:
            continue
        out[short] = {
            "capture_rate": summary["capture_rate"],
            "final_dist_mean": summary["final_dist_mean"],
            "collision_rate": summary["collision_rate"],
            "return_mean": summary["return_mean"],
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--phase", choices=("1m", "5m"), default="1m")
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="显式 run 目录名；默认扫描 rsc_{phase}_*",
    )
    p.add_argument("--list-promote", action="store_true")
    p.add_argument("--max-promote", type=int, default=2)
    args = p.parse_args()
    root = Path(args.logdir)

    if args.runs:
        names = list(args.runs)
    else:
        prefix = f"rsc_{args.phase}_"
        names = (
            sorted(
                d.name
                for d in root.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            )
            if root.exists()
            else []
        )

    headers = [
        "run",
        "final_capture",
        "final_dist",
        "final_coll",
        "final_return",
        "best_capture",
        "best_dist",
        "best_coll",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for name in names:
        summary = load_run_eval_summary(root / name)
        if summary is None:
            print(
                f"| {name} | "
                + " | ".join(["missing"] * (len(headers) - 1))
                + " |"
            )
            continue
        print(
            f"| {name} | "
            f"{summary['capture_rate']*100:.1f}% | "
            f"{summary['final_dist_mean']:.1f} | "
            f"{summary['collision_rate']*100:.1f}% | "
            f"{summary['return_mean']:.1f} | "
            f"{summary['best_capture_rate']*100:.1f}% | "
            f"{summary['best_final_dist_mean']:.1f} | "
            f"{summary['best_collision_rate']*100:.1f}% |"
        )

    if args.list_promote:
        metrics = load_rsc_final_metrics(root, phase=args.phase)
        try:
            promo = select_rsc_promotions(metrics, max_promote=args.max_promote)
        except ValueError as exc:
            print(f"promote: {exc}")
            return 1
        print("promote:", " ".join(promo) if promo else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
