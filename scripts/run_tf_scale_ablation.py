#!/usr/bin/env python3
"""串行跑 Transformer 规模消融（见 docs/tf_scale_ablation.md）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TF_SIZE_PRESETS, list_tf_size_presets  # noqa: E402

ORDER = ["S", "M", "L"]


def main() -> int:
    p = argparse.ArgumentParser(description="串行跑 TF 规模消融")
    p.add_argument("--sizes", nargs="+", default=ORDER, choices=ORDER)
    p.add_argument("--total-steps", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--env-backend", type=str, default="cuda")
    p.add_argument("--num-envs", type=int, default=12288)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--minibatch-size", type=int, default=65536)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-summarize", action="store_true")
    p.add_argument("--log-file", type=str, default="outputs/logs/tf_scale_ablation.log")
    args = p.parse_args()

    known = set(list_tf_size_presets())
    for size in args.sizes:
        if size not in TF_SIZE_PRESETS or size not in known:
            raise SystemExit(f"Unknown size {size!r}. Known: {', '.join(sorted(known))}")

    log_path = _ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmds: list[list[str]] = []
    for size in args.sizes:
        run_name = f"tf_scale_{size}_r120"
        cmds.append([
            sys.executable,
            str(_ROOT / "scripts" / "train.py"),
            "--arch", "transformer",
            "--tf-size", size,
            "--init-radius", "120",
            "--slot-assignment", "minimax",
            "--run-name", run_name,
            "--total-steps", str(args.total_steps),
            "--seed", str(args.seed),
            "--device", args.device,
            "--env-backend", args.env_backend,
            "--num-envs", str(args.num_envs),
            "--rollout-steps", str(args.rollout_steps),
            "--minibatch-size", str(args.minibatch_size),
            "--eval-workers", str(args.eval_workers),
        ])

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"=== tf scale ablation {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for cmd in cmds:
            line = " ".join(cmd)
            print(line)
            f.write(line + "\n")
            f.flush()
            if args.dry_run:
                continue
            proc = subprocess.run(cmd, cwd=str(_ROOT))
            if proc.returncode != 0:
                return int(proc.returncode)

    if not args.dry_run and not args.skip_summarize:
        summary = [
            sys.executable,
            str(_ROOT / "scripts" / "summarize_tf_scale.py"),
            "--runs",
            *[f"tf_scale_{s}_r120" for s in args.sizes],
        ]
        return int(subprocess.run(summary, cwd=str(_ROOT)).returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
