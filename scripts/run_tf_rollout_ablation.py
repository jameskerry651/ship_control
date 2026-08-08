#!/usr/bin/env python3
"""串行跑 Transformer-S 的 rollout_steps 消融（见 design spec）。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ORDER = [32, 64, 128]


def main() -> int:
    p = argparse.ArgumentParser(description="串行跑 TF-S rollout_steps 消融")
    p.add_argument("--rollouts", nargs="+", type=int, default=ORDER)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--env-backend", type=str, default="cuda")
    # 小并行以便 1M steps 下有多次 PPO update（12288 时每 update 已超 1M）
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--log-file",
        type=str,
        default="outputs/logs/tf_rollout_ablation.log",
    )
    args = p.parse_args()

    for r in args.rollouts:
        if r <= 0:
            raise SystemExit(f"rollout_steps must be positive, got {r}")

    log_path = _ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmds: list[list[str]] = []
    for rollout in args.rollouts:
        run_name = f"tf_S_roll{rollout}_r120"
        cmds.append([
            sys.executable, "-u",
            str(_ROOT / "scripts" / "train.py"),
            "--arch", "transformer",
            "--tf-size", "S",
            "--init-radius", "120",
            "--slot-assignment", "minimax",
            "--run-name", run_name,
            "--total-steps", str(args.total_steps),
            "--seed", str(args.seed),
            "--device", args.device,
            "--env-backend", args.env_backend,
            "--num-envs", str(args.num_envs),
            "--rollout-steps", str(rollout),
            "--minibatch-size", str(args.minibatch_size),
            "--eval-workers", str(args.eval_workers),
        ])

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"=== tf rollout ablation {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for cmd in cmds:
            line = " ".join(cmd)
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
            if args.dry_run:
                continue
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)
            if proc.returncode != 0:
                return int(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
