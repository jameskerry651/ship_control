#!/usr/bin/env python3
"""串行跑奖励超参 preset 消融（调用 scripts/train.py）。

默认协议见 docs/reward_presets.md：
  transformer + init_radius=100 + 1M steps + seed=42
  env-backend=sync（避免 subproc 在 macOS/MPS 上被杀）

用法：
  python scripts/run_reward_preset_ablation.py
  python scripts/run_reward_preset_ablation.py --presets rw_baseline rw_combo
  python scripts/run_reward_preset_ablation.py --dry-run
"""

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

from config import REWARD_PRESETS, list_reward_presets  # noqa: E402


# 与 docs/reward_presets.md 协议顺序一致（非字母序）
DEFAULT_PRESETS = [
    "rw_baseline",
    "rw_dist_up",
    "rw_ship_safe_dn",
    "rw_coll_soft",
    "rw_shape_up",
    "rw_combo",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="串行跑奖励超参 preset 消融（见 docs/reward_presets.md）",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=DEFAULT_PRESETS,
        help=f"要跑的 preset id（默认全部: {', '.join(DEFAULT_PRESETS)}）",
    )
    parser.add_argument("--arch", type=str, default="transformer")
    parser.add_argument("--init-radius", type=float, default=100.0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--env-backend",
        type=str,
        default="sync",
        choices=["sync", "subproc"],
        help="默认 sync；subproc 在本机易被系统杀掉",
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--python", type=str, default=sys.executable, help="训练用 Python 解释器")
    parser.add_argument(
        "--log-file",
        type=str,
        default="outputs/logs/reward_preset_ablation.log",
        help="汇总日志路径（相对项目根）",
    )
    parser.add_argument(
        "--skip-summarize",
        action="store_true",
        help="结束后不跑 summarize_reward_presets.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不训练",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="额外传给 train.py 的参数（写在 -- 之后）",
    )
    return parser.parse_args()


def _build_train_cmd(args: argparse.Namespace, preset: str) -> list[str]:
    cmd = [
        args.python,
        "-u",
        str(_ROOT / "scripts" / "train.py"),
        "--arch",
        args.arch,
        "--init-radius",
        str(args.init_radius),
        "--reward-preset",
        preset,
        "--run-name",
        preset,
        "--total-steps",
        str(args.total_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--env-backend",
        args.env_backend,
        "--num-envs",
        str(args.num_envs),
    ]
    extra = list(args.train_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd.extend(extra)
    return cmd


def main() -> int:
    args = _parse_args()
    unknown = [p for p in args.presets if p not in REWARD_PRESETS]
    if unknown:
        known = ", ".join(list_reward_presets())
        print(f"[error] unknown preset(s): {unknown}. Known: {known}", file=sys.stderr)
        return 2

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = _ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== reward preset ablation {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"presets={args.presets}\n")
        f.write(
            f"arch={args.arch} init_radius={args.init_radius} "
            f"steps={args.total_steps} seed={args.seed} device={args.device} "
            f"env_backend={args.env_backend} num_envs={args.num_envs}\n"
        )

    for preset in args.presets:
        cmd = _build_train_cmd(args, preset)
        log(f"\n=== BEGIN {preset} {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        log("cmd: " + " ".join(cmd))
        if args.dry_run:
            log(f"=== SKIP {preset} (dry-run) ===")
            continue
        proc = subprocess.run(cmd, cwd=str(_ROOT))
        log(f"=== END {preset} exit={proc.returncode} {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        if proc.returncode != 0:
            log(f"=== ABORT after {preset} ===")
            return int(proc.returncode)

    if not args.dry_run and not args.skip_summarize:
        summarize = [
            args.python,
            "-u",
            str(_ROOT / "scripts" / "summarize_reward_presets.py"),
            "--logdir",
            "runs",
            "--runs",
            *args.presets,
        ]
        log("\n=== summarize ===")
        log("cmd: " + " ".join(summarize))
        subprocess.run(summarize, cwd=str(_ROOT))

    log(f"\n=== ALL DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
