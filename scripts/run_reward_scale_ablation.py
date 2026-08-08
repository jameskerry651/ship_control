#!/usr/bin/env python3
"""串行跑奖励尺度碰撞消融（见 design spec）。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.reward_scale_ablation import (  # noqa: E402
    RSC_PRESET_ORDER,
    rsc_run_name,
    select_rsc_promotions,
)


def _load_summarize_mod():
    spec = importlib.util.spec_from_file_location(
        "summarize_reward_scale",
        _ROOT / "scripts" / "summarize_reward_scale.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_cmd(
    *,
    preset_id: str,
    run_name: str,
    total_steps: int,
    seed: int,
    device: str,
    env_backend: str,
    eval_backend: str,
    num_envs: int,
    rollout_steps: int,
    minibatch_size: int,
    eval_workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(_ROOT / "scripts" / "train.py"),
        "--arch",
        "transformer",
        "--tf-size",
        "S",
        "--init-radius",
        "120",
        "--slot-assignment",
        "minimax",
        "--reward-preset",
        preset_id,
        "--run-name",
        run_name,
        "--total-steps",
        str(total_steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--env-backend",
        env_backend,
        "--eval-backend",
        eval_backend,
        "--num-envs",
        str(num_envs),
        "--rollout-steps",
        str(rollout_steps),
        "--minibatch-size",
        str(minibatch_size),
        "--eval-workers",
        str(eval_workers),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="串行跑 reward scale collision 消融")
    p.add_argument(
        "--presets",
        nargs="+",
        default=list(RSC_PRESET_ORDER),
        help="rsc_* preset ids（默认设计表全序）",
    )
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument(
        "--phase",
        choices=("1m", "5m"),
        default="1m",
        help="决定 run-name 前缀；与 --total-steps 独立，调用方负责一致",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--env-backend", type=str, default="cuda")
    p.add_argument("--eval-backend", type=str, default="cuda")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument(
        "--promote",
        action="store_true",
        help="读取 1M runs，按规则最多选 2 个 short name 跑 5M（忽略 --presets）",
    )
    p.add_argument("--max-promote", type=int, default=2)
    p.add_argument("--promote-steps", type=int, default=5_000_000)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--log-file",
        type=str,
        default="outputs/logs/reward_scale_ablation.log",
    )
    args = p.parse_args()

    jobs: list[tuple[str, str, int]] = []
    # (preset_id, run_name, steps)

    if args.promote:
        mod = _load_summarize_mod()
        metrics = mod.load_rsc_final_metrics(Path(args.logdir), phase="1m")
        try:
            shorts = select_rsc_promotions(metrics, max_promote=args.max_promote)
        except ValueError as exc:
            print(f"promote failed: {exc}", flush=True)
            return 1
        if not shorts:
            print("No presets promoted; stopping.", flush=True)
            return 0
        print(f"Promoting: {shorts}", flush=True)
        for short in shorts:
            preset_id = f"rsc_{short}"
            jobs.append(
                (
                    preset_id,
                    rsc_run_name(preset_id, phase="5m"),
                    args.promote_steps,
                )
            )
    else:
        for preset_id in args.presets:
            if not str(preset_id).startswith("rsc_"):
                raise SystemExit(f"expected rsc_* preset, got {preset_id!r}")
            jobs.append(
                (
                    preset_id,
                    rsc_run_name(preset_id, phase=args.phase),
                    args.total_steps,
                )
            )

    log_path = _ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"=== reward scale ablation {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        for preset_id, run_name, steps in jobs:
            cmd = _train_cmd(
                preset_id=preset_id,
                run_name=run_name,
                total_steps=steps,
                seed=args.seed,
                device=args.device,
                env_backend=args.env_backend,
                eval_backend=args.eval_backend,
                num_envs=args.num_envs,
                rollout_steps=args.rollout_steps,
                minibatch_size=args.minibatch_size,
                eval_workers=args.eval_workers,
            )
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
