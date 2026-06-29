"""Evaluate a checkpoint separately for each mixed-ready tug count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from curricula.loader import apply_course, load_course
from env.formation_env import FormationEnv
from rl.ppo import MAPPOActorCritic
from scripts.train import _load_checkpoint, evaluate_policy


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def _build_env_cfg(
    course_path: Path,
    *,
    ready_count: int,
    pos_tol_m: float | None,
    hold_time_s: float | None,
) -> EnvConfig:
    course = load_course(course_path)
    env_cfg = EnvConfig()
    apply_course(env_cfg, course)
    env_cfg.tug_init_mixed_ready_counts = (int(ready_count),)
    if pos_tol_m is not None:
        env_cfg.pos_tol_m = float(pos_tol_m)
    if hold_time_s is not None:
        env_cfg.hold_time_s = float(hold_time_s)
    return env_cfg


def _load_model(checkpoint: Path, env_cfg: EnvConfig) -> tuple[MAPPOActorCritic, dict[str, Any]]:
    probe = FormationEnv(env_cfg, seed=0)
    model_kwargs = {
        "obs_dim": probe.obs_dim,
        "action_dim": probe.action_dim,
        "n_agents": env_cfg.n_tugs,
        "global_state_dim": probe.global_state_dim,
    }
    model = MAPPOActorCritic(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _load_checkpoint(model, optimizer, ckpt, reset_progress=True)
    model.eval()
    return model, model_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MAPPO checkpoint by ready-count bucket."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--course",
        type=Path,
        default=Path("curricula/strict_pos/c04_pos10m.py"),
    )
    parser.add_argument("--ready-counts", type=_parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes-per-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=92_000)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--pos-tol", type=float, default=None)
    parser.add_argument("--hold-time", type=float, default=None)
    parser.add_argument("--target-success", type=float, default=0.90)
    parser.add_argument("--max-collision", type=float, default=0.08)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    course_path = args.course if args.course.is_absolute() else PROJECT_ROOT / args.course
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not course_path.exists():
        raise FileNotFoundError(f"course not found: {course_path}")

    results: dict[str, dict[str, float]] = {}
    total_success = 0.0
    total_collision = 0.0
    total_episodes = 0
    all_pass = True

    for idx, ready_count in enumerate(args.ready_counts):
        env_cfg = _build_env_cfg(
            course_path,
            ready_count=ready_count,
            pos_tol_m=args.pos_tol,
            hold_time_s=args.hold_time,
        )
        model, model_kwargs = _load_model(checkpoint, env_cfg)
        stats = evaluate_policy(
            model,
            env_cfg,
            n_episodes=int(args.episodes_per_count),
            device=torch.device("cpu"),
            seed=int(args.seed) + 10_000 * idx,
            eval_workers=int(args.eval_workers),
            model_kwargs=model_kwargs,
        )
        success_rate = float(stats["eval/success_rate"])
        collision_rate = float(stats["eval/collision_rate"])
        count_key = str(int(ready_count))
        results[count_key] = {key: float(value) for key, value in stats.items()}
        total_success += success_rate * int(args.episodes_per_count)
        total_collision += collision_rate * int(args.episodes_per_count)
        total_episodes += int(args.episodes_per_count)
        count_pass = success_rate >= args.target_success and collision_rate <= args.max_collision
        all_pass = all_pass and count_pass
        print(
            f"[ready={ready_count}] "
            f"success={success_rate:.4f}, collision={collision_rate:.4f}, "
            f"final_dist={stats['eval/final_dist_mean']:.2f}m, pass={count_pass}"
        )

    aggregate = {
        "eval/success_rate": total_success / max(total_episodes, 1),
        "eval/collision_rate": total_collision / max(total_episodes, 1),
        "episodes": float(total_episodes),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "course": str(course_path),
        "target_success": float(args.target_success),
        "max_collision": float(args.max_collision),
        "aggregate": aggregate,
        "by_ready_count": results,
        "pass": bool(all_pass),
    }

    print(
        "[aggregate] "
        f"success={aggregate['eval/success_rate']:.4f}, "
        f"collision={aggregate['eval/collision_rate']:.4f}, pass={all_pass}"
    )
    if args.json_out is not None:
        json_out = args.json_out if args.json_out.is_absolute() else PROJECT_ROOT / args.json_out
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"[artifact] json={json_out}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
