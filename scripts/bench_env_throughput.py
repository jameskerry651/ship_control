#!/usr/bin/env python3
"""Sweep num_envs for env throughput (CudaVecEnv / SyncVecEnv / SubprocVecEnv).

Example:
  python scripts/bench_env_throughput.py --backend cuda --device cuda \\
      --num-envs 32,64,128,256 --rollout-steps 128
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from env.gpu import CudaVecEnv


def _load_train_vec_envs():
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("ship_control_train", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SyncVecEnv, mod.SubprocVecEnv


def _make_env(backend: str, n_envs: int, device: str, seed: int):
    cfg = EnvConfig()
    if backend == "cuda":
        return CudaVecEnv(cfg, n_envs=n_envs, base_seed=seed, device=device, dtype=torch.float32)
    SyncVecEnv, SubprocVecEnv = _load_train_vec_envs()
    if backend == "subproc":
        return SubprocVecEnv(cfg, n_envs=n_envs, base_seed=seed, n_workers=n_envs, start_method="fork")
    return SyncVecEnv(cfg, n_envs=n_envs, base_seed=seed)


def _gpu_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def bench_one(backend: str, n_envs: int, device: str, rollout_steps: int, seed: int) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    env = _make_env(backend, n_envs, device, seed)
    obs = env.reset()
    k = env.n_tugs
    rng = np.random.default_rng(seed)
    # warmup
    for _ in range(2):
        actions = rng.uniform(-1, 1, size=(n_envs, k, 4)).astype(np.float32)
        env.step(actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rollout_steps):
        actions = rng.uniform(-1, 1, size=(n_envs, k, 4)).astype(np.float32)
        env.step(actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    samples = rollout_steps * n_envs * k
    env.close()
    return {
        "backend": backend,
        "num_envs": n_envs,
        "rollout_steps": rollout_steps,
        "samples": samples,
        "dt_s": dt,
        "sps": samples / max(dt, 1e-9),
        "peak_mem_mb": _gpu_mem_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda", "sync", "subproc"], default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=str, default="32,64,128,256")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "cuda" and args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    grid = [int(x) for x in args.num_envs.split(",") if x.strip()]
    print(f"backend={args.backend} device={args.device} rollout_steps={args.rollout_steps}")
    print(f"{'num_envs':>10} {'sps':>12} {'dt_s':>10} {'peak_mb':>10}")
    best = None
    for n in grid:
        try:
            row = bench_one(args.backend, n, args.device, args.rollout_steps, args.seed)
        except RuntimeError as exc:
            print(f"{n:>10} FAILED: {exc}")
            break
        mem = row["peak_mem_mb"]
        mem_s = f"{mem:.1f}" if mem is not None else "-"
        print(f"{n:>10} {row['sps']:>12.1f} {row['dt_s']:>10.3f} {mem_s:>10}")
        if best is None or row["sps"] > best["sps"]:
            best = row
    if best is not None:
        print(
            f"\nbest sps={best['sps']:.1f} at num_envs={best['num_envs']} "
            f"(recommend starting train with this; re-check with short MAPPO update)"
        )


if __name__ == "__main__":
    main()
