"""可视化 FormationEnv 在 reset 后的初始场景与 waypoint 路线。

用法（项目根目录）::

    python env/visualize_init_scene.py
    python env/visualize_init_scene.py --num-samples 6 --init-mode mixed_slot_approach
    python env/visualize_init_scene.py --output outputs/init_scenes.png

每张图包含世界系（左）与大船船体系（右），便于检查初始拖轮分布与 route 几何。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from env.formation_env import FormationEnv
from utils.mpl_fonts import configure_matplotlib_fonts

SLOT_NAMES = ("S0 船首左", "S1 船首右", "S2 船尾左", "S3 船尾右")
TUG_COLORS = ("#e74c3c", "#3498db", "#2ecc71", "#f1c40f")
ROUTE_COLORS = ("#c0392b", "#2980b9", "#27ae60", "#d68910")


def _heading_arrow(ax, x: float, y: float, psi: float, length_m: float, **kwargs) -> None:
    dx = length_m * math.cos(psi)
    dy = length_m * math.sin(psi)
    ax.annotate(
        "",
        xy=(y + dy, x + dx),
        xytext=(y, x),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, **kwargs),
    )


def _plot_frame_xy(ax, *, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, lw=0.6)


def _collect_snapshot(env: FormationEnv) -> dict:
    ship = env.ship
    n = env.n_tugs
    slots_world = ship.slot_positions_world()
    tugs = []
    routes_world = []
    routes_body = []
    for i in range(n):
        tug = env.tugs[i]
        slot_idx = int(env.tug_to_slot[i])
        wp_w = env._route_waypoints_world_for_tug(i)
        wp_b = env._route_waypoints_body_for_tug(i)
        stage = int(env.route_stage[i])
        routes_world.append(wp_w)
        routes_body.append(wp_b)
        tugs.append(
            {
                "idx": i,
                "slot_idx": slot_idx,
                "x": float(tug.eta.x),
                "y": float(tug.eta.y),
                "psi": float(tug.eta.z),
                "u": float(tug.nu.x),
                "v": float(tug.nu.y),
                "route_stage": stage,
                "route_len": len(wp_b),
                "ready_at_init": stage >= max(len(wp_b) - 1, 0),
            }
        )
    return {
        "ship": {
            "x": float(ship.x),
            "y": float(ship.y),
            "psi": float(ship.psi),
            "u": float(ship.u),
            "length_m": float(ship.length_m),
            "beam_m": float(ship.beam_m),
            "hull": ship.hull_polygon_world(),
        },
        "slots_world": slots_world,
        "slots_body": ship.slot_positions_body(),
        "tugs": tugs,
        "routes_world": routes_world,
        "routes_body": routes_body,
        "init_mode": env._init_mode(),
    }


def _world_to_body_xy(snap: dict, x_w: float, y_w: float) -> tuple[float, float]:
    ship = snap["ship"]
    dx = x_w - ship["x"]
    dy = y_w - ship["y"]
    psi = ship["psi"]
    c = math.cos(psi)
    s = math.sin(psi)
    return c * dx + s * dy, -s * dx + c * dy


def _draw_scene_on_axes(ax, snap: dict, *, frame: str, panel_title: str) -> None:
    """frame: 'world' 或 'body'（body 系下大船在原点、psi=0 的展示）。"""
    ship = snap["ship"]
    if frame == "world":
        to_xy = lambda x, y: (y, x)
        hull = snap["ship"]["hull"]
        ship_xy = (ship["y"], ship["x"])
        ship_psi = ship["psi"]
    else:
        def to_xy(x_b: float, y_b: float) -> tuple[float, float]:
            return (y_b, x_b)

        l_half = ship["length_m"] / 2.0
        b_half = ship["beam_m"] / 2.0
        hull = np.array(
            [
                [+l_half, 0.0],
                [+l_half * 0.78, -b_half],
                [-l_half, -b_half],
                [-l_half, +b_half],
                [+l_half * 0.78, +b_half],
            ],
            dtype=np.float64,
        )
        ship_xy = (0.0, 0.0)
        ship_psi = 0.0

    hull_xy = np.array([to_xy(p[0], p[1]) for p in hull])
    ax.add_patch(
        Polygon(
            hull_xy,
            closed=True,
            facecolor="#5d6d7e",
            edgecolor="#2c3e50",
            alpha=0.55,
            lw=1.5,
            zorder=1,
        )
    )
    _heading_arrow(ax, ship_xy[1], ship_xy[0], ship_psi, 35.0, color="#2c3e50")

    if frame == "world":
        slots = snap["slots_world"]
        for k in range(len(slots)):
            sx, sy = to_xy(float(slots[k, 0]), float(slots[k, 1]))
            ax.plot(sx, sy, marker="*", ms=10, color="#1e8449", zorder=3)
            ax.text(sx + 4, sy + 4, SLOT_NAMES[k], fontsize=7, color="#145a32")
    else:
        slots = snap["slots_body"]
        for k in range(len(slots)):
            sx, sy = to_xy(float(slots[k, 0]), float(slots[k, 1]))
            ax.plot(sx, sy, marker="*", ms=10, color="#1e8449", zorder=3)
            ax.text(sx + 3, sy + 3, SLOT_NAMES[k], fontsize=7, color="#145a32")

    for tug in snap["tugs"]:
        i = tug["idx"]
        color = TUG_COLORS[i % len(TUG_COLORS)]
        route_color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        if frame == "world":
            tx, ty = to_xy(tug["x"], tug["y"])
            psi = tug["psi"]
            route = snap["routes_world"][i]
        else:
            x_b, y_b = _world_to_body_xy(snap, tug["x"], tug["y"])
            tx, ty = to_xy(x_b, y_b)
            psi = tug["psi"] - ship["psi"]
            route = snap["routes_body"][i]

        label = f"T{i}→{tug['slot_idx']}"
        if snap["init_mode"] == "mixed_slot_approach":
            label += " 就位" if tug["ready_at_init"] else " 赶路"
        ax.plot(tx, ty, "o", color=color, ms=8, zorder=4)
        ax.text(tx + 5, ty + 5, label, fontsize=7, color=color)
        _heading_arrow(ax, ty, tx, psi, 22.0, color=color)

        if len(route) >= 2:
            rx = [to_xy(p[0], p[1])[0] for p in route]
            ry = [to_xy(p[0], p[1])[1] for p in route]
            ax.plot(rx, ry, "-", color=route_color, lw=1.4, alpha=0.85, zorder=2)
            ax.plot(rx, ry, "o", color=route_color, ms=3.5, alpha=0.9, zorder=2)
            for k, (px, py) in enumerate(zip(rx, ry)):
                ax.text(px + 2, py + 2, str(k), fontsize=6, color=route_color)

            stage = int(np.clip(tug["route_stage"], 0, len(route) - 1))
            ax.plot(rx[stage], ry[stage], "D", color=route_color, ms=7, zorder=5)

    _plot_frame_xy(ax, title=panel_title)


def _draw_sample(fig, grid_pos: int, nrows: int, ncols: int, snap: dict, seed: int) -> None:
    ax_w = fig.add_subplot(nrows, ncols * 2, grid_pos * 2 - 1)
    ax_b = fig.add_subplot(nrows, ncols * 2, grid_pos * 2)

    ship = snap["ship"]
    ready_n = sum(1 for t in snap["tugs"] if t["ready_at_init"])
    meta = (
        f"seed={seed} | {snap['init_mode']} | "
        f"L={ship['length_m']:.0f}m B={ship['beam_m']:.0f}m | "
        f"ψ={math.degrees(ship['psi']):.0f}° u={ship['u']:.2f}m/s"
    )
    if snap["init_mode"] == "mixed_slot_approach":
        meta += f" | 就位 {ready_n}/{len(snap['tugs'])}"

    _draw_scene_on_axes(ax_w, snap, frame="world", panel_title=f"{meta}\n世界系")
    _draw_scene_on_axes(ax_b, snap, frame="body", panel_title="船体系（route 生成坐标）")


def render_init_scenes(
    *,
    env_cfg: EnvConfig,
    num_samples: int,
    base_seed: int,
    output: Path,
    dpi: int,
) -> Path:
    configure_matplotlib_fonts()

    ncols = min(2, num_samples)
    nrows = int(math.ceil(num_samples / ncols))
    fig_w = 7.0 * ncols
    fig_h = 6.2 * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        "FormationEnv reset 初始场景（左：世界系，右：船体系；菱形=当前 route stage）",
        fontsize=11,
        y=0.995,
    )

    env = FormationEnv(cfg=env_cfg)
    for k in range(num_samples):
        seed = base_seed + k
        env.reset(seed=seed)
        snap = _collect_snapshot(env)
        _draw_sample(fig, k + 1, nrows, ncols, snap, seed)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 reset 初始场景与 waypoint")
    parser.add_argument("--num-samples", type=int, default=4, help="随机场景数量（网格子图数）")
    parser.add_argument("--seed", type=int, default=0, help="第一个场景的随机种子")
    parser.add_argument("--init-mode", type=str, default=None,
                        choices=["mixed_slot_approach"],
                        help="覆盖 EnvConfig.tug_init_mode")
    parser.add_argument("--no-ship-size-randomize", action="store_true",
                        help="关闭大船尺度随机化")
    parser.add_argument("--output", type=str, default="outputs/init_scenes.png",
                        help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=150, help="输出分辨率")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig()
    if args.init_mode is not None:
        env_cfg = replace(env_cfg, tug_init_mode=args.init_mode)
    if args.no_ship_size_randomize:
        env_cfg = replace(env_cfg, ship_size_randomize=False)

    out = PROJECT_ROOT / args.output
    saved = render_init_scenes(
        env_cfg=env_cfg,
        num_samples=max(1, args.num_samples),
        base_seed=args.seed,
        output=out,
        dpi=args.dpi,
    )
    print(f"[ok] saved: {saved}")
    print(
        f"     init_mode={env_cfg.tug_init_mode}, "
        f"ship_size_randomize={env_cfg.ship_size_randomize}"
    )


if __name__ == "__main__":
    main()
