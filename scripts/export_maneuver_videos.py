"""导出 Z 字试验、S 形试验与回转试验动画视频。

用法（项目根目录）:
    python scripts/export_maneuver_videos.py
    python scripts/export_maneuver_videos.py --maneuver turning --fps 30 --video-duration 45
    python scripts/export_maneuver_videos.py --format gif
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon

from utils.mpl_fonts import configure_matplotlib_fonts


def _load_maneuver_module():
    tests_dir = str(PROJECT_ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_maneuvers as mod

    return mod


def _hull_body_polygon(length_m: float, beam_m: float) -> np.ndarray:
    l_half = length_m / 2.0
    b_half = beam_m / 2.0
    return np.array(
        [
            [+l_half, 0.0],
            [+l_half * 0.78, -b_half],
            [-l_half, -b_half],
            [-l_half, +b_half],
            [+l_half * 0.78, +b_half],
        ],
        dtype=np.float64,
    )


def _body_to_world(
    x_n: float,
    y_e: float,
    psi: float,
    body_pts: np.ndarray,
) -> np.ndarray:
    c = math.cos(psi)
    s = math.sin(psi)
    north = x_n + c * body_pts[:, 0] - s * body_pts[:, 1]
    east = y_e + s * body_pts[:, 0] + c * body_pts[:, 1]
    return np.column_stack([east, north])


def _frame_indices(n_samples: int, target_frames: int) -> np.ndarray:
    target_frames = max(2, target_frames)
    if n_samples <= target_frames:
        return np.arange(n_samples, dtype=int)
    return np.linspace(0, n_samples - 1, target_frames, dtype=int)


def _pick_writer(fmt: str, fps: int):
    if fmt == "mp4":
        return FFMpegWriter(fps=fps, bitrate=2400, metadata={"artist": "ship_control"})
    return PillowWriter(fps=fps)


def export_maneuver_video(
    log,
    *,
    title: str,
    out_path: Path,
    length_m: float = 36.0,
    beam_m: float = 11.0,
    fps: int = 30,
    video_duration_s: float = 40.0,
    fmt: str = "mp4",
) -> Path:
    configure_matplotlib_fonts()

    n = len(log.t)
    if n < 2:
        raise ValueError("maneuver log is too short for video export")

    frame_idx = _frame_indices(n, int(round(fps * video_duration_s)))
    body_hull = _hull_body_polygon(length_m, beam_m)

    x_arr = np.asarray(log.x, dtype=np.float64)
    y_arr = np.asarray(log.y, dtype=np.float64)
    psi_arr = np.radians(np.asarray(log.heading_deg, dtype=np.float64))
    t_arr = np.asarray(log.t, dtype=np.float64)

    margin = max(length_m, beam_m) * 2.0
    east_min, east_max = y_arr.min() - margin, y_arr.max() + margin
    north_min, north_max = x_arr.min() - margin, x_arr.max() + margin

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#d6eaf8")
    ax.set_xlim(east_min, east_max)
    ax.set_ylim(north_min, north_max)
    ax.set_xlabel("East  y [m]")
    ax.set_ylabel("North  x [m]")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35, color="#7fb3d5")

    full_trail, = ax.plot([], [], color="#5dade2", lw=1.0, alpha=0.55, label="trajectory")
    active_trail, = ax.plot([], [], color="#1f618d", lw=2.0, label="progress")
    hull_patch = Polygon(
        np.zeros((len(body_hull), 2)),
        closed=True,
        facecolor="#5d6d7e",
        edgecolor="#1b2631",
        lw=2.0,
        zorder=5,
    )
    ax.add_patch(hull_patch)
    heading_line, = ax.plot([], [], color="#c0392b", lw=2.5, zorder=6)
    start_pt = ax.scatter([y_arr[0]], [x_arr[0]], s=70, color="#27ae60", zorder=7, label="start")
    ax.scatter([y_arr[-1]], [x_arr[-1]], s=70, color="#e74c3c", zorder=7, label="end")
    hud = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        zorder=8,
    )
    ax.legend(loc="lower right", fontsize=9)

    def _update(i: int):
        idx = int(frame_idx[i])
        east = y_arr[: idx + 1]
        north = x_arr[: idx + 1]
        full_trail.set_data(y_arr, x_arr)
        active_trail.set_data(east, north)

        psi = psi_arr[idx]
        hull_world = _body_to_world(x_arr[idx], y_arr[idx], psi, body_hull)
        hull_patch.set_xy(hull_world)

        arrow_len = length_m * 0.55
        hx0, hy0 = y_arr[idx], x_arr[idx]
        hx1 = hx0 + math.sin(psi) * arrow_len
        hy1 = hy0 + math.cos(psi) * arrow_len
        heading_line.set_data([hx0, hx1], [hy0, hy1])

        az_cmd = log.az_cmd_deg[idx]
        speed = log.speed_ms[idx]
        yaw_rate = log.yaw_rate_degs[idx]
        hud.set_text(
            f"t = {t_arr[idx]:.1f} s\n"
            f"heading = {log.heading_deg[idx]:+.1f}°\n"
            f"yaw rate = {yaw_rate:+.2f}°/s\n"
            f"speed = {speed:.2f} m/s\n"
            f"azimuth cmd = {az_cmd:+.1f}°"
        )
        artists = (full_trail, active_trail, hull_patch, heading_line, hud)
        return artists

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frame_idx),
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _pick_writer(fmt, fps)
    try:
        anim.save(str(out_path), writer=writer, dpi=120)
    except Exception as exc:
        if fmt == "mp4":
            fallback = out_path.with_suffix(".gif")
            print(f"mp4 export failed ({exc}); falling back to gif: {fallback}")
            anim.save(str(fallback), writer=PillowWriter(fps=fps), dpi=120)
            out_path = fallback
        else:
            raise
    finally:
        plt.close(fig)

    print(f"saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export maneuver test videos.")
    parser.add_argument(
        "--maneuver",
        choices=("all", "both", "zigzag", "sine", "turning"),
        default="all",
        help="which maneuver to export (all/both = all three)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "maneuvers",
        help="directory for exported videos",
    )
    parser.add_argument("--fps", type=int, default=30, help="video frame rate")
    parser.add_argument(
        "--video-duration",
        type=float,
        default=80.0,
        help="output video length in seconds (larger = slower playback)",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="override video-duration as sim_seconds * time_scale (e.g. 0.3)",
    )
    parser.add_argument(
        "--format",
        choices=("mp4", "gif"),
        default="mp4",
        help="video container (mp4 requires ffmpeg)",
    )
    args = parser.parse_args()

    mod = _load_maneuver_module()
    export_all = args.maneuver in ("all", "both")

    def _video_duration(log) -> float:
        if args.time_scale is not None:
            return max(2.0, float(log.t[-1]) * args.time_scale)
        return args.video_duration

    if export_all or args.maneuver == "zigzag":
        print("running zigzag test ...")
        zigzag_log = mod.run_zigzag()
        export_maneuver_video(
            zigzag_log,
            title="Z字试验 (Zigzag) — 15° 方位角, ±10° 航向切换, 90 RPM",
            out_path=args.output_dir / f"zigzag_test.{args.format}",
            fps=args.fps,
            video_duration_s=_video_duration(zigzag_log),
            fmt=args.format,
        )

    if export_all or args.maneuver == "sine":
        print("running sinusoidal (S-shape) test ...")
        sine_log = mod.run_sinusoidal()
        export_maneuver_video(
            sine_log,
            title="S形试验 (Sinusoidal) — ±25° 方位角, T=90 s, 90 RPM",
            out_path=args.output_dir / f"s_shape_test.{args.format}",
            fps=args.fps,
            video_duration_s=_video_duration(sine_log),
            fmt=args.format,
        )

    if export_all or args.maneuver == "turning":
        print("running turning circle test ...")
        turning_log = mod.run_turning_circle()
        export_maneuver_video(
            turning_log,
            title="回转试验 (Turning circle) — 20° 方位角右转, 90 RPM",
            out_path=args.output_dir / f"turning_circle_test.{args.format}",
            fps=args.fps,
            video_duration_s=_video_duration(turning_log),
            fmt=args.format,
        )


if __name__ == "__main__":
    main()
