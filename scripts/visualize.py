"""pygame visualization for tugboat formation env."""
from __future__ import annotations
import argparse, math, sys, time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
from config import EnvConfig, PPOConfig, VizConfig
from env.formation_env import FormationEnv
from utils.mpl_fonts import configure_matplotlib_fonts
from rl.ppo import MAPPOActorCritic

C_SEA_TOP    = (8,   14,  30)
C_SEA_BOT    = (18,  28,  48)
C_GRID       = (25,  45,  65)
C_GRID_SUB   = (18,  32,  50)
C_GRID_LABEL = (80,  110, 140)
C_SHIP_HULL  = (80, 100, 140)
C_SHIP_EDGE  = (120, 150, 200)
C_SLOT_RING  = (60, 200, 120)
C_SLOT_FILL  = (30, 100,  60)
C_SLOT_DONE  = (255, 220,  50)
C_TUG_COLORS = [
    (255, 100,  80),
    (80,  180, 255),
    (100, 230, 100),
    (255, 200,  60),
]
C_THRUST     = (255, 255, 255)
C_TEXT       = (220, 220, 220)
C_TEXT_DIM   = (120, 120, 120)
C_WARN       = (255,  80,  80)
C_SUCCESS    = (80,  255, 120)
C_CPA_WARN   = (255, 120,  80)
C_COMPASS    = (100, 130, 160)
C_PANEL_BG   = (20,  26,  45)
C_PANEL_EDGE = (50,  65,  100)
C_AXIS       = (60,  75,  110)
C_ZERO_LINE  = (50,  65,  100)
C_REF_LINE   = (180, 140, 60)

C_CURVES = [
    (255, 220,  60),
    (100, 200, 255),
    (255, 130, 200),
    (100, 255, 160),
    (255, 160,  80),
    (200, 100,  80),
    (140, 200, 255),
    (80,  255, 120),
    (200, 160, 255),
]

CHART_W = 310
CHART_HISTORY = 300


class Camera:
    def __init__(self, w: int, h: int, mpp: float) -> None:
        self.w = w
        self.h = h
        self.mpp = mpp
        self.cx = w // 2
        self.cy = h // 2

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int(self.cx + y / self.mpp)
        sy = int(self.cy - x / self.mpp)
        return sx, sy

    def scale(self, meters: float) -> int:
        return max(1, int(meters / self.mpp))

    def center_on(self, x: float, y: float) -> None:
        self.cx = self.w // 2
        self.cy = self.h // 2
        self._ox = x
        self._oy = y

    def world_to_screen_with_offset(self, x: float, y: float) -> tuple[int, int]:
        ox = getattr(self, "_ox", 0.0)
        oy = getattr(self, "_oy", 0.0)
        sx = int(self.cx + (y - oy) / self.mpp)
        sy = int(self.cy - (x - ox) / self.mpp)
        return sx, sy


def _rot_poly(verts_body: list[tuple[float, float]], psi: float,
              tx: float, ty: float, cam: Camera) -> list[tuple[int, int]]:
    cos_p = math.cos(psi)
    sin_p = math.sin(psi)
    result = []
    for bx, by in verts_body:
        wx = tx + cos_p * bx - sin_p * by
        wy = ty + sin_p * bx + cos_p * by
        result.append(cam.world_to_screen_with_offset(wx, wy))
    return result


def _load_pygame_ui_font(size: int, bold: bool = False) -> pygame.font.Font:
    names = (
        "pingfangsc,pingfangtc,hiragino sans gb,stheitisc,stheitilight,"
        "songtisc,songti sc,heiti sc,simhei,microsoft yahei,"
        "notosanscjksc,notosanscjktc,wenquanyizenhei,arial unicode ms"
    )
    path = pygame.font.match_font(names, bold=bold)
    if path is None:
        path = pygame.font.match_font("arial")
    if path is None:
        return pygame.font.SysFont(None, size, bold=bold)
    f = pygame.font.Font(path, size)
    if bold:
        f.set_bold(True)
    return f


def _draw_sea_background(surf: pygame.Surface) -> None:
    h = surf.get_height()
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(C_SEA_TOP[0] + (C_SEA_BOT[0] - C_SEA_TOP[0]) * t)
        g = int(C_SEA_TOP[1] + (C_SEA_BOT[1] - C_SEA_TOP[1]) * t)
        b = int(C_SEA_TOP[2] + (C_SEA_BOT[2] - C_SEA_TOP[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (surf.get_width(), y))


def _draw_compass(surf: pygame.Surface, cx: int, cy: int, radius: int,
                  ship_psi: float, font_small: pygame.font.Font) -> None:
    pygame.draw.circle(surf, C_COMPASS, (cx, cy), radius, 1)
    pygame.draw.circle(surf, C_COMPASS, (cx, cy), radius - 8, 1)
    nav = (-math.sin(ship_psi), -math.cos(ship_psi))
    eas = (math.cos(ship_psi), -math.sin(ship_psi))
    for direction, vx, vy in [("N", *nav), ("S", -nav[0], -nav[1]),
                               ("E", *eas), ("W", -eas[0], -eas[1])]:
        ex = int(cx + vx * radius * 0.75)
        ey = int(cy + vy * radius * 0.75)
        ex2 = int(cx + vx * (radius - 1))
        ey2 = int(cy + vy * (radius - 1))
        pygame.draw.line(surf, C_COMPASS, (ex, ey), (ex2, ey2), 1)
        lx = int(cx + vx * (radius - 16))
        ly = int(cy + vy * (radius - 16))
        lbl = font_small.render(direction, True, C_COMPASS)
        surf.blit(lbl, (lx - lbl.get_width() // 2, ly - lbl.get_height() // 2))
    pygame.draw.circle(surf, C_COMPASS, (cx, cy), 2, 0)


def _draw_dashed_line(surf: pygame.Surface, color: tuple, p1: tuple, p2: tuple,
                      dash_len: int = 4, gap_len: int = 3) -> None:
    x1, y = p1
    x2, _ = p2
    x = x1
    on = True
    while x < x2:
        end = min(x + dash_len, x2) if on else x + gap_len
        if on and end > x:
            pygame.draw.line(surf, color, (x, y), (end, y), 1)
        x = end
        on = not on


def _draw_grid(surf: pygame.Surface, cam: Camera, spacing_m: float = 100.0) -> None:
    ox = getattr(cam, "_ox", 0.0)
    oy = getattr(cam, "_oy", 0.0)
    half_w_m = cam.w / 2 * cam.mpp
    half_h_m = cam.h / 2 * cam.mpp
    x_min = ox - half_h_m
    x_max = ox + half_h_m
    y_min = oy - half_w_m
    y_max = oy + half_w_m
    import math as _m
    x0 = _m.floor(x_min / spacing_m) * spacing_m
    y0 = _m.floor(y_min / spacing_m) * spacing_m
    x = x0
    while x <= x_max:
        p1 = cam.world_to_screen_with_offset(x, y_min)
        p2 = cam.world_to_screen_with_offset(x, y_max)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)
        x += spacing_m
    y = y0
    while y <= y_max:
        p1 = cam.world_to_screen_with_offset(x_min, y)
        p2 = cam.world_to_screen_with_offset(x_max, y)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)
        y += spacing_m


def _draw_ship(surf: pygame.Surface, ship_data: dict, cam: Camera) -> None:
    hull_pts = [cam.world_to_screen_with_offset(float(ship_data["hull"][k][0]),
                                                 float(ship_data["hull"][k][1]))
                for k in range(len(ship_data["hull"]))]
    if len(hull_pts) >= 3:
        pygame.draw.polygon(surf, C_SHIP_HULL, hull_pts)
        pygame.draw.polygon(surf, C_SHIP_EDGE, hull_pts, 2)
    cx, cy = cam.world_to_screen_with_offset(ship_data["x"], ship_data["y"])
    psi = ship_data["psi"]
    arrow_len = cam.scale(30.0)
    ex = int(cx + arrow_len * math.sin(psi))
    ey = int(cy - arrow_len * math.cos(psi))
    pygame.draw.line(surf, C_SHIP_EDGE, (cx, cy), (ex, ey), 2)


def _draw_slot(surf: pygame.Surface, slot: np.ndarray, idx: int,
               in_zone: bool, cam: Camera) -> None:
    sx, sy = cam.world_to_screen_with_offset(float(slot[0]), float(slot[1]))
    r = cam.scale(5.0)
    color = C_SLOT_DONE if in_zone else C_SLOT_RING
    pygame.draw.circle(surf, C_SLOT_FILL if in_zone else C_SEA_TOP, (sx, sy), r)
    pygame.draw.circle(surf, color, (sx, sy), r, 2)
    psi = float(slot[2])
    line_len = cam.scale(12.0)
    ex = int(sx + line_len * math.sin(psi))
    ey = int(sy - line_len * math.cos(psi))
    pygame.draw.line(surf, color, (sx, sy), (ex, ey), 2)
    font_small = pygame.font.SysFont("monospace", 11)
    lbl = font_small.render(f"S{idx}", True, color)
    surf.blit(lbl, (sx + r + 2, sy - 6))


def _draw_tug(surf: pygame.Surface, tug_data: dict, tug_idx: int,
              trail: deque, cam: Camera, show_thrust: bool) -> None:
    color = C_TUG_COLORS[tug_idx % len(C_TUG_COLORS)]
    x, y, psi = tug_data["x"], tug_data["y"], tug_data["psi"]
    L = tug_data["length"] / 2.0
    B = tug_data["beam"] / 2.0
    if len(trail) >= 2:
        pts = [cam.world_to_screen_with_offset(p[0], p[1]) for p in trail]
        alpha_surf = pygame.Surface((cam.w, cam.h), pygame.SRCALPHA)
        for k in range(len(pts) - 1):
            alpha = int(80 * (k + 1) / len(pts))
            pygame.draw.line(alpha_surf, (*color, alpha), pts[k], pts[k + 1], 1)
        surf.blit(alpha_surf, (0, 0))
    verts_body = [(+L, 0.0), (+L * 0.7, -B), (-L, -B), (-L, +B), (+L * 0.7, +B)]
    pts = _rot_poly(verts_body, psi, x, y, cam)
    pygame.draw.polygon(surf, (*color, 180), pts)
    pygame.draw.polygon(surf, color, pts, 2)
    if show_thrust:
        thr = tug_data["thruster"]
        scale_n = cam.scale(1.0) / 50000.0
        for side in ("port", "starboard"):
            fp = thr[f"{side}_force_body_n"]
            pp = thr[f"{side}_position_body_m"]
            cos_p = math.cos(psi)
            sin_p = math.sin(psi)
            fx_w = cos_p * fp.x - sin_p * fp.y
            fy_w = sin_p * fp.x + cos_p * fp.y
            px_w = x + cos_p * pp.x - sin_p * pp.y
            py_w = y + sin_p * pp.x + cos_p * pp.y
            sx0, sy0 = cam.world_to_screen_with_offset(px_w, py_w)
            sx1 = int(sx0 + fy_w * scale_n)
            sy1 = int(sy0 - fx_w * scale_n)
            if abs(sx1 - sx0) + abs(sy1 - sy0) > 2:
                pygame.draw.line(surf, C_THRUST, (sx0, sy0), (sx1, sy1), 2)
                pygame.draw.circle(surf, C_THRUST, (sx1, sy1), 2)
    sx, sy = cam.world_to_screen_with_offset(x, y)
    font_small = pygame.font.SysFont("monospace", 12, bold=True)
    lbl = font_small.render(f"T{tug_idx}", True, color)
    surf.blit(lbl, (sx + cam.scale(L) + 2, sy - 6))


def _draw_cpa_warnings(surf: pygame.Surface, snap: dict, cam: "Camera") -> None:
    cpa_pairs = snap.get("cpa_pairs", [])
    cpa_ship = snap.get("cpa_ship", [])
    cpa_warn_m = 40.0
    cpa_ship_warn_m = 30.0
    for pair in cpa_pairs:
        if pair["dcpa"] >= cpa_warn_m or pair["tcpa"] <= 0:
            continue
        i, j = pair["i"], pair["j"]
        ti = snap["tugs"][i]
        tj = snap["tugs"][j]
        cpx, cpy = cam.world_to_screen_with_offset(pair["cpa_x"], pair["cpa_y"])
        six, siy = cam.world_to_screen_with_offset(ti["x"], ti["y"])
        sjx, sjy = cam.world_to_screen_with_offset(tj["x"], tj["y"])
        pygame.draw.line(surf, (255, 80, 60), (six, siy), (cpx, cpy), 2)
        pygame.draw.line(surf, (255, 80, 60), (sjx, sjy), (cpx, cpy), 2)
        r_cpa = max(3, int(cam.scale(2.0)))
        pygame.draw.circle(surf, C_CPA_WARN, (cpx, cpy), r_cpa)
    for cs in cpa_ship:
        if cs["dcpa"] >= cpa_ship_warn_m or cs["tcpa"] <= 0:
            continue
        ti = snap["tugs"][cs["tug_idx"]]
        cpx, cpy = cam.world_to_screen_with_offset(cs["cpa_x"], cs["cpa_y"])
        six, siy = cam.world_to_screen_with_offset(ti["x"], ti["y"])
        pygame.draw.line(surf, (255, 160, 40), (six, siy), (cpx, cpy), 2)
        r_cpa = max(4, int(cam.scale(3.0)))
        pygame.draw.circle(surf, (255, 160, 40), (cpx, cpy), r_cpa)


def _tug_pair_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _min_hull_dist(tx: float, ty: float, hull: list[tuple[float, float]]) -> float:
    if len(hull) < 3:
        return 100.0
    best = float("inf")
    n = len(hull)
    for k in range(n):
        x1, y1 = hull[k]
        x2, y2 = hull[(k + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        seg_sq = dx * dx + dy * dy
        if seg_sq < 1e-12:
            d = math.hypot(tx - x1, ty - y1)
        else:
            t_proj = max(0.0, min(1.0, ((tx - x1) * dx + (ty - y1) * dy) / seg_sq))
            px = x1 + t_proj * dx
            py = y1 + t_proj * dy
            d = math.hypot(tx - px, ty - py)
        if d < best:
            best = d
    return best


def _draw_hud(surf: pygame.Surface, snap: dict, ep_ret: float,
              paused: bool, speed: float, font: pygame.font.Font,
              font_small: pygame.font.Font) -> None:
    lines = [
        f"Step: {snap['step']:5d}   Ep.Ret: {ep_ret:+8.2f}",
        f"Ship  u={snap['ship']['u']:.2f}m/s  r={math.degrees(snap['ship']['r']):.2f}\u00b0/s",
        "",
    ]
    for i, t in enumerate(snap["tugs"]):
        slot_idx = t["slot_idx"]
        in_zone = t["in_zone_steps"] > 0
        zone_str = f"IN_ZONE({t['in_zone_steps']})" if in_zone else "       "
        d_hull = _min_hull_dist(t["x"], t["y"], snap["ship"]["hull"])
        d_tug = min(
            _tug_pair_dist(t["x"], t["y"], snap["tugs"][j]["x"], snap["tugs"][j]["y"])
            for j in range(len(snap["tugs"])) if j != i
        ) if len(snap["tugs"]) > 1 else 100.0
        lines.append(
            f"T{i} S{slot_idx} "
            f"r={math.degrees(t['r']):+.1f}\u00b0/s  h:{d_hull:.0f}m t:{d_tug:.0f}m  {zone_str}"
        )
    if snap.get("_success"):
        lines.append("")
        lines.append(">>> SUCCESS <<<")
    if snap.get("_collision"):
        lines.append("")
        lines.append(">>> COLLISION <<<")
    if paused:
        lines.append("")
        lines.append("[PAUSED]")
    if speed > 1:
        lines.append(f"[Speed: {speed}x]")
    y_off = 8
    for line in lines:
        color = C_SUCCESS if "SUCCESS" in line else C_WARN if "COLLISION" in line else C_TEXT
        lbl = font_small.render(line, True, color)
        surf.blit(lbl, (8, y_off))
        y_off += 14


class TugChartHistory:
    LABELS = ["Heading/\u03c0", "Port Az", "Stbd Az", "Port RPM", "Stbd RPM",
              "DCPA min", "TCPA near", "r_escort", "dist/slot"]
    UNITS  = ["\u00d7\u03c0 rad", "norm", "norm", "norm", "norm",
              "m", "s", "\u00d70.5", "\u00d7tol"]

    def __init__(self, maxlen: int = CHART_HISTORY) -> None:
        self.maxlen = maxlen
        self.series: list[deque] = [deque(maxlen=maxlen) for _ in range(9)]
        self.pos_tol_m: float = 60.0

    def push(self, tug_data: dict, psi_ship: float, rpm_limit: float, az_limit: float,
             dcpa_min: float = 100.0, tcpa_near: float = 0.0) -> None:
        ctrl = tug_data["ctrl"]
        dpsi = (tug_data["psi"] - psi_ship + math.pi) % (2 * math.pi) - math.pi
        self.series[0].append(dpsi / math.pi)
        self.series[1].append(ctrl["port_azimuth_actual_deg"] / az_limit)
        self.series[2].append(ctrl["starboard_azimuth_actual_deg"] / az_limit)
        self.series[3].append(ctrl["port_rpm_actual"] / rpm_limit)
        self.series[4].append(ctrl["starboard_rpm_actual"] / rpm_limit)
        self.series[5].append(min(dcpa_min / 100.0, 1.0))
        self.series[6].append(math.tanh(tcpa_near / 60.0))
        self.series[7].append(min(tug_data.get("r_escort", 0.0) / 0.5, 1.0))
        self.series[8].append(min(tug_data.get("dist_to_slot", 0.0) / max(self.pos_tol_m, 1e-3), 2.0))

    def clear(self) -> None:
        for s in self.series:
            s.clear()


def _draw_chart_panel(
    surf: pygame.Surface,
    history: TugChartHistory,
    tug_idx: int,
    panel_rect: pygame.Rect,
    font_small: pygame.font.Font,
    az_limit: float,
    rpm_limit: float,
) -> None:
    px, py, pw, ph = panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height
    pygame.draw.rect(surf, C_PANEL_BG, panel_rect)
    pygame.draw.rect(surf, C_PANEL_EDGE, panel_rect, 1)
    color = C_TUG_COLORS[tug_idx % len(C_TUG_COLORS)]
    title = font_small.render(f"Tug {tug_idx} \u2014   [0=\u5173\u95ed]", True, color)
    surf.blit(title, (px + 6, py + 4))
    n_charts = 9
    margin_top = 22
    margin_bot = 4
    gap = 2
    chart_h = (ph - margin_top - margin_bot - gap * (n_charts - 1)) // n_charts
    chart_x = px + 6
    chart_w = pw - 12
    y_ranges = [
        (-1.0, 1.0, "rad", lambda v: f"{v*math.pi:.2f}"),
        (-1.0, 1.0, "\u00b0", lambda v: f"{v*az_limit:.0f}"),
        (-1.0, 1.0, "\u00b0", lambda v: f"{v*az_limit:.0f}"),
        (-1.0, 1.0, "rpm", lambda v: f"{v*rpm_limit:.0f}"),
        (-1.0, 1.0, "rpm", lambda v: f"{v*rpm_limit:.0f}"),
        ( 0.0, 1.0, "m", lambda v: f"{v*100.0:.0f}"),
        (-1.0, 1.0, "s", lambda v: f"{v*60.0:.0f}"),
        ( 0.0, 1.0, "\u00d70.5", lambda v: f"{v*0.5:.2f}"),
        ( 0.0, 2.0, "\u00d7tol", lambda v: f"{v*history.pos_tol_m:.0f}m"),
    ]
    tol_idx = 8
    for ci in range(n_charts):
        cy = py + margin_top + ci * (chart_h + gap)
        sub_rect = pygame.Rect(chart_x, cy, chart_w, chart_h)
        pygame.draw.rect(surf, (12, 16, 28), sub_rect)
        pygame.draw.rect(surf, C_AXIS, sub_rect, 1)
        zero_y = cy + chart_h // 2
        pygame.draw.line(surf, C_ZERO_LINE,
                         (chart_x, zero_y), (chart_x + chart_w, zero_y), 1)
        if ci == tol_idx:
            ref_v = 1.0
            ref_y = cy + chart_h - 2 - int((ref_v / 2.0) * (chart_h - 4))
            ref_y = max(cy + 1, min(cy + chart_h - 1, ref_y))
            _draw_dashed_line(surf, C_REF_LINE, (chart_x, ref_y), (chart_x + chart_w, ref_y), 4, 3)
            ref_lbl = font_small.render("tol", True, C_REF_LINE)
            surf.blit(ref_lbl, (chart_x + 2, ref_y - 10))
        lbl = font_small.render(TugChartHistory.LABELS[ci], True, C_TEXT_DIM)
        surf.blit(lbl, (chart_x + 2, cy + 1))
        series = history.series[ci]
        if series:
            cur_val = series[-1]
            _, _, unit, fmt = y_ranges[ci]
            val_str = fmt(cur_val) + unit
            val_surf = font_small.render(val_str, True, C_CURVES[ci % len(C_CURVES)])
            surf.blit(val_surf, (chart_x + chart_w - val_surf.get_width() - 2, cy + 1))
        pts = list(series)
        if len(pts) < 2:
            continue
        curve_pts = []
        for k, v in enumerate(pts):
            sx = chart_x + int(k * chart_w / max(len(pts) - 1, 1))
            sy = cy + chart_h - 2 - int((v + 1.0) / 2.0 * (chart_h - 4))
            sy = max(cy + 1, min(cy + chart_h - 1, sy))
            curve_pts.append((sx, sy))
        if len(curve_pts) >= 2:
            pygame.draw.lines(surf, C_CURVES[ci % len(C_CURVES)], False, curve_pts, 1)


class EpisodeRecorder:
    def __init__(self, n_tugs: int, dt_ctrl: float) -> None:
        self.n_tugs = n_tugs
        self.dt_ctrl = dt_ctrl
        self.history: list[dict] = []

    def push(self, snap: dict) -> None:
        self.history.append(snap)

    def clear(self) -> None:
        self.history.clear()


def _setup_paper_style() -> None:
    configure_matplotlib_fonts()
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.labelsize": 11,
    })


def export_tug_curves(rec: EpisodeRecorder, tug_idx: int, out_path: Path) -> None:
    _setup_paper_style()
    if not rec.history:
        return
    dt = rec.dt_ctrl
    n = len(rec.history)
    t = np.arange(n) * dt
    data = {key: [] for key in ["psi_deg", "psi_rel_deg", "r_deg",
                                  "port_rpm_cmd", "port_rpm_actual",
                                  "stbd_rpm_cmd", "stbd_rpm_actual",
                                  "port_az_cmd", "port_az_actual",
                                  "stbd_az_cmd", "stbd_az_actual"]}
    for snap in rec.history:
        tug = snap["tugs"][tug_idx]
        data["psi_deg"].append(math.degrees(tug["psi"]))
        data["psi_rel_deg"].append(math.degrees(tug["psi"] - snap["ship"]["psi"]))
        data["r_deg"].append(math.degrees(tug["r"]))
        ctrl = tug["ctrl"]
        data["port_rpm_cmd"].append(ctrl["port_rpm_cmd"])
        data["port_rpm_actual"].append(ctrl["port_rpm_actual"])
        data["stbd_rpm_cmd"].append(ctrl["starboard_rpm_cmd"])
        data["stbd_rpm_actual"].append(ctrl["starboard_rpm_actual"])
        data["port_az_cmd"].append(ctrl["port_azimuth_cmd_deg"])
        data["port_az_actual"].append(ctrl["port_azimuth_actual_deg"])
        data["stbd_az_cmd"].append(ctrl["starboard_azimuth_cmd_deg"])
        data["stbd_az_actual"].append(ctrl["starboard_azimuth_actual_deg"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    color = plt.cm.tab10(tug_idx)
    axes[0].plot(t, data["psi_deg"], color=color, lw=1.4, label="\u03c8 (abs)")
    axes[0].plot(t, data["psi_rel_deg"], color="gray", lw=1.0, ls="--", label="\u03c8 - \u03c8_ship")
    axes[0].set_ylabel("Heading [deg]")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, data["r_deg"], color=color, lw=1.4)
    axes[1].set_ylabel("Yaw rate [deg/s]")
    axes[2].plot(t, data["port_rpm_cmd"], color="#1f77b4", lw=0.8, ls="--", label="port cmd")
    axes[2].plot(t, data["port_rpm_actual"], color="#1f77b4", lw=1.4, label="port actual")
    axes[2].plot(t, data["stbd_rpm_cmd"], color="#d62728", lw=0.8, ls="--", label="stbd cmd")
    axes[2].plot(t, data["stbd_rpm_actual"], color="#d62728", lw=1.4, label="stbd actual")
    axes[2].set_ylabel("Propeller RPM")
    axes[2].legend(fontsize=8)
    axes[3].plot(t, data["port_az_cmd"], color="#1f77b4", lw=0.8, ls="--", label="port cmd")
    axes[3].plot(t, data["port_az_actual"], color="#1f77b4", lw=1.4, label="port actual")
    axes[3].plot(t, data["stbd_az_cmd"], color="#d62728", lw=0.8, ls="--", label="stbd cmd")
    axes[3].plot(t, data["stbd_az_actual"], color="#d62728", lw=1.4, label="stbd actual")
    axes[3].set_ylabel("Azimuth angle [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_trajectory(rec: EpisodeRecorder, out_path: Path) -> None:
    _setup_paper_style()
    if not rec.history:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ship_x, ship_y = [], []
    for snap in rec.history:
        s = snap["ship"]
        ship_x.append(s["x"])
        ship_y.append(s["y"])
    ax.plot(ship_y, ship_x, color="#444", lw=1.6, label="Ship path")
    ax.scatter(ship_y[-1], ship_x[-1], color="#444", s=30, marker="s")
    n_tugs = rec.n_tugs
    tug_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]
    for i in range(n_tugs):
        tx = [snap["tugs"][i]["x"] for snap in rec.history]
        ty = [snap["tugs"][i]["y"] for snap in rec.history]
        ax.plot(ty, tx, color=tug_colors[i % len(tug_colors)], lw=1.2, label=f"Tug {i}")
    final = rec.history[-1]
    for i, slot in enumerate(final["slots"]):
        ax.plot(float(slot[1]), float(slot[0]), "o", markersize=4, color=tug_colors[i % len(tug_colors)],
                label=f"Slot {i}" if i == 0 else "")
    ax.set_xlabel("East  [m]")
    ax.set_ylabel("North [m]")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_visualization(
    checkpoint_path: str | None = None,
    random_policy: bool = False,
    speed: float = 1.0,
) -> None:
    pygame.init()
    env_cfg = EnvConfig()
    viz_cfg = VizConfig()
    info = pygame.display.Info()
    win_w, win_h = info.current_w, info.current_h
    win_w = min(win_w, 1920)
    win_h = min(win_h, 1200)
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Tugboat Formation \u2014 MAPPO Visualization")
    clock = pygame.time.Clock()
    font = _load_pygame_ui_font(16, bold=True)
    font_small = _load_pygame_ui_font(13)

    # --- model loading ---
    policy = None
    device = torch.device("cpu")
    checkpoint: dict = {}
    if checkpoint_path and not random_policy:
        checkpoint = torch.load(
            str(_project_path(checkpoint_path)), map_location="cpu", weights_only=False
        )
        if str(checkpoint.get("algo", "")).lower() != "mappo":
            raise ValueError("only MAPPO checkpoints are supported")

        ckpt_env_cfg = checkpoint.get("env_cfg", {})
        for key, value in ckpt_env_cfg.items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
        if ckpt_env_cfg:
            if "tug_init_mode" not in ckpt_env_cfg:
                env_cfg.tug_init_mode = "astern_approach"
            if "obs_include_route" not in ckpt_env_cfg:
                env_cfg.obs_include_route = False
            if "route_planner" not in ckpt_env_cfg:
                env_cfg.route_planner = "manual"
            if "ship_size_randomize" not in ckpt_env_cfg:
                env_cfg.ship_size_randomize = False
            if "obs_include_ship_size" not in ckpt_env_cfg:
                env_cfg.obs_include_ship_size = False
            if "obs_include_ego_accel" not in ckpt_env_cfg:
                env_cfg.obs_include_ego_accel = False

        ppo_cfg = PPOConfig()
        for key, value in checkpoint.get("ppo_cfg", {}).items():
            if hasattr(ppo_cfg, key):
                setattr(ppo_cfg, key, value)

        state = checkpoint["model"]
        if "actor_trunk.0.weight" not in state:
            raise RuntimeError("checkpoint does not contain MAPPO actor weights")
        actual_obs_dim = int(state["actor_trunk.0.weight"].shape[1])
        dummy_env = FormationEnv(cfg=env_cfg)
        if actual_obs_dim != dummy_env.obs_dim:
            for flag in (
                "obs_include_ego_accel",
                "obs_include_ship_size",
                "obs_include_cpa_ship",
                "obs_include_cpa",
                "obs_include_route",
                "obs_include_slot_onehot",
            ):
                if actual_obs_dim == dummy_env.obs_dim:
                    break
                if getattr(env_cfg, flag, False):
                    setattr(env_cfg, flag, False)
                    dummy_env = FormationEnv(cfg=env_cfg)
        if actual_obs_dim != dummy_env.obs_dim:
            raise RuntimeError(
                f"checkpoint obs_dim={actual_obs_dim}, env obs_dim={dummy_env.obs_dim}; "
                "cannot match observation dimensions"
            )

        critic_hidden = ppo_cfg.critic_hidden_dims
        critic_in_dim: int | None = None
        if "critic.0.weight" in state:
            c_hidden: list[int] = []
            ci = 0
            while f"critic.{ci}.weight" in state:
                c_hidden.append(int(state[f"critic.{ci}.weight"].shape[0]))
                ci += 2
            if c_hidden and c_hidden[-1] <= 8:
                c_hidden.pop()
            if c_hidden:
                critic_hidden = tuple(c_hidden)
            critic_in_dim = int(state["critic.0.weight"].shape[1])
        policy = MAPPOActorCritic(
            obs_dim=actual_obs_dim,
            action_dim=dummy_env.action_dim,
            n_agents=env_cfg.n_tugs,
            hidden_dims=ppo_cfg.hidden_dims,
            critic_hidden_dims=critic_hidden,
            activation=ppo_cfg.activation,
            log_std_init=ppo_cfg.log_std_init,
            global_state_dim=critic_in_dim,
        )
        policy.load_state_dict(state)
        print(f"[viz] loaded MAPPO checkpoint: {checkpoint_path}")
    policy_eval = policy
    if policy is not None:
        policy_eval.eval()

    # --- env ---
    env = FormationEnv(cfg=env_cfg, seed=42)

    # --- state ---
    trails: list[deque] = [deque(maxlen=500) for _ in range(env_cfg.n_tugs)]
    chart_histories: list[TugChartHistory] = [
        TugChartHistory(maxlen=CHART_HISTORY) for _ in range(env_cfg.n_tugs)
    ]
    for h in chart_histories:
        h.pos_tol_m = env_cfg.pos_tol_m
    recorder = EpisodeRecorder(n_tugs=env_cfg.n_tugs, dt_ctrl=env_cfg.dt_ctrl)
    selected_tug = None
    paused = False
    ep_ret = 0.0
    ep_done_flag: dict = {}
    rpm_limit = 2500.0
    az_limit = 90.0
    if hasattr(env.tugs[0], "rpm_limit"):
        rpm_limit = float(env.tugs[0].rpm_limit)
    if hasattr(env.tugs[0], "azimuth_limit_deg"):
        az_limit = float(env.tugs[0].azimuth_limit_deg)

    # --- initial reset ---
    obs = env.reset()
    ep_ret = 0.0
    ep_done_flag = {}
    for t in trails:
        t.clear()
    for h in chart_histories:
        h.clear()
    recorder.clear()

    running = True
    while running:
        # --- event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    ep_ret = 0.0
                    ep_done_flag = {}
                    for t in trails:
                        t.clear()
                    for h in chart_histories:
                        h.clear()
                    recorder.clear()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    speed = min(speed * 1.5, 64.0)
                elif event.key == pygame.K_MINUS:
                    speed = max(speed / 1.5, 0.25)
                elif event.key == pygame.K_0:
                    selected_tug = None
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    idx = event.key - pygame.K_1
                    if idx < env_cfg.n_tugs:
                        selected_tug = idx
                elif event.key == pygame.K_s:
                    if selected_tug is not None:
                        export_tug_curves(
                            recorder, selected_tug,
                            _project_path(f"exports/tug{selected_tug}_curves.png"),
                        )
                elif event.key == pygame.K_a:
                    out_dir = _project_path("exports")
                    out_dir.mkdir(exist_ok=True)
                    for i in range(env_cfg.n_tugs):
                        export_tug_curves(recorder, i, out_dir / f"tug{i}_curves.png")
                elif event.key == pygame.K_t:
                    export_trajectory(recorder, _project_path("exports/trajectory.png"))
                elif event.key == pygame.K_e:
                    out_dir = _project_path("exports")
                    out_dir.mkdir(exist_ok=True)
                    for i in range(env_cfg.n_tugs):
                        export_tug_curves(recorder, i, out_dir / f"tug{i}_curves.png")
                    export_trajectory(recorder, out_dir / "trajectory.png")

        if not running:
            break

        # --- step ---
        if not paused:
            if policy_eval is not None:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    actions_t, _, _ = policy_eval.act(obs_t, deterministic=True)
                    actions = actions_t.cpu().numpy()
            else:
                actions = np.random.uniform(-1, 1, (env_cfg.n_tugs, 4)).astype(np.float32)

            obs, rewards, dones, info = env.step(actions)
            ep_ret += float(np.mean(rewards))
            if dones.any():
                if info.get("success"):
                    ep_done_flag["success"] = True
                elif info.get("collision"):
                    ep_done_flag["collision"] = True
                else:
                    ep_done_flag["timeout"] = True
            if dones.all():
                obs = env.reset()
                for t in trails:
                    t.clear()
                for h in chart_histories:
                    h.clear()
                recorder.clear()
                ep_ret = 0.0
                ep_done_flag = {}

            # --- update chart/recorder after step ---
            snap = env.render_snapshot()
            snap["_success"] = ep_done_flag.get("success", False)
            snap["_collision"] = ep_done_flag.get("collision", False)
            cpa_pairs = snap.get("cpa_pairs", [])
            dcpa_per_tug = {i: 100.0 for i in range(env_cfg.n_tugs)}
            tcpa_per_tug = {i: 0.0 for i in range(env_cfg.n_tugs)}
            for pair in cpa_pairs:
                i, j = pair["i"], pair["j"]
                if pair["dcpa"] < dcpa_per_tug[i]:
                    dcpa_per_tug[i] = pair["dcpa"]
                    tcpa_per_tug[i] = pair["tcpa"]
                if pair["dcpa"] < dcpa_per_tug[j]:
                    dcpa_per_tug[j] = pair["dcpa"]
                    tcpa_per_tug[j] = pair["tcpa"]
            for i, t in enumerate(snap["tugs"]):
                trails[i].append((t["x"], t["y"]))
                chart_histories[i].push(t, snap["ship"]["psi"], rpm_limit, az_limit,
                                        dcpa_per_tug[i], tcpa_per_tug[i])
            recorder.push(snap)
        else:
            snap = env.render_snapshot()

        # --- rendering ---
        cur_w, cur_h = screen.get_size()
        viewport_w = cur_w - CHART_W if selected_tug is not None else cur_w
        surf = pygame.Surface((cur_w, cur_h))
        _draw_sea_background(surf)
        cam = Camera(viewport_w, cur_h, viz_cfg.meters_per_pixel)
        if viz_cfg.follow_ship:
            cam.center_on(snap["ship"]["x"], snap["ship"]["y"])
        _draw_grid(surf, cam, spacing_m=100.0)
        _draw_ship(surf, snap["ship"], cam)

        hold_steps = int(round(env_cfg.hold_time_s / env_cfg.dt_ctrl))
        for i, slot in enumerate(snap["slots"]):
            in_zone = any(
                t["slot_idx"] == i and t["in_zone_steps"] >= hold_steps
                for t in snap["tugs"]
            )
            _draw_slot(surf, slot, i, in_zone, cam)

        for i, t in enumerate(snap["tugs"]):
            if i == selected_tug:
                sx, sy = cam.world_to_screen_with_offset(t["x"], t["y"])
                r_hl = cam.scale(t["length"] * 0.8)
                pygame.draw.circle(surf, C_TUG_COLORS[i], (sx, sy), r_hl, 2)
            _draw_tug(surf, t, i, trails[i], cam, viz_cfg.show_thrust)

        _draw_cpa_warnings(surf, snap, cam)
        _draw_hud(surf, snap, ep_ret, paused, speed, font, font_small)

        surf.set_clip(None)

        key_hint = font_small.render(
            "1-4=\u9009\u62d6\u8f6e  0=\u5173\u9762\u677f  S=\u5bfc\u51fa\u66f2\u7ebf  A=\u5168\u90e8\u66f2\u7ebf  T=\u8f68\u8ff9  E=\u5168\u90e8  Space=\u6682\u505c  R=\u91cd\u7f6e  +/-=\u901f\u5ea6  Q=\u9000\u51fa",
            True, C_TEXT_DIM,
        )
        surf.blit(key_hint, (8, cur_h - 18))

        # compass rose (bottom-right)
        cx_comp = viewport_w - 60
        cy_comp = cur_h - 60
        _draw_compass(surf, cx_comp, cy_comp, 35, snap["ship"]["psi"], font_small)

        # chart panel
        if selected_tug is not None:
            panel_rect = pygame.Rect(viewport_w, 0, CHART_W, cur_h)
            pygame.draw.line(surf, C_PANEL_EDGE, (viewport_w, 0), (viewport_w, cur_h), 1)
            _draw_chart_panel(
                surf, chart_histories[selected_tug],
                selected_tug, panel_rect, font_small, az_limit, rpm_limit,
            )

        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tugboat formation visualization")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    run_visualization(checkpoint_path=args.ckpt, random_policy=args.random, speed=args.speed)


if __name__ == "__main__":
    main()
