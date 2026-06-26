"""pygame 渲染：复用项目可视化风格绘制单艘拖轮、方位推进器力矢量与 HUD。"""

from __future__ import annotations

import math
from typing import Any

import pygame

from .config import SimConfig
from .wheel import ControlState

# ---------- 配色（取自 scripts/visualize.py 风格） ----------
C_SEA_TOP = (8, 14, 30)
C_SEA_BOT = (18, 28, 48)
C_GRID = (25, 45, 65)
C_SHIP_HULL = (80, 100, 140)
C_SHIP_EDGE = (120, 150, 200)
C_BIGSHIP_HULL = (70, 80, 110)
C_BIGSHIP_EDGE = (150, 170, 210)
C_BIGSHIP_VEL = (120, 220, 255)
C_THRUST = (255, 255, 255)
C_THRUST_PORT = (80, 180, 255)
C_THRUST_STBD = (255, 120, 80)
C_TEXT = (220, 220, 220)
C_TEXT_DIM = (120, 120, 120)
C_WARN = (255, 180, 60)
C_OK = (80, 255, 120)
C_COMPASS = (100, 130, 160)
C_PANEL_BG = (20, 26, 45)
C_PANEL_EDGE = (50, 65, 100)


_FONT_CACHE: dict[tuple[int, bool], pygame.font.Font] = {}


def get_ui_font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    f = _FONT_CACHE.get(key)
    if f is None:
        f = load_ui_font(size, bold)
        _FONT_CACHE[key] = f
    return f


def load_ui_font(size: int, bold: bool = False) -> pygame.font.Font:
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


class Camera:
    def __init__(self, w: int, h: int, mpp: float) -> None:
        self.w = w
        self.h = h
        self.mpp = mpp
        self.cx = w // 2
        self.cy = h // 2
        self._ox = 0.0
        self._oy = 0.0

    def center_on(self, x: float, y: float) -> None:
        self.cx = self.w // 2
        self.cy = self.h // 2
        self._ox = x
        self._oy = y

    def scale(self, meters: float) -> int:
        return max(1, int(meters / self.mpp))

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int(self.cx + (y - self._oy) / self.mpp)
        sy = int(self.cy - (x - self._ox) / self.mpp)
        return sx, sy


def _rot_poly(
    verts_body: list[tuple[float, float]], psi: float, tx: float, ty: float, cam: Camera
) -> list[tuple[int, int]]:
    cos_p = math.cos(psi)
    sin_p = math.sin(psi)
    out = []
    for bx, by in verts_body:
        wx = tx + cos_p * bx - sin_p * by
        wy = ty + sin_p * bx + cos_p * by
        out.append(cam.world_to_screen(wx, wy))
    return out


def draw_sea_background(surf: pygame.Surface) -> None:
    h = surf.get_height()
    w = surf.get_width()
    for y in range(0, h, 2):
        t = y / max(h - 1, 1)
        r = int(C_SEA_TOP[0] + (C_SEA_BOT[0] - C_SEA_TOP[0]) * t)
        g = int(C_SEA_TOP[1] + (C_SEA_BOT[1] - C_SEA_TOP[1]) * t)
        b = int(C_SEA_TOP[2] + (C_SEA_BOT[2] - C_SEA_TOP[2]) * t)
        pygame.draw.rect(surf, (r, g, b), (0, y, w, 2))


def draw_grid(surf: pygame.Surface, cam: Camera, spacing_m: float) -> None:
    half_w_m = cam.w / 2 * cam.mpp
    half_h_m = cam.h / 2 * cam.mpp
    x_min = cam._ox - half_h_m
    x_max = cam._ox + half_h_m
    y_min = cam._oy - half_w_m
    y_max = cam._oy + half_w_m
    x = math.floor(x_min / spacing_m) * spacing_m
    while x <= x_max:
        p1 = cam.world_to_screen(x, y_min)
        p2 = cam.world_to_screen(x, y_max)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)
        x += spacing_m
    y = math.floor(y_min / spacing_m) * spacing_m
    while y <= y_max:
        p1 = cam.world_to_screen(x_min, y)
        p2 = cam.world_to_screen(x_max, y)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)
        y += spacing_m


def draw_compass(
    surf: pygame.Surface, cx: int, cy: int, radius: int, psi: float, font: pygame.font.Font
) -> None:
    pygame.draw.circle(surf, C_COMPASS, (cx, cy), radius, 1)
    nav = (-math.sin(psi), -math.cos(psi))
    eas = (math.cos(psi), -math.sin(psi))
    for label, vx, vy in [
        ("N", *nav),
        ("S", -nav[0], -nav[1]),
        ("E", *eas),
        ("W", -eas[0], -eas[1]),
    ]:
        lx = int(cx + vx * (radius - 12))
        ly = int(cy + vy * (radius - 12))
        lbl = font.render(label, True, C_COMPASS)
        surf.blit(lbl, (lx - lbl.get_width() // 2, ly - lbl.get_height() // 2))
    pygame.draw.circle(surf, C_COMPASS, (cx, cy), 2, 0)


def draw_ship(
    surf: pygame.Surface,
    state: dict[str, Any],
    thr: dict[str, Any],
    length_m: float,
    beam_m: float,
    cam: Camera,
) -> None:
    x = state["position_m"].x
    y = state["position_m"].y
    psi = state["heading_rad"]
    half_l = length_m / 2.0
    half_b = beam_m / 2.0
    verts_body = [
        (+half_l, 0.0),
        (+half_l * 0.7, -half_b),
        (-half_l, -half_b),
        (-half_l, +half_b),
        (+half_l * 0.7, +half_b),
    ]
    pts = _rot_poly(verts_body, psi, x, y, cam)
    pygame.draw.polygon(surf, C_SHIP_HULL, pts)
    pygame.draw.polygon(surf, C_SHIP_EDGE, pts, 2)

    # 航向箭头
    cx, cy = cam.world_to_screen(x, y)
    arrow = cam.scale(length_m * 0.7)
    ex = int(cx + arrow * math.sin(psi))
    ey = int(cy - arrow * math.cos(psi))
    pygame.draw.line(surf, C_SHIP_EDGE, (cx, cy), (ex, ey), 2)

    # 两个方位推进器：位置（船体系）+ 力矢量（船体系）
    cos_p = math.cos(psi)
    sin_p = math.sin(psi)
    force_scale = cam.scale(1.0) / 6000.0
    for side, color in (("port", C_THRUST_PORT), ("starboard", C_THRUST_STBD)):
        pp = thr[f"{side}_position_body_m"]
        fp = thr[f"{side}_force_body_n"]
        px_w = x + cos_p * pp.x - sin_p * pp.y
        py_w = y + sin_p * pp.x + cos_p * pp.y
        sx0, sy0 = cam.world_to_screen(px_w, py_w)
        pygame.draw.circle(surf, color, (sx0, sy0), 4)
        fx_w = cos_p * fp.x - sin_p * fp.y
        fy_w = sin_p * fp.x + cos_p * fp.y
        sx1 = int(sx0 + fy_w * force_scale)
        sy1 = int(sy0 - fx_w * force_scale)
        if abs(sx1 - sx0) + abs(sy1 - sy0) > 2:
            pygame.draw.line(surf, C_THRUST, (sx0, sy0), (sx1, sy1), 3)
            pygame.draw.circle(surf, C_THRUST, (sx1, sy1), 3)


def draw_large_ship(surf: pygame.Surface, ship, cam: Camera) -> None:
    hull = ship.hull_polygon_world()  # (N,2) -> [x_north, y_east]
    pts = [cam.world_to_screen(float(p[0]), float(p[1])) for p in hull]
    if len(pts) >= 3:
        pygame.draw.polygon(surf, C_BIGSHIP_HULL, pts)
        pygame.draw.polygon(surf, C_BIGSHIP_EDGE, pts, 2)
    cx, cy = cam.world_to_screen(ship.x, ship.y)
    psi = ship.psi
    # 航向/速度箭头
    arrow = cam.scale(ship.length_m * 0.45)
    ex = int(cx + arrow * math.sin(psi))
    ey = int(cy - arrow * math.cos(psi))
    pygame.draw.line(surf, C_BIGSHIP_VEL, (cx, cy), (ex, ey), 3)
    pygame.draw.circle(surf, C_BIGSHIP_VEL, (ex, ey), 4)
    lbl = get_ui_font(13).render("大船", True, C_BIGSHIP_EDGE)
    surf.blit(lbl, (cx + 6, cy + 6))


C_SLOT_RING = (60, 200, 120)
C_SLOT_FILL = (25, 70, 45)
C_SLOT_NEAR = (255, 220, 50)
_SLOT_LABELS = ["首左", "首右", "尾左", "尾右"]


def draw_slots(surf: pygame.Surface, slots, cam: Camera, near_idx: int = -1,
               radius_m: float = 10.0) -> None:
    """绘制大船 4 个 slot（船首左/右、船尾左/右）及期望航向。"""
    font = get_ui_font(12)
    for i in range(len(slots)):
        sx_w, sy_w, psi = float(slots[i][0]), float(slots[i][1]), float(slots[i][2])
        cx, cy = cam.world_to_screen(sx_w, sy_w)
        r = cam.scale(radius_m)
        is_near = (i == near_idx)
        ring = C_SLOT_NEAR if is_near else C_SLOT_RING
        pygame.draw.circle(surf, C_SLOT_FILL, (cx, cy), r)
        pygame.draw.circle(surf, ring, (cx, cy), r, 2)
        # 期望航向短线
        line_len = cam.scale(radius_m * 1.6)
        ex = int(cx + line_len * math.sin(psi))
        ey = int(cy - line_len * math.cos(psi))
        pygame.draw.line(surf, ring, (cx, cy), (ex, ey), 2)
        lbl = font.render(f"S{i} {_SLOT_LABELS[i]}", True, ring)
        surf.blit(lbl, (cx + r + 3, cy - 7))


def draw_hud(
    surf: pygame.Surface,
    state: dict[str, Any],
    ctrl: dict[str, Any],
    control: ControlState,
    sim_speed: float,
    sim_time_s: float,
    paused: bool,
    input_name: str,
    az_limit_deg: float,
    rpm_limit: float,
    font: pygame.font.Font,
    font_small: pygame.font.Font,
    escort: dict[str, Any] | None = None,
) -> None:
    steer_deg = control.steer * az_limit_deg
    port_rev = " [反转]" if control.port_reverse else ""
    stbd_rev = " [反转]" if control.starboard_reverse else ""
    lines = [
        ("拖轮手动驾驶仿真", C_TEXT),
        (f"仿真时间 {sim_time_s:6.1f}s   倍速 {sim_speed:.2f}x" + ("   [暂停]" if paused else ""),
         C_WARN if paused else C_TEXT),
        (f"输入设备: {input_name}", C_TEXT_DIM),
        ("", C_TEXT),
        (f"方向盘转角  {steer_deg:+6.1f}°", C_TEXT),
        (f"油门(左桨)  {control.throttle * 100:5.1f}%{port_rev}", C_WARN if control.port_reverse else C_TEXT),
        (f"刹车(右桨)  {control.brake * 100:5.1f}%{stbd_rev}", C_WARN if control.starboard_reverse else C_TEXT),
        ("", C_TEXT),
        (f"左桨 RPM   cmd {ctrl['port_rpm_cmd']:+7.1f}  act {ctrl['port_rpm_actual']:+7.1f}", C_THRUST_PORT),
        (f"右桨 RPM   cmd {ctrl['starboard_rpm_cmd']:+7.1f}  act {ctrl['starboard_rpm_actual']:+7.1f}", C_THRUST_STBD),
        (f"左舵方位角 cmd {ctrl['port_azimuth_cmd_deg']:+6.1f}°  act {ctrl['port_azimuth_actual_deg']:+6.1f}°", C_THRUST_PORT),
        (f"右舵方位角 cmd {ctrl['starboard_azimuth_cmd_deg']:+6.1f}°  act {ctrl['starboard_azimuth_actual_deg']:+6.1f}°", C_THRUST_STBD),
        ("", C_TEXT),
        (f"航向 {state['heading_deg']:+6.1f}°   艏摇 {state['yaw_rate_degs']:+5.2f}°/s", C_TEXT),
        (f"u {state['u_ms']:+5.2f}  v {state['v_ms']:+5.2f}  |V| {state['speed_ms']:5.2f} m/s", C_TEXT),
        (f"位置 N {state['position_m'].x:8.1f}  E {state['position_m'].y:8.1f} m", C_TEXT_DIM),
    ]
    if escort is not None:
        dspeed = escort["tug_speed"] - escort["ship_speed"]
        dist_color = C_OK if escort["hull_dist"] < 40.0 else C_TEXT
        lines += [
            ("", C_TEXT),
            ("—— 伴航 (相对大船) ——", C_BIGSHIP_VEL),
            (f"船体距离 {escort['hull_dist']:6.1f} m", dist_color),
            (f"纵向偏移 {escort['lon_off']:+7.1f} m (+前/-后)", C_TEXT),
            (f"横向偏移 {escort['lat_off']:+7.1f} m (+右舷/-左舷)", C_TEXT),
            (f"艇速 {escort['tug_speed']:.2f}  船速 {escort['ship_speed']:.2f}  Δ {dspeed:+.2f} m/s", C_TEXT),
        ]
    y_off = 8
    for text, color in lines:
        if text:
            surf.blit(font_small.render(text, True, color), (10, y_off))
        y_off += 18


def draw_axis_debug(
    surf: pygame.Surface,
    cfg: SimConfig,
    raw_axes: list[float],
    calibrating: bool,
    font: pygame.font.Font,
    pressed_buttons: list[int] | None = None,
) -> None:
    w = 280
    h = 30 + max(1, len(raw_axes)) * 18 + 48
    x0 = surf.get_width() - w - 10
    y0 = 10
    panel = pygame.Rect(x0, y0, w, h)
    pygame.draw.rect(surf, C_PANEL_BG, panel)
    pygame.draw.rect(surf, C_PANEL_EDGE, panel, 1)
    title = "轴/按钮调试" + ("  [校准中: 踩满两踏板后按 C 结束]" if calibrating else "  (C 校准)")
    surf.blit(font.render(title, True, C_WARN if calibrating else C_TEXT), (x0 + 8, y0 + 6))
    assign = {
        cfg.steer_axis: "转向",
        cfg.throttle_axis: "油门/左桨",
        cfg.brake_axis: "刹车/右桨",
    }
    yy = y0 + 30
    if not raw_axes:
        surf.blit(font.render("无方向盘 (键盘模式)", True, C_TEXT_DIM), (x0 + 8, yy))
    for i, v in enumerate(raw_axes):
        tag = assign.get(i, "")
        color = C_OK if tag else C_TEXT_DIM
        bar_x = x0 + 120
        bar_w = w - 130
        mid = bar_x + bar_w // 2
        surf.blit(font.render(f"axis{i} {v:+.2f} {tag}", True, color), (x0 + 8, yy))
        pygame.draw.line(surf, C_PANEL_EDGE, (bar_x, yy + 14), (bar_x + bar_w, yy + 14), 1)
        px = int(mid + v * bar_w / 2)
        pygame.draw.circle(surf, color, (px, yy + 14), 3)
        yy += 18
    btn_str = ",".join(str(b) for b in (pressed_buttons or [])) or "-"
    surf.blit(font.render(f"按下按钮: {btn_str}", True, C_TEXT), (x0 + 8, yy + 2))
    surf.blit(
        font.render(f"拨片(反转) 左={cfg.paddle_left_button} 右={cfg.paddle_right_button}", True, C_TEXT_DIM),
        (x0 + 8, yy + 20),
    )


def draw_key_hints(surf: pygame.Surface, font: pygame.font.Font) -> None:
    hint = "Space 暂停  R 重置  +/- 倍速  D 调试  C 校准  拨片/Shift 反转左右桨  方向键/Q-Z/E-C 键盘  Esc 退出"
    lbl = font.render(hint, True, C_TEXT_DIM)
    surf.blit(lbl, (10, surf.get_height() - 20))
