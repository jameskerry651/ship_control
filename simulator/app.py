"""主循环：方向盘/键盘输入 -> 拖轮动力学模型 -> pygame 渲染，支持可调倍速。"""

from __future__ import annotations

import math

import pygame

from physics.large_ship_model import LargeShipModel
from physics.tugboat_dynamics_model import TugboatDynamicsModel

from . import render
from .config import SimConfig
from .wheel import KeyboardInput, WheelInput, create_input


class SimApp:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        pygame.init()
        self.screen = pygame.display.set_mode(
            (cfg.window_w, cfg.window_h), pygame.RESIZABLE
        )
        pygame.display.set_caption("拖轮手动驾驶仿真 — G29 方向盘")
        self.clock = pygame.time.Clock()
        self.font = render.load_ui_font(16, bold=True)
        self.font_small = render.load_ui_font(14)
        self.font_tiny = render.load_ui_font(12)

        self.model = TugboatDynamicsModel()
        self.model.reset()
        self.model.snap_actuators_to_commands()

        self.ship: LargeShipModel | None = None
        if cfg.show_ship:
            self.ship = LargeShipModel(
                length_m=cfg.ship_length_m, beam_m=cfg.ship_beam_m
            )
            self._reset_ship()

        self.input = create_input(cfg)
        self.sim_speed = cfg.sim_speed
        self.sim_time_s = 0.0
        self.paused = False
        self.show_axis_debug = not isinstance(self.input, KeyboardInput)
        self.running = True

    def _reset_ship(self) -> None:
        """把大船设为匀速直线运动状态。"""
        ship = self.ship
        if ship is None:
            return
        speed = self.cfg.ship_speed_ms
        ship.speed_min = speed
        ship.speed_max = speed
        ship.yaw_rate_max = 0.0
        ship.x = self.cfg.ship_init_north_m
        ship.y = self.cfg.ship_init_east_m
        ship.psi = math.radians(self.cfg.ship_heading_deg)
        ship.u = speed
        ship.v = 0.0
        ship.r = 0.0
        ship.u_dot = 0.0
        ship._u_target = speed
        ship._r_target = 0.0
        ship._time_to_resample = 1e9  # 不重采样，保持恒定

    def _reset(self) -> None:
        self.model.reset()
        self.model.snap_actuators_to_commands()
        self._reset_ship()
        self.sim_time_s = 0.0

    def _toggle_calibration(self) -> None:
        if isinstance(self.input, WheelInput) and self.input.connected:
            if self.input.calibrating:
                self.input.end_calibration()
            else:
                self.input.begin_calibration()
                self.show_axis_debug = True

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.sim_speed = min(
                        self.sim_speed * self.cfg.sim_speed_step, self.cfg.sim_speed_max
                    )
                elif event.key == pygame.K_MINUS:
                    self.sim_speed = max(
                        self.sim_speed / self.cfg.sim_speed_step, self.cfg.sim_speed_min
                    )
                elif event.key == pygame.K_d:
                    self.show_axis_debug = not self.show_axis_debug
                elif event.key == pygame.K_c:
                    self._toggle_calibration()

    def _apply_controls(self, control) -> None:
        az_cmd = control.steer * self.model.azimuth_limit_deg
        port_sign = -1.0 if control.port_reverse else 1.0
        starboard_sign = -1.0 if control.starboard_reverse else 1.0
        port_rpm = control.throttle * self.model.rpm_limit * port_sign
        starboard_rpm = control.brake * self.model.rpm_limit * starboard_sign
        self.model.set_control_commands(
            port_rpm_cmd=port_rpm,
            starboard_rpm_cmd=starboard_rpm,
            port_azimuth_cmd_deg=az_cmd,
            starboard_azimuth_cmd_deg=az_cmd,
        )

    def _nearest_slot_idx(self, state, slots) -> int:
        tx = state["position_m"].x
        ty = state["position_m"].y
        best_i = -1
        best_d = float("inf")
        for i in range(len(slots)):
            d = math.hypot(tx - float(slots[i][0]), ty - float(slots[i][1]))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _escort_metrics(self, state) -> dict:
        ship = self.ship
        tx = state["position_m"].x
        ty = state["position_m"].y
        dx = tx - ship.x
        dy = ty - ship.y
        cos_p = math.cos(ship.psi)
        sin_p = math.sin(ship.psi)
        # 旋转到大船船体系：x_b 纵向(+前), y_b 横向(+右舷)
        lon_off = cos_p * dx + sin_p * dy
        lat_off = -sin_p * dx + cos_p * dy
        return {
            "hull_dist": ship.distance_from_hull(tx, ty),
            "lon_off": lon_off,
            "lat_off": lat_off,
            "tug_speed": state["speed_ms"],
            "ship_speed": ship.u,
        }

    def _render(self, control) -> None:
        w, h = self.screen.get_size()
        surf = pygame.Surface((w, h))
        render.draw_sea_background(surf)

        cam = render.Camera(w, h, self.cfg.meters_per_pixel)
        state = self.model.get_state_snapshot()
        if self.cfg.follow_ship:
            cam.center_on(state["position_m"].x, state["position_m"].y)
        render.draw_grid(surf, cam, self.cfg.grid_spacing_m)

        escort = None
        if self.ship is not None:
            render.draw_large_ship(surf, self.ship, cam)
            slots = self.ship.slot_positions_world()
            near_idx = self._nearest_slot_idx(state, slots)
            render.draw_slots(surf, slots, cam, near_idx)
            escort = self._escort_metrics(state)

        thr = self.model.get_thruster_snapshot()
        render.draw_ship(surf, state, thr, self.model.length_m, self.model.beam_m, cam)

        ctrl = self.model.get_control_snapshot()
        render.draw_hud(
            surf, state, ctrl, control, self.sim_speed, self.sim_time_s,
            self.paused, self.input.name, self.model.azimuth_limit_deg,
            self.model.rpm_limit, self.font, self.font_small, escort,
        )

        if self.show_axis_debug:
            pressed = (
                self.input.pressed_buttons()
                if isinstance(self.input, WheelInput)
                else None
            )
            render.draw_axis_debug(
                surf, self.cfg, self.input.raw_axes(),
                isinstance(self.input, WheelInput) and self.input.calibrating,
                self.font_tiny, pressed,
            )

        render.draw_compass(
            surf, w - 60, h - 70, 35, state["heading_rad"], self.font_small
        )
        render.draw_key_hints(surf, self.font_tiny)

        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            real_dt = self.clock.tick(self.cfg.fps) / 1000.0
            self._handle_events()
            if not self.running:
                break

            control = self.input.poll(real_dt)
            self._apply_controls(control)

            if not self.paused:
                sim_dt = real_dt * self.sim_speed
                if sim_dt > 0.0:
                    self.model.step(sim_dt)
                    if self.ship is not None:
                        self.ship.step(sim_dt)
                    self.sim_time_s += sim_dt

            self._render(control)

        # macOS 上 Cocoa 视频子系统的销毁顺序较敏感，先关 display 再整体 quit，
        # 可避免关闭窗口后在原生销毁阶段段错误。
        try:
            pygame.display.quit()
        finally:
            pygame.quit()


def run(cfg: SimConfig | None = None) -> None:
    SimApp(cfg or SimConfig()).run()
