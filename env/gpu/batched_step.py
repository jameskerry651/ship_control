"""Device-resident reward, termination, and observation path for ``CudaVecEnv``.

The CPU environment remains the authoritative reset sampler.  Between resets,
however, all mutable episode state lives in tensors shaped ``(num_envs, n_tugs)``.
This keeps the hot loop independent of ``FormationEnv`` object graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from config import EnvConfig
from env.obs_spec import (
    ACTION_DIM,
    ObservationSpec,
    _GLOBAL_ACCEL_PER_TUG_DIM,
    _GLOBAL_PER_TUG_DIM,
    _GLOBAL_SHIP_DIM,
    _NEIGHBOR_REL_SPEED_EPS,
    _NEIGHBOR_TCPA_SCALE_S,
    _SHIP_LINEAR_ACCEL_SCALE,
    _TUG_LINEAR_ACCEL_SCALE,
    _TUG_YAW_ACCEL_SCALE,
)
from env.state import MutableEpisodeState
from physics.batched.geometry import (
    distance_from_rectangular_hull,
    slot_positions_world,
    wrap_pi,
)
from physics.batched.ship import step_ships
from physics.batched.tugboat import set_control_commands, step_tugs


@dataclass
class BatchedEpisodeState:
    """All episode bookkeeping, in struct-of-arrays form on one device."""

    in_zone_steps: torch.Tensor
    prev_dist: torch.Tensor
    capture_done: torch.Tensor
    just_captured: torch.Tensor
    track_streak_steps: torch.Tensor
    track_steps_total: torch.Tensor
    track_steps_all_in_zone: torch.Tensor
    step_count: torch.Tensor
    tug_to_slot: torch.Tensor
    last_actions: torch.Tensor
    motion_history: torch.Tensor
    action_history: torch.Tensor


def _tensor_like(value: np.ndarray | torch.Tensor, ref: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(value, device=ref.device, dtype=dtype or ref.dtype)


class FastBatchedStep:
    """Vectorized post-dynamics FormationEnv semantics.

    ``GpuEnvBatch`` owns dynamics state.  This class owns the state which the
    legacy environment previously kept in Python/numpy objects.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        batch: Any,
        n_envs: int,
        n_tugs: int,
        spec: ObservationSpec,
    ) -> None:
        self.cfg = cfg
        self.batch = batch
        self.n_envs = n_envs
        self.n_tugs = n_tugs
        self.spec = spec
        self.device = batch.device
        self.dtype = batch.dtype
        z = lambda *shape, dtype=None: torch.zeros(*shape, device=self.device, dtype=dtype or self.dtype)
        self.episode = BatchedEpisodeState(
            in_zone_steps=z(n_envs, n_tugs, dtype=torch.int64),
            prev_dist=z(n_envs, n_tugs),
            capture_done=z(n_envs, dtype=torch.bool),
            just_captured=z(n_envs, dtype=torch.bool),
            track_streak_steps=z(n_envs, dtype=torch.int64),
            track_steps_total=z(n_envs, dtype=torch.int64),
            track_steps_all_in_zone=z(n_envs, dtype=torch.int64),
            step_count=z(n_envs, dtype=torch.int64),
            tug_to_slot=z(n_envs, n_tugs, dtype=torch.int64),
            last_actions=z(n_envs, n_tugs, ACTION_DIM),
            motion_history=z(n_envs, n_tugs, spec.history_len, spec.motion_dim),
            action_history=z(n_envs, n_tugs, spec.history_len, spec.action_history_dim),
        )

    # ---------------------------------------------------------------- reset
    def reset_from_env(self, env: Any, row: int) -> None:
        """Copy one CPU reset's episode bookkeeping to its tensor row."""
        ep: MutableEpisodeState = env._episode
        e = self.episode
        e.in_zone_steps[row].copy_(_tensor_like(ep.in_zone_steps, e.in_zone_steps, dtype=torch.int64))
        e.prev_dist[row].copy_(_tensor_like(ep.prev_dist, e.prev_dist))
        e.capture_done[row] = False
        e.just_captured[row] = False
        e.track_streak_steps[row] = 0
        e.track_steps_total[row] = 0
        e.track_steps_all_in_zone[row] = 0
        e.step_count[row] = 0
        e.tug_to_slot[row].copy_(_tensor_like(env.tug_to_slot, e.tug_to_slot, dtype=torch.int64))
        e.last_actions[row].copy_(_tensor_like(env.last_actions, e.last_actions))
        e.motion_history[row].copy_(_tensor_like(env.motion_history, e.motion_history))
        e.action_history[row].copy_(_tensor_like(env.action_history, e.action_history))

    # -------------------------------------------------------------- dynamics
    def dynamics_step(self, actions: torch.Tensor, rng_envs: list[Any] | None = None) -> torch.Tensor:
        """Advance batched tug/ship dynamics and return pre-step tug velocity."""
        actions = actions.clamp(-1.0, 1.0)
        prev_nu = self.batch.tugs.nu.clone()
        set_control_commands(self.batch.tugs, actions, self.batch.tug_params)
        dt = float(self.cfg.dt_ctrl)

        # Replay the exact per-environment NumPy streams used by FormationEnv.
        # This sparse host round-trip occurs only at ship resample boundaries.
        need = self.batch.ships.time_to_resample - dt <= 0.0
        u_samples = interval_samples = None
        if rng_envs is not None and bool(need.any()):
            needed = torch.nonzero(need, as_tuple=False).flatten().cpu().tolist()
            u_values = np.zeros(self.n_envs, dtype=np.float64)
            interval_values = np.zeros(self.n_envs, dtype=np.float64)
            for i in needed:
                ship = rng_envs[i].ship
                u_values[i] = float(ship.rng.uniform(ship.speed_min, ship.speed_max))
                interval_values[i] = float(
                    ship.rng.uniform(ship.target_resample_min_s, ship.target_resample_max_s)
                )
            u_samples = torch.as_tensor(
                u_values, device=self.device, dtype=self.dtype
            )
            interval_samples = torch.as_tensor(
                interval_values,
                device=self.device,
                dtype=self.batch.ships.time_to_resample.dtype,
            )
        step_tugs(self.batch.tugs, self.batch.tug_params, dt)
        step_ships(
            self.batch.ships,
            dt,
            speed_min=float(self.cfg.ship_speed_min),
            speed_max=float(self.cfg.ship_speed_max),
            speed_tau=float(self.cfg.ship_speed_tau_s),
            target_resample_min_s=float(self.cfg.ship_target_resample_min_s),
            target_resample_max_s=float(self.cfg.ship_target_resample_max_s),
            u_target_samples=u_samples,
            resample_interval_samples=interval_samples,
        )
        self.episode.step_count.add_(1)
        return prev_nu

    # --------------------------------------------------------------- geometry
    def _slots(self) -> torch.Tensor:
        s = self.batch.ships
        return slot_positions_world(
            s.x, s.y, s.psi, float(self.cfg.ship_length_m), float(self.cfg.ship_beam_m),
            float(self.cfg.slot_lon_offset_m), float(self.cfg.slot_lat_offset_m),
        )

    def _derived(self) -> dict[str, torch.Tensor]:
        t = self.batch.tugs
        s = self.batch.ships
        slots_all = self._slots()
        slots = slots_all.gather(
            1, self.episode.tug_to_slot[..., None].expand(-1, -1, 3)
        )
        dx = t.eta[..., 0] - slots[..., 0]
        dy = t.eta[..., 1] - slots[..., 1]
        dist = torch.hypot(dx, dy)
        heading_err = wrap_pi(slots[..., 2] - t.eta[..., 2])
        ct, st = torch.cos(t.eta[..., 2]), torch.sin(t.eta[..., 2])
        tug_vx = ct * t.nu[..., 0] - st * t.nu[..., 1]
        tug_vy = st * t.nu[..., 0] + ct * t.nu[..., 1]
        cs, ss = torch.cos(s.psi), torch.sin(s.psi)
        ship_vx = cs * s.u - ss * s.v
        ship_vy = ss * s.u + cs * s.v
        speed_err = torch.hypot(tug_vx - ship_vx[:, None], tug_vy - ship_vy[:, None])
        hull_dist = distance_from_rectangular_hull(
            t.eta[..., 0], t.eta[..., 1], s.x[:, None], s.y[:, None], s.psi[:, None],
            float(self.cfg.ship_length_m), float(self.cfg.ship_beam_m),
        )
        return {
            "slots_all": slots_all, "slots": slots, "dist": dist, "heading_err": heading_err,
            "tug_vx": tug_vx, "tug_vy": tug_vy, "ship_vx": ship_vx, "ship_vy": ship_vy,
            "speed_err": speed_err, "hull_dist": hull_dist,
        }

    @staticmethod
    def _barrier(distance: torch.Tensor, collision: float, safe: float) -> torch.Tensor:
        return ((safe - distance) / max(safe - collision, 1e-6)).clamp(0.0, 1.0)

    def _cpa_risk(
        self, rx: torch.Tensor, ry: torch.Tensor, rvx: torch.Tensor, rvy: torch.Tensor,
        collision: float, safe: float, horizon: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speed_sq = rvx.square() + rvy.square()
        closing = rx * rvx + ry * rvy
        active = (speed_sq > 1e-9) & (closing < 0.0)
        tcpa_raw = -closing / speed_sq.clamp_min(1e-9)
        tcpa = torch.where(active, tcpa_raw, torch.full_like(tcpa_raw, horizon))
        dcpa = torch.hypot(rx + rvx * tcpa, ry + rvy * tcpa)
        risk = self._barrier(dcpa, collision, safe) * (1.0 - tcpa / max(horizon, 1e-6)).clamp(0.0, 1.0)
        return tcpa, dcpa, torch.where(active & (tcpa <= horizon), risk, torch.zeros_like(risk))

    # --------------------------------------------------------------- rewards
    def compute_rewards_batched(
        self, actions: torch.Tensor, derived: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Match ``FormationRewardComputer.compute_rewards`` over N environments."""
        d = derived or self._derived()
        cfg, ep, t, s = self.cfg, self.episode, self.batch.tugs, self.batch.ships
        dist, dpsi, speed_err = d["dist"], d["heading_err"], d["speed_err"]
        hold_start = max(float(cfg.reward_hold_start_m), 1e-6)
        target_tol = max(float(cfg.pos_tol_m), 1e-6)
        target_x = (1.0 - dist / target_tol).clamp(0.0, 1.0)
        target_gate = target_x.square() * (3.0 - 2.0 * target_x)
        head_score = (1.0 - dpsi.abs() / max(float(cfg.heading_tol_rad), 1e-6)).clamp_min(0.0)
        speed_score = (1.0 - speed_err / max(float(cfg.speed_tol_ms), 1e-6)).clamp_min(0.0)
        r_hold = target_gate * (0.5 + 0.25 * head_score + 0.25 * speed_score)
        in_zone = (dist < cfg.pos_tol_m) & (dpsi.abs() < cfg.heading_tol_rad) & (speed_err < cfg.speed_tol_ms)

        progress = ((ep.prev_dist - dist) / max(float(cfg.reward_dist_progress_clip_m), 1e-6)).clamp(-1.0, 1.0)
        r_dist = (1.0 - target_gate) * progress
        p_distance = (1.0 - target_gate) * (
            dist / max(float(cfg.reward_dist_scale_m), 1e-6)
        ).clamp(0.0, 1.0)

        speed_scale = max(float(getattr(cfg, "reward_velocity_speed_scale_ms", cfg.speed_tol_ms)), 1e-6)
        yaw_scale = max(float(getattr(cfg, "reward_velocity_yaw_scale_rads", 0.05)), 1e-6)
        speed_pen = 1.0 - torch.exp(-(speed_err / speed_scale).square())
        yaw_pen = 1.0 - torch.exp(-((t.nu[..., 2] - s.r[:, None]).abs() / yaw_scale).square())
        r_vel = -target_gate * (0.8 * speed_pen + 0.2 * yaw_pen)

        ship_safe = max(float(cfg.reward_collision_ship_safe_m), float(cfg.ship_collision_dist_m) + 1e-6)
        tug_safe = max(float(cfg.reward_collision_tug_safe_m), float(cfg.tug_collision_dist_m) + 1e-6)
        horizon = max(float(cfg.reward_cpa_horizon_s), 1e-6)
        cpa_w = max(float(cfg.reward_collision_cpa_w), 0.0)
        p_ship = self._barrier(d["hull_dist"], float(cfg.ship_collision_dist_m), ship_safe)
        tcpa_ship, _, _ = self._cpa_risk(
            t.eta[..., 0] - s.x[:, None], t.eta[..., 1] - s.y[:, None],
            d["tug_vx"] - d["ship_vx"][:, None], d["tug_vy"] - d["ship_vy"][:, None],
            float(cfg.ship_collision_dist_m), ship_safe, horizon,
        )
        future_tx = t.eta[..., 0] + d["tug_vx"] * tcpa_ship
        future_ty = t.eta[..., 1] + d["tug_vy"] * tcpa_ship
        future_sx = s.x[:, None] + d["ship_vx"][:, None] * tcpa_ship
        future_sy = s.y[:, None] + d["ship_vy"][:, None] * tcpa_ship
        future_hull = distance_from_rectangular_hull(
            future_tx, future_ty, future_sx, future_sy, s.psi[:, None] + s.r[:, None] * tcpa_ship,
            float(cfg.ship_length_m), float(cfg.ship_beam_m),
        )
        ship_active = tcpa_ship <= horizon
        ship_risk = self._barrier(future_hull, float(cfg.ship_collision_dist_m), ship_safe) * (1.0 - tcpa_ship / horizon).clamp(0.0, 1.0)
        p_ship = p_ship + cpa_w * torch.where(ship_active, ship_risk, torch.zeros_like(ship_risk))

        slot_to_tug_x = t.eta[..., 0] - d["slots"][..., 0]
        slot_to_tug_y = t.eta[..., 1] - d["slots"][..., 1]
        axis_x = d["slots"][..., 0] - s.x[:, None]
        axis_y = d["slots"][..., 1] - s.y[:, None]
        axis_norm = torch.hypot(axis_x, axis_y)
        ex, ey = axis_x / axis_norm.clamp_min(1e-6), axis_y / axis_norm.clamp_min(1e-6)
        axial = slot_to_tug_x * ex + slot_to_tug_y * ey
        lateral = torch.hypot(slot_to_tug_x - axial * ex, slot_to_tug_y - axial * ey)
        lateral_u = (1.0 - lateral / max(float(cfg.reward_corridor_half_width_m), 1e-6)).clamp(0.0, 1.0)
        corridor = lateral_u.square() * (3.0 - 2.0 * lateral_u)
        corridor = torch.where(
            (dist < hold_start) & (axis_norm > 1e-6) & (axial >= -max(float(cfg.reward_corridor_axial_slack_m), 0.0)),
            corridor, torch.zeros_like(corridor),
        )
        soft = 1.0 - (1.0 - float(np.clip(cfg.reward_ship_soft_min_scale, 0.0, 1.0))) * corridor
        p_ship = p_ship * soft

        px = t.eta[..., 0]
        py = t.eta[..., 1]
        pair_dx, pair_dy = px[:, None, :] - px[:, :, None], py[:, None, :] - py[:, :, None]
        pair_dist = torch.hypot(pair_dx, pair_dy)
        eye = torch.eye(self.n_tugs, dtype=torch.bool, device=self.device)[None]
        prox = self._barrier(pair_dist, float(cfg.tug_collision_dist_m), tug_safe).masked_fill(eye, 0.0).sum(dim=-1)
        rvx = d["tug_vx"][:, None, :] - d["tug_vx"][:, :, None]
        rvy = d["tug_vy"][:, None, :] - d["tug_vy"][:, :, None]
        _, _, pair_risk = self._cpa_risk(
            pair_dx, pair_dy, rvx, rvy, float(cfg.tug_collision_dist_m), tug_safe, horizon
        )
        p_tug = prox + cpa_w * pair_risk.masked_fill(eye, 0.0).sum(dim=-1)
        p_coll = (p_ship + p_tug).clamp_max(float(cfg.reward_collision_cap))

        rewards = (
            float(cfg.reward_dist_w) * r_dist
            - max(float(cfg.reward_distance_cost_w), 0.0) * p_distance
            + float(cfg.reward_hold_w) * r_hold
            + float(cfg.reward_velocity_w) * r_vel
            - float(cfg.reward_collision_w) * p_coll
        )
        # FormationRewardComputer writes each per-tug total into a float32
        # array before adding the team reward. Preserve that rounding in the
        # float64 parity mode while retaining float32 CUDA throughput.
        rewards = rewards.to(torch.float32).to(self.dtype)
        r_team = torch.zeros_like(rewards)
        if cfg.reward_team_w > 0.0:
            beta = max(float(cfg.reward_team_softmin_beta), 1e-6)
            logits = beta * p_distance
            weights = torch.softmax(logits, dim=1)
            team_cost = (weights * p_distance).sum(dim=1, keepdim=True)
            r_team = (-float(cfg.reward_team_w) * team_cost).to(torch.float32).to(self.dtype)
            rewards = rewards + r_team

        ep.in_zone_steps = torch.where(in_zone, ep.in_zone_steps + 1, (ep.in_zone_steps - 2).clamp_min(0))
        components = {
            "r_total": rewards, "r_dist": r_dist, "p_distance": p_distance,
            "r_hold": r_hold, "r_velocity": r_vel, "r_team": r_team.expand(-1, self.n_tugs),
            "p_collision": p_coll, "p_ship_collision": p_ship, "p_tug_collision": p_tug,
            "dist_to_slot": dist,
            "heading_err_deg": dpsi.abs() * (180.0 / math.pi), "speed_err": speed_err,
            "hull_dist": d["hull_dist"], "in_zone": in_zone,
            "corridor_gate": corridor, "ship_soft_scale": soft,
        }
        return rewards, components, d

    # ----------------------------------------------------------- termination
    def check_termination_batched(self, derived: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update capture/track state and return vectorized termination data."""
        ep, cfg = self.episode, self.cfg
        ep.just_captured.zero_()
        hold_steps = max(1, int(round(cfg.hold_time_s / cfg.dt_ctrl)))
        track_steps = max(1, int(round(float(cfg.track_horizon_s) / cfg.dt_ctrl)))
        all_in_zone = (ep.in_zone_steps >= 1).all(dim=1)
        all_held = (ep.in_zone_steps >= hold_steps).all(dim=1)
        captured_now = ~ep.capture_done & all_held
        ep.capture_done |= captured_now
        ep.just_captured = captured_now
        was_tracking = ep.capture_done & ~captured_now
        ep.track_steps_total += was_tracking.to(torch.int64)
        ep.track_streak_steps = torch.where(
            was_tracking & all_in_zone, ep.track_streak_steps + 1,
            torch.where(was_tracking, torch.zeros_like(ep.track_streak_steps), ep.track_streak_steps),
        )
        ep.track_steps_all_in_zone += (was_tracking & all_in_zone).to(torch.int64)

        ship_hits = derived["hull_dist"] < float(cfg.ship_collision_dist_m)
        ship_collision = ship_hits.any(dim=1)
        px, py = self.batch.tugs.eta[..., 0], self.batch.tugs.eta[..., 1]
        pair_distance = torch.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
        upper = torch.triu(torch.ones((self.n_tugs, self.n_tugs), dtype=torch.bool, device=self.device), diagonal=1)
        pair_hits = (pair_distance < float(cfg.tug_collision_dist_m)) & upper
        tug_collision = pair_hits.any(dim=(1, 2))
        collision = ship_collision | tug_collision
        success = ep.capture_done & (ep.track_streak_steps >= track_steps) & ~collision
        timeout = (ep.step_count >= int(cfg.max_episode_steps)) & ~collision & ~success
        done = collision | success | timeout
        ratio = torch.where(
            ep.track_steps_total > 0,
            ep.track_steps_all_in_zone.to(self.dtype) / ep.track_steps_total.to(self.dtype),
            torch.zeros(self.n_envs, device=self.device, dtype=self.dtype),
        )

        ship_culprit = ship_hits.to(torch.int64).argmax(dim=1)
        pair_flat = pair_hits.reshape(self.n_envs, -1).to(torch.int64).argmax(dim=1)
        pair_a, pair_b = pair_flat // self.n_tugs, pair_flat % self.n_tugs
        terminal = torch.zeros_like(derived["dist"])
        culprit_pen = float(getattr(cfg, "reward_collision_pen_culprit", cfg.reward_collision_pen))
        bystander_pen = float(getattr(cfg, "reward_collision_pen_bystander", culprit_pen))
        terminal = torch.where(collision[:, None], torch.full_like(terminal, -bystander_pen), terminal)
        row = torch.arange(self.n_envs, device=self.device)
        ship_rows = torch.nonzero(ship_collision, as_tuple=False).flatten()
        terminal[ship_rows, ship_culprit[ship_rows]] = -culprit_pen
        tug_rows = torch.nonzero(~ship_collision & tug_collision, as_tuple=False).flatten()
        terminal[tug_rows, pair_a[tug_rows]] = -culprit_pen
        terminal[tug_rows, pair_b[tug_rows]] = -culprit_pen
        terminal += ep.just_captured[:, None].to(self.dtype) * float(cfg.reward_arrival_bonus)
        return {
            "done": done, "success": success, "collision": collision, "ship_collision": ship_collision,
            "tug_collision": tug_collision, "ship_culprit": ship_culprit, "pair_a": pair_a, "pair_b": pair_b,
            "timeout": timeout, "terminated": collision | success, "truncated": timeout,
            "capture": ep.capture_done, "just_captured": ep.just_captured, "track_streak_steps": ep.track_streak_steps,
            "track_steps_total": ep.track_steps_total, "track_in_zone_ratio": ratio, "terminal_reward": terminal,
        }

    # ----------------------------------------------------------- bookkeeping
    def update_episode_state(self, actions: torch.Tensor, prev_nu: torch.Tensor, derived: dict[str, torch.Tensor]) -> None:
        ep, t = self.episode, self.batch.tugs
        ep.last_actions.copy_(actions)
        ep.motion_history[:, :, 1:].copy_(ep.motion_history[:, :, :-1].clone())
        ep.action_history[:, :, 1:].copy_(ep.action_history[:, :, :-1].clone())
        nu, old = t.nu, prev_nu
        motion = torch.stack(
            [nu[..., 0] / 5.0, nu[..., 1] / 5.0, nu[..., 2] / 0.5,
             (nu[..., 0] - old[..., 0]) / 5.0, (nu[..., 1] - old[..., 1]) / 5.0,
             (nu[..., 2] - old[..., 2]) / 0.5],
            dim=-1,
        )
        ep.motion_history[:, :, 0].copy_(motion.to(torch.float32))
        ep.action_history[:, :, 0].copy_(actions.to(torch.float32))
        # Legacy episode arrays are float32 even when batched dynamics is
        # float64; quantizing these tracked values is required for exact L2.
        ep.prev_dist.copy_(derived["dist"].to(torch.float32))

    # ------------------------------------------------------------- observer
    def build_obs_batched(self, derived: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Build all agent observations in the same layout as ``Observer``."""
        d = derived or self._derived()
        ep, t, s, spec = self.episode, self.batch.tugs, self.batch.ships, self.spec
        obs = torch.zeros(self.n_envs, self.n_tugs, spec.total_dim, device=self.device, dtype=self.dtype)
        obs[..., spec.motion_history_slice] = ep.motion_history.reshape(self.n_envs, self.n_tugs, -1)
        obs[..., spec.action_history_slice] = ep.action_history.reshape(self.n_envs, self.n_tugs, -1)
        if spec.thruster_state_dim:
            obs[..., spec.thruster_state_slice] = torch.cat(
                [t.rpm_actual / self.batch.tug_params.rpm_limit, t.az_actual_deg / self.batch.tug_params.azimuth_limit_deg], dim=-1
            )
        psi = t.eta[..., 2]
        ct, st = torch.cos(psi), torch.sin(psi)
        def local(dx: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return ct * dx + st * dy, -st * dx + ct * dy

        sl = spec.ship_relative_slice
        x_local, y_local = local(s.x[:, None] - t.eta[..., 0], s.y[:, None] - t.eta[..., 1])
        ship_dpsi = wrap_pi(s.psi[:, None] - psi)
        obs[..., sl] = torch.stack([x_local / 100.0, y_local / 100.0, s.u[:, None].expand_as(x_local) / 3.0, torch.sin(ship_dpsi), torch.cos(ship_dpsi)], dim=-1)

        preview = []
        for tau in tuple(self.cfg.obs_ship_preview_times_s):
            tt = float(tau)
            near_zero = s.r.abs() < 1e-6
            dx_body = (s.u * torch.sin(s.r * tt) + s.v * (torch.cos(s.r * tt) - 1.0)) / s.r.clamp_min(1e-12)
            dy_body = (s.u * (1.0 - torch.cos(s.r * tt)) + s.v * torch.sin(s.r * tt)) / s.r.clamp_min(1e-12)
            dx_lin, dy_lin = d["ship_vx"] * tt, d["ship_vy"] * tt
            xf = torch.where(near_zero, s.x + dx_lin, s.x + torch.cos(s.psi) * dx_body - torch.sin(s.psi) * dy_body)
            yf = torch.where(near_zero, s.y + dy_lin, s.y + torch.sin(s.psi) * dx_body + torch.cos(s.psi) * dy_body)
            px, py = local(xf[:, None] - t.eta[..., 0], yf[:, None] - t.eta[..., 1])
            preview.extend([px / 100.0, py / 100.0])
        if preview:
            obs[..., spec.ship_preview_slice] = torch.stack(preview, dim=-1)

        sl = spec.slot_target_slice
        sx, sy = local(d["slots"][..., 0] - t.eta[..., 0], d["slots"][..., 1] - t.eta[..., 1])
        slot_dpsi = wrap_pi(d["slots"][..., 2] - psi)
        obs[..., sl] = torch.stack([sx / 100.0, sy / 100.0, torch.hypot(sx, sy).clamp_max(1000.0) / 100.0, torch.sin(slot_dpsi), torch.cos(slot_dpsi)], dim=-1)

        # Closest rectangular hull point, including the interior tie-breaking used by env.state.
        dx_ship, dy_ship = t.eta[..., 0] - s.x[:, None], t.eta[..., 1] - s.y[:, None]
        xb, yb = torch.cos(s.psi)[:, None] * dx_ship + torch.sin(s.psi)[:, None] * dy_ship, -torch.sin(s.psi)[:, None] * dx_ship + torch.cos(s.psi)[:, None] * dy_ship
        lh, bh = float(cfg_or(self.cfg, "ship_length_m")) / 2.0, float(cfg_or(self.cfg, "ship_beam_m")) / 2.0
        hx, hy = xb.clamp(-lh, lh), yb.clamp(-bh, bh)
        inside = (xb.abs() <= lh) & (yb.abs() <= bh)
        choose_x = (lh - xb.abs()) < (bh - yb.abs())
        sign_x = torch.where(xb == 0.0, torch.ones_like(xb), torch.sign(xb))
        sign_y = torch.where(yb == 0.0, torch.ones_like(yb), torch.sign(yb))
        hx = torch.where(inside & choose_x, sign_x * lh, hx)
        hy = torch.where(inside & ~choose_x, sign_y * bh, hy)
        hxw = s.x[:, None] + torch.cos(s.psi)[:, None] * hx - torch.sin(s.psi)[:, None] * hy
        hyw = s.y[:, None] + torch.sin(s.psi)[:, None] * hx + torch.cos(s.psi)[:, None] * hy
        hxl, hyl = local(hxw - t.eta[..., 0], hyw - t.eta[..., 1])
        obs[..., spec.hull_clearance_slice] = torch.stack([hxl / 50.0, hyl / 50.0, d["hull_dist"] / 50.0], dim=-1)

        others = torch.tensor(
            [[j for j in range(self.n_tugs) if j != i] for i in range(self.n_tugs)],
            device=self.device, dtype=torch.long,
        )
        count = min(spec.neighbor_count, self.n_tugs - 1)
        if count:
            oi = others[:, :count]
            ox = t.eta[:, oi, 0]
            oy = t.eta[:, oi, 1]
            ovx = d["tug_vx"][:, oi]
            ovy = d["tug_vy"][:, oi]
            ndx, ndy = ox - t.eta[..., 0, None], oy - t.eta[..., 1, None]
            nx, ny = ct[..., None] * ndx + st[..., None] * ndy, -st[..., None] * ndx + ct[..., None] * ndy
            dvx, dvy = ovx - d["tug_vx"][..., None], ovy - d["tug_vy"][..., None]
            du, dv = ct[..., None] * dvx + st[..., None] * dvy, -st[..., None] * dvx + ct[..., None] * dvy
            ndist = torch.hypot(nx, ny)
            bearing = torch.atan2(ny, nx)
            range_rate = torch.where(ndist > _NEIGHBOR_REL_SPEED_EPS, (nx * du + ny * dv) / ndist, torch.zeros_like(ndist))
            rel_sq = du.square() + dv.square()
            tcpa = torch.where(rel_sq > _NEIGHBOR_REL_SPEED_EPS, (-(nx * du + ny * dv) / rel_sq).clamp_min(0.0), torch.full_like(rel_sq, _NEIGHBOR_TCPA_SCALE_S))
            dcpa = torch.where(rel_sq > _NEIGHBOR_REL_SPEED_EPS, torch.hypot(nx + du * tcpa, ny + dv * tcpa), ndist)
            neighbor = torch.stack([nx / 100.0, ny / 100.0, (ndist / 100.0).clamp_max(10.0), torch.sin(bearing), torch.cos(bearing), du / 5.0, dv / 5.0, range_rate / 5.0, (tcpa / _NEIGHBOR_TCPA_SCALE_S).clamp_max(10.0), (dcpa / 100.0).clamp_max(10.0)], dim=-1)
            obs[..., spec.neighbor_slice.start:spec.neighbor_slice.start + count * spec.neighbor_dim] = neighbor.reshape(self.n_envs, self.n_tugs, -1)
        return obs.clamp(-10.0, 10.0).to(torch.float32)

    def build_global_state_batched(self) -> torch.Tensor:
        """Build centralized critic state without CPU snapshots."""
        e, t, s = self.episode, self.batch.tugs, self.batch.ships
        total = _GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * self.n_tugs + _GLOBAL_ACCEL_PER_TUG_DIM * self.n_tugs
        out = torch.zeros(self.n_envs, total, device=self.device, dtype=self.dtype)
        out[:, 0], out[:, 1] = s.u / 5.0, s.u_dot / _SHIP_LINEAR_ACCEL_SCALE
        cs, ss = torch.cos(s.psi), torch.sin(s.psi)
        dx, dy = t.eta[..., 0] - s.x[:, None], t.eta[..., 1] - s.y[:, None]
        xb, yb = cs[:, None] * dx + ss[:, None] * dy, -ss[:, None] * dx + cs[:, None] * dy
        ct, st = torch.cos(t.eta[..., 2]), torch.sin(t.eta[..., 2])
        vx, vy = ct * t.nu[..., 0] - st * t.nu[..., 1], st * t.nu[..., 0] + ct * t.nu[..., 1]
        ub, vb = cs[:, None] * vx + ss[:, None] * vy, -ss[:, None] * vx + cs[:, None] * vy
        hd = distance_from_rectangular_hull(t.eta[..., 0], t.eta[..., 1], s.x[:, None], s.y[:, None], s.psi[:, None], float(self.cfg.ship_length_m), float(self.cfg.ship_beam_m))
        blocks = torch.stack([
            xb / 100.0, yb / 100.0, ub / 5.0, vb / 5.0,
            torch.sin(wrap_pi(t.eta[..., 2] - s.psi[:, None])), torch.cos(wrap_pi(t.eta[..., 2] - s.psi[:, None])),
            t.nu[..., 2] / 0.5, t.rpm_actual[..., 0] / self.batch.tug_params.rpm_limit,
            t.rpm_actual[..., 1] / self.batch.tug_params.rpm_limit, t.az_actual_deg[..., 0] / self.batch.tug_params.azimuth_limit_deg,
            t.az_actual_deg[..., 1] / self.batch.tug_params.azimuth_limit_deg, e.last_actions[..., 0], e.last_actions[..., 1],
            e.last_actions[..., 2], e.last_actions[..., 3],
            e.in_zone_steps.to(self.dtype) / max(1, int(round(self.cfg.hold_time_s / self.cfg.dt_ctrl))), hd / 50.0,
        ], dim=-1)
        out[:, _GLOBAL_SHIP_DIM:_GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * self.n_tugs] = blocks.reshape(self.n_envs, -1)
        ax, ay = ct * t.last_nu_dot[..., 0] - st * t.last_nu_dot[..., 1], st * t.last_nu_dot[..., 0] + ct * t.last_nu_dot[..., 1]
        acc = torch.stack([(cs[:, None] * ax + ss[:, None] * ay) / _TUG_LINEAR_ACCEL_SCALE, (-ss[:, None] * ax + cs[:, None] * ay) / _TUG_LINEAR_ACCEL_SCALE, t.last_nu_dot[..., 2] / _TUG_YAW_ACCEL_SCALE], dim=-1)
        out[:, _GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * self.n_tugs:] = acc.reshape(self.n_envs, -1)
        return out.clamp(-10.0, 10.0).to(torch.float32)


def cfg_or(cfg: EnvConfig, name: str) -> Any:
    """Keep type checkers happy around dataclass configuration access."""
    return getattr(cfg, name)
