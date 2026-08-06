"""Batched env state container and CPU↔GPU sync helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from physics.batched.ship import BatchedShipState
from physics.batched.tugboat import BatchedTugParams, BatchedTugState
from physics.tugboat_dynamics_model import Vec3


@dataclass
class GpuEnvBatch:
    """Device-resident dynamics state for N envs × K tugs."""

    tugs: BatchedTugState
    ships: BatchedShipState
    tug_params: BatchedTugParams
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def create(
        cls,
        n_envs: int,
        n_tugs: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "GpuEnvBatch":
        return cls(
            tugs=BatchedTugState.zeros(n_envs, n_tugs, device=device, dtype=dtype),
            ships=BatchedShipState.zeros(n_envs, device=device, dtype=dtype),
            tug_params=BatchedTugParams.from_model(),
            device=device,
            dtype=dtype,
        )


def pull_env_to_gpu(env, batch: GpuEnvBatch, env_idx: int) -> None:
    """Copy one ``FormationEnv`` dynamics state onto the GPU batch row."""
    i = env_idx
    for k, tug in enumerate(env.tugs):
        batch.tugs.eta[i, k, 0] = tug.eta.x
        batch.tugs.eta[i, k, 1] = tug.eta.y
        batch.tugs.eta[i, k, 2] = tug.eta.z
        batch.tugs.nu[i, k, 0] = tug.nu.x
        batch.tugs.nu[i, k, 1] = tug.nu.y
        batch.tugs.nu[i, k, 2] = tug.nu.z
        batch.tugs.rpm_cmd[i, k, 0] = tug._port_rpm_cmd
        batch.tugs.rpm_cmd[i, k, 1] = tug._starboard_rpm_cmd
        batch.tugs.az_cmd_deg[i, k, 0] = tug._port_azimuth_cmd_deg
        batch.tugs.az_cmd_deg[i, k, 1] = tug._starboard_azimuth_cmd_deg
        batch.tugs.rpm_actual[i, k, 0] = tug._port_rpm_actual
        batch.tugs.rpm_actual[i, k, 1] = tug._starboard_rpm_actual
        batch.tugs.az_actual_deg[i, k, 0] = tug._port_azimuth_actual_deg
        batch.tugs.az_actual_deg[i, k, 1] = tug._starboard_azimuth_actual_deg
        tau = tug.get_last_tau()
        nd = tug.get_last_nu_dot()
        batch.tugs.last_tau[i, k, 0] = tau.x
        batch.tugs.last_tau[i, k, 1] = tau.y
        batch.tugs.last_tau[i, k, 2] = tau.z
        batch.tugs.last_nu_dot[i, k, 0] = nd.x
        batch.tugs.last_nu_dot[i, k, 1] = nd.y
        batch.tugs.last_nu_dot[i, k, 2] = nd.z

    ship = env.ship
    batch.ships.x[i] = ship.x
    batch.ships.y[i] = ship.y
    batch.ships.psi[i] = ship.psi
    batch.ships.u[i] = ship.u
    batch.ships.v[i] = ship.v
    batch.ships.r[i] = ship.r
    batch.ships.u_dot[i] = ship.u_dot
    batch.ships.v_dot[i] = ship.v_dot
    batch.ships.r_dot[i] = ship.r_dot
    batch.ships.u_target[i] = ship._u_target
    batch.ships.r_target[i] = ship._r_target
    batch.ships.time_to_resample[i] = ship._time_to_resample


def push_env_from_gpu(env, batch: GpuEnvBatch, env_idx: int) -> None:
    """Write GPU batch row back into one ``FormationEnv`` (post-dynamics)."""
    i = env_idx
    eta = batch.tugs.eta[i].detach().cpu().numpy()
    nu = batch.tugs.nu[i].detach().cpu().numpy()
    rpm_cmd = batch.tugs.rpm_cmd[i].detach().cpu().numpy()
    az_cmd = batch.tugs.az_cmd_deg[i].detach().cpu().numpy()
    rpm_a = batch.tugs.rpm_actual[i].detach().cpu().numpy()
    az_a = batch.tugs.az_actual_deg[i].detach().cpu().numpy()
    tau = batch.tugs.last_tau[i].detach().cpu().numpy()
    nd = batch.tugs.last_nu_dot[i].detach().cpu().numpy()

    for k, tug in enumerate(env.tugs):
        tug.eta = Vec3(float(eta[k, 0]), float(eta[k, 1]), float(eta[k, 2]))
        tug.nu = Vec3(float(nu[k, 0]), float(nu[k, 1]), float(nu[k, 2]))
        tug._port_rpm_cmd = float(rpm_cmd[k, 0])
        tug._starboard_rpm_cmd = float(rpm_cmd[k, 1])
        tug._port_azimuth_cmd_deg = float(az_cmd[k, 0])
        tug._starboard_azimuth_cmd_deg = float(az_cmd[k, 1])
        tug._port_rpm_actual = float(rpm_a[k, 0])
        tug._starboard_rpm_actual = float(rpm_a[k, 1])
        tug._port_azimuth_actual_deg = float(az_a[k, 0])
        tug._starboard_azimuth_actual_deg = float(az_a[k, 1])
        tug._last_tau = Vec3(float(tau[k, 0]), float(tau[k, 1]), float(tau[k, 2]))
        tug._last_nu_dot = Vec3(float(nd[k, 0]), float(nd[k, 1]), float(nd[k, 2]))

    ship = env.ship
    ship.x = float(batch.ships.x[i].detach().cpu())
    ship.y = float(batch.ships.y[i].detach().cpu())
    ship.psi = float(batch.ships.psi[i].detach().cpu())
    ship.u = float(batch.ships.u[i].detach().cpu())
    ship.v = float(batch.ships.v[i].detach().cpu())
    ship.r = float(batch.ships.r[i].detach().cpu())
    ship.u_dot = float(batch.ships.u_dot[i].detach().cpu())
    ship.v_dot = float(batch.ships.v_dot[i].detach().cpu())
    ship.r_dot = float(batch.ships.r_dot[i].detach().cpu())
    ship._u_target = float(batch.ships.u_target[i].detach().cpu())
    ship._r_target = float(batch.ships.r_target[i].detach().cpu())
    ship._time_to_resample = float(batch.ships.time_to_resample[i].detach().cpu())
