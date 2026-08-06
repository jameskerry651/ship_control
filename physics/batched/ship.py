"""Batched large-ship kinematics matching ``LargeShipModel``."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BatchedShipState:
    x: torch.Tensor  # (N,)
    y: torch.Tensor
    psi: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor
    u_dot: torch.Tensor
    v_dot: torch.Tensor
    r_dot: torch.Tensor
    u_target: torch.Tensor
    r_target: torch.Tensor
    time_to_resample: torch.Tensor

    @classmethod
    def zeros(
        cls,
        n: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "BatchedShipState":
        device = device or torch.device("cpu")
        z = torch.zeros(n, device=device, dtype=dtype)
        o = torch.ones(n, device=device, dtype=dtype)
        return cls(
            x=z.clone(),
            y=z.clone(),
            psi=z.clone(),
            u=o.clone(),
            v=z.clone(),
            r=z.clone(),
            u_dot=z.clone(),
            v_dot=z.clone(),
            r_dot=z.clone(),
            u_target=o.clone(),
            r_target=z.clone(),
            time_to_resample=torch.full(
                (n,), 25.0, device=device, dtype=torch.float64
            ),
        )


def step_ships(
    state: BatchedShipState,
    dt: float,
    *,
    speed_min: float,
    speed_max: float,
    speed_tau: float,
    target_resample_min_s: float,
    target_resample_max_s: float,
    u_target_samples: torch.Tensor | None = None,
    resample_interval_samples: torch.Tensor | None = None,
) -> None:
    """Advance ships by ``dt``.

    When resample triggers, optional pre-sampled tensors supply RNG values so
    parity tests can replay NumPy draws. Shapes: (N,) for each sample tensor.
    If samples are None, uses torch.rand on the state device (non-parity path).
    """
    state.time_to_resample = state.time_to_resample - dt
    need = state.time_to_resample <= 0.0
    if bool(need.any()):
        n = int(state.x.shape[0])
        if u_target_samples is None:
            u_new = torch.rand(n, device=state.x.device, dtype=state.x.dtype) * (
                speed_max - speed_min
            ) + speed_min
        else:
            u_new = u_target_samples
        if resample_interval_samples is None:
            t_new = torch.rand(
                n,
                device=state.x.device,
                dtype=state.time_to_resample.dtype,
            ) * (target_resample_max_s - target_resample_min_s) + target_resample_min_s
        else:
            t_new = resample_interval_samples
        state.u_target = torch.where(need, u_new, state.u_target)
        state.r_target = torch.where(need, torch.zeros_like(state.r_target), state.r_target)
        state.time_to_resample = torch.where(need, t_new, state.time_to_resample)

    state.u_dot = (state.u_target - state.u) / max(speed_tau, 1e-3)
    state.u = state.u + state.u_dot * dt
    state.v_dot = torch.zeros_like(state.v_dot)
    state.r_dot = torch.zeros_like(state.r_dot)
    state.r = torch.zeros_like(state.r)

    cos_p = torch.cos(state.psi)
    sin_p = torch.sin(state.psi)
    state.x = state.x + (cos_p * state.u - sin_p * state.v) * dt
    state.y = state.y + (sin_p * state.u + cos_p * state.v) * dt
