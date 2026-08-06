"""Batched geometry helpers (torch), matching CPU physics conventions."""

from __future__ import annotations

import torch


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    """Map angle to (-pi, pi]."""
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def distance_from_rectangular_hull(
    x_world: torch.Tensor,
    y_world: torch.Tensor,
    ship_x: torch.Tensor,
    ship_y: torch.Tensor,
    ship_psi: torch.Tensor,
    length_m: float,
    beam_m: float,
) -> torch.Tensor:
    """External distance to rectangular hull; 0 when inside. Broadcast-friendly."""
    dx = x_world - ship_x
    dy = y_world - ship_y
    cos_p = torch.cos(ship_psi)
    sin_p = torch.sin(ship_psi)
    x_b = cos_p * dx + sin_p * dy
    y_b = -sin_p * dx + cos_p * dy
    ex = (x_b.abs() - length_m / 2.0).clamp(min=0.0)
    ey = (y_b.abs() - beam_m / 2.0).clamp(min=0.0)
    return torch.hypot(ex, ey)


def slot_positions_world(
    ship_x: torch.Tensor,
    ship_y: torch.Tensor,
    ship_psi: torch.Tensor,
    length_m: float,
    beam_m: float,
    slot_lon_offset_m: float,
    slot_lat_offset_m: float,
) -> torch.Tensor:
    """Return slots (N, 4, 3) = [x, y, psi] in world frame."""
    # body offsets (4, 2)
    L = length_m / 2.0
    lon = slot_lon_offset_m
    lat = slot_lat_offset_m + beam_m / 2.0
    body = ship_x.new_tensor(
        [
            [+L + lon, -lat],
            [+L + lon, +lat],
            [-L - lon, -lat],
            [-L - lon, +lat],
        ]
    )  # (4, 2)
    cos_p = torch.cos(ship_psi)
    sin_p = torch.sin(ship_psi)
    # world = R @ body; R = [[c,-s],[s,c]]
    # wx = c*bx - s*by; wy = s*bx + c*by
    bx = body[:, 0]  # (4,)
    by = body[:, 1]
    # broadcast: (N,1) * (4,)
    wx = cos_p.unsqueeze(-1) * bx - sin_p.unsqueeze(-1) * by + ship_x.unsqueeze(-1)
    wy = sin_p.unsqueeze(-1) * bx + cos_p.unsqueeze(-1) * by + ship_y.unsqueeze(-1)
    psi = ship_psi.unsqueeze(-1).expand(-1, 4)
    return torch.stack([wx, wy, psi], dim=-1)
