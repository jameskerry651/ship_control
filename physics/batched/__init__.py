"""Batched torch physics matching CPU OOP models (parity-first)."""

from physics.batched.geometry import distance_from_rectangular_hull, slot_positions_world, wrap_pi
from physics.batched.ship import BatchedShipState, step_ships
from physics.batched.tugboat import BatchedTugParams, BatchedTugState, step_tugs

__all__ = [
    "wrap_pi",
    "distance_from_rectangular_hull",
    "slot_positions_world",
    "BatchedTugParams",
    "BatchedTugState",
    "step_tugs",
    "BatchedShipState",
    "step_ships",
]
