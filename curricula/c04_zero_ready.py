"""Course 04: zero-ready near/mid route task.

The long rear-lane and opposite-stern starts are intentionally left out of c04:
diagnostics show the current policy solves side/gate/outer zero-ready starts
reliably, while rear/opposite starts need a separate harder stage.
"""

from __future__ import annotations


COURSE = {
    "name": "c04_zero_ready",
    "description": "No ready tugs; near/mid route starts before rear/opposite starts.",
    "total_steps": 1_500_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (0,),
        "tug_init_mixed_zones": ("stern_gate", "side_lane", "outer_slot"),
        "hold_time_s": 2.0,
    },
}
