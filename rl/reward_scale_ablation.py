"""Helpers for reward-scale collision ablation (naming + 1M promotion)."""

from __future__ import annotations

RSC_PRESET_ORDER: list[str] = [
    "rsc_baseline",
    "rsc_dist_soft",
    "rsc_coll_mid",
    "rsc_coll_hi",
    "rsc_balanced",
    "rsc_corridor_hard",
]


def rsc_short_name(preset_id: str) -> str:
    key = str(preset_id).strip()
    if not key.startswith("rsc_"):
        raise ValueError(f"preset id must start with 'rsc_', got {preset_id!r}")
    return key[len("rsc_") :]


def rsc_run_name(preset_id: str, *, phase: str) -> str:
    phase_norm = str(phase).strip().lower()
    if phase_norm not in {"1m", "5m"}:
        raise ValueError(f"phase must be '1m' or '5m', got {phase!r}")
    return f"rsc_{phase_norm}_{rsc_short_name(preset_id)}"


def select_rsc_promotions(
    metrics_by_short: dict[str, dict[str, float]],
    *,
    max_promote: int = 2,
) -> list[str]:
    if "baseline" not in metrics_by_short:
        raise ValueError("metrics_by_short must include 'baseline'")
    if max_promote < 0:
        raise ValueError("max_promote must be >= 0")

    base = metrics_by_short["baseline"]
    base_dist = float(base["final_dist_mean"])
    base_coll = float(base["collision_rate"])

    qualified: list[tuple[float, float, float, str]] = []
    for short, m in metrics_by_short.items():
        if short == "baseline":
            continue
        cap = float(m["capture_rate"])
        dist = float(m["final_dist_mean"])
        coll = float(m["collision_rate"])
        coll_delta = base_coll - coll
        dist_ok = dist < 200.0 or dist <= base_dist + 20.0
        coll_ok = coll_delta >= 0.15
        if cap > 0.0 or (dist_ok and coll_ok):
            # sort key: capture first (0 if capt, 1 else), -coll_delta, dist
            qualified.append(
                (0.0 if cap > 0.0 else 1.0, -coll_delta, dist, short)
            )

    qualified.sort()
    return [short for *_rest, short in qualified[:max_promote]]
