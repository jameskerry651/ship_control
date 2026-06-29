"""Shared helpers for building course dictionaries."""

from __future__ import annotations

from typing import Any


def make_course(
    *,
    name: str,
    description: str,
    total_steps: int,
    env_overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "total_steps": total_steps,
        "env_overrides": env_overrides,
    }


def merge_overrides(*layers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer)
    return merged
