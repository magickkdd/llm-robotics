from __future__ import annotations

import math

import numpy as np

SPIRAL_POINTS = 576
SPIRAL_MAX_RADIUS_M = 0.004
SPIRAL_TURNS = 16.0 / 3.0
SPIRAL_COVERAGE_SWEEPS = 4
SPIRAL_SEAT_TRAVEL_M = 0.003
SPIRAL_SEAT_FORCE_DROP_N = 0.75
SPIRAL_SEAT_CONFIRM_STEPS = 20
SPIRAL_MAX_FORCE_N = 20.0
SPIRAL_MAX_TILT_RAD = math.radians(8.0)


def spiral_target(center_xy: np.ndarray, index: int) -> np.ndarray:
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("spiral center must be a finite XY vector")
    if index < 0:
        raise ValueError("spiral index must be non-negative")
    radius = SPIRAL_MAX_RADIUS_M * min(index + 1, SPIRAL_POINTS) / SPIRAL_POINTS
    angle = index * 2.0 * math.pi * SPIRAL_TURNS / SPIRAL_POINTS
    return center + radius * np.array([math.cos(angle), math.sin(angle)])


def seated_from_sensor_trace(axial_force: list[float], z_positions: list[float],
                             baseline_force: float) -> bool:
    if len(axial_force) < SPIRAL_SEAT_CONFIRM_STEPS or len(z_positions) < SPIRAL_SEAT_CONFIRM_STEPS:
        return False
    travel = z_positions[0] - z_positions[-1]
    force_drop = baseline_force - min(axial_force)
    stable_force = abs(axial_force[-1]) >= max(0.2, abs(baseline_force) * 0.1)
    return (travel >= SPIRAL_SEAT_TRAVEL_M and force_drop >= SPIRAL_SEAT_FORCE_DROP_N
            and stable_force)
