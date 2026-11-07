"""Panda-specific outer-loop Cartesian admittance helpers."""

from __future__ import annotations

import numpy as np


PANDA_MAX_CARTESIAN_STEP_M = 0.01
PANDA_COMPLIANT_AXIS_STEP_M = 0.0025
PANDA_MAX_FORCE_CORRECTION_M = 0.0003
PANDA_POSITION_BOUNDS_M = np.array([[-0.6, 0.6], [-0.6, 0.6], [0.05, 1.6]])
PANDA_FORCE_RAMP_STEPS = 20


def force_setpoint_for_contact(
    requested_force_n: float,
    force_delta_world: np.ndarray,
    *,
    axis: int = 2,
    contact_threshold_n: float = 1.0,
    contact_latched: bool = False,
    ramp_step: int = 0,
    ramp_steps: int = PANDA_FORCE_RAMP_STEPS,
) -> tuple[float, bool, int]:
    """Return a latched, smoothly-ramped signed force target.

    No force target is applied in free space. Once the signed measured force
    reaches the contact threshold, the contact state latches and the target
    advances to the requested value over control cycles. This state is
    independent of image confidence, so a transient vision dropout cannot
    disable contact compliance.
    """
    force = np.asarray(force_delta_world, dtype=float)
    if force.shape != (3,) or axis not in (0, 1, 2):
        raise ValueError("force must have shape (3,) and axis must be 0, 1, or 2")
    if not np.isfinite(force).all() or not np.isfinite(
        [requested_force_n, contact_threshold_n, ramp_step, ramp_steps]
    ).all() or contact_threshold_n <= 0 or ramp_steps < 1:
        raise ValueError("invalid force ramp parameters")
    direction = np.sign(requested_force_n or -1.0)
    contact_force = direction * float(force[axis])
    engaged = bool(contact_latched or contact_force >= contact_threshold_n)
    if not engaged:
        return 0.0, False, 0
    next_step = min(int(ramp_steps), int(ramp_step) + 1)
    fraction = next_step / float(ramp_steps)
    return float(requested_force_n) * fraction, True, next_step


def compliant_target(
    current_position: np.ndarray,
    requested_position: np.ndarray,
    force_delta_world: np.ndarray,
    *,
    axis: int = 2,
    target_force_n: float = -2.0,
    contact_threshold_n: float = 1.0,
    gain_m_per_n: float = 0.0003,
    max_correction_m: float = PANDA_MAX_FORCE_CORRECTION_M,
    contact_latched: bool = False,
) -> tuple[np.ndarray, float, bool]:
    """Adjust a world-axis target using signed, bias-subtracted force.

    Contact engages above the force threshold and stays engaged while the target
    continues in that direction. It releases when the requested motion reverses.
    """
    current = np.asarray(current_position, dtype=float)
    requested = np.asarray(requested_position, dtype=float)
    force = np.asarray(force_delta_world, dtype=float)
    if current.shape != (3,) or requested.shape != (3,) or force.shape != (3,):
        raise ValueError("positions and world force must each have shape (3,)")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (x), 1 (y), or 2 (z)")
    values = (target_force_n, contact_threshold_n, gain_m_per_n, max_correction_m)
    if not np.isfinite(values).all() or contact_threshold_n <= 0 or gain_m_per_n < 0 or max_correction_m < 0:
        raise ValueError("compliance parameters must be finite and non-negative")

    corrected = requested.copy()
    sensed_force = float(force[axis])
    same_direction_force = np.sign(target_force_n or 1.0) * sensed_force
    requested_axis_delta = float(requested[axis] - current[axis])
    follows_force_direction = requested_axis_delta * np.sign(target_force_n or 1.0) >= 0.0
    active = bool(
        follows_force_direction
        and (contact_latched or same_direction_force >= contact_threshold_n)
    )
    if active:
        correction = float(np.clip(
            gain_m_per_n * (target_force_n - sensed_force),
            -min(max_correction_m, PANDA_COMPLIANT_AXIS_STEP_M),
            min(max_correction_m, PANDA_COMPLIANT_AXIS_STEP_M),
        ))
        corrected[axis] = current[axis] + correction
    else:
        corrected[axis] = current[axis] + float(np.clip(
            requested_axis_delta,
            -PANDA_COMPLIANT_AXIS_STEP_M,
            PANDA_COMPLIANT_AXIS_STEP_M,
        ))

    corrected = np.clip(
        corrected,
        current - PANDA_MAX_CARTESIAN_STEP_M,
        current + PANDA_MAX_CARTESIAN_STEP_M,
    )
    corrected = np.clip(corrected, PANDA_POSITION_BOUNDS_M[:, 0], PANDA_POSITION_BOUNDS_M[:, 1])
    applied_correction = float(corrected[axis] - requested[axis])
    return corrected, applied_correction, active
