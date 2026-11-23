"""Map policy tool calls to task-specific UR20 actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ur20_screw.contracts import Action, Observation


def sequence_targets(arguments: dict[str, Any]) -> list[dict[str, float]]:
    """Validate the bounded waypoint segment used by the action-stream tool."""
    note = arguments.get("note")
    waypoints = arguments.get("waypoints")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("note is required; no simulation step was advanced")
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 4:
        raise ValueError("waypoints must contain between one and four targets")
    validated = []
    for waypoint in waypoints:
        targets = waypoint.get("targets") if isinstance(waypoint, dict) else None
        if not isinstance(targets, dict) or not targets:
            raise ValueError("each waypoint must contain one or more of x, y, z")
        if set(targets) - {"x", "y", "z"}:
            raise ValueError("waypoint targets may contain only x, y, z")
        values = {axis: float(value) for axis, value in targets.items()}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("waypoint coordinates must be finite")
        validated.append(values)
    return validated


def action_for(name: str, arguments: dict[str, Any], observation: Observation) -> Action | None:
    current = np.asarray(observation.state["eef_pos"], dtype=float)
    if name == "move_to":
        if not isinstance(arguments.get("note"), str) or not arguments["note"].strip():
            raise ValueError("note is required; no simulation step was advanced")
        targets = arguments.get("targets")
        if not isinstance(targets, dict) or not targets:
            raise ValueError("targets must contain one or more of x, y, z")
        target = current.copy()
        for index, axis in enumerate(("x", "y", "z")):
            if axis in targets:
                target[index] = float(targets[axis])
        return Action(target)
    if name == "spiral_search":
        if not isinstance(arguments.get("note"), str) or not arguments["note"].strip():
            raise ValueError("note is required; no simulation step was advanced")
        force = np.asarray(observation.state.get("force_torque_compensated", np.zeros(6)))
        if force.shape != (6,) or abs(float(force[2])) < 1.0:
            raise ValueError("spiral_search rejected: axial contact is not confirmed; no simulation step was advanced")
        return Action(current, {"ur20_spiral_search": True,
                               "ur20_axial_contact_force_n": float(abs(force[2]))})
    if name == "turn_socket":
        if not isinstance(arguments.get("note"), str) or not arguments["note"].strip():
            raise ValueError("note is required; no simulation step was advanced")
        tactile = np.asarray(observation.state.get("spiral_tactile_state", []), dtype=float)
        if tactile.shape != (7,) or tactile[1] < 0.5:
            raise ValueError("turn_socket rejected: tactile seating is not confirmed; no simulation step was advanced")
        return Action(current, {"ur20_turn_socket": True})
    if name == "done":
        return None
    raise ValueError(f"unknown UR20 tool {name!r}")
