"""Map policy tool calls to task-specific UR20 actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ur20_screw.contracts import Action, Observation


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
