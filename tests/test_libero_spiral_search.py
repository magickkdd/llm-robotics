import json

import numpy as np
import pytest
from inspect_robots.types import Observation
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import build_toolset

from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.spiral import spiral_target, seated_from_sensor_trace


def _observation(engaged=True):
    return Observation(
        images={},
        state={
            "eef_state": np.zeros(10),
            "gripper_qpos": np.zeros(2),
            "compliance_state": np.array([float(engaged), 20.0]),
        },
        instruction="", state_time=0.0, extra={},
    )


def test_spiral_points_are_small_and_cover_bounded_radius():
    origin = np.zeros(2)
    points = np.array([spiral_target(origin, index) for index in range(576)])
    assert np.linalg.norm(points, axis=1).max() <= 0.004
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() < 0.001


def test_seating_requires_force_drop_travel_and_confirmation_window():
    force = [1.0] * 20
    z = [0.0 - index * 0.0002 for index in range(20)]
    assert not seated_from_sensor_trace(force, z, 1.0)
    force[5] = 0.1
    force[-1] = 0.5
    assert seated_from_sensor_trace(force, z, 1.0)


def test_libero_spiral_is_opt_in_and_requires_force_compliance():
    with pytest.raises(Exception, match="requires force_feedback and compliant_control"):
        LiberoEmbodiment(spiral_search=True)
    embodiment = LiberoEmbodiment(
        force_feedback=True, compliant_control=True, compliant_target_force_n=-1.0,
        spiral_search=True,
    )
    assert "atomic_spiral_search" in embodiment.info.capabilities
    assert "spiral_search_state" in {field.key for field in embodiment.info.observation_space.state.fields}


def test_libero_agent_exposes_atomic_spiral_tool_and_contact_gate():
    embodiment = LiberoEmbodiment(
        force_feedback=True, compliant_control=True, compliant_target_force_n=-1.0,
        spiral_search=True,
    )
    toolset = build_toolset(
        embodiment.info.action_space, embodiment.info.observation_space,
        embodiment.info.control_hz,
    )
    toolset.enable_libero_spiral_search()
    assert "spiral_search" in {schema["function"]["name"] for schema in toolset.schemas()}
    call = ToolCall(id="spiral", name="spiral_search", arguments=json.dumps({"note": "contact"}))
    rejected = toolset.execute(call, _observation(engaged=False))
    assert rejected.error and "contact is not confirmed" in rejected.error
    accepted = toolset.execute(call, _observation(engaged=True))
    assert accepted.chunk is not None
    assert accepted.chunk.actions[0].meta["libero_spiral_search"] is True
    assert accepted.chunk.actions[0].meta["chunk_final"] is True
