import base64

import numpy as np
from inspect_robots.types import Action, Observation
from inspect_robots_agent._llm import ToolCall

from inspect_robots_libero.ur20_embodiment import (
    CONTACT_FORCE_THRESHOLD_N,
    COMPLIANT_APPROACH_DISTANCE_M,
    COMPLIANT_FULL_FORCE_DISTANCE_M,
    LOCKED_APPROACH_STEP_M,
    MAX_CARTESIAN_TARGET_DELTA_M,
    XY_ALIGNMENT_TOLERANCE_M,
    UR20ScrewEmbodiment,
    amplify_cartesian_target,
    constrain_locked_approach,
    constrain_uncertain_lateral_target,
    limit_cartesian_target,
    prevent_unaligned_descent,
    force_setpoint_for_vertical_gap,
)
from inspect_robots_agent._tools import build_toolset


def test_force_observations_and_compliance_controls_are_opt_in():
    default = UR20ScrewEmbodiment(output_dir="/tmp/ur20-ft-default")
    force_only = UR20ScrewEmbodiment(output_dir="/tmp/ur20-ft-force", include_force=True)
    compliant = UR20ScrewEmbodiment(
        output_dir="/tmp/ur20-ft-compliant", include_force=True, compliant_control=True
    )
    spiral = UR20ScrewEmbodiment(
        output_dir="/tmp/ur20-ft-spiral", include_force=True,
        compliant_control=True, spiral_search=True
    )
    try:
        assert default.info.action_space.shape == (3,)
        assert force_only.info.action_space.shape == (3,)
        assert compliant.info.action_space.shape == (3,)
        assert spiral.info.action_space.shape == (3,)
        assert default.info.action_space.semantics.control_mode == "eef_abs_pose"
        assert default.info.action_space.semantics.frame == "world"
        assert default.info.action_space.semantics.dim_labels == ("x", "y", "z")
        assert default.info.action_space.semantics.max_step == (0.05, 0.05, 0.05)
        default_fields = {field.key for field in default.info.observation_space.state.fields}
        force_fields = {field.key for field in force_only.info.observation_space.state.fields}
        compliant_fields = {field.key for field in compliant.info.observation_space.state.fields}
        spiral_fields = {field.key for field in spiral.info.observation_space.state.fields}
        assert "force_torque" not in default_fields
        assert {"force_torque", "force_torque_compensated",
                "force_torque_peak_compensated"} <= force_fields
        assert "force_control_state" in compliant_fields
        assert "spiral_search_state" in spiral_fields
        assert "force_torque_compensated" in spiral_fields
        assert "spiral_axial_force_delta_n" in spiral_fields
        assert "spiral_tip_z_displacement_m" not in spiral_fields
        assert "spiral_tactile_state" in spiral_fields
        assert {"eef_pos", "vision_screw_center_px", "vision_screw_xy_estimate_m",
                "vision_xy_error_m", "vision_confidence", "joint_pos", "joint_vel"} <= default_fields
        assert not {"qpos", "qvel", "sites", "task_state"} & default_fields
        compliant_tools = build_toolset(
            compliant.info.action_space, compliant.info.observation_space,
            compliant.info.control_hz,
        )
        spiral_tools = build_toolset(
            spiral.info.action_space, spiral.info.observation_space,
            spiral.info.control_hz,
        )
        assert compliant_tools._move_tool == "move_to"
        assert spiral_tools._move_tool == "move_to"
        assert compliant_tools._state_key == "eef_pos"
        assert spiral_tools._state_key == "eef_pos"
        schema = compliant_tools.schemas()[0]["function"]
        assert schema["name"] == "move_to"
        assert "targets" in schema["parameters"]["properties"]
        assert "x, y, z" in schema["parameters"]["properties"]["targets"]["description"]
    finally:
        default.close()
        force_only.close()
        compliant.close()
        spiral.close()


def test_cartesian_target_is_limited_to_five_centimeters_without_forcing_minimum():
    origin = np.array([0.0, 0.0, 0.0])
    far_target = np.array([0.10, 0.10, 0.0])
    limited = limit_cartesian_target(origin, far_target)
    assert np.isclose(np.linalg.norm(limited - origin), MAX_CARTESIAN_TARGET_DELTA_M)
    small_target = np.array([0.01, 0.0, 0.0])
    np.testing.assert_allclose(limit_cartesian_target(origin, small_target), small_target)


def test_low_confidence_vision_freezes_lateral_motion_but_keeps_z_target():
    current = np.array([0.8016, 1.0008, 0.1617])
    target = np.array([0.8063, 1.0048, 0.1530])

    constrained = constrain_uncertain_lateral_target(current, target, 0.5764)
    np.testing.assert_allclose(constrained, [current[0], current[1], target[2]])
    np.testing.assert_allclose(
        constrain_uncertain_lateral_target(current, target, 0.8), target
    )


def test_xy_alignment_is_recomputed_from_each_confident_two_millimeter_observation():
    aligned = {"vision_confidence": 0.9, "vision_xy_error_m": [0.0012, 0.0012]}
    boundary = {"vision_confidence": 0.9, "vision_xy_error_m": [0.002, 0.0]}
    outside = {"vision_confidence": 0.9, "vision_xy_error_m": [0.0021, 0.0]}
    uncertain = {"vision_confidence": 0.4, "vision_xy_error_m": [0.0, 0.0]}

    assert UR20ScrewEmbodiment._vision_xy_aligned(aligned)
    assert UR20ScrewEmbodiment._vision_xy_aligned(boundary)
    assert not UR20ScrewEmbodiment._vision_xy_aligned(outside)
    assert not UR20ScrewEmbodiment._vision_xy_aligned(uncertain)
    assert XY_ALIGNMENT_TOLERANCE_M == 0.002

    # The lock is a per-observation deadband, not a one-way latch.
    xy_locked = UR20ScrewEmbodiment._vision_xy_aligned(aligned)
    assert xy_locked
    xy_locked = UR20ScrewEmbodiment._vision_xy_aligned(outside)
    assert not xy_locked
    xy_locked = UR20ScrewEmbodiment._vision_xy_aligned(aligned)
    assert xy_locked


def test_spiral_search_contact_gate_requires_axial_force_threshold():
    no_contact = {"force_torque_compensated": [0.0, 0.0, 0.99, 0.0, 0.0, 0.0]}
    axial_contact = {"force_torque_compensated": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    lateral_contact = {"force_torque_compensated": [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]}

    assert not UR20ScrewEmbodiment._has_axial_contact(no_contact)
    assert UR20ScrewEmbodiment._has_axial_contact(axial_contact)
    assert not UR20ScrewEmbodiment._has_axial_contact(lateral_contact)
    assert CONTACT_FORCE_THRESHOLD_N == 1.0


def test_contact_with_lost_vision_keeps_force_and_leaves_spiral_invocation_to_model(tmp_path):
    image = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x02\x00"
        b"\x00\x05\x00\x01\xa5\xf6E\x40\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode()

    def worker_result(steps, spiral_state, engaged=False, stable=False):
        return {
            "steps": steps,
            "time": steps * 0.005,
            "images": {"agentview": image, "wristview": image},
            "qpos": [0.0] * 9,
            "qvel": [0.0] * 9,
            "sites": {"tool_tip": [0.8, 1.0, 0.16]},
            "vision_screw_center_px": [128.0, 128.0],
            "vision_screw_xy_estimate_m": [0.81, 1.0],
            "vision_screw_top_z_estimate_m": 0.153,
            "vision_xy_error_m": [0.01, 0.0],
            "vision_confidence": 0.0,
            "force_torque": [0.0] * 6,
            "force_torque_compensated": [0.0, 0.0, 1.2, 0.0, 0.0, 0.0],
            "force_torque_peak_compensated": [0.0, 0.0, 1.2, 0.0, 0.0, 0.0],
            "force_control_state": [1.0, 2.0, 0.0],
            "force_control_active": True,
            "force_control_target_n": 2.0,
            "force_control_offset_m": 0.0,
            "spiral_search_state": spiral_state,
            "socket_engaged": engaged,
            "socket_engagement_stable": stable,
            "socket_contact": True,
            "socket_axial_error": 0.0,
            "socket_radial_error": 0.01,
            "tool_axis_alignment": 1.0,
            "screw_rotation": 0.0,
            "screw_lift": 0.0,
            "engaged_seen": False,
        }

    embodiment = UR20ScrewEmbodiment(
        output_dir=tmp_path, include_force=True, compliant_control=True,
        spiral_search=True, action_amplification=1.0,
    )
    embodiment._last_tool_tip = np.array([0.8, 1.0, 0.183])
    embodiment._last_vision_confidence = 0.0
    embodiment._last_vision_xy_error = np.array([0.01, 0.0])
    embodiment._last_screw_top_z_estimate = 0.153
    embodiment._last_screw_xy_estimate = np.array([0.8, 1.0])
    embodiment._force_proximity_latched = True
    embodiment._last_axial_contact = False
    embodiment._xy_alignment_locked = False
    embodiment.instruction = "seat socket"
    responses = [worker_result(25, [0.0, 0.0, 0.0]),
                 worker_result(50, [0.0, 0.0, 0.0])]
    rpc_requests = []

    def fake_rpc(request):
        rpc_requests.append(request)
        return responses.pop(0)

    embodiment._rpc = fake_rpc
    try:
        result = embodiment.step(Action(data=np.array([0.8, 1.0, 0.16])))
        assert np.isclose(rpc_requests[0]["target_force_n"], 1.0)
        result = embodiment.step(Action(data=np.array([0.8, 1.0, 0.16])))
        assert [request["spiral_search"] for request in rpc_requests] == [False, False]
        assert rpc_requests[1]["target_force_n"] == 2.0
        assert not result.terminated
        assert not result.info["xy_alignment_locked"]
        assert not result.info["success"]
        assert result.info["spiral_internal_increments"] == 0
        assert result.info["spiral_internal_increments_total"] == 0
        assert embodiment._contact_seating_latched
    finally:
        embodiment.close()


def test_spiral_search_tool_gate_uses_compensated_axial_force():
    embodiment = UR20ScrewEmbodiment(
        output_dir="/tmp/ur20-ft-spiral-tool-gate", include_force=True,
        compliant_control=True, spiral_search=True,
    )
    toolset = build_toolset(
        embodiment.info.action_space, embodiment.info.observation_space,
        embodiment.info.control_hz,
    )
    toolset.enable_ur20_spiral_search()
    call = ToolCall(
        id="spiral-1", name="spiral_search", arguments='{"note":"start search"}'
    )
    try:
        rejected = toolset.execute(
            call,
            Observation(state={
                "eef_pos": np.array([0.8, 1.0, 0.17]),
                "force_torque_compensated": np.zeros(6),
            }),
        )
        assert rejected.chunk is None
        assert "No simulation step was advanced" in rejected.error
        assert "Continue the compliant downward approach" in rejected.error

        accepted = toolset.execute(
            call,
            Observation(state={
                "eef_pos": np.array([0.8, 1.0, 0.16]),
                "force_torque_compensated": np.array([0.0, 0.0, 2.209, 0.0, 0.0, 0.0]),
            }),
        )
        assert accepted.error is None
        assert accepted.chunk is not None
        assert accepted.chunk.actions[0].meta["ur20_spiral_search"] is True
        assert accepted.chunk.actions[0].meta["ur20_axial_contact_force_n"] == 2.209
    finally:
        embodiment.close()


def test_locked_xy_enforces_downward_progress_unless_force_guarded():
    current = np.array([0.8, 1.0, 0.16])
    lateral_or_stationary = np.array([0.803, 1.002, 0.16])
    constrained = constrain_locked_approach(current, lateral_or_stationary)
    np.testing.assert_allclose(constrained[:2], current[:2])
    assert np.isclose(constrained[2], current[2] - LOCKED_APPROACH_STEP_M)

    guarded = constrain_locked_approach(current, np.array([0.8, 1.0, 0.18]), force_guarded=True)
    assert guarded[2] == 0.18


def test_confident_visual_misalignment_blocks_descent_but_allows_xy_correction():
    current = np.array([0.79, 1.0, 0.16])
    target = np.array([0.803, 1.001, 0.11])
    constrained = prevent_unaligned_descent(current, target, 0.9, np.array([0.014, 0.0]))
    np.testing.assert_allclose(constrained, [0.803, 1.001, 0.16])

    aligned = prevent_unaligned_descent(current, target, 0.9, np.array([0.001, 0.001]))
    np.testing.assert_allclose(aligned, target)

    uncertain = prevent_unaligned_descent(current, target, 0.3, np.array([0.014, 0.0]))
    np.testing.assert_allclose(uncertain, target)


def test_compliance_smoothly_ramps_from_zero_to_two_newtons_near_surface():
    surface_z = 0.153
    assert force_setpoint_for_vertical_gap(0.21, surface_z, True) == 0.0
    assert force_setpoint_for_vertical_gap(0.203, surface_z, True) == 0.0
    assert np.isclose(force_setpoint_for_vertical_gap(0.183, surface_z, True), 1.0)
    assert force_setpoint_for_vertical_gap(0.163, surface_z, True) == 2.0
    assert force_setpoint_for_vertical_gap(0.21, surface_z, True, contact_detected=True) == 2.0
    assert force_setpoint_for_vertical_gap(0.18, surface_z, False) == 0.0
    assert COMPLIANT_APPROACH_DISTANCE_M == 0.05
    assert COMPLIANT_FULL_FORCE_DISTANCE_M == 0.01


def test_action_amplification_increases_requested_motion_before_safety_limit():
    origin = np.array([0.0, 0.0, 0.0])
    small_target = np.array([0.001, 0.0, 0.0])
    amplified = amplify_cartesian_target(origin, small_target, 3.0)
    np.testing.assert_allclose(amplified, [0.003, 0.0, 0.0])
    limited = limit_cartesian_target(origin, amplify_cartesian_target(
        origin, np.array([0.03, 0.0, 0.0]), 3.0))
    np.testing.assert_allclose(limited, [MAX_CARTESIAN_TARGET_DELTA_M, 0.0, 0.0])
