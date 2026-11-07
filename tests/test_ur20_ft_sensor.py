import mujoco
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


SCENE = "assets/ur20_screw_station/scene.xml"
VERTICAL_IK_SEED = np.array(
    [-2.44392306, -1.04459098, 1.15111654, -1.67729567, -1.57079662, -0.87312675]
)


def _sensor_reading(model, data):
    force_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_force")
    torque_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_torque")
    force_adr = model.sensor_adr[force_id]
    torque_adr = model.sensor_adr[torque_id]
    return np.r_[data.sensordata[force_adr:force_adr + 3],
                 data.sensordata[torque_adr:torque_adr + 3]].copy()


def _move_tip_kinematically(model, data, target_position, target_rotation):
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    for _ in range(250):
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[tip_id].reshape(3, 3)
        position_error = target_position - data.site_xpos[tip_id]
        rotation_error = 0.5 * sum(
            (np.cross(rotation[:, axis], target_rotation[:, axis]) for axis in range(3)),
            np.zeros(3),
        )
        if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 1e-3:
            return
        position_jacobian = np.zeros((3, model.nv))
        rotation_jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, position_jacobian, rotation_jacobian, tip_id)
        jacobian = np.vstack((position_jacobian[:, :6], rotation_jacobian[:, :6] * 0.2))
        error = np.r_[position_error, rotation_error * 0.2]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-4 * np.eye(6), error
        )
        data.qpos[:6] += np.clip(delta, -0.08, 0.08)
    raise AssertionError("test IK failed to reach target")


def test_flange_ft_sensor_axes_and_contact_response():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:6] = VERTICAL_IK_SEED
    mujoco.mj_forward(model, data)

    force_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_force")
    torque_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_torque")
    assert model.sensor_dim[force_id] == 3
    assert model.sensor_dim[torque_id] == 3

    sensor_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    sensor_rotation = data.site_xmat[sensor_site].reshape(3, 3)
    tip_rotation = data.site_xmat[tip_site].reshape(3, 3)
    assert np.dot(sensor_rotation[:, 2], tip_rotation[:, 0]) > 0.999

    free_space_wrench = _sensor_reading(model, data)
    assert free_space_wrench.shape == (6,)
    assert np.isfinite(free_space_wrench).all()

    screw_head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
    socket_walls = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"socket_wall_{index}")
        for index in range(6)
    }
    head_position = data.geom_xpos[screw_head].copy()
    start_rotation = tip_rotation.copy()
    assert not any(
        (contact.geom1 == screw_head and contact.geom2 in socket_walls)
        or (contact.geom2 == screw_head and contact.geom1 in socket_walls)
        for contact in data.contact[:data.ncon]
    )

    _move_tip_kinematically(
        model, data, head_position + np.array([0.008, 0.0, 0.0]), start_rotation
    )
    mujoco.mj_forward(model, data)
    assert any(
        (contact.geom1 == screw_head and contact.geom2 in socket_walls)
        or (contact.geom2 == screw_head and contact.geom1 in socket_walls)
        for contact in data.contact[:data.ncon]
    )
    contact_wrench = _sensor_reading(model, data)
    assert np.linalg.norm(contact_wrench[:3] - free_space_wrench[:3]) > 1.0
    assert np.linalg.norm(contact_wrench[3:] - free_space_wrench[3:]) > 0.1


def test_worker_reports_contact_compensated_wrench_and_interval_peak():
    import ur20_screw_worker
    import view_ur20_socket_start

    original_offsets = ur20_screw_worker.INITIAL_OFFSETS
    original_rotations = ur20_screw_worker.INITIAL_ROTATIONS
    ur20_screw_worker.INITIAL_OFFSETS = ((0.001, 0.0),)
    ur20_screw_worker.INITIAL_ROTATIONS = (np.deg2rad(30.0),)
    worker = ur20_screw_worker.Worker()
    try:
        observation = worker.reset(0, include_force=True)
        pose = view_ur20_socket_start.set_socket_start_pose(
            worker, 0.001, 30.0, 0.0005
        )
        assert abs(pose["center_offset_xy_mm"][0] - 1.0) < 0.01
        assert abs(pose["center_offset_xy_mm"][1] - 0.5) < 0.01
        assert abs(pose["opening_plane_height_m"] - pose["screw_head_top_height_m"]) < 1e-5
        assert pose["socket_wall_contact_at_pose"]
        assert abs(observation["force_torque_compensated"][2]) < 1e-6

        worker.osc_target_position = worker.data.site_xpos[worker.tool_tip_id].copy()
        preload = worker._run_control_steps(500, True, 2.0, 0.0)
        assert preload["force_control_active"]
        assert 1.0 <= abs(preload["force_torque_compensated"][2]) <= 3.0
    finally:
        worker.renderer.close()
        ur20_screw_worker.INITIAL_OFFSETS = original_offsets
        ur20_screw_worker.INITIAL_ROTATIONS = original_rotations


def test_force_guard_stops_inside_control_batch():
    import ur20_screw_worker

    worker = ur20_screw_worker.Worker()
    original_contact_wrench = worker._contact_wrench
    original_observe = worker.observe
    worker.reset(0, include_force=True)
    reads = 0

    def controlled_wrench():
        nonlocal reads
        reads += 1
        external = np.zeros(6)
        if reads >= 4:
            external[2] = 51.0
        return np.zeros(6), external

    worker._contact_wrench = controlled_wrench
    worker.observe = lambda include_force=False: {
        "force_torque_peak_compensated": worker.ft_peak_external.tolist(),
        "force_control_offset_m": worker.force_control_offset_m,
    }
    try:
        result = worker._run_control_steps(25, True, 2.0, 0.0)
        assert worker.steps == 2
        assert result["force_torque_peak_compensated"][2] == 51.0
    finally:
        worker._contact_wrench = original_contact_wrench
        worker.observe = original_observe
        worker.renderer.close()


def test_spiral_search_uses_cartesian_osc_and_rotates_socket():
    import ur20_screw_worker
    import view_ur20_socket_start

    original_offsets = ur20_screw_worker.INITIAL_OFFSETS
    original_rotations = ur20_screw_worker.INITIAL_ROTATIONS
    ur20_screw_worker.INITIAL_OFFSETS = ((0.001, 0.0),)
    ur20_screw_worker.INITIAL_ROTATIONS = (np.deg2rad(30.0),)
    worker = ur20_screw_worker.Worker()
    try:
        worker.reset(0, include_force=True)
        pose = view_ur20_socket_start.set_socket_start_pose(worker, 0.001, 30.0)
        assert pose["socket_wall_contact_at_pose"]
        worker.osc_target_position = worker.data.site_xpos[worker.tool_tip_id].copy()
        worker._run_control_steps(500, True, 2.0, 0.0)
        assert not worker._confirmed_socket_seating()

        socket_qpos = worker.socket_joint_qpos
        initial_socket_angle = worker.data.qpos[socket_qpos]
        result = worker.step([0.0, 0.0, 0.0], count=25, include_force=True,
                             target_force_n=2.0, spiral_search=True)

        started, increments, _ = result["spiral_search_state"]
        angle_delta = worker.data.qpos[socket_qpos] - initial_socket_angle
        assert started == 1.0
        assert 1.0 <= increments <= 6.0
        assert np.isclose(angle_delta, -np.pi / 3.0, atol=0.08)
        assert np.linalg.norm(result["force_torque_peak_compensated"][:3]) < 50.0
        assert len(result["spiral_force_trace"]) > 0
        assert {
            "axial_drop_from_baseline_n",
            "lateral_force_xy_n",
            "lateral_rise_from_baseline_n",
            "socket_engaged_diagnostic",
        } <= result["spiral_force_trace"][0].keys()
    finally:
        worker.renderer.close()
        ur20_screw_worker.INITIAL_OFFSETS = original_offsets
        ur20_screw_worker.INITIAL_ROTATIONS = original_rotations


def test_spiral_search_covers_radius_with_submillimeter_increments():
    import ur20_screw_worker

    point_count = ur20_screw_worker.SPIRAL_SEARCH_POINTS
    max_radius = ur20_screw_worker.SPIRAL_SEARCH_MAX_RADIUS_M
    total_turns = ur20_screw_worker.SPIRAL_SEARCH_TURNS
    angles = np.arange(point_count) * 2.0 * np.pi * total_turns / point_count
    radii = max_radius * (np.arange(point_count) + 1) / point_count
    positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    positions = np.vstack((np.zeros((1, 2)), positions))
    increments = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    assert np.isclose(radii[-1], 0.008)
    assert np.max(increments) < 0.001
    assert max_radius / total_turns <= 0.002


def test_socket_seating_trigger_uses_tactile_signal_without_geometry_oracle():
    import ur20_screw_worker

    worker = ur20_screw_worker.Worker.__new__(ur20_screw_worker.Worker)
    from ur20_screw_worker import SpiralTactileDetector
    worker.spiral_tactile_detector = SpiralTactileDetector(2.0)
    worker.spiral_tactile_detector.candidate_detected = True
    worker._valid_socket_engagement = lambda: (_ for _ in ()).throw(
        AssertionError("online seating trigger must not query geometry")
    )
    assert worker._confirmed_socket_seating()

    worker.spiral_tactile_detector.candidate_detected = False
    assert not worker._confirmed_socket_seating()


def test_spiral_tactile_detector_requires_axial_drop_and_sustained_lateral_rise():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=2.0,
                                     reference_lateral_force_xy=(0.2, 0.0),
                                     thresholds_calibrated=True)
    assert not detector.update(1.2, (0.2, 0.0))
    assert detector.phase == "seeking_lateral_force"
    assert detector.force_drop_detected
    assert not detector.update(1.2, (0.2, 1.0))
    for _ in range(ur20_screw_worker.SPIRAL_LATERAL_FORCE_CONFIRM_STEPS - 2):
        assert not detector.update(1.2, (0.2, 1.0))
    assert detector.update(1.2, (0.2, 1.0))
    assert detector.phase == "seated"


def test_spiral_tactile_detector_ignores_lateral_spike_and_requires_both_signals():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=2.0,
                                     reference_lateral_force_xy=(0.2, 0.0))
    assert not detector.update(2.0, (0.2, 1.5))
    assert detector.confirmation_steps == 0
    assert not detector.update(1.2, (0.2, 1.5))
    assert detector.confirmation_steps == 1
    assert not detector.update(1.2, (0.2, 0.0))
    assert detector.confirmation_steps == 0
    for _ in range(ur20_screw_worker.SPIRAL_LATERAL_FORCE_CONFIRM_STEPS - 1):
        assert not detector.update(1.2, (0.2, 1.5))
    assert not detector.seated
    assert not detector.update(1.2, (0.2, 1.5))
    assert detector.candidate_detected
    assert not detector.seated


def test_spiral_tactile_detector_requires_axial_drop_before_lateral_confirmation():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=2.0,
                                     reference_lateral_force_xy=(0.2, 0.0))
    for _ in range(ur20_screw_worker.SPIRAL_LATERAL_FORCE_CONFIRM_STEPS * 2):
        assert not detector.update(1.5, (0.2, 1.5))
    assert not detector.force_drop_detected
    assert detector.confirmation_steps == 0
    assert not detector.seated


def test_spiral_tactile_detector_requires_axial_drop_during_lateral_confirmation():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=2.0,
                                     reference_lateral_force_xy=(0.0, 0.0))
    for _ in range(ur20_screw_worker.SPIRAL_LATERAL_FORCE_CONFIRM_STEPS * 2):
        assert not detector.update(2.0, (0.0, 1.0))
    assert detector.confirmation_steps == 0
    assert not detector.candidate_detected
    assert not detector.update(1.0, (0.0, 0.0))
    for _ in range(ur20_screw_worker.SPIRAL_LATERAL_FORCE_CONFIRM_STEPS):
        detector.update(1.0, (0.0, 1.0))
    assert detector.candidate_detected


def test_spiral_tactile_detector_expires_old_axial_drop_event():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=2.0,
                                     reference_lateral_force_xy=(0.0, 0.0))
    assert not detector.update(1.0, (0.0, 0.0))
    for _ in range(ur20_screw_worker.SPIRAL_FORCE_DROP_EVENT_WINDOW_STEPS + 1):
        assert not detector.update(2.0, (0.0, 0.0))
    assert not detector.force_drop_detected
    assert detector.phase == "seeking_force_drop"
    for _ in range(ur20_screw_worker.SPIRAL_LATERAL_FORCE_CONFIRM_STEPS):
        assert not detector.update(2.0, (0.0, 1.0))
    assert not detector.seated


def test_spiral_tactile_detector_does_not_treat_lateral_force_loss_as_rise():
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=9.57,
                                     reference_lateral_force_xy=(1.63, 0.0))
    assert not detector.update(0.0, (0.0, 0.0))
    assert detector.lateral_force_rise_n < 0.0
    assert detector.confirmation_steps == 0
    assert not detector.candidate_detected


def test_spiral_tactile_detector_uses_signed_axial_unloading_and_sequential_cues():
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=-1.5)
    assert not detector.update(-0.6, (0.0, 0.0))
    assert detector.force_drop_detected
    assert detector.axial_force_drop_n == 0.9
    for _ in range(5):
        detector.update(-0.6, (0.0, 0.8))
    assert detector.candidate_detected


def test_spiral_tactile_candidate_does_not_retrigger_until_cue_releases():
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=1.5)
    for _ in range(5):
        detector.update(0.5, (0.0, 0.8))
    assert detector.candidate_detected

    detector.candidate_detected = False
    detector.candidate_latched = False
    detector.confirmation_steps = 0
    detector.candidate_rearm_blocked = True
    for _ in range(20):
        detector.update(0.5, (0.0, 0.8))
    assert not detector.candidate_detected

    detector.update(1.5, (0.0, 0.0))
    for _ in range(5):
        detector.update(0.5, (0.0, 0.8))
    assert detector.candidate_detected


def test_spiral_tactile_detector_uses_encoder_downward_travel_with_lateral_resistance():
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(
        reference_force_n=1.5, reference_lateral_force_xy=(0.0, 0.0),
        reference_tool_z_m=0.153,
    )
    for _ in range(5):
        detector.update(1.5, (0.0, 0.2), tool_z_m=0.1495)
    assert not detector.candidate_detected
    for _ in range(5):
        detector.update(1.5, (0.0, 0.35), tool_z_m=0.1495)
    assert detector.candidate_detected


def test_spiral_tactile_detector_confirms_seating_from_encoder_travel_and_axial_load():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    detector = SpiralTactileDetector(reference_force_n=1.0, reference_tool_z_m=0.153)
    for _ in range(ur20_screw_worker.SPIRAL_SEATED_CONFIRM_STEPS - 1):
        assert not detector.update(0.8, (0.0, 0.0), tool_z_m=0.150)
    assert detector.update(0.8, (0.0, 0.0), tool_z_m=0.150)
    assert detector.seated
    assert detector.phase == "seated"

    assert not detector.thresholds_calibrated

    not_seated = SpiralTactileDetector(reference_force_n=1.0, reference_tool_z_m=0.153)
    for _ in range(ur20_screw_worker.SPIRAL_SEATED_CONFIRM_STEPS + 1):
        not_seated.update(0.09, (0.0, 0.0), tool_z_m=0.149)
    assert not not_seated.seated


def test_worker_publishes_encoder_confirmed_seating_without_calibrated_lateral_thresholds():
    import ur20_screw_worker
    from ur20_screw_worker import SpiralTactileDetector

    worker = ur20_screw_worker.Worker()
    try:
        worker.reset(0, include_force=True)
        detector = SpiralTactileDetector(
            reference_force_n=1.0,
            reference_tool_z_m=float(worker.data.site_xpos[worker.tool_tip_id, 2]),
        )
        seated_z = detector.reference_tool_z_m - ur20_screw_worker.SPIRAL_MIN_SOCKET_INSERTION_M
        for _ in range(ur20_screw_worker.SPIRAL_SEATED_CONFIRM_STEPS):
            detector.update(0.8, (0.0, 0.0), tool_z_m=seated_z)
        worker.spiral_tactile_detector = detector

        observation = worker.observe(include_force=True)

        assert observation["spiral_tactile_seated"]
        assert observation["spiral_tactile_state"][1] == 1.0
        assert observation["spiral_tactile_phase"] == "seated"
    finally:
        worker.close()


def test_spiral_baseline_accepts_soft_preload_and_detects_half_depth_seating():
    import ur20_screw_worker

    worker = ur20_screw_worker.Worker.__new__(ur20_screw_worker.Worker)
    worker.force_control_offset_m = 0.0
    worker.spiral_force_baseline_settled = False
    worker.spiral_force_baseline_settle_steps = 0
    worker.debug_tactile_diagnostics = True
    worker.spiral_search_guarded = False
    worker.spiral_search_active = False
    worker.spiral_search_state = np.zeros(3)
    worker.spiral_tactile_detector = None
    worker._viewer_lock = lambda: __import__("contextlib").nullcontext()
    worker.data = type("Data", (), {
        "site_xpos": np.array([[0.0, 0.0, 0.0]]),
        "site_xmat": np.eye(3).reshape(1, 9),
        "geom_xpos": np.array([[0.0, 0.0, 0.0]]),
    })()
    worker.tool_tip_id = 0
    worker.screw_head_geom_id = 0
    worker.osc_target_position = None
    worker.osc_target_rotation = None
    worker._contact_wrench = lambda: (np.zeros(6), np.array([0., 0., 25., 0., 0., 0.]))
    def settle_with_soft_preload(count, include_force, target_force_n, socket_velocity, **kwargs):
        worker.data.site_xpos[0, 2] = -0.0035
        return {
            "force_samples": [
                np.array([0.05, 0.0, 0.7, 0.0, 0.0, 0.0]) for _ in range(count)
            ]
        }

    worker._run_control_steps = settle_with_soft_preload
    worker.model = type("Model", (), {"opt": type("Opt", (), {"timestep": 0.002})()})()
    worker.spiral_search_center = None
    worker.spiral_search_rotation = None
    worker.spiral_reference_force_n = None
    worker.spiral_force_baseline_settle_steps = 0
    worker.spiral_force_baseline_guard_reason = None
    worker.spiral_search_start_axial = None
    worker.socket_engagement_ticks = 0
    worker.socket_joint_qpos = 0
    worker.spiral_phase_offset = 0.0
    worker.data.qpos = np.zeros(1)
    worker._valid_socket_engagement = lambda: (False, False, 0.0, 0.0, 1.0)
    worker.observe = lambda include_force=False: {"force_torque_compensated": [0.] * 6}

    result = worker._spiral_search(25, include_force=True, target_force_n=1.0)

    assert not worker.spiral_search_guarded
    assert not worker.spiral_search_active
    assert worker.spiral_tactile_detector is not None
    assert worker.spiral_tactile_detector.seated
    assert np.isclose(worker.spiral_reference_force_n, 0.7)
    assert worker.spiral_search_state.tolist() == [0.0, 0.0, 0.0]
    assert worker.spiral_force_baseline_settled
    assert worker.spiral_force_baseline_settle_steps >= (
        ur20_screw_worker.SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS
    )


def test_spiral_search_stops_after_one_complete_coverage():
    import ur20_screw_worker

    worker = ur20_screw_worker.Worker.__new__(ur20_screw_worker.Worker)
    worker.spiral_search_index = ur20_screw_worker.SPIRAL_SEARCH_POINTS - 1
    worker.spiral_search_exhausted = False
    worker.spiral_search_active = True

    worker._advance_spiral_search_index()

    assert worker.spiral_search_index == ur20_screw_worker.SPIRAL_SEARCH_POINTS
    assert worker.spiral_search_exhausted
    assert not worker.spiral_search_active


def test_osc_anchor_compensates_force_offset_for_tip_target():
    from ur20_screw_worker import osc_anchor_for_tip_target

    desired_tip = np.array([0.4, -0.2, 0.7])
    tool_axis = np.array([0.0, 0.0, -1.0])
    anchor = osc_anchor_for_tip_target(desired_tip, tool_axis, 0.005)

    np.testing.assert_allclose(anchor - tool_axis * 0.005, desired_tip)


def test_socket_screw_coupling_preserves_world_rotation_with_opposed_axes():
    from ur20_screw_worker import socket_screw_coupling_parameters

    screw_start = 0.27
    socket_start = -0.41
    intercept, slope = socket_screw_coupling_parameters(screw_start, socket_start, -1.0)
    socket_next = socket_start - 0.8
    screw_next = intercept + slope * socket_next

    assert slope == -1.0
    assert np.isclose(screw_next - screw_start, -(socket_next - socket_start))
