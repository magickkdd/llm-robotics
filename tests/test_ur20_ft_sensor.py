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

    original_offsets = ur20_screw_worker.INITIAL_OFFSETS
    ur20_screw_worker.INITIAL_OFFSETS = ((0.0, 0.0),)
    worker = ur20_screw_worker.Worker()
    try:
        observation = worker.reset(0, include_force=True)
        assert np.allclose(observation["force_torque_compensated"], 0.0, atol=1e-6)

        model = worker.model
        data = worker.data
        tip_id = worker.tool_tip_id
        head_position = data.geom_xpos[worker.screw_head_geom_id].copy()
        target_position = head_position + np.array([0.008, 0.0, 0.0])
        target_rotation = data.site_xmat[tip_id].reshape(3, 3).copy()
        _move_tip_kinematically(model, data, target_position, target_rotation)
        mujoco.mj_forward(model, data)
        worker.osc_target_position = data.site_xpos[tip_id].copy()
        worker.osc_target_rotation = data.site_xmat[tip_id].reshape(3, 3).copy()
        assert worker._socket_contact()

        _, external_wrench = worker._contact_wrench()
        assert np.linalg.norm(external_wrench[:3]) > 1.0
        initial_force = 0.0
        final_result = None
        for interval in range(60):
            final_result = worker.step([0.0, 0.0, 0.0], count=25,
                                       include_force=True, target_force_n=2.0)
            if interval == 0:
                initial_force = np.linalg.norm(
                    final_result["force_torque_compensated"][:3]
                )
        assert final_result["force_control_active"]
        assert initial_force > np.linalg.norm(final_result["force_torque_compensated"][:3]) * 2.0
        assert abs(final_result["force_torque_compensated"][2] - 2.0) < 1.0
        assert np.linalg.norm(final_result["force_torque_peak_compensated"][:3]) > 1.0
    finally:
        worker.renderer.close()
        ur20_screw_worker.INITIAL_OFFSETS = original_offsets


def test_spiral_search_uses_cartesian_osc_and_rotates_socket():
    import ur20_screw_worker

    original_offsets = ur20_screw_worker.INITIAL_OFFSETS
    ur20_screw_worker.INITIAL_OFFSETS = ((0.0, 0.0),)
    worker = ur20_screw_worker.Worker()
    try:
        worker.reset(0, include_force=True)
        model = worker.model
        data = worker.data
        head_position = data.geom_xpos[worker.screw_head_geom_id].copy()
        tip_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3).copy()
        _move_tip_kinematically(
            model, data, head_position + np.array([0.004, 0.0, 0.0]), tip_rotation
        )
        mujoco.mj_forward(model, data)
        worker.osc_target_position = data.site_xpos[worker.tool_tip_id].copy()
        worker.osc_target_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3).copy()
        assert not worker._valid_socket_engagement()[0]

        socket_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "socket_joint")
        socket_qpos = model.jnt_qposadr[socket_joint]
        initial_socket_angle = data.qpos[socket_qpos]
        result = worker.step([0.0, 0.0, 0.0], count=25, include_force=True,
                             target_force_n=2.0, spiral_search=True)

        started, increments, _ = result["spiral_search_state"]
        angle_delta = data.qpos[socket_qpos] - initial_socket_angle
        assert started == 1.0
        assert 1.0 <= increments <= 6.0
        assert -1.2 < angle_delta < -0.1
        assert np.linalg.norm(result["force_torque_peak_compensated"][:3]) < 50.0
    finally:
        worker.renderer.close()
        ur20_screw_worker.INITIAL_OFFSETS = original_offsets
