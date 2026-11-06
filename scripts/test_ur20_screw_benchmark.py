"""Offline checks for deterministic starts and physical socket engagement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

WORKER_PATH = Path(__file__).with_name("ur20_screw_worker.py")
SPEC = importlib.util.spec_from_file_location("ur20_screw_worker", WORKER_PATH)
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WORKER_MODULE)


def main() -> None:
    worker = WORKER_MODULE.Worker()
    try:
        model, data = worker.model, worker.data
        worker.reset(0)
        equalities = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
            for index in range(model.neq)
        }
        assert equalities == {"screw_rotation_lock", "screw_height_lock"}
        assert model.geom_type[worker.screw_head_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tool_lift") == -1
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wristview") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "closeup") == -1
        wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wristview")
        assert 0.40 <= model.cam_pos[wrist_camera_id, 0] <= 0.41
        assert -0.06 <= model.cam_pos[wrist_camera_id, 1] <= -0.05
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.cam_bodyid[wrist_camera_id]) == "end_effector"
        assert model.cam_mode[wrist_camera_id] == mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.cam_targetbodyid[wrist_camera_id]) == "screw_m6"
        battery_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "battery_collision")
        assert abs(model.geom_pos[battery_geom_id, 2] + model.geom_size[battery_geom_id, 2] - 0.148) < 1e-6
        battery_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "battery")
        screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw_m6")
        assert model.nexclude == 1
        excluded_pair = int(model.exclude_signature[0])
        assert {excluded_pair >> 16, excluded_pair & 0xFFFF} == {battery_body_id, screw_body_id}
        flange_axis = data.site_xmat[worker.site_ids["attachment_site"]].reshape(3, 3)[:, 2]
        tool_axis = data.site_xmat[worker.tool_tip_id].reshape(3, 3)[:, 0]
        mount_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mount")
        assert np.dot(flange_axis, tool_axis) > 0.999
        assert np.linalg.norm(data.site_xpos[worker.site_ids["attachment_site"]] -
                              data.site_xpos[mount_id]) < 1e-6

        starts = []
        rotations = []
        for state_id in range(10):
            initial = worker.reset(state_id)
            repeated = worker.reset(state_id)
            assert initial["qpos"] == repeated["qpos"]
            assert not initial["socket_engaged"]
            assert len(initial["qpos"]) == 9
            tool_axis = worker.data.site_xmat[worker.tool_tip_id].reshape(3, 3)[:, 0]
            assert np.dot(tool_axis, np.array([0.0, 0.0, -1.0])) > 0.999
            assert np.linalg.norm(np.asarray(initial["sites"]["tool_tip"]) -
                                  worker.data.geom_xpos[worker.screw_head_geom_id]) >= 0.13
            starts.append(initial["sites"]["tool_tip"][:2])
            rotations.append(float(worker.data.qpos[worker.screw_spin_qpos]))
        assert len({tuple(np.round(position, 5)) for position in starts}) == 10
        assert len({round(angle, 6) for angle in rotations}) == 10
        assert all(abs(angle % (np.pi / 3.0)) > 1e-6 for angle in rotations)
        assert WORKER_MODULE.SPIRAL_SEARCH_MAX_RADIUS_M == 0.008
        assert np.isclose(
            WORKER_MODULE.SPIRAL_SEARCH_MAX_RADIUS_M / WORKER_MODULE.SPIRAL_SEARCH_POINTS,
            0.00025,
        )

        worker.reset(0)
        target = np.array([0.8, 1.0, 0.145])
        for _ in range(100):
            mujoco.mj_forward(model, data)
            error = target - data.site_xpos[worker.tool_tip_id]
            if np.linalg.norm(error) < 1e-6:
                break
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacobian, None, worker.tool_tip_id)
            arm_jacobian = jacobian[:, :6]
            data.qpos[:6] += arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + 1e-5 * np.eye(3), error
            )
        data.ctrl[:6] = 0.0
        worker.osc_target_position = data.site_xpos[worker.tool_tip_id].copy()
        worker.osc_target_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3).copy()
        result = worker.step([0.0] * 3, socket_velocity=-1.0, count=1000)
        assert result["engaged_seen"]
        assert abs(float(data.qpos[worker.screw_spin_qpos]) -
                   WORKER_MODULE.INITIAL_ROTATIONS[0]) < 1e-5
        assert abs(float(data.qpos[worker.screw_lift_qpos])) < 1e-5

        worker.reset(0)
        target = np.array([0.8, 1.0, 0.12])
        for _ in range(100):
            mujoco.mj_forward(model, data)
            error = target - data.site_xpos[worker.tool_tip_id]
            if np.linalg.norm(error) < 1e-6:
                break
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacobian, None, worker.tool_tip_id)
            arm_jacobian = jacobian[:, :6]
            data.qpos[:6] += arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + 1e-5 * np.eye(3), error
            )
        target_joints = data.qpos[:6].copy()
        worker.reset(0)
        data.qpos[:6] = target_joints
        data.ctrl[:6] = 0.0
        mujoco.mj_forward(model, data)
        worker.osc_target_position = data.site_xpos[worker.tool_tip_id].copy()
        worker.osc_target_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3).copy()
        worker.step([0.0] * 3, count=2000)
        battery_top = model.geom_pos[battery_geom_id, 2] + model.geom_size[battery_geom_id, 2]
        assert data.site_xpos[worker.tool_tip_id, 2] >= battery_top - 0.003
        assert any(
            battery_geom_id in (contact.geom1, contact.geom2)
            for contact in data.contact[:data.ncon]
        )
        print("UR20 benchmark checks passed: rigid tool mount, wrist camera, 10 clear starts, and seated socket contact.")
    finally:
        worker.renderer.close()


if __name__ == "__main__":
    main()
