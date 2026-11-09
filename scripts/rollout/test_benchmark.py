"""Offline checks for deterministic starts and physical socket engagement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

from ur20_screw.target import SCREW_TARGET
from ur20_screw.vision.estimation import SCREW_HEAD_TOP_Z_M

WORKER_PATH = Path(__file__).resolve().parents[2] / "src/ur20_screw/simulation/worker.py"
SPEC = importlib.util.spec_from_file_location("ur20_screw_worker", WORKER_PATH)
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WORKER_MODULE)

BATTERY_PLATE_TOP_Z_M = 0.058
BATTERY_BODY_TOP_Z_M = 0.148
HOVER_HEIGHT_M = 0.03
SOCKET_INSERTION_DEPTH_M = 0.006


def _head_top_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start:start + count])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return float(data.geom_xpos[geom_id][2] + np.max((vertices @ rotation.T)[:, 2]))


def _contact_geom_ids(data: mujoco.MjData) -> set[int]:
    return {
        geom for contact in data.contact[:data.ncon]
        for geom in (contact.geom1, contact.geom2)
    }


def main() -> None:
    worker = WORKER_MODULE.Worker()
    try:
        model, data = worker.model, worker.data
        worker.reset(0)
        equalities = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
            for index in range(model.neq)
        }
        assert equalities == {
            SCREW_TARGET.rotation_lock,
            SCREW_TARGET.socket_screw_coupling,
            SCREW_TARGET.height_lock,
            SCREW_TARGET.thread_pitch_coupling,
        }
        assert bool(model.eq_active0[worker.screw_rotation_lock_id])
        assert bool(model.eq_active0[worker.screw_height_lock_id])
        assert not bool(model.eq_active0[worker.socket_screw_coupling_id])
        assert not bool(model.eq_active0[worker.screw_thread_pitch_coupling_id])
        assert model.geom_type[worker.screw_head_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        assert abs(_head_top_z(model, data, worker.screw_head_geom_id)
                   - SCREW_HEAD_TOP_Z_M) < 1e-6
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tool_lift") == -1
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wristview") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "closeup") == -1
        wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wristview")
        assert 0.40 <= model.cam_pos[wrist_camera_id, 0] <= 0.41
        assert -0.06 <= model.cam_pos[wrist_camera_id, 1] <= -0.05
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.cam_bodyid[wrist_camera_id]) == "end_effector"
        assert model.cam_mode[wrist_camera_id] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        agentview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        assert model.cam_mode[agentview_id] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        plate_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "battery_collision_plate"
        )
        body_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "battery_collision_body"
        )
        assert abs(model.geom_pos[plate_geom_id, 2]
                   + model.geom_size[plate_geom_id, 2] - BATTERY_PLATE_TOP_Z_M) < 1e-6
        assert abs(model.geom_pos[body_geom_id, 2]
                   + model.geom_size[body_geom_id, 2] - BATTERY_BODY_TOP_Z_M) < 1e-6
        battery_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "battery")
        screw_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, SCREW_TARGET.body
        )
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
        point_count = WORKER_MODULE.SPIRAL_SEARCH_POINTS
        angles = np.arange(point_count) * (
            2.0 * np.pi * WORKER_MODULE.SPIRAL_SEARCH_TURNS / point_count
        )
        radii = (WORKER_MODULE.SPIRAL_SEARCH_MAX_RADIUS_M
                 * (np.arange(point_count) + 1) / point_count)
        positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        positions = np.vstack((np.zeros((1, 2)), positions))
        assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.001

        worker.reset(0)
        screw_xy = data.xpos[worker.screw_body_id, :2].copy()
        worker.step(
            [screw_xy[0], screw_xy[1], SCREW_HEAD_TOP_Z_M + HOVER_HEIGHT_M],
            count=1200,
        )
        engaged = worker.step(
            [screw_xy[0], screw_xy[1], SCREW_HEAD_TOP_Z_M - SOCKET_INSERTION_DEPTH_M],
            socket_velocity=-1.0, count=500,
        )
        assert engaged["engaged_seen"]
        assert engaged["socket_engaged"]
        assert abs(float(data.qpos[worker.screw_spin_qpos]) -
                   WORKER_MODULE.INITIAL_ROTATIONS[0]) < 1e-5
        assert float(data.qpos[worker.screw_lift_qpos]) > -0.002

        worker.reset(0)
        screw_xy = data.xpos[worker.screw_body_id, :2].copy()
        worker.step(
            [screw_xy[0], screw_xy[1], SCREW_HEAD_TOP_Z_M + HOVER_HEIGHT_M],
            count=1200,
        )
        worker.step([screw_xy[0], screw_xy[1], BATTERY_PLATE_TOP_Z_M - 0.020], count=1500)
        assert float(data.site_xpos[worker.tool_tip_id, 2]) >= BATTERY_PLATE_TOP_Z_M - 0.003
        assert plate_geom_id in _contact_geom_ids(data)
        assert float(data.qpos[worker.screw_lift_qpos]) > -0.002
        print("UR20 benchmark checks passed: rigid tool mount, wrist camera, 10 clear starts, "
              "socket engagement, and flange-plane blocking.")
    finally:
        worker.renderer.close()


if __name__ == "__main__":
    main()
