"""Show the UR20 socket at a deliberately offset screw-head start pose."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from ur20_screw.simulation import worker as worker_module
from ur20_screw.vision.estimation import SCREW_HEAD_TOP_Z_M

# 开口面压到头顶下方几微米，保证盒体碰撞在起始位姿就判定套筒壁-螺钉头接触；
# 偏移量小于测试中 1e-5 m 的高度一致性容差。
SOCKET_START_CONTACT_BIAS_M = 8e-6


def solve_tip_ik(worker, target_position: np.ndarray) -> None:
    model, data = worker.model, worker.data
    target_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3).copy()
    for _ in range(500):
        mujoco.mj_forward(model, data)
        position_error = target_position - data.site_xpos[worker.tool_tip_id]
        rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3)
        rotation_error = 0.5 * sum(
            (np.cross(rotation[:, axis], target_rotation[:, axis]) for axis in range(3)),
            np.zeros(3),
        )
        if np.linalg.norm(position_error) < 1e-6 and np.linalg.norm(rotation_error) < 1e-5:
            return
        position_jacobian = np.zeros((3, model.nv))
        rotation_jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(
            model, data, position_jacobian, rotation_jacobian, worker.tool_tip_id
        )
        jacobian = np.vstack((position_jacobian[:, :6], rotation_jacobian[:, :6] * 0.2))
        error = np.concatenate((position_error, rotation_error * 0.2))
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-5 * np.eye(6), error
        )
        data.qpos[:6] += np.clip(delta, -0.08, 0.08)
    raise RuntimeError("could not solve the socket-start pose")


def set_socket_start_pose(worker, offset_x_m: float, screw_angle_deg: float,
                          offset_y_m: float = 0.0) -> dict:
    model, data = worker.model, worker.data
    mujoco.mj_resetDataKeyframe(model, data, 0)
    worker._place_start(0)
    screw_angle = np.deg2rad(screw_angle_deg)
    model.eq_data[worker.screw_rotation_lock_id, 0] = screw_angle
    data.qpos[worker.screw_spin_qpos] = screw_angle
    data.qpos[worker.socket_joint_qpos] = 0.0
    mujoco.mj_forward(model, data)
    screw_center = data.xpos[worker.screw_body_id, :2].copy()
    target_tip = np.array([
        screw_center[0] + offset_x_m,
        screw_center[1] + offset_y_m,
        SCREW_HEAD_TOP_Z_M - SOCKET_START_CONTACT_BIAS_M,
    ])
    solve_tip_ik(worker, target_tip)
    mujoco.mj_forward(model, data)
    actual_tip = data.site_xpos[worker.tool_tip_id].copy()
    opening_center = actual_tip.copy()
    socket_axis = data.site_xmat[worker.tool_tip_id].reshape(3, 3)[:, 0]
    head_delta = data.geom_xpos[worker.screw_head_geom_id] - actual_tip
    opening_offset_xy = opening_center[:2] - screw_center
    return {
        "screw_center_xy_m": screw_center.tolist(),
        "tool_tip_world_m": actual_tip.tolist(),
        "socket_opening_center_world_m": opening_center.tolist(),
        "center_offset_xy_mm": ((actual_tip[:2] - screw_center) * 1000.0).tolist(),
        "opening_plane_height_m": float(opening_center[2]),
        "opening_center_offset_xy_mm": (opening_offset_xy * 1000.0).tolist(),
        "screw_head_top_height_m": float(SCREW_HEAD_TOP_Z_M),
        "socket_screw_axis_angle_deg": float(screw_angle_deg),
        "socket_axis_alignment": float(np.dot(socket_axis, np.array([0.0, 0.0, -1.0]))),
        "head_axial_offset_m": float(np.dot(head_delta, socket_axis)),
        "head_top_contact_at_pose": any(
            (contact.geom1 == worker.screw_head_geom_id and contact.geom2 in worker.socket_wall_ids)
            or (contact.geom2 == worker.screw_head_geom_id and contact.geom1 in worker.socket_wall_ids)
            for contact in data.contact[:data.ncon]
        ),
        "socket_wall_contact_at_pose": worker._socket_contact(),
        "contact_count": int(data.ncon),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-x-mm", type=float, default=1.0)
    parser.add_argument("--offset-y-mm", type=float, default=0.0)
    parser.add_argument("--screw-angle-deg", type=float, default=30.0)
    parser.add_argument("--check-only", action="store_true",
                        help="print the pose without opening a graphical viewer")
    args = parser.parse_args()

    worker_module.INITIAL_OFFSETS = ((args.offset_x_mm / 1000.0,
                                     args.offset_y_mm / 1000.0),)
    worker_module.INITIAL_ROTATIONS = (np.deg2rad(args.screw_angle_deg),)
    worker = worker_module.Worker(render=False)
    try:
        pose = set_socket_start_pose(worker, args.offset_x_mm / 1000.0,
                                     args.screw_angle_deg, args.offset_y_mm / 1000.0)
        for key, value in pose.items():
            print(f"{key}: {value}")
        if args.check_only:
            return

        import mujoco.viewer

        worker.viewer = mujoco.viewer.launch_passive(worker.model, worker.data)
        while worker.viewer.is_running():
            worker.viewer.sync()
            time.sleep(1.0 / 60.0)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
