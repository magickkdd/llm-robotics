"""JSON-lines worker for the UR20 battery-disassembly MuJoCo scene."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/ur20_screw_station/scene.xml"
CONTROL_SUBSTEPS = 25
TOOL_TIP_REACH = 0.4979
DOWNWARD_FLANGE_ROTATION = np.diag([1.0, -1.0, -1.0])
VERTICAL_IK_SEED = np.array([-2.44392306, -1.04459098, 1.15111654,
                             -1.67729567, -1.57079662, -0.87312675])
INITIAL_OFFSETS = (
    (0.080, 0.000), (0.0566, 0.0566), (0.000, 0.080), (-0.0566, 0.0566),
    (-0.080, 0.000), (-0.0566, -0.0566), (0.000, -0.080), (0.0566, -0.0566),
    (0.100, 0.000), (0.000, 0.100),
)


class Worker:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.model.site_rgba[:, 3] = 0.0
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.camera_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("agentview", "wristview")
        }
        self.site_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("attachment_site", "tool_tip", "socket_center")
        }
        self.flange_site_id = self.site_ids["attachment_site"]
        self.steps = 0
        self.engaged_seen = False
        self.screw_lift_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "screw_lift")
        self.tool_tip_id = self.site_ids["tool_tip"]
        self.screw_head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
        self.screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "screw_m6")
        self.screw_spin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "screw_spin")
        self.screw_spin_qpos = self.model.jnt_qposadr[self.screw_spin_id]
        self.screw_lift_qpos = self.model.jnt_qposadr[self.screw_lift_id]
        self.socket_wall_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"socket_wall_{index}")
            for index in range(6)
        }

    def _socket_contact(self) -> bool:
        return any(
            (contact.geom1 == self.screw_head_geom_id and contact.geom2 in self.socket_wall_ids)
            or (contact.geom2 == self.screw_head_geom_id and contact.geom1 in self.socket_wall_ids)
            for contact in self.data.contact[:self.data.ncon]
        )

    def _valid_socket_engagement(self) -> tuple[bool, bool, float, float, float]:
        contact = self._socket_contact()
        axis = self.data.site_xmat[self.tool_tip_id].reshape(3, 3)[:, 0]
        alignment = float(np.dot(axis, np.array([0.0, 0.0, -1.0])))
        delta = self.data.geom_xpos[self.screw_head_geom_id] - self.data.site_xpos[self.tool_tip_id]
        axial = float(np.dot(delta, axis))
        radial = float(np.linalg.norm(delta - axial * axis))
        engaged = contact and alignment >= 0.985 and radial <= 0.003 and -0.012 <= axial <= 0.003
        return engaged, contact, axial, radial, alignment

    def _place_start(self, state_id: int) -> None:
        if not 0 <= state_id < len(INITIAL_OFFSETS):
            raise ValueError(f"state_id must be in [0, {len(INITIAL_OFFSETS) - 1}]")
        offset_x, offset_y = INITIAL_OFFSETS[state_id]
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[:6] = VERTICAL_IK_SEED
        mujoco.mj_forward(self.model, self.data)
        screw_position = self.data.xpos[self.screw_body_id]
        target_tip = screw_position + np.array([offset_x, offset_y, 0.150])
        target_mount = target_tip - DOWNWARD_FLANGE_ROTATION[:, 2] * TOOL_TIP_REACH
        for _ in range(250):
            mujoco.mj_forward(self.model, self.data)
            current_rotation = self.data.site_xmat[self.flange_site_id].reshape(3, 3)
            position_error = target_mount - self.data.site_xpos[self.flange_site_id]
            rotation_error = 0.5 * sum(
                (np.cross(current_rotation[:, axis], DOWNWARD_FLANGE_ROTATION[:, axis])
                 for axis in range(3)), np.zeros(3)
            )
            if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 1e-3:
                break
            position_jacobian = np.zeros((3, self.model.nv))
            rotation_jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, position_jacobian,
                              rotation_jacobian, self.flange_site_id)
            arm_jacobian = np.vstack((position_jacobian[:, :6], rotation_jacobian[:, :6] * 0.2))
            pose_error = np.concatenate((position_error, rotation_error * 0.2))
            delta = arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + 1e-4 * np.eye(6), pose_error
            )
            self.data.qpos[:6] += np.clip(delta, -0.08, 0.08)
        else:
            raise RuntimeError(f"could not solve initial pose for state {state_id}")
        self.data.ctrl[:6] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)

    def _image(self, camera_id: int) -> str:
        self.renderer.update_scene(self.data, camera=camera_id)
        image = Image.fromarray(self.renderer.render()).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def observe(self) -> dict:
        mujoco.mj_forward(self.model, self.data)
        socket_engaged, socket_contact, socket_axial, socket_radial, axis_alignment = self._valid_socket_engagement()
        self.engaged_seen = self.engaged_seen or socket_engaged
        return {
            "images": {name: self._image(camera_id) for name, camera_id in self.camera_ids.items()},
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "sites": {name: self.data.site_xpos[index].tolist() for name, index in self.site_ids.items()},
            "time": float(self.data.time),
            "steps": self.steps,
            "screw_lift": float(self.data.qpos[self.screw_lift_qpos]),
            "screw_rotation": float(self.data.qpos[self.screw_spin_qpos]),
            "socket_engaged": socket_engaged,
            "socket_contact": socket_contact,
            "socket_axial_error": socket_axial,
            "socket_radial_error": socket_radial,
            "tool_axis_alignment": axis_alignment,
            "engaged_seen": self.engaged_seen,
        }

    def reset(self, state_id: int = 0) -> dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._place_start(int(state_id))
        self.steps = 0
        self.engaged_seen = False
        return self.observe()

    def step(self, ctrl: list[float], count: int = CONTROL_SUBSTEPS) -> dict:
        if len(ctrl) != self.model.nu:
            raise ValueError(f"expected {self.model.nu} controls, got {len(ctrl)}")
        self.data.ctrl[:] = np.asarray(ctrl, dtype=float)
        for _ in range(max(1, int(count))):
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            engaged, _, _, _, _ = self._valid_socket_engagement()
            self.engaged_seen = self.engaged_seen or engaged
        return self.observe()


def main() -> None:
    worker = Worker()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "reset":
                result = worker.reset(request.get("state_id", 0))
            elif operation == "observe":
                result = worker.observe()
            elif operation == "step":
                result = worker.step(request["ctrl"], request.get("count", 1))
            elif operation == "close":
                break
            else:
                raise ValueError(f"unknown operation: {operation}")
            print(json.dumps(result), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
