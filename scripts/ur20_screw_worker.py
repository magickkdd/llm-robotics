"""JSON-lines worker for the UR20 battery-disassembly MuJoCo scene."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from ur20_vision import SCREW_HEAD_TOP_Z_M, estimate_pair, project_pixel_to_plane
from ur20_timing import wait_for_simulated_time

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/ur20_screw_station/scene.xml"
CONTROL_SUBSTEPS = 25
SPIRAL_SEARCH_POINTS = 32
SPIRAL_SEARCH_MAX_RADIUS_M = 0.008
TOOL_TIP_REACH = 0.4979
DOWNWARD_FLANGE_ROTATION = np.diag([1.0, -1.0, -1.0])
ARM_TORQUE_LIMITS = np.array([738.0, 738.0, 433.0, 107.0, 107.0, 107.0])
OSC_KP = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
OSC_KD = np.array([20.0, 20.0, 20.0, 12.0, 12.0, 12.0])
VERTICAL_IK_SEED = np.array([-2.44392306, -1.04459098, 1.15111654,
                             -1.67729567, -1.57079662, -0.87312675])
INITIAL_OFFSETS = (
    (0.080, 0.000), (0.0566, 0.0566), (0.000, 0.080), (-0.0566, 0.0566),
    (-0.080, 0.000), (-0.0566, -0.0566), (0.000, -0.080), (0.0566, -0.0566),
    (0.100, 0.000), (0.000, 0.100),
)
INITIAL_ROTATIONS = tuple(np.deg2rad(5.0 * (index + 1)) for index in range(10))


class Worker:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.model.site_rgba[:, 3] = 0.0
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.viewer = None
        self.viewer_last_sync = None
        self.camera_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("agentview", "wristview")
        }
        self.site_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("attachment_site", "tool_tip", "socket_center")
        }
        self.flange_site_id = self.site_ids["attachment_site"]
        self.ft_sensor_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site"
        )
        self.steps = 0
        self.engaged_seen = False
        self.ft_reference_model = None
        self.ft_reference_data = None
        self.ft_peak_external = np.zeros(6, dtype=float)
        self.force_control_active = False
        self.force_control_target_n = 0.0
        self.force_control_offset_m = 0.0
        self.spiral_phase_offset = 0.0
        self.spiral_search_state = np.zeros(3, dtype=float)
        self.spiral_search_active = False
        self.spiral_search_guarded = False
        self.spiral_search_index = 0
        self.spiral_search_center = None
        self.spiral_search_rotation = None
        self.spiral_reference_force_n = None
        self.spiral_reference_z_m = None
        self.socket_engagement_ticks = 0
        self.osc_target_position = None
        self.osc_target_rotation = None
        self.screw_lift_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "screw_lift")
        self.screw_rotation_lock_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "screw_rotation_lock"
        )
        self.screw_height_lock_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "screw_height_lock"
        )
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
        force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_force")
        torque_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_torque")
        self.ft_sensor_adr = int(self.model.sensor_adr[force_sensor_id])
        self.torque_sensor_adr = int(self.model.sensor_adr[torque_sensor_id])

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

    def _wrench(self) -> np.ndarray:
        return np.concatenate((
            self.data.sensordata[self.ft_sensor_adr:self.ft_sensor_adr + 3],
            self.data.sensordata[self.torque_sensor_adr:self.torque_sensor_adr + 3],
        )).copy()

    def _initialize_force_reference(self) -> None:
        if self.ft_reference_model is not None:
            return
        self.ft_reference_model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.ft_reference_model.geom_contype[:] = 0
        self.ft_reference_model.geom_conaffinity[:] = 0
        self.ft_reference_data = mujoco.MjData(self.ft_reference_model)

    def _contact_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self._wrench()
        self._initialize_force_reference()
        reference_model = self.ft_reference_model
        reference_data = self.ft_reference_data
        for field in ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied",
                      "mocap_pos", "mocap_quat", "userdata"):
            target = getattr(reference_data, field)
            source = getattr(self.data, field)
            if target.size:
                target[:] = source
        reference_data.time = self.data.time
        mujoco.mj_forward(reference_model, reference_data)
        reference_force_adr = int(reference_model.sensor_adr[
            mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_force")
        ])
        reference_torque_adr = int(reference_model.sensor_adr[
            mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_ft_torque")
        ])
        free = np.concatenate((
            reference_data.sensordata[reference_force_adr:reference_force_adr + 3],
            reference_data.sensordata[reference_torque_adr:reference_torque_adr + 3],
        ))
        external = raw - free
        external[np.abs(external) < 1e-6] = 0.0
        return raw, external

    def _place_start(self, state_id: int) -> None:
        if not 0 <= state_id < len(INITIAL_OFFSETS):
            raise ValueError(f"state_id must be in [0, {len(INITIAL_OFFSETS) - 1}]")
        offset_x, offset_y = INITIAL_OFFSETS[state_id]
        screw_phase = INITIAL_ROTATIONS[state_id]
        self.model.eq_data[self.screw_rotation_lock_id, 0] = screw_phase
        self.data.qpos[self.screw_spin_qpos] = screw_phase
        self.data.qpos[self.screw_lift_qpos] = 0.0
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
        self.data.ctrl[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _render_rgb(self, camera_id: int) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_id)
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    @staticmethod
    def _encode_image(rgb: np.ndarray) -> str:
        image = Image.fromarray(rgb).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _start_viewer(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            return
        import mujoco.viewer

        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:
            self.viewer = None
            raise RuntimeError(
                "could not open MuJoCo viewer; verify WSLg/X11 desktop access "
                f"(DISPLAY={os.environ.get('DISPLAY')!r}): {exc}"
            ) from exc
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.camera_ids["agentview"]
        self.viewer.sync()
        self.viewer_last_sync = time.perf_counter()

    def _viewer_lock(self):
        if self.viewer is not None and self.viewer.is_running():
            return self.viewer.lock()
        return nullcontext()

    def _sync_viewer_realtime(self, started: float, simulated_seconds: float) -> None:
        wait_for_simulated_time(started, simulated_seconds)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            self.viewer_last_sync = time.perf_counter()

    def close(self) -> None:
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def observe(self, include_force: bool = False) -> dict:
        with self._viewer_lock():
            mujoco.mj_forward(self.model, self.data)
            socket_engaged, socket_contact, socket_axial, socket_radial, axis_alignment = self._valid_socket_engagement()
            self.engaged_seen = self.engaged_seen or socket_engaged
            frames = {name: self._render_rgb(camera_id)
                      for name, camera_id in self.camera_ids.items()}
            vision_estimate = estimate_pair(frames["agentview"], frames["wristview"])
            wrist_camera_id = self.camera_ids["wristview"]
            screw_pixel = vision_estimate.wristview_screw.center_px
            screw_target = None
            if screw_pixel is not None and vision_estimate.wristview_screw.confidence >= 0.5:
                screw_target = project_pixel_to_plane(
                    screw_pixel,
                    self.data.cam_xpos[wrist_camera_id],
                    self.data.cam_xmat[wrist_camera_id].reshape(3, 3),
                    float(self.model.cam_fovy[wrist_camera_id]),
                    (frames["wristview"].shape[1], frames["wristview"].shape[0]),
                )
            tool_tip = self.data.site_xpos[self.tool_tip_id].copy()
            vision_valid = screw_target is not None
            result = {
                "images": {name: self._encode_image(frame) for name, frame in frames.items()},
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
                "sites": {name: self.data.site_xpos[index].tolist() for name, index in self.site_ids.items()},
                "vision_screw_center_px": list(screw_pixel) if screw_pixel is not None else [-1.0, -1.0],
                "vision_screw_xy_estimate_m": screw_target[:2].tolist() if vision_valid else tool_tip[:2].tolist(),
                "vision_screw_top_z_estimate_m": float(SCREW_HEAD_TOP_Z_M),
                "vision_xy_error_m": (screw_target[:2] - tool_tip[:2]).tolist() if vision_valid else [0.0, 0.0],
                "vision_confidence": float(vision_estimate.wristview_screw.confidence if vision_valid else 0.0),
                "vision_valid": vision_valid,
                "vision_processing_ms": float(vision_estimate.processing_ms),
                "time": float(self.data.time),
                "steps": self.steps,
                "screw_lift": float(self.data.qpos[self.screw_lift_qpos]),
                "screw_rotation": float(self.data.qpos[self.screw_spin_qpos]),
                "socket_engaged": socket_engaged,
                "socket_engagement_stable": self.socket_engagement_ticks >= 20,
                "socket_contact": socket_contact,
                "socket_axial_error": socket_axial,
                "socket_radial_error": socket_radial,
                "tool_axis_alignment": axis_alignment,
                "engaged_seen": self.engaged_seen,
            }
            if include_force:
                raw, external = self._contact_wrench()
                result["force_torque"] = raw.tolist()
                result["force_torque_compensated"] = external.tolist()
                result["force_torque_peak_compensated"] = self.ft_peak_external.tolist()
                result["force_control_active"] = self.force_control_active
                result["force_control_target_n"] = self.force_control_target_n
                result["force_control_offset_m"] = self.force_control_offset_m
                result["spiral_search_state"] = self.spiral_search_state.tolist()
                result["spiral_axial_force_delta_n"] = (
                    float(external[2] - self.spiral_reference_force_n)
                    if self.spiral_reference_force_n is not None else 0.0
                )
                result["spiral_tip_z_displacement_m"] = (
                    float(tool_tip[2] - self.spiral_reference_z_m)
                    if self.spiral_reference_z_m is not None else 0.0
                )
            return result

    def reset(self, state_id: int = 0, include_force: bool = False,
              open_viewer: bool = False) -> dict:
        with self._viewer_lock():
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self._place_start(int(state_id))
            mujoco.mj_forward(self.model, self.data)
            self.osc_target_position = self.data.site_xpos[self.tool_tip_id].copy()
            self.osc_target_rotation = self.data.site_xmat[self.tool_tip_id].reshape(3, 3).copy()
        self.steps = 0
        self.engaged_seen = False
        self.ft_peak_external = np.zeros(6, dtype=float)
        self.force_control_active = False
        self.force_control_target_n = 0.0
        self.force_control_offset_m = 0.0
        self.spiral_phase_offset = 0.0
        self.spiral_search_state = np.zeros(3, dtype=float)
        self.spiral_search_active = False
        self.spiral_search_guarded = False
        self.spiral_search_center = None
        self.spiral_search_rotation = None
        self.spiral_reference_force_n = None
        self.spiral_reference_z_m = None
        self.socket_engagement_ticks = 0
        if open_viewer:
            self._start_viewer()
        if include_force:
            self._contact_wrench()
        return self.observe(include_force)

    def _osc_torque(self, target_position: np.ndarray,
                    target_rotation: np.ndarray) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.tool_tip_id].reshape(3, 3)
        position_error = target_position - self.data.site_xpos[self.tool_tip_id]
        rotation_error = 0.5 * sum(
            (np.cross(rotation[:, axis], target_rotation[:, axis]) for axis in range(3)),
            np.zeros(3),
        )
        position_jacobian = np.zeros((3, self.model.nv))
        rotation_jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, position_jacobian,
                          rotation_jacobian, self.tool_tip_id)
        jacobian = np.vstack((position_jacobian[:, :6], rotation_jacobian[:, :6]))
        twist = jacobian @ self.data.qvel[:6]
        desired_acceleration = np.concatenate((
            OSC_KP[:3] * position_error - OSC_KD[:3] * twist[:3],
            OSC_KP[3:] * rotation_error - OSC_KD[3:] * twist[3:],
        ))
        mass = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        arm_mass = mass[:6, :6]
        mass_inverse_jacobian_t = np.linalg.solve(arm_mass, jacobian.T)
        operational_inertia = np.linalg.solve(
            jacobian @ mass_inverse_jacobian_t + 1e-5 * np.eye(6), np.eye(6)
        )
        torque = jacobian.T @ (operational_inertia @ desired_acceleration)
        torque += self.data.qfrc_bias[:6] - self.data.qfrc_passive[:6]
        return np.clip(torque, -ARM_TORQUE_LIMITS, ARM_TORQUE_LIMITS)

    def _spiral_search(self, count: int, include_force: bool,
                       target_force_n: float) -> dict:
        if not include_force or target_force_n <= 0:
            raise ValueError("spiral search requires force feedback and a positive axial force setpoint")
        with self._viewer_lock():
            self.force_control_target_n = float(np.clip(target_force_n, 0.0, 5.0))
            current_engagement = self._valid_socket_engagement()[0]
            tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            tip_rotation_now = self.data.site_xmat[self.tool_tip_id].reshape(3, 3).copy()
        if current_engagement:
            if not self.spiral_search_active:
                self.spiral_search_index = 0
                self.spiral_search_center = tip_position.copy()
                self.spiral_search_rotation = tip_rotation_now.copy()
                _, reference_wrench = self._contact_wrench()
                self.spiral_reference_force_n = float(reference_wrench[2])
                self.spiral_reference_z_m = float(tip_position[2])
            self.osc_target_position = tip_position
            self.osc_target_rotation = tip_rotation_now
            result = self._run_control_steps(count, include_force=True,
                                             target_force_n=target_force_n,
                                             socket_velocity=0.0)
            engaged = bool(result["socket_engaged"])
            stable = bool(result["socket_engagement_stable"])
            peak = np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
            self.spiral_search_guarded = not engaged and np.linalg.norm(peak) >= 50.0
            self.spiral_search_active = not stable and not self.spiral_search_guarded
            self.spiral_search_state = np.array([
                float(self.spiral_search_active), float(self.spiral_search_index),
                float(self.spiral_search_guarded),
            ])
            return result

        if not self.spiral_search_active:
            self.spiral_search_active = True
            self.spiral_search_index = 0
            self.spiral_search_center = tip_position
            self.spiral_search_rotation = tip_rotation_now
            _, reference_wrench = self._contact_wrench()
            self.spiral_reference_force_n = float(reference_wrench[2])
            self.spiral_reference_z_m = float(tip_position[2])
        index = self.spiral_search_index
        tip_position = self.spiral_search_center
        tip_rotation = self.spiral_search_rotation
        sensor_rotation = self.data.site_xmat[self.ft_sensor_site_id].reshape(3, 3).copy()
        radius = SPIRAL_SEARCH_MAX_RADIUS_M * (index + 1) / SPIRAL_SEARCH_POINTS
        angle = self.spiral_phase_offset + index * np.pi / 3.0
        target_position = tip_position + radius * (
            np.cos(angle) * sensor_rotation[:, 0]
            + np.sin(angle) * sensor_rotation[:, 1]
        )
        self.osc_target_position = target_position
        self.osc_target_rotation = tip_rotation
        search_duration = max(1, int(count)) * self.model.opt.timestep
        socket_speed = -(np.pi / 3.0) / search_duration
        result = self._run_control_steps(count, include_force=True,
                                         target_force_n=target_force_n,
                                         socket_velocity=socket_speed)
        peak = np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
        engaged = bool(result["socket_engaged"])
        force_guarded = np.linalg.norm(peak) >= 50.0
        if engaged or force_guarded:
            self.data.ctrl[6] = 0.0
            self.spiral_search_active = False
            self.spiral_search_guarded = force_guarded and not engaged
        else:
            self.spiral_search_index += 1
            if self.spiral_search_index >= SPIRAL_SEARCH_POINTS:
                self.spiral_phase_offset = (self.spiral_phase_offset + np.pi / 6.0) % (2.0 * np.pi)
                self.spiral_search_index = 0
                self.spiral_search_active = True
            else:
                self.spiral_search_active = True
        self.spiral_search_state = np.array([
            float(self.spiral_search_active), float(index + 1),
            float(self.spiral_search_guarded),
        ])
        return self.observe(include_force)

    def _run_control_steps(self, count: int, include_force: bool,
                           target_force_n: float, socket_velocity: float) -> dict:
        started = time.perf_counter()
        peak_external = np.zeros(6, dtype=float)
        self.force_control_target_n = float(np.clip(target_force_n, 0.0, 5.0))
        if self.force_control_target_n == 0.0:
            self.force_control_active = False
            self.force_control_offset_m = 0.0
        with self._viewer_lock():
            for _ in range(max(1, int(count))):
                if include_force and self.force_control_target_n > 0.0:
                    mujoco.mj_forward(self.model, self.data)
                    _, external = self._contact_wrench()
                    self.force_control_active = True
                    force_error = self.force_control_target_n - external[2]
                    self.force_control_offset_m = float(np.clip(
                        self.force_control_offset_m + 0.0005 * force_error * self.model.opt.timestep,
                        -0.010, 0.010,
                    ))
                tool_axis = self.data.site_xmat[self.tool_tip_id].reshape(3, 3)[:, 0]
                target_position = self.osc_target_position + tool_axis * self.force_control_offset_m
                self.data.ctrl[:6] = self._osc_torque(target_position, self.osc_target_rotation)
                self.data.ctrl[6] = float(np.clip(socket_velocity, -60.0, 60.0))
                mujoco.mj_step(self.model, self.data)
                self.steps += 1
                if include_force:
                    mujoco.mj_forward(self.model, self.data)
                    _, external = self._contact_wrench()
                    peak_external = np.where(np.abs(external) > np.abs(peak_external),
                                             external, peak_external)
                engaged, _, _, _, _ = self._valid_socket_engagement()
                self.engaged_seen = self.engaged_seen or engaged
                self.socket_engagement_ticks = self.socket_engagement_ticks + 1 if engaged else 0
        self.ft_peak_external = peak_external
        self._sync_viewer_realtime(started, max(1, int(count)) * self.model.opt.timestep)
        return self.observe(include_force)

    def step(self, target_position: list[float], socket_velocity: float = 0.0,
             count: int = CONTROL_SUBSTEPS,
             include_force: bool = False, target_force_n: float = 0.0,
             spiral_search: bool = False) -> dict:
        if len(target_position) != 3:
            raise ValueError(f"expected 3 absolute Cartesian position targets, got {len(target_position)}")
        target = np.asarray(target_position, dtype=float)
        if not np.isfinite(target).all() or not np.isfinite(socket_velocity):
            raise ValueError("Cartesian position targets and socket velocity must be finite")
        if np.any(target < np.array([-1.0, -1.0, 0.0])) or np.any(target > np.array([1.0, 1.2, 2.0])):
            raise ValueError("absolute Cartesian target is outside the UR20 workspace bounds")
        if spiral_search:
            return self._spiral_search(count, include_force, target_force_n)
        self.osc_target_position = target.copy()
        self.spiral_search_state = np.array([
            0.0, 0.0, float(self.spiral_search_guarded),
        ])
        return self._run_control_steps(count, include_force, target_force_n, socket_velocity)


def main() -> None:
    try:
        worker = Worker()
    except Exception as exc:
        print(json.dumps({"error": f"worker initialization failed: {type(exc).__name__}: {exc}"}),
              flush=True)
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "reset":
                result = worker.reset(request.get("state_id", 0), request.get("include_force", False),
                                      request.get("open_viewer", False))
            elif operation == "observe":
                result = worker.observe(request.get("include_force", False))
            elif operation == "step":
                result = worker.step(request["target_position"], request.get("socket_velocity", 0.0),
                                     request.get("count", 1),
                                     request.get("include_force", False),
                                     request.get("target_force_n", 0.0),
                                     request.get("spiral_search", False))
            elif operation == "close":
                break
            else:
                raise ValueError(f"unknown operation: {operation}")
            print(json.dumps(result), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
    worker.close()


if __name__ == "__main__":
    main()
