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
SPIRAL_SEARCH_POINTS = 576
SPIRAL_SEARCH_MAX_RADIUS_M = 0.008
SPIRAL_SEARCH_TURNS = 16.0 / 3.0
SPIRAL_SEARCH_SOCKET_SPEED_RAD_S = 3.0
SPIRAL_SEARCH_SOCKET_STEP_RAD = np.pi / 3.0
SPIRAL_SEARCH_POSITION_TOLERANCE_M = 0.00025
SPIRAL_FORCE_BASELINE_SETTLE_MAX_STEPS = 2000
SPIRAL_FORCE_BASELINE_SETTLE_TOLERANCE_N = 0.2
SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS = 125
FORCE_ADMITTANCE_GAIN_M_PER_N_S = 0.001
FORCE_ADMITTANCE_MAX_OFFSET_M = 0.005
SOCKET_TURN_SPEED_RAD_S = 0.6
SOCKET_TURN_ANGLE_RAD = 2.0 * np.pi
SOCKET_TURN_DIRECTION = -1.0
SPIRAL_FORCE_DROP_THRESHOLD_N = 0.75
SPIRAL_LATERAL_FORCE_RISE_THRESHOLD_N = 0.75
SPIRAL_LATERAL_FORCE_CONFIRM_STEPS = 5
SPIRAL_FORCE_DROP_EVENT_WINDOW_STEPS = 25
SPIRAL_TACTILE_THRESHOLDS_CALIBRATED = False
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


def osc_anchor_for_tip_target(tip_target: np.ndarray, tool_axis: np.ndarray,
                              force_offset_m: float) -> np.ndarray:
    tip_target = np.asarray(tip_target, dtype=float)
    tool_axis = np.asarray(tool_axis, dtype=float)
    if tip_target.shape != (3,) or tool_axis.shape != (3,):
        raise ValueError("tip target and tool axis must be 3D vectors")
    if not np.isfinite(tip_target).all() or not np.isfinite(tool_axis).all():
        raise ValueError("tip target and tool axis must be finite")
    return tip_target + tool_axis * float(force_offset_m)


def socket_screw_coupling_parameters(start_screw_angle: float,
                                     start_socket_angle: float,
                                     axis_alignment: float) -> tuple[float, float]:
    if not np.isfinite(axis_alignment) or abs(axis_alignment) < 0.99:
        raise ValueError("socket and screw rotation axes must be parallel")
    slope = 1.0 if axis_alignment > 0.0 else -1.0
    return float(start_screw_angle - slope * start_socket_angle), slope


class SpiralTactileDetector:
    def __init__(self, reference_force_n: float,
                 reference_lateral_force_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
                 thresholds_calibrated: bool = SPIRAL_TACTILE_THRESHOLDS_CALIBRATED) -> None:
        self.reference_force_n = float(reference_force_n)
        self.reference_lateral_force_xy = np.asarray(reference_lateral_force_xy, dtype=float)
        if (self.reference_lateral_force_xy.shape != (2,)
                or not np.isfinite(self.reference_lateral_force_xy).all()):
            raise ValueError("reference lateral force must be a finite XY vector")
        self.current_force_n = self.reference_force_n
        self.current_lateral_force_xy = self.reference_lateral_force_xy.copy()
        self.axial_force_drop_n = 0.0
        self.lateral_force_rise_n = 0.0
        self.phase = "seeking_force_drop"
        self.confirmation_steps = 0
        self.force_drop_age_steps = None
        self.force_drop_detected = False
        self.axial_drop_condition_previous = False
        self.thresholds_calibrated = bool(thresholds_calibrated)
        self.candidate_detected = False
        self.seated = False

    def update(self, force_z_n: float,
               lateral_force_xy: np.ndarray | tuple[float, float] = (0.0, 0.0)) -> bool:
        force_z_n = float(force_z_n)
        lateral_force_xy = np.asarray(lateral_force_xy, dtype=float)
        if lateral_force_xy.shape != (2,) or not np.isfinite(lateral_force_xy).all():
            raise ValueError("lateral force must be a finite XY vector")
        self.current_force_n = force_z_n
        self.current_lateral_force_xy = lateral_force_xy
        self.axial_force_drop_n = abs(self.reference_force_n) - abs(force_z_n)
        self.lateral_force_rise_n = float(
            np.linalg.norm(lateral_force_xy)
            - np.linalg.norm(self.reference_lateral_force_xy)
        )
        if self.seated:
            return True
        axial_drop_condition = (
            abs(self.reference_force_n) - abs(force_z_n) >= SPIRAL_FORCE_DROP_THRESHOLD_N
        )
        if axial_drop_condition and not self.axial_drop_condition_previous:
            self.force_drop_detected = True
            self.force_drop_age_steps = 0
            self.phase = "seeking_lateral_force"
        elif self.force_drop_detected:
            self.force_drop_age_steps += 1
            if self.force_drop_age_steps > SPIRAL_FORCE_DROP_EVENT_WINDOW_STEPS:
                self.force_drop_detected = False
                self.force_drop_age_steps = None
                self.confirmation_steps = 0
                self.phase = "seeking_force_drop"
        self.axial_drop_condition_previous = axial_drop_condition
        lateral_force_rise = (
            self.lateral_force_rise_n >= SPIRAL_LATERAL_FORCE_RISE_THRESHOLD_N
        )
        if axial_drop_condition and self.force_drop_detected and lateral_force_rise:
            self.confirmation_steps += 1
            if self.confirmation_steps >= SPIRAL_LATERAL_FORCE_CONFIRM_STEPS:
                self.candidate_detected = True
                if self.thresholds_calibrated:
                    self.phase = "seated"
                    self.seated = True
        else:
            self.confirmation_steps = 0
        return self.seated


class Worker:
    def __init__(self, render: bool = True) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.model.site_rgba[:, 3] = 0.0
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=256, width=256) if render else None
        self.viewer = None
        self.viewer_last_sync = None
        self.camera_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("agentview", "wristview", "screw_front", "screw_side", "screw_back")
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
        self.force_control_direction = 1.0
        self.force_control_offset_m = 0.0
        self.spiral_phase_offset = 0.0
        self.spiral_search_state = np.zeros(3, dtype=float)
        self.spiral_search_active = False
        self.spiral_search_guarded = False
        self.spiral_search_index = 0
        self.spiral_search_center = None
        self.spiral_search_start_axial = None
        self.spiral_search_rotation = None
        self.spiral_reference_force_n = None
        self.spiral_force_baseline_settled = False
        self.spiral_force_baseline_settle_steps = 0
        self.spiral_force_baseline_guard_reason = None
        self.spiral_tactile_detector = None
        self.spiral_force_trace = []
        self.debug_tactile_diagnostics = False
        self.record_socket_opening_trajectory = False
        self.socket_opening_trajectory = []
        self.socket_engagement_ticks = 0
        self.socket_turn_active = False
        self.socket_turn_start_angle = 0.0
        self.socket_turn_start_screw_angle = 0.0
        self.socket_turn_completed = False
        self.osc_target_position = None
        self.osc_target_rotation = None
        self.screw_lift_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "screw_lift")
        self.screw_rotation_lock_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "screw_rotation_lock"
        )
        self.socket_screw_coupling_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "socket_screw_rotation_coupling"
        )
        self.screw_height_lock_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "screw_height_lock"
        )
        self.tool_tip_id = self.site_ids["tool_tip"]
        self.screw_head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
        self.screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "screw_m6")
        self.screw_spin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "screw_spin")
        self.screw_spin_qpos = self.model.jnt_qposadr[self.screw_spin_id]
        self.socket_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "socket_joint"
        )
        self.socket_joint_qpos = self.model.jnt_qposadr[self.socket_joint_id]
        socket_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "socket_velocity"
        )
        self.socket_actuator_id = socket_actuator_id
        self.screw_lift_qpos = self.model.jnt_qposadr[self.screw_lift_id]
        self.socket_wall_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"socket_wall_{index}")
            for index in range(6)
        }
        self.socket_wall_ids.update(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"socket_lip_{index}")
            for index in range(6)
        )
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
        screw_head_center = self.data.geom_xpos[self.screw_head_geom_id].copy()
        head_vertices = np.asarray(self.model.mesh_vert[
            self.model.mesh_vertadr[self.model.geom_dataid[self.screw_head_geom_id]]:
            self.model.mesh_vertadr[self.model.geom_dataid[self.screw_head_geom_id]]
            + self.model.mesh_vertnum[self.model.geom_dataid[self.screw_head_geom_id]]
        ])
        head_rotation = self.data.geom_xmat[self.screw_head_geom_id].reshape(3, 3)
        screw_head_top = screw_head_center.copy()
        screw_head_top[2] += float(np.max((head_vertices @ head_rotation.T)[:, 2]))
        delta = screw_head_top - self.data.site_xpos[self.tool_tip_id]
        axial = float(np.dot(delta, axis))
        radial = float(np.linalg.norm(delta - axial * axis))
        engaged = contact and alignment >= 0.985 and radial <= 0.003 and -0.014 <= axial <= -0.002
        return engaged, contact, axial, radial, alignment

    def _confirmed_socket_seating(self) -> bool:
        detector = self.spiral_tactile_detector
        return bool(detector is not None and detector.candidate_detected)

    def _wrench(self) -> np.ndarray:
        return np.concatenate((
            self.data.sensordata[self.ft_sensor_adr:self.ft_sensor_adr + 3],
            self.data.sensordata[self.torque_sensor_adr:self.torque_sensor_adr + 3],
        )).copy()

    def _spiral_tactile_state(self) -> list[float]:
        detector = self.spiral_tactile_detector
        if detector is None:
            return [0.0] * 7
        candidate = detector.candidate_detected
        confirmed = detector.seated
        phase = {
            "seeking_force_drop": 0.0,
            "seeking_lateral_force": 1.0,
            "seated": 2.0,
        }[detector.phase] if not candidate or confirmed else 3.0
        return [phase, float(confirmed),
                detector.axial_force_drop_n,
                detector.lateral_force_rise_n,
                float(detector.confirmation_steps),
                float(detector.force_drop_detected),
                float(detector.force_drop_age_steps or 0)]

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
        self.viewer.sync()
        self.viewer_last_sync = time.perf_counter()

    def _viewer_lock(self):
        if self.viewer is not None and self.viewer.is_running():
            return self.viewer.lock()
        return nullcontext()

    def _sync_viewer_realtime(self, started: float, simulated_seconds: float) -> None:
        if self.spiral_search_active:
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
                self.viewer_last_sync = time.perf_counter()
            return
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
                "spiral_tactile_seated": bool(
                    self.spiral_tactile_detector and self.spiral_tactile_detector.seated
                    and SPIRAL_TACTILE_THRESHOLDS_CALIBRATED
                ),
                "spiral_tactile_phase": (
                    ("candidate_uncalibrated"
                     if self.spiral_tactile_detector
                     and self.spiral_tactile_detector.candidate_detected
                     and not self.spiral_tactile_detector.seated
                     else self.spiral_tactile_detector.phase)
                    if self.spiral_tactile_detector else "idle"
                ),
                "spiral_tactile_state": self._spiral_tactile_state(),
                "spiral_force_baseline_settled": self.spiral_force_baseline_settled,
                "spiral_force_trace": list(self.spiral_force_trace),
                "socket_turn_state": [
                    float(self.socket_turn_active),
                    float(self.data.qpos[self.socket_joint_qpos] - self.socket_turn_start_angle)
                    if self.socket_turn_active or self.socket_turn_completed else 0.0,
                    float(self.socket_turn_completed),
                ],
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
            return result

    def reset(self, state_id: int = 0, include_force: bool = False,
              open_viewer: bool = False) -> dict:
        with self._viewer_lock():
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self._place_start(int(state_id))
            mujoco.mj_forward(self.model, self.data)
            self.osc_target_rotation = self.data.site_xmat[self.tool_tip_id].reshape(3, 3).copy()
            self.osc_target_position = osc_anchor_for_tip_target(
                self.data.site_xpos[self.tool_tip_id],
                self.osc_target_rotation[:, 0],
                self.force_control_offset_m,
            )
        self.steps = 0
        self.engaged_seen = False
        self.ft_peak_external = np.zeros(6, dtype=float)
        self.force_control_active = False
        self.force_control_target_n = 0.0
        self.force_control_offset_m = 0.0
        self.spiral_phase_offset = 0.0
        self.spiral_search_state = np.zeros(3, dtype=float)
        self.socket_turn_active = False
        self.socket_turn_start_angle = float(self.data.qpos[self.socket_joint_qpos])
        self.socket_turn_start_screw_angle = float(self.data.qpos[self.screw_spin_qpos])
        self.socket_turn_completed = False
        self.spiral_search_active = False
        self.spiral_search_guarded = False
        self.spiral_search_center = None
        self.spiral_search_start_axial = None
        self.spiral_search_rotation = None
        self.spiral_reference_force_n = None
        self.spiral_force_baseline_settled = False
        self.spiral_force_baseline_settle_steps = 0
        self.spiral_force_baseline_guard_reason = None
        self.spiral_tactile_detector = None
        self.spiral_force_trace = []
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
        position_kp = OSC_KP[:3].copy()
        position_kd = OSC_KD[:3].copy()
        if self.spiral_search_active:
            position_kp[:2] *= 16.0
            position_kd[:2] *= 4.0
        desired_acceleration = np.concatenate((
            position_kp * position_error - position_kd * twist[:3],
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

    def _begin_spiral_search(self, target_force_n: float) -> dict:
        with self._viewer_lock():
            self.force_control_target_n = float(np.clip(target_force_n, 0.0, 5.0))
            tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            tip_rotation_now = self.data.site_xmat[self.tool_tip_id].reshape(3, 3).copy()
        tool_axis = tip_rotation_now[:, 0]
        self.osc_target_position = tip_position.copy()
        self.osc_target_rotation = tip_rotation_now
        initial_raw, initial_wrench = self._contact_wrench()
        if abs(float(initial_wrench[2])) > 0.1:
            self.force_control_direction = float(np.sign(initial_wrench[2]))
        self.spiral_force_baseline_settle_steps = 0
        self.spiral_force_baseline_settled = abs(float(initial_wrench[2])) >= 0.2
        if np.linalg.norm(initial_wrench[:3]) >= 50.0:
            self.spiral_search_guarded = True
            self.spiral_force_baseline_guard_reason = "force_limit"
            self.spiral_search_state = np.array([0.0, 0.0, 1.0])
            return self.observe(include_force=True)
        result = self.observe(include_force=True)
        with self._viewer_lock():
            tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            tip_rotation_now = self.data.site_xmat[self.tool_tip_id].reshape(3, 3).copy()
        self.spiral_search_center = tip_position.copy()
        self.spiral_search_rotation = tip_rotation_now
        self.spiral_search_start_axial = float(
            np.dot(
                self.data.geom_xpos[self.screw_head_geom_id]
                - self.data.site_xpos[self.tool_tip_id],
                tip_rotation_now[:, 0],
            )
        )
        self.spiral_reference_force_n = float(initial_wrench[2])
        self.spiral_tactile_detector = SpiralTactileDetector(
            self.spiral_reference_force_n,
            initial_wrench[:2],
            SPIRAL_TACTILE_THRESHOLDS_CALIBRATED,
        )
        self.spiral_search_index = 0
        self.spiral_search_active = True
        if self.debug_tactile_diagnostics:
            result["spiral_baseline_diagnostic"] = {
                "initial_axial_force_n": float(initial_wrench[2]),
                "settled_axial_force_n": float(initial_wrench[2]),
                "target_force_n": float(target_force_n),
                "settle_steps": 0,
                "settle_force_trace_n": [],
                "baseline_settled": self.spiral_force_baseline_settled,
                "force_offset_m": float(self.force_control_offset_m),
                "raw_initial_wrench": initial_raw.tolist(),
            }
        return result

    def _spiral_search(self, count: int, include_force: bool,
                       target_force_n: float) -> dict:
        if not include_force or target_force_n <= 0:
            raise ValueError("spiral search requires force feedback and a positive axial force setpoint")
        if not self.spiral_search_active:
            settle_result = self._begin_spiral_search(target_force_n)
            if self.spiral_search_guarded:
                return settle_result
        self.spiral_force_trace = []
        index = self.spiral_search_index
        tip_position = self.spiral_search_center
        tip_rotation = self.spiral_search_rotation
        sensor_rotation = self.data.site_xmat[self.ft_sensor_site_id].reshape(3, 3).copy()
        radius = SPIRAL_SEARCH_MAX_RADIUS_M * (index + 1) / SPIRAL_SEARCH_POINTS
        angle = (self.spiral_phase_offset + index * 2.0 * np.pi
                 * SPIRAL_SEARCH_TURNS / SPIRAL_SEARCH_POINTS)
        desired_tip_position = tip_position + radius * (
            np.cos(angle) * sensor_rotation[:, 0]
            + np.sin(angle) * sensor_rotation[:, 1]
        )
        tool_axis = tip_rotation[:, 0]
        self.osc_target_position = desired_tip_position.copy()
        self.osc_target_rotation = tip_rotation
        rotation_start = float(self.data.qpos[self.socket_joint_qpos])
        rotation_target = SPIRAL_SEARCH_SOCKET_STEP_RAD
        rotation_speed = SOCKET_TURN_DIRECTION * SPIRAL_SEARCH_SOCKET_SPEED_RAD_S
        rotation_steps = 0
        max_rotation_steps = max(1, int(np.ceil(
            rotation_target / (SPIRAL_SEARCH_SOCKET_SPEED_RAD_S * self.model.opt.timestep)
        )))
        max_point_steps = max_rotation_steps * 20
        result = None
        while rotation_steps < max_point_steps:
            angle_progress = SOCKET_TURN_DIRECTION * (
                float(self.data.qpos[self.socket_joint_qpos]) - rotation_start
            )
            actual_tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            point_reached = np.linalg.norm(
                actual_tip_position - desired_tip_position
            ) <= SPIRAL_SEARCH_POSITION_TOLERANCE_M
            socket_stepped = angle_progress >= rotation_target - (
                SPIRAL_SEARCH_SOCKET_SPEED_RAD_S * self.model.opt.timestep
            )
            if socket_stepped:
                break
            result = self._run_control_steps(
                min(count, max_point_steps - rotation_steps,
                    max(1, max_rotation_steps - int(angle_progress / (
                        SPIRAL_SEARCH_SOCKET_SPEED_RAD_S * self.model.opt.timestep
                    ))) if not socket_stepped else count),
                include_force=True, target_force_n=target_force_n,
                socket_velocity=rotation_speed if not socket_stepped else 0.0,
            )
            rotation_steps += min(count, max_point_steps - rotation_steps)
            peak = np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
            tactile_candidate = bool(
                self.spiral_tactile_detector
                and self.spiral_tactile_detector.candidate_detected
            )
            if np.linalg.norm(peak) >= 50.0 or tactile_candidate:
                break
        if result is None:
            result = self.observe(include_force=True)
        else:
            actual_tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            point_reached = np.linalg.norm(
                actual_tip_position - desired_tip_position
            ) <= SPIRAL_SEARCH_POSITION_TOLERANCE_M
            socket_stepped = (
                SOCKET_TURN_DIRECTION
                * (float(self.data.qpos[self.socket_joint_qpos]) - rotation_start)
                >= rotation_target - SPIRAL_SEARCH_SOCKET_SPEED_RAD_S * self.model.opt.timestep
            )
            if not socket_stepped:
                if np.linalg.norm(np.asarray(result["force_torque_peak_compensated"][:3])) < 50.0:
                    self.spiral_search_guarded = True
                    self.spiral_force_baseline_guard_reason = "socket_step_incomplete"
            result["spiral_target_tracking_error_m"] = float(
                np.linalg.norm(actual_tip_position - desired_tip_position)
            )
        self.data.ctrl[6] = 0.0
        peak = np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
        force_guarded = np.linalg.norm(peak) >= 50.0
        tactile_candidate = bool(
            self.spiral_tactile_detector
            and self.spiral_tactile_detector.candidate_detected
        )
        if force_guarded and self.spiral_tactile_detector is not None:
            self.spiral_tactile_detector.seated = False
            self.spiral_tactile_detector.candidate_detected = False
            self.spiral_tactile_detector.phase = "seeking_force_drop"
            self.spiral_tactile_detector.confirmation_steps = 0
            self.spiral_tactile_detector.force_drop_detected = False
            self.spiral_tactile_detector.force_drop_age_steps = None
            self.spiral_tactile_detector.axial_force_drop_n = 0.0
            self.spiral_tactile_detector.lateral_force_rise_n = 0.0
            tactile_candidate = False
        seating_confirmed = tactile_candidate
        step_failed = self.spiral_force_baseline_guard_reason in {
            "spiral_point_unreached", "socket_step_incomplete"
        }
        if seating_confirmed or force_guarded or step_failed:
            self.data.ctrl[6] = 0.0
            self.spiral_search_active = False
            self.spiral_search_guarded = (force_guarded and not seating_confirmed) or step_failed
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
        result.update(self.observe(include_force))
        if self.debug_tactile_diagnostics:
            result["spiral_force_trace"] = self.spiral_force_trace
        return result

    def socket_turn_step(self, count: int = CONTROL_SUBSTEPS,
                         target_force_n: float = 2.0) -> dict:
        if not self.socket_turn_active:
            engaged, _, _, _, _ = self._valid_socket_engagement()
            if not (self.spiral_tactile_detector and self.spiral_tactile_detector.seated
                    and SPIRAL_TACTILE_THRESHOLDS_CALIBRATED):
                raise ValueError("socket turning requires confirmed tactile seating")
            self.socket_turn_active = True
            self.socket_turn_completed = False
            self.socket_turn_start_angle = float(self.data.qpos[self.socket_joint_qpos])
            self.socket_turn_start_screw_angle = float(self.data.qpos[self.screw_spin_qpos])
            self.data.eq_active[self.screw_rotation_lock_id] = 0
            axis_alignment = float(
                np.dot(
                    self.data.xaxis[self.screw_spin_id],
                    self.data.xaxis[self.socket_joint_id],
                )
            )
            self.model.eq_data[self.socket_screw_coupling_id, :2] = (
                socket_screw_coupling_parameters(
                    self.socket_turn_start_screw_angle,
                    self.socket_turn_start_angle,
                    axis_alignment,
                )
            )
            self.data.eq_active[self.socket_screw_coupling_id] = 1
            self.osc_target_position = self.data.site_xpos[self.tool_tip_id].copy()
            self.osc_target_rotation = self.data.site_xmat[self.tool_tip_id].reshape(3, 3).copy()

        result = self._run_control_steps(
            count, include_force=True, target_force_n=target_force_n,
            socket_velocity=SOCKET_TURN_DIRECTION * SOCKET_TURN_SPEED_RAD_S,
        )
        elapsed_angle = float(self.data.qpos[self.socket_joint_qpos] - self.socket_turn_start_angle)
        force_peak = np.linalg.norm(np.asarray(result["force_torque_peak_compensated"][:3]))
        still_engaged = bool(result["socket_engaged"])
        if force_peak >= 50.0 or not still_engaged:
            self._finish_socket_turn()
            result["socket_turn_stop_reason"] = (
                "force_guard" if force_peak >= 50.0 else "socket_engagement_lost"
            )
        elif abs(elapsed_angle) >= SOCKET_TURN_ANGLE_RAD:
            self._finish_socket_turn(completed=True)
            result["socket_turn_stop_reason"] = "one_revolution_completed"
        else:
            result["socket_turn_stop_reason"] = None
        result["socket_turn_state"] = [
            float(self.socket_turn_active),
            float(self.data.qpos[self.socket_joint_qpos] - self.socket_turn_start_angle),
            float(self.socket_turn_completed),
        ]
        result["screw_rotation_delta"] = float(
            self.data.qpos[self.screw_spin_qpos] - self.socket_turn_start_screw_angle
        )
        return result

    def _finish_socket_turn(self, completed: bool = False) -> None:
        with self._viewer_lock():
            self.data.ctrl[6] = 0.0
            self.data.eq_active[self.socket_screw_coupling_id] = 0
            self.model.eq_data[self.screw_rotation_lock_id, 0] = self.data.qpos[self.screw_spin_qpos]
            self.data.eq_active[self.screw_rotation_lock_id] = 1
            self.socket_turn_active = False
            self.socket_turn_completed = completed

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
                    measured_force_n = external[2]
                    force_error = self.force_control_direction * (
                        self.force_control_target_n
                        - self.force_control_direction * measured_force_n
                    )
                    self.force_control_offset_m = float(np.clip(
                        self.force_control_offset_m
                        + FORCE_ADMITTANCE_GAIN_M_PER_N_S * force_error
                        * self.model.opt.timestep,
                        -FORCE_ADMITTANCE_MAX_OFFSET_M,
                        FORCE_ADMITTANCE_MAX_OFFSET_M,
                    ))
                tool_axis = self.data.site_xmat[self.tool_tip_id].reshape(3, 3)[:, 0]
                target_position = self.osc_target_position + tool_axis * self.force_control_offset_m
                self.data.ctrl[:6] = self._osc_torque(target_position, self.osc_target_rotation)
                self.data.ctrl[6] = float(np.clip(socket_velocity, -60.0, 60.0))
                mujoco.mj_step(self.model, self.data)
                self.steps += 1
                if self.record_socket_opening_trajectory:
                    opening_center = self.data.site_xpos[self.tool_tip_id].copy()
                    self.socket_opening_trajectory.append({
                        "sim_time_s": float(self.data.time),
                        "world_m": opening_center.tolist(),
                    })
                if include_force:
                    mujoco.mj_forward(self.model, self.data)
                    _, external = self._contact_wrench()
                    peak_external = np.where(np.abs(external) > np.abs(peak_external),
                                             external, peak_external)
                    force_guard_triggered = np.linalg.norm(external[:3]) >= 50.0
                    engaged_diagnostic = self._valid_socket_engagement()[0]
                    engagement_ticks_diagnostic = (
                        self.socket_engagement_ticks + 1 if engaged_diagnostic else 0
                    )
                    if self.spiral_search_active and self.spiral_tactile_detector is not None:
                        self.spiral_tactile_detector.update(
                            external[2], external[:2]
                        )
                        socket_screw_normal_force_n = 0.0
                        socket_screw_contact_count = 0
                        if self.debug_tactile_diagnostics:
                            for contact_index, contact in enumerate(
                                self.data.contact[:self.data.ncon]
                            ):
                                is_socket_screw_contact = (
                                    contact.geom1 == self.screw_head_geom_id
                                    and contact.geom2 in self.socket_wall_ids
                                ) or (
                                    contact.geom2 == self.screw_head_geom_id
                                    and contact.geom1 in self.socket_wall_ids
                                )
                                if is_socket_screw_contact:
                                    contact_force = np.zeros(6, dtype=float)
                                    mujoco.mj_contactForce(
                                        self.model, self.data, contact_index, contact_force
                                    )
                                    socket_screw_normal_force_n += max(
                                        0.0, float(contact_force[0])
                                    )
                                    socket_screw_contact_count += 1
                        trace_sample = {
                            "time": float(self.data.time),
                            "axial_force_n": float(external[2]),
                            "axial_drop_from_baseline_n": float(
                                self.spiral_tactile_detector.axial_force_drop_n
                            ),
                            "lateral_force_xy_n": external[:2].tolist(),
                            "lateral_force_magnitude_n": float(np.linalg.norm(external[:2])),
                            "lateral_rise_from_baseline_n": float(
                                self.spiral_tactile_detector.lateral_force_rise_n
                            ),
                            "force_drop_detected": bool(
                                self.spiral_tactile_detector.force_drop_detected
                            ),
                            "lateral_confirmation_steps": int(
                                self.spiral_tactile_detector.confirmation_steps
                            ),
                            "axial_drop_event_age_steps": (
                                self.spiral_tactile_detector.force_drop_age_steps
                            ),
                            "socket_engaged_diagnostic": bool(engaged_diagnostic),
                            "socket_engagement_ticks_diagnostic": int(
                                engagement_ticks_diagnostic
                            ),
                            "socket_seated_diagnostic": engagement_ticks_diagnostic >= 20,
                            "socket_screw_normal_contact_force_n_private": (
                                socket_screw_normal_force_n
                            ),
                            "socket_screw_contact_count_private": socket_screw_contact_count,
                        }
                        if self.debug_tactile_diagnostics and (
                            self.spiral_tactile_detector.force_drop_detected
                            and self.spiral_tactile_detector.axial_force_drop_n
                            >= SPIRAL_FORCE_DROP_THRESHOLD_N
                            and self.spiral_tactile_detector.lateral_force_rise_n
                            >= SPIRAL_LATERAL_FORCE_RISE_THRESHOLD_N
                        ):
                            trace_sample["contact_pairs"] = [
                                {
                                    "geom1": mujoco.mj_id2name(
                                        self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                                    ),
                                    "geom2": mujoco.mj_id2name(
                                        self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                                    ),
                                    "distance_m": float(contact.dist),
                                }
                                for contact in self.data.contact[:self.data.ncon]
                            ]
                        if self.debug_tactile_diagnostics:
                            trace_sample["moment_xyz_nm_private"] = external[3:].tolist()
                            trace_sample["socket_motor_torque_nm_private"] = float(
                                self.data.actuator_force[self.socket_actuator_id]
                            )
                            trace_sample["tool_tip_xy_private_m"] = (
                                self.data.site_xpos[self.tool_tip_id, :2].tolist()
                            )
                            trace_sample["socket_angle_private_rad"] = float(
                                self.data.qpos[self.socket_joint_qpos]
                            )
                        self.spiral_force_trace.append(trace_sample)
                    if force_guard_triggered:
                        break
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
        self.force_control_offset_m = 0.0
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
            elif operation == "socket_turn_step":
                result = worker.socket_turn_step(request.get("count", CONTROL_SUBSTEPS),
                                                 request.get("target_force_n", 2.0))
            elif operation == "socket_turn_stop":
                worker._finish_socket_turn()
                result = worker.observe(include_force=True)
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
