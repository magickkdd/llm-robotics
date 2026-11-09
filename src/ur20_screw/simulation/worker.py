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
from ur20_screw.vision.estimation import (
    estimate_pair,
    project_pixel_with_depth,
)
from ur20_screw.simulation.timing import wait_for_simulated_time
from ur20_screw.target import SCREW_TARGET

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "assets/ur20_screw_station/scene.xml"
CONTROL_SUBSTEPS = 25
SPIRAL_SEARCH_POINTS = 576
SPIRAL_SEARCH_MAX_RADIUS_M = 0.008
SPIRAL_SEARCH_TURNS = 16.0 / 3.0
SPIRAL_SEARCH_SOCKET_SPEED_RAD_S = 3.0
SPIRAL_SEARCH_SOCKET_STEP_RAD = np.pi / 3.0
SPIRAL_SEARCH_POSITION_TOLERANCE_M = 0.00025
SPIRAL_FORCE_BASELINE_SETTLE_MAX_STEPS = 2000
SPIRAL_FORCE_BASELINE_SETTLE_TOLERANCE_N = 0.4
SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS = 125
FORCE_ADMITTANCE_GAIN_M_PER_N_S = 0.003
FORCE_ADMITTANCE_MAX_OFFSET_M = 0.005
# 速度伺服 kv=0.5 在接触摩擦下约有 0.2 rad/s 静差：命令 2.0 rad/s 实际约
# 1.8 rad/s，21 mm 旋出约 75 s（原 0.6 rad/s 实测约 5.4 min）。
SOCKET_TURN_SPEED_RAD_S = 2.0
SCREW_THREAD_PITCH_M = 0.001
SCREW_EXTRACTION_DISTANCE_M = 0.021
SOCKET_TURN_ANGLE_RAD = (
    SCREW_EXTRACTION_DISTANCE_M / SCREW_THREAD_PITCH_M * 2.0 * np.pi
)
SOCKET_TURN_DIRECTION = -1.0
SPIRAL_FORCE_DROP_THRESHOLD_N = 0.75
SPIRAL_LATERAL_FORCE_RISE_THRESHOLD_N = 0.30
SPIRAL_FORCE_TACTILE_LATERAL_RISE_THRESHOLD_N = 0.75
SPIRAL_LATERAL_FORCE_CONFIRM_STEPS = 5
SPIRAL_FORCE_DROP_EVENT_WINDOW_STEPS = 25
SPIRAL_SEATING_DOWNWARD_TRAVEL_M = 0.003
SOCKET_HEX_ENGAGEMENT_DEPTH_M = 0.006
SPIRAL_MIN_SOCKET_INSERTION_M = SOCKET_HEX_ENGAGEMENT_DEPTH_M / 2.0
SPIRAL_SEATED_DOWNWARD_TRAVEL_M = SPIRAL_MIN_SOCKET_INSERTION_M
SPIRAL_SEATED_MIN_AXIAL_FORCE_RATIO = 0.1
SPIRAL_SEATED_CONFIRM_STEPS = 20
SPIRAL_TACTILE_THRESHOLDS_CALIBRATED = False
# 每个螺旋点转动完成后停转下压驻留，再回到接触高度进入下一点；
# 直接带压转动会被六角棱边卡住（socket_step_incomplete）。预压保持轻微：
# 对齐后靠轴向力即可自行套入，过大预压会顶住法兰，横向阻力使螺旋跟踪误差累积。
SPIRAL_DOWNWARD_PRELOAD_M = 0.001
# 轻压下探时保持自转：套筒六角只有 6 重对称，每个螺旋点转 60° 不改变相对相位，
# 必须靠下压期间的自转扫过相位，套筒才能沿棱边滑入（实测可容忍约 1.2 mm 横向偏差）。
# 每个点驻留 0.7 s、1.5 rad/s，正好扫过约 60°，保证任意起始相位都能被覆盖一次。
SPIRAL_PRESS_ROTATION_RAD_S = 1.5
# 用相机估计（不是仿真真值）否定“压在法兰上”的假 seated。
VISION_SEATING_RADIUS_M = 0.005
# 螺钉在旋出前不会横向移动；新估计若相对上一次高置信估计跳变超过该距离，
# 视为误检（例如锁到相邻装饰螺钉），不更新对准目标。
VISION_XY_JUMP_LIMIT_M = 0.005
SPIRAL_PRESS_STEPS = 350
SPIRAL_RELEASE_STEPS = 50
TOOL_TIP_REACH = 0.4979
DOWNWARD_FLANGE_ROTATION = np.diag([1.0, -1.0, -1.0])
ARM_TORQUE_LIMITS = np.array([738.0, 738.0, 433.0, 107.0, 107.0, 107.0])
OSC_KP = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
OSC_KD = np.array([40.0, 40.0, 40.0, 48.0, 48.0, 48.0])
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
                 thresholds_calibrated: bool = SPIRAL_TACTILE_THRESHOLDS_CALIBRATED,
                 reference_tool_z_m: float | None = None) -> None:
        self.reference_force_n = float(reference_force_n)
        self.reference_tool_z_m = reference_tool_z_m
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
        self.candidate_latched = False
        self.candidate_rearm_blocked = False
        self.seated = False
        self.downward_travel_m = 0.0
        self.seated_confirmation_steps = 0
        self.candidate_cooldown_steps = 0

    def update(self, force_z_n: float,
               lateral_force_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
               tool_z_m: float | None = None) -> bool:
        force_z_n = float(force_z_n)
        lateral_force_xy = np.asarray(lateral_force_xy, dtype=float)
        if lateral_force_xy.shape != (2,) or not np.isfinite(lateral_force_xy).all():
            raise ValueError("lateral force must be a finite XY vector")
        self.current_force_n = force_z_n
        self.current_lateral_force_xy = lateral_force_xy
        self.candidate_cooldown_steps = max(0, self.candidate_cooldown_steps - 1)
        if tool_z_m is not None and self.reference_tool_z_m is not None:
            self.downward_travel_m = self.reference_tool_z_m - float(tool_z_m)
        self.axial_force_drop_n = (
            abs(self.reference_force_n)
            - np.sign(self.reference_force_n or 1.0) * force_z_n
        )
        self.lateral_force_rise_n = float(
            np.linalg.norm(lateral_force_xy)
            - np.linalg.norm(self.reference_lateral_force_xy)
        )
        if self.seated:
            return True
        seated_sensor_condition = (
            self.downward_travel_m >= SPIRAL_SEATED_DOWNWARD_TRAVEL_M
            and abs(force_z_n) >= (
                abs(self.reference_force_n) * SPIRAL_SEATED_MIN_AXIAL_FORCE_RATIO
            )
        )
        self.seated_confirmation_steps = (
            self.seated_confirmation_steps + 1 if seated_sensor_condition else 0
        )
        if self.seated_confirmation_steps >= SPIRAL_SEATED_CONFIRM_STEPS:
            self.seated = True
            self.candidate_detected = True
            self.candidate_latched = True
            self.phase = "seated"
            return True
        axial_drop_condition = self.axial_force_drop_n >= SPIRAL_FORCE_DROP_THRESHOLD_N
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
        force_tactile_cue = (
            axial_drop_condition
            and self.lateral_force_rise_n
            >= SPIRAL_FORCE_TACTILE_LATERAL_RISE_THRESHOLD_N
        )
        encoder_cue = (
            self.downward_travel_m >= SPIRAL_SEATING_DOWNWARD_TRAVEL_M
            and lateral_force_rise
        )
        candidate_cue = force_tactile_cue or encoder_cue
        if self.candidate_rearm_blocked and not candidate_cue:
            self.candidate_rearm_blocked = False
        if (not self.candidate_rearm_blocked and self.candidate_cooldown_steps == 0
                and candidate_cue):
            self.confirmation_steps += 1
            if self.confirmation_steps >= SPIRAL_LATERAL_FORCE_CONFIRM_STEPS:
                self.candidate_detected = True
                self.candidate_latched = True
                if self.thresholds_calibrated:
                    self.phase = "seated"
                    self.seated = True
        elif self.candidate_latched:
            self.candidate_detected = True
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
        self.spiral_search_exhausted = False
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
        self.socket_fit_ticks = 0
        self.max_socket_fit_ticks = 0
        self.socket_turn_active = False
        self.socket_turn_start_angle = 0.0
        self.socket_turn_start_screw_angle = 0.0
        self.socket_turn_completed = False
        self.osc_target_position = None
        self.osc_target_rotation = None
        self.last_valid_screw_xy = None
        self.screw_lift_id = self._named_id(
            mujoco.mjtObj.mjOBJ_JOINT, SCREW_TARGET.lift_joint
        )
        self.screw_rotation_lock_id = self._named_id(
            mujoco.mjtObj.mjOBJ_EQUALITY, SCREW_TARGET.rotation_lock
        )
        self.socket_screw_coupling_id = self._named_id(
            mujoco.mjtObj.mjOBJ_EQUALITY, SCREW_TARGET.socket_screw_coupling
        )
        self.screw_height_lock_id = self._named_id(
            mujoco.mjtObj.mjOBJ_EQUALITY, SCREW_TARGET.height_lock
        )
        self.screw_thread_pitch_coupling_id = self._named_id(
            mujoco.mjtObj.mjOBJ_EQUALITY, SCREW_TARGET.thread_pitch_coupling
        )
        self.tool_tip_id = self.site_ids["tool_tip"]
        self.tool_base_body_id = self._named_id(
            mujoco.mjtObj.mjOBJ_BODY, "end_effector"
        )
        self.screw_head_geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM, SCREW_TARGET.head_geom
        )
        self.screw_body_id = self._named_id(
            mujoco.mjtObj.mjOBJ_BODY, SCREW_TARGET.body
        )
        self.screw_spin_id = self._named_id(
            mujoco.mjtObj.mjOBJ_JOINT, SCREW_TARGET.spin_joint
        )
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

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(
                f"目标螺钉配置引用了模型中不存在的对象 {name!r}；"
                "请同步修改 scene.xml 与 src/ur20_screw/target.py"
            )
        return object_id

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

    def _sensor_triggered_seating_turn(self, target_force_n: float) -> tuple[dict, float]:
        detector = self.spiral_tactile_detector
        if detector is None or not detector.candidate_detected:
            return self.observe(include_force=True), 0.0
        tip_position = self.data.site_xpos[self.tool_tip_id].copy()
        tool_axis = self.data.site_xmat[self.tool_tip_id].reshape(3, 3)[:, 0]
        self.osc_target_position = tip_position - tool_axis * self.force_control_offset_m
        self.osc_target_rotation = self.spiral_search_rotation
        rotation_start = float(self.data.qpos[self.socket_joint_qpos])
        target_rotation = SPIRAL_SEARCH_SOCKET_STEP_RAD
        rotation_speed = min(SPIRAL_SEARCH_SOCKET_SPEED_RAD_S, SOCKET_TURN_SPEED_RAD_S)
        max_steps = 3 * max(1, int(np.ceil(
            target_rotation / (rotation_speed * self.model.opt.timestep)
        )))
        steps = 0
        result = None
        while steps < max_steps:
            angle_progress = SOCKET_TURN_DIRECTION * (
                float(self.data.qpos[self.socket_joint_qpos]) - rotation_start
            )
            if angle_progress >= target_rotation - (
                rotation_speed * self.model.opt.timestep
            ):
                break
            batch = min(CONTROL_SUBSTEPS, max_steps - steps, max(1, int(np.ceil(
                (target_rotation - angle_progress)
                / (rotation_speed * self.model.opt.timestep)
            ))))
            result = self._run_control_steps(
                batch, True, target_force_n, SOCKET_TURN_DIRECTION * rotation_speed
            )
            steps += batch
            if np.linalg.norm(np.asarray(result["force_torque_peak_compensated"][:3])) >= 50.0:
                break
        self.data.ctrl[6] = 0.0
        if result is None:
            result = self.observe(include_force=True)
        rotation_delta = SOCKET_TURN_DIRECTION * (
            float(self.data.qpos[self.socket_joint_qpos]) - rotation_start
        )
        return result, float(np.degrees(rotation_delta))

    def _advance_spiral_search_index(self) -> None:
        self.spiral_search_index += 1
        if self.spiral_search_index >= SPIRAL_SEARCH_POINTS:
            self.spiral_search_exhausted = True
            self.spiral_search_active = False
        else:
            self.spiral_search_active = True

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

    def _render_rgb_depth(self, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.update_scene(self.data, camera=camera_id)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_id)
        depth = np.asarray(self.renderer.render(), dtype=np.float32)
        self.renderer.disable_depth_rendering()
        return rgb, depth

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

    def _head_top_centroid(self, depth_image: np.ndarray, camera_id: int,
                           bbox_xywh: tuple[int, int, int, int],
                           pad: int = 2, band_m: float = 0.0015) -> np.ndarray | None:
        """Return the world-frame centroid of the screw head-top surface.

        The bright-blob centroid is biased toward the visible head side face and
        flange, and the bbox may also clip the socket above. Selecting the depth
        samples on the calibrated head-top plane removes both effects. Only the
        station's nominal head-top height is used, never the screw's true pose.
        """
        height, width = depth_image.shape[:2]
        focal = height / (2.0 * np.tan(
            np.deg2rad(float(self.model.cam_fovy[camera_id])) / 2.0))
        x, y, box_width, box_height = bbox_xywh
        x0, x1 = max(0, x - pad), min(width, x + box_width + pad)
        y0, y1 = max(0, y - pad), min(height, y + box_height + pad)
        window = depth_image[y0:y1, x0:x1]
        rows, columns = np.nonzero(
            np.isfinite(window) & (window > 0.02) & (window < 10.0))
        if rows.size < 5:
            return None
        depth = window[rows, columns]
        rays = np.stack((
            (columns + x0 - width / 2.0) / focal,
            -(rows + y0 - height / 2.0) / focal,
            -np.ones_like(depth),
        ), axis=1)
        rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        points = self.data.cam_xpos[camera_id] + (rotation @ rays.T).T * depth[:, None]
        selected = np.abs(points[:, 2] - SCREW_TARGET.head_top_z_m) <= band_m
        if selected.sum() < 5:
            return None
        patch = points[selected]
        # 头顶面被工具本体挡住一部分时，可见区域只是月牙，质心会朝未遮挡一侧偏移
        # （wristview 在近距离实测可达 3.7 mm）。要求可见面两个方向都足够大，否则
        # 视为“看不到完整头顶面”，返回 None 交由另一台相机处理。
        centered = patch[:, :2] - patch[:, :2].mean(axis=0)
        extents = np.sqrt(np.linalg.eigvalsh(np.cov(centered.T))) * 4.0
        if float(extents.min()) < 0.005 or patch.shape[0] < 8:
            return None
        return patch.mean(axis=0)

    def observe(self, include_force: bool = False) -> dict:
        with self._viewer_lock():
            mujoco.mj_forward(self.model, self.data)
            socket_engaged, socket_contact, socket_axial, socket_radial, axis_alignment = self._valid_socket_engagement()
            self.engaged_seen = self.engaged_seen or socket_engaged
            rendered = {name: self._render_rgb_depth(camera_id)
                      for name, camera_id in self.camera_ids.items()}
            frames = {name: pair[0] for name, pair in rendered.items()}
            vision_estimate = estimate_pair(frames["agentview"], frames["wristview"])
            # 按要求优先使用刚性固定在腕部工具上的 wristview（近距离、分辨率高）；
            # 它看不到螺钉（初始高度或套筒遮挡）时回退到固定对准孔位的 agentview。
            # 两者都是标定相机，不使用仿真真值。
            source = None
            detection = None
            for candidate in ("wristview", "agentview"):
                candidate_detection = getattr(vision_estimate, f"{candidate}_screw")
                if (candidate_detection.center_px is not None
                        and candidate_detection.confidence >= 0.5):
                    source, detection = candidate, candidate_detection
                    break
            screw_pixel = detection.center_px if detection is not None else None
            screw_position = None
            screw_depth = None
            plane_verified = False
            if source is not None:
                camera_id = self.camera_ids[source]
                depth_image = rendered[source][1]
                screw_position = self._head_top_centroid(
                    depth_image, camera_id, detection.bbox_xywh)
                if screw_position is not None:
                    plane_verified = True
                    screw_depth = float(np.linalg.norm(
                        screw_position - self.data.cam_xpos[camera_id]))
                else:
                    # 头顶面被遮挡或深度不足时退回到像素-深度投影，仅用于显示；
                    # 该估计可能落在相邻螺钉或工具上，不参与对准与螺旋中心。
                    px, py = (int(round(value)) for value in screw_pixel)
                    y0, y1 = max(0, py - 2), min(depth_image.shape[0], py + 3)
                    x0, x1 = max(0, px - 2), min(depth_image.shape[1], px + 3)
                    samples = depth_image[y0:y1, x0:x1]
                    valid_depth = samples[np.isfinite(samples) & (samples > 0.02) & (samples < 10.0)]
                    screw_depth = float(np.median(valid_depth)) if valid_depth.size else None
                    screw_position = project_pixel_with_depth(
                        screw_pixel,
                        screw_depth,
                        self.data.cam_xpos[camera_id],
                        self.data.cam_xmat[camera_id].reshape(3, 3),
                        float(self.model.cam_fovy[camera_id]),
                        (frames[source].shape[1], frames[source].shape[0]),
                    )
            tool_tip = self.data.site_xpos[self.tool_tip_id].copy()
            vision_valid = screw_position is not None and plane_verified
            if (vision_valid and self.last_valid_screw_xy is not None
                    and np.linalg.norm(
                        screw_position[:2] - self.last_valid_screw_xy
                    ) > VISION_XY_JUMP_LIMIT_M):
                # 螺钉在旋出前横向固定；跳跃过大说明锁到了相邻螺钉/装饰件。
                vision_valid = False
            if vision_valid and detection.confidence >= 0.7:
                self.last_valid_screw_xy = np.asarray(
                    screw_position[:2], dtype=float
                ).copy()
            result = {
                "images": {name: self._encode_image(frame) for name, frame in frames.items()},
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
                "sites": {name: self.data.site_xpos[index].tolist() for name, index in self.site_ids.items()},
                "vision_screw_center_px": list(screw_pixel) if screw_pixel is not None else [-1.0, -1.0],
                "vision_screw_position_m": screw_position.tolist() if vision_valid else tool_tip.tolist(),
                "vision_screw_xy_estimate_m": screw_position[:2].tolist() if vision_valid else tool_tip[:2].tolist(),
                "vision_screw_top_z_estimate_m": float(screw_position[2]) if vision_valid else float(tool_tip[2]),
                "vision_screw_depth_m": float(screw_depth) if screw_depth is not None else 0.0,
                "vision_xy_error_m": (screw_position[:2] - tool_tip[:2]).tolist() if vision_valid else [0.0, 0.0],
                "vision_confidence": float(detection.confidence if vision_valid else 0.0),
                "vision_valid": vision_valid,
                # 套筒套入后会遮住头顶面，当前帧无法再给出估计；保留最后一次高置信
                # 估计供“套筒是否仍在孔心”的视觉校验使用。
                "vision_screw_last_valid_xy_m": (
                    np.asarray(self.last_valid_screw_xy, dtype=float).tolist()
                    if self.last_valid_screw_xy is not None else None
                ),
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
                result["spiral_search_exhausted"] = self.spiral_search_exhausted
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
            self.osc_target_rotation = self._tool_base_rotation().copy()
            self.osc_target_position = osc_anchor_for_tip_target(
                self.data.site_xpos[self.tool_tip_id],
                self.osc_target_rotation[:, 0],
                self.force_control_offset_m,
            )
        self.steps = 0
        self.engaged_seen = False
        self.last_valid_screw_xy = None
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
        self.spiral_search_exhausted = False
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
        self.socket_fit_ticks = 0
        self.max_socket_fit_ticks = 0
        if open_viewer:
            self._start_viewer()
        if include_force:
            self._contact_wrench()
        return self.observe(include_force)

    def _tool_base_rotation(self) -> np.ndarray:
        """Tool frame orientation, excluding the socket's own spin."""
        return self.data.xmat[self.tool_base_body_id].reshape(3, 3)

    def _osc_torque(self, target_position: np.ndarray,
                    target_rotation: np.ndarray) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        rotation = self._tool_base_rotation()
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

    def _begin_spiral_search(self, target_force_n: float,
                             reference_tool_z_m: float | None = None) -> dict:
        with self._viewer_lock():
            self.force_control_target_n = float(np.clip(target_force_n, 0.0, 5.0))
            tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            tip_rotation_now = self._tool_base_rotation().copy()
        tool_axis = tip_rotation_now[:, 0]
        self.osc_target_position = tip_position.copy()
        self.osc_target_rotation = tip_rotation_now
        initial_raw, initial_wrench = self._contact_wrench()
        # 只有真实轴向接触（>=1 N）才采信力的符号；0.1 N 的偏置噪声会把
        # 顺应方向判反，导致下压变成抬升、力基线永不收敛。
        if abs(float(initial_wrench[2])) >= 1.0:
            self.force_control_direction = float(np.sign(initial_wrench[2]))
        self.spiral_force_baseline_settle_steps = 0
        self.spiral_force_baseline_settled = False
        self.spiral_search_guarded = False
        self.spiral_search_exhausted = False
        if np.linalg.norm(initial_wrench[:3]) >= 50.0:
            self.spiral_search_guarded = True
            self.spiral_force_baseline_guard_reason = "force_limit"
            self.spiral_search_state = np.array([0.0, 0.0, 1.0])
            return self.observe(include_force=True)
        result = self.observe(include_force=True)
        with self._viewer_lock():
            tip_position = self.data.site_xpos[self.tool_tip_id].copy()
            tip_rotation_now = self._tool_base_rotation().copy()
        # 螺旋中心优先用最后一次高置信视觉估计（现在由深度点云给出，误差约 0.3–0.5 mm）：
        # 纯 z 下探时机械臂有约 2–3 mm 的下垂漂移，接触点未必落在孔心；阿基米德螺旋对
        # 同一半径只经过一次，若以套筒中心为中心很难再经过孔心。视觉失效时回退到套筒中心。
        center_xy = tip_position[:2].copy()
        last_valid_screw_xy = getattr(self, "last_valid_screw_xy", None)
        if last_valid_screw_xy is not None:
            candidate = np.asarray(last_valid_screw_xy, dtype=float)
            if np.linalg.norm(candidate - tip_position[:2]) <= SPIRAL_SEARCH_MAX_RADIUS_M:
                center_xy = candidate
        self.spiral_search_center = np.array(
            [center_xy[0], center_xy[1], tip_position[2]], dtype=float
        )
        self.spiral_search_rotation = tip_rotation_now
        self.spiral_search_start_axial = float(
            np.dot(
                self.data.geom_xpos[self.screw_head_geom_id]
                - self.data.site_xpos[self.tool_tip_id],
                tip_rotation_now[:, 0],
            )
        )
        self.spiral_reference_force_n = float(initial_wrench[2])
        settled_wrenches: list[np.ndarray] = []
        recent_wrenches: list[np.ndarray] = []
        stable_steps = 0
        settle_samples = 0
        while settle_samples < SPIRAL_FORCE_BASELINE_SETTLE_MAX_STEPS:
            batch_size = min(CONTROL_SUBSTEPS,
                             SPIRAL_FORCE_BASELINE_SETTLE_MAX_STEPS - settle_samples)
            settle_result = self._run_control_steps(
                batch_size, True, target_force_n, 0.0, capture_force_samples=True
            )
            samples = settle_result.pop("force_samples")
            for sample in samples:
                recent_wrenches.append(sample)
                if len(recent_wrenches) > CONTROL_SUBSTEPS:
                    recent_wrenches.pop(0)
                settle_samples += 1
                if len(recent_wrenches) == CONTROL_SUBSTEPS:
                    recent_mean = np.mean(recent_wrenches, axis=0)
                    target_signed_force = self.force_control_direction * target_force_n
                    if abs(recent_mean[2] - target_signed_force) <= (
                        SPIRAL_FORCE_BASELINE_SETTLE_TOLERANCE_N
                    ):
                        stable_steps += 1
                        settled_wrenches.append(sample)
                        if len(settled_wrenches) > SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS:
                            settled_wrenches.pop(0)
                    else:
                        stable_steps = 0
                        settled_wrenches.clear()
                    if stable_steps >= SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS:
                        break
            self.spiral_force_baseline_settle_steps = settle_samples
            result = settle_result
            if stable_steps >= SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS:
                break
        if stable_steps < SPIRAL_FORCE_BASELINE_SETTLE_STABLE_STEPS:
            self.spiral_search_guarded = True
            self.spiral_force_baseline_guard_reason = "force_baseline_unsettled"
            self.spiral_search_state = np.array([0.0, 0.0, 1.0])
            result.update(self.observe(include_force=True))
            if self.debug_tactile_diagnostics:
                result["spiral_baseline_diagnostic"] = {
                    "initial_axial_force_n": float(initial_wrench[2]),
                    "settled_axial_force_n": None,
                    "target_force_n": float(target_force_n),
                    "settle_steps": settle_samples,
                    "baseline_settled": False,
                    "force_offset_m": float(self.force_control_offset_m),
                }
            return result
        settled_baseline = np.mean(settled_wrenches, axis=0)
        self.spiral_reference_force_n = float(settled_baseline[2])
        self.spiral_force_baseline_settled = True
        self.spiral_tactile_detector = SpiralTactileDetector(
            self.spiral_reference_force_n,
            settled_baseline[:2],
            SPIRAL_TACTILE_THRESHOLDS_CALIBRATED,
            # 插入深度以标称头顶面为基准：若以螺旋起始的尖端高度为基准，起始时已
            # 被预压到接近头顶面，随后即使真正套入也凑不满 3 mm 行程，触觉判据失效。
            reference_tool_z_m=(
                SCREW_TARGET.head_top_z_m if reference_tool_z_m is None
                else float(reference_tool_z_m)
            ),
        )
        self.spiral_search_index = 0
        settled_tip_z_m = float(self.data.site_xpos[self.tool_tip_id, 2])
        for sample in settled_wrenches[-SPIRAL_SEATED_CONFIRM_STEPS:]:
            self.spiral_tactile_detector.update(
                sample[2], sample[:2], tool_z_m=settled_tip_z_m
            )
        self.spiral_search_active = not self.spiral_tactile_detector.seated
        if self.debug_tactile_diagnostics:
            result["spiral_baseline_diagnostic"] = {
                "initial_axial_force_n": float(initial_wrench[2]),
                "settled_axial_force_n": self.spiral_reference_force_n,
                "target_force_n": float(target_force_n),
                "settle_steps": settle_samples,
                "settle_force_trace_n": [float(sample[2]) for sample in settled_wrenches],
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
            if self.spiral_tactile_detector and self.spiral_tactile_detector.seated:
                self.spiral_search_state = np.array([
                    0.0, float(self.spiral_search_index), 0.0,
                ])
                return self.observe(include_force=True)
        self.spiral_force_trace = []
        index = self.spiral_search_index
        tip_position = self.spiral_search_center
        tip_rotation = self.spiral_search_rotation
        radius = SPIRAL_SEARCH_MAX_RADIUS_M * (index + 1) / SPIRAL_SEARCH_POINTS
        angle = (self.spiral_phase_offset + index * 2.0 * np.pi
                 * SPIRAL_SEARCH_TURNS / SPIRAL_SEARCH_POINTS)
        desired_tip_position = tip_position + radius * (
            np.cos(angle) * tip_rotation[:, 1]
            + np.sin(angle) * tip_rotation[:, 2]
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
            if np.linalg.norm(peak) >= 50.0:
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
        if not self.spiral_search_guarded:
            # 轻压下探并保持慢速自转：60° 步进对六角相位是恒等操作，只有下压期间
            # 慢转才能扫过棱边/平面相位，使轻微偏心的套筒滑入开口。
            press_target = desired_tip_position.copy()
            press_target[2] -= SPIRAL_DOWNWARD_PRELOAD_M
            self.osc_target_position = press_target
            press_result = self._run_control_steps(
                SPIRAL_PRESS_STEPS, True, target_force_n,
                SOCKET_TURN_DIRECTION * SPIRAL_PRESS_ROTATION_RAD_S
            )
            press_peak = np.asarray(
                press_result["force_torque_peak_compensated"], dtype=float
            )
            result = press_result
            if not (self.spiral_tactile_detector
                    and self.spiral_tactile_detector.seated):
                self.osc_target_position = desired_tip_position.copy()
                result = self._run_control_steps(
                    SPIRAL_RELEASE_STEPS, True, target_force_n, 0.0
                )
            result["force_torque_peak_compensated"] = np.maximum(
                press_peak,
                np.asarray(result["force_torque_peak_compensated"], dtype=float),
            ).tolist()
        self.data.ctrl[6] = 0.0
        peak = np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
        force_guarded = np.linalg.norm(peak) >= 50.0
        tactile_candidate = bool(
            self.spiral_tactile_detector
            and self.spiral_tactile_detector.candidate_detected
        )
        sensor_settle_rotation_deg = 0.0
        if tactile_candidate and not force_guarded:
            result, sensor_settle_rotation_deg = self._sensor_triggered_seating_turn(
                target_force_n
            )
            peak = np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
            force_guarded = np.linalg.norm(peak) >= 50.0
            result["sensor_settle_rotation_deg"] = sensor_settle_rotation_deg
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
        if seating_confirmed and not self.spiral_tactile_detector.seated:
            self.spiral_tactile_detector.candidate_detected = False
            self.spiral_tactile_detector.candidate_latched = False
            self.spiral_tactile_detector.confirmation_steps = 0
            self.spiral_tactile_detector.force_drop_detected = False
            self.spiral_tactile_detector.force_drop_age_steps = None
            self.spiral_tactile_detector.phase = "seeking_force_drop"
            seating_confirmed = False
        if seating_confirmed or force_guarded or step_failed:
            self.data.ctrl[6] = 0.0
            self.spiral_search_active = False
            self.spiral_search_guarded = (force_guarded and not seating_confirmed) or step_failed
        else:
            self._advance_spiral_search_index()
        self.spiral_search_state = np.array([
            float(self.spiral_search_active), float(index + 1),
            float(self.spiral_search_guarded),
        ])
        result.update(self.observe(include_force))
        if self.debug_tactile_diagnostics:
            result["spiral_force_trace"] = self.spiral_force_trace
        return result

    def socket_turn_step(self, count: int = CONTROL_SUBSTEPS,
        target_force_n: float = 1.0) -> dict:
        if not self.socket_turn_active:
            if not (self.spiral_tactile_detector and self.spiral_tactile_detector.seated):
                raise ValueError("socket turning requires confirmed tactile seating")
            self.socket_turn_active = True
            self.socket_turn_completed = False
            self.socket_turn_start_angle = float(self.data.qpos[self.socket_joint_qpos])
            self.socket_turn_start_screw_angle = float(self.data.qpos[self.screw_spin_qpos])
            self.data.eq_active[self.screw_rotation_lock_id] = 0
            thread_lift_per_radian = SCREW_THREAD_PITCH_M / (2.0 * np.pi)
            self.model.eq_data[self.screw_thread_pitch_coupling_id, :2] = (
                float(self.data.qpos[self.screw_lift_qpos])
                - thread_lift_per_radian * self.data.qpos[self.screw_spin_qpos],
                thread_lift_per_radian,
            )
            self.data.eq_active[self.screw_height_lock_id] = 0
            self.data.eq_active[self.screw_thread_pitch_coupling_id] = 1
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
            self.osc_target_rotation = self._tool_base_rotation().copy()

        result = self._run_control_steps(
            count, include_force=True, target_force_n=target_force_n,
            socket_velocity=SOCKET_TURN_DIRECTION * SOCKET_TURN_SPEED_RAD_S,
            follow_screw_lift=True,
        )
        elapsed_angle = float(self.data.qpos[self.socket_joint_qpos] - self.socket_turn_start_angle)
        force_peak = np.linalg.norm(np.asarray(result["force_torque_peak_compensated"][:3]))
        if force_peak >= 50.0:
            self._finish_socket_turn()
            result["socket_turn_stop_reason"] = "force_guard"
        elif abs(elapsed_angle) >= SOCKET_TURN_ANGLE_RAD:
            self._finish_socket_turn(completed=True)
            result["socket_turn_stop_reason"] = "screw_extracted"
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
            self.data.eq_active[self.screw_thread_pitch_coupling_id] = 0
            self.model.eq_data[self.screw_rotation_lock_id, 0] = self.data.qpos[self.screw_spin_qpos]
            self.data.eq_active[self.screw_rotation_lock_id] = 1
            self.model.eq_data[self.screw_height_lock_id, 0] = self.data.qpos[self.screw_lift_qpos]
            self.data.eq_active[self.screw_height_lock_id] = 1
            self.socket_turn_active = False
            self.socket_turn_completed = completed

    def _run_control_steps(self, count: int, include_force: bool,
                           target_force_n: float, socket_velocity: float,
                           follow_screw_lift: bool = False,
                           capture_force_samples: bool = False) -> dict:
        started = time.perf_counter()
        peak_external = np.zeros(6, dtype=float)
        force_samples = []
        tool_tip_trajectory = []
        self.force_control_target_n = float(np.clip(target_force_n, 0.0, 5.0))
        tracked_screw_lift = float(self.data.qpos[self.screw_lift_qpos])
        if self.force_control_target_n == 0.0:
            self.force_control_active = False
            self.force_control_offset_m = 0.0
        with self._viewer_lock():
            for _ in range(max(1, int(count))):
                if follow_screw_lift:
                    screw_lift = float(self.data.qpos[self.screw_lift_qpos])
                    self.osc_target_position[2] += screw_lift - tracked_screw_lift
                    tracked_screw_lift = screw_lift
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
                tool_tip_trajectory.append({
                    "time": float(self.data.time),
                    "step": int(self.steps),
                    "world_m": self.data.site_xpos[self.tool_tip_id].tolist(),
                })
                if self.record_socket_opening_trajectory:
                    opening_center = self.data.site_xpos[self.tool_tip_id].copy()
                    self.socket_opening_trajectory.append({
                        "sim_time_s": float(self.data.time),
                        "world_m": opening_center.tolist(),
                    })
                if include_force:
                    mujoco.mj_forward(self.model, self.data)
                    _, external = self._contact_wrench()
                    if capture_force_samples:
                        force_samples.append(external.copy())
                    peak_external = np.where(np.abs(external) > np.abs(peak_external),
                                             external, peak_external)
                    force_guard_triggered = np.linalg.norm(external[:3]) >= 50.0
                    engaged_diagnostic = self._valid_socket_engagement()[0]
                    engagement_ticks_diagnostic = (
                        self.socket_engagement_ticks + 1 if engaged_diagnostic else 0
                    )
                    if self.spiral_search_active and self.spiral_tactile_detector is not None:
                        self.spiral_tactile_detector.update(
                            external[2], external[:2],
                            float(self.data.site_xpos[self.tool_tip_id, 2]),
                        )
                        if (self.spiral_tactile_detector.seated
                                and self.last_valid_screw_xy is not None
                                and np.linalg.norm(
                                    np.asarray(self.last_valid_screw_xy, dtype=float)
                                    - self.data.site_xpos[self.tool_tip_id][:2]
                                ) > VISION_SEATING_RADIUS_M):
                            # 触觉判据可能把压在法兰/棱边误判为 seated；用相机估计
                            # 的孔心与套筒口位置做否定校验，避免提前终止螺旋。
                            self.spiral_tactile_detector.seated = False
                            self.spiral_tactile_detector.candidate_detected = False
                            self.spiral_tactile_detector.candidate_latched = False
                            self.spiral_tactile_detector.confirmation_steps = 0
                            self.spiral_tactile_detector.force_drop_detected = False
                            self.spiral_tactile_detector.force_drop_age_steps = None
                            self.spiral_tactile_detector.phase = "seeking_force_drop"
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
                            "spiral_search_index": int(self.spiral_search_index),
                            "tool_tip_world_m": self.data.site_xpos[self.tool_tip_id].tolist(),
                            "tool_tip_xy_private_m": self.data.site_xpos[self.tool_tip_id, :2].tolist(),
                            "spiral_target_world_m": self.osc_target_position.tolist(),
                            "spiral_center_xy_m": (
                                self.spiral_search_center[:2].tolist()
                                if self.spiral_search_center is not None else None
                            ),
                            "spiral_target_radius_m": float(np.linalg.norm(
                                self.osc_target_position - self.spiral_search_center
                            )) if self.spiral_search_center is not None else 0.0,
                            "spiral_target_xy_radius_m": float(np.linalg.norm(
                                self.osc_target_position[:2] - self.spiral_search_center[:2]
                            )) if self.spiral_search_center is not None else 0.0,
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
                            trace_sample["socket_angle_private_rad"] = float(
                                self.data.qpos[self.socket_joint_qpos]
                            )
                        self.spiral_force_trace.append(trace_sample)
                    if force_guard_triggered:
                        break
                engaged, _, axial, radial, alignment = self._valid_socket_engagement()
                self.engaged_seen = self.engaged_seen or engaged
                self.socket_engagement_ticks = self.socket_engagement_ticks + 1 if engaged else 0
                seated_geometry = (
                    alignment >= 0.995 and -0.014 <= axial <= -SPIRAL_MIN_SOCKET_INSERTION_M
                    and radial <= 0.0015
                )
                self.socket_fit_ticks = self.socket_fit_ticks + 1 if seated_geometry else 0
                self.max_socket_fit_ticks = max(
                    self.max_socket_fit_ticks, self.socket_fit_ticks
                )
        self.ft_peak_external = peak_external
        self._sync_viewer_realtime(started, max(1, int(count)) * self.model.opt.timestep)
        result = self.observe(include_force)
        result["tool_tip_trajectory"] = tool_tip_trajectory
        if capture_force_samples:
            result["force_samples"] = force_samples
        return result

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
        if target_force_n <= 0.0:
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
                                                 request.get("target_force_n", 1.0))
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
