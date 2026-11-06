"""Inspect Robots embodiment for the UR20 battery-disassembly scene."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "ur20_screw_worker.py"
DEFAULT_MUJOCO_PYTHON = Path.home() / "miniconda3" / "envs" / "libero" / "bin" / "python"
PYTHON = os.environ.get(
    "MUJOCO_PYTHON",
    str(DEFAULT_MUJOCO_PYTHON if DEFAULT_MUJOCO_PYTHON.is_file() else sys.executable),
)
MAX_CARTESIAN_TARGET_DELTA_M = 0.05
XY_ALIGNMENT_TOLERANCE_M = 0.002
VISION_XY_MOTION_MIN_CONFIDENCE = 0.7
CONTACT_FORCE_THRESHOLD_N = 1.0
LOCKED_APPROACH_STEP_M = 0.01
COMPLIANT_APPROACH_DISTANCE_M = 0.05
COMPLIANT_FULL_FORCE_DISTANCE_M = 0.01
COMPLIANT_LATERAL_RELEASE_DISTANCE_M = 0.05
COMPLIANT_FORCE_SETPOINT_N = 2.0


def force_setpoint_for_vertical_gap(current_z: float, estimated_surface_z: float,
                                    enabled: bool,
                                    contact_detected: bool = False) -> float:
    if not enabled:
        return 0.0
    if contact_detected:
        return COMPLIANT_FORCE_SETPOINT_N
    gap = abs(float(current_z) - float(estimated_surface_z))
    if gap >= COMPLIANT_APPROACH_DISTANCE_M:
        return 0.0
    if gap <= COMPLIANT_FULL_FORCE_DISTANCE_M:
        return COMPLIANT_FORCE_SETPOINT_N
    ramp = (COMPLIANT_APPROACH_DISTANCE_M - gap) / (
        COMPLIANT_APPROACH_DISTANCE_M - COMPLIANT_FULL_FORCE_DISTANCE_M
    )
    smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    return COMPLIANT_FORCE_SETPOINT_N * smooth_ramp


def limit_cartesian_target(current: np.ndarray, target: np.ndarray,
                           max_distance: float = MAX_CARTESIAN_TARGET_DELTA_M) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("Cartesian current position and target must have shape (3,)")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Cartesian current position and target must be finite")
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance > max_distance:
        return current + delta * (max_distance / distance)
    return target.copy()


def amplify_cartesian_target(current: np.ndarray, target: np.ndarray,
                             amplification: float) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("Cartesian current position and target must have shape (3,)")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Cartesian current position and target must be finite")
    if not np.isfinite(amplification) or amplification < 1.0:
        raise ValueError("action amplification must be finite and at least 1")
    return current + (target - current) * amplification


def constrain_locked_approach(current: np.ndarray, target: np.ndarray,
                              force_guarded: bool = False) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("Cartesian current position and target must have shape (3,)")
    result = target.copy()
    result[:2] = current[:2]
    if not force_guarded:
        result[2] = min(result[2], current[2] - LOCKED_APPROACH_STEP_M)
    return result


def constrain_uncertain_lateral_target(current: np.ndarray, target: np.ndarray,
                                       vision_confidence: float) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float).copy()
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("Cartesian current position and target must have shape (3,)")
    if vision_confidence < VISION_XY_MOTION_MIN_CONFIDENCE:
        target[:2] = current[:2]
    return target


def prevent_unaligned_descent(current: np.ndarray, target: np.ndarray,
                              vision_confidence: float,
                              vision_xy_error: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float).copy()
    error = np.asarray(vision_xy_error, dtype=float)
    if current.shape != (3,) or target.shape != (3,) or error.shape != (2,):
        raise ValueError("expected 3D positions and a 2D vision XY error")
    if (vision_confidence >= VISION_XY_MOTION_MIN_CONFIDENCE
            and np.linalg.norm(error) > XY_ALIGNMENT_TOLERANCE_M):
        target[2] = max(target[2], current[2])
    return target


class UR20ScrewEmbodiment(EmbodimentBase):
    def __init__(self, output_dir: str | os.PathLike[str] = "outputs/ur20_screw", state_id: int = 0,
                 max_steps: int = 600, include_force: bool = False,
                 compliant_control: bool = False, spiral_search: bool = False,
                 mujoco_viewer: bool = False, action_amplification: float = 3.0,
                 twist_after_seating: bool = False) -> None:
        self.output = Path(output_dir)
        self.spiral_force_trace_path = self.output / "spiral_force_trace.jsonl"
        self.video_frames_dir = self.output / "video_frames"
        self.video_frame_index = 0
        self.state_id = state_id
        self.max_steps = max_steps
        self.include_force = include_force
        self.compliant_control = compliant_control
        self.spiral_search = spiral_search
        self.twist_after_seating = twist_after_seating
        if twist_after_seating and (not spiral_search or not compliant_control or not include_force):
            raise ValueError("twist_after_seating requires force feedback, compliant control, and spiral search")
        self.mujoco_viewer = mujoco_viewer
        if not np.isfinite(action_amplification) or not 1.0 <= action_amplification <= 5.0:
            raise ValueError("action_amplification must be between 1.0 and 5.0")
        self.action_amplification = float(action_amplification)
        self._last_vision_confidence = 0.0
        self._last_vision_xy_error = np.zeros(2, dtype=float)
        self._last_screw_top_z_estimate = 0.153
        self._last_screw_xy_estimate: np.ndarray | None = None
        self._force_proximity_latched = False
        if compliant_control and not include_force:
            raise ValueError("compliant_control requires force feedback")
        if spiral_search and not compliant_control:
            raise ValueError("spiral_search requires compliant control")
        self.environment_steps = 0
        self.physics_steps = 0
        self.spiral_internal_increments = 0
        self._last_socket_engaged = False
        self._last_engaged_seen = False
        self._last_tool_tip: np.ndarray | None = None
        self._xy_alignment_locked = False
        self._spiral_search_active = False
        self._spiral_search_guarded = False
        self._contact_seating_latched = False
        self._last_axial_contact = False
        self.process: subprocess.Popen[str] | None = None
        self._worker_env = None
        self.instruction = ""
        self.task_success = False
        self.socket_seated = False
        self.socket_turn_completed = False
        self.socket_turn_screw_delta = 0.0
        self.socket_turn_stop_reason: str | None = None
        self.max_screw_lift = 0.0
        self.engaged_seen = False
        self.info = EmbodimentInfo(
            name="ur20_screw_station",
            is_simulated=True,
            control_hz=20.0,
            capabilities=frozenset({"resettable", "renderable"}),
            action_space=Box(
                shape=(3,),
                low=np.array([-1.0, -1.0, 0.0]),
                high=np.array([1.0, 1.2, 2.0]),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose", frame="world",
                    dim_labels=("x", "y", "z"),
                    max_step=(MAX_CARTESIAN_TARGET_DELTA_M,) * 3,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 256, 256) for name in ("agentview", "wristview")),
                state=StateSpec((StateField("eef_pos", (3,)),
                                 StateField("vision_screw_center_px", (2,)),
                                 StateField("vision_screw_xy_estimate_m", (2,)),
                                 StateField("vision_screw_top_z_estimate_m", (1,)),
                                 StateField("vision_xy_error_m", (2,)),
                                 StateField("vision_confidence", (1,)),
                                 StateField("joint_pos", (6,)),
                                 StateField("joint_vel", (6,)),
                                 *((StateField("force_torque", (6,)),
                                    StateField("force_torque_compensated", (6,)),
                                    StateField("force_torque_peak_compensated", (6,))) if include_force else ()),
                                 *((StateField("force_control_state", (3,)),) if compliant_control else ()),
                                 *((StateField("spiral_axial_force_delta_n", (1,)),)
                                   if include_force and spiral_search else ()),
                                 *((StateField("spiral_search_state", (3,)),) if spiral_search else ()),
                                 *((StateField("spiral_tactile_state", (7,)),)
                                   if include_force and spiral_search else ()),
                                 *((StateField("socket_turn_state", (3,)),) if twist_after_seating else ()))),
            ),
        docs=("UR20 with a rigidly wrist-mounted screwdriver and labeled agentview and "
              "wristview cameras. Use move_to with absolute world-frame Cartesian x, y, z "
              "targets in meters; each target change is capped at 5 cm in Euclidean "
              "distance. This is a hard safety maximum, not a recommended step size: "
              "when the target is clearly distant and the path is visibly clear, use "
              "the full allowed displacement; reduce motion size only near contact, "
              "uncertainty, or an obstacle. "
                 "The wristview screw detector and calibrated camera-ray projection provide "
              "vision_screw_xy_estimate_m and vision_xy_error_m on the nominal screw-head "
              "plane; vision_screw_top_z_estimate_m is that plane's calibrated nominal Z. "
              "These are calibrated estimates, not simulator task state. Use them "
              "only when vision_confidence is at least 0.7. Below that, hold x/y fixed and "
              "do not treat a zero error as alignment. First align x/y while holding "
              "the current z, using the full safe displacement when far from the target "
              "and unobstructed. "
              + ("In force-controlled spiral-search mode, reassess image-derived x/y "
                 "error at every observation: lock x/y only while confident error is "
                 "at most 2 mm, and resume x/y correction if it later exceeds 2 mm. "
                 "While aligned, the controller enforces continued downward progress; "
                 "the move_to z target is absolute and "
                 "smaller z means down. Do not retract unless the force guard activates. "
                 if spiral_search else "")
              + "With x/y aligned, continue descending toward the visible screw top while "
              "there is clearance. "
              "The tool orientation stays fixed at the task pose "
              "and a low-level "
              "operational-space torque controller tracks that target. Use larger Cartesian "
              "motions when clearance is obvious and the target is distant; use small motions "
              "only near contact or when resolving uncertainty. Continue until visual and "
              "force evidence supports approach alignment. After axial contact, call "
              "spiral_search once only when the observed absolute Z component of "
              "force_torque_compensated reaches 1 N. Below 1 N, the tool rejects the request "
              "without advancing simulation and tells you to continue the compliant downward "
              "approach with move_to, then reassess force. After contact, do not use move_to "
              "to push through it. The search runs to completion without "
              "intermediate LLM calls. "
              "A useful visual cue is that, after the socket is properly seated over the "
              "external-hex screw head, the screw head will normally be occluded by the "
              "socket in agentview; treat this as supporting evidence, not proof by itself. "
              "Search runs until the tactile seating sequence is detected, a force guard fires, or the configured step limit is reached. Geometric simulator engagement is diagnostic only and does not end the search. "
              "Never infer contact or success from hidden "
              "simulator state. "
              + ("force_torque is the raw sensor wrench and force_torque_compensated is "
                 "the estimated external wrench; use compensated force and its interval "
                 "peak as contact evidence. " if include_force else "")
              + ("One OSC torque controller is used throughout. Axial force admittance "
                 "stays at 0 N while more than 5 cm above the calibrated nominal screw-top "
                 "plane, then its target follows a smooth ramp from 0 N at 5 cm to 2 N at "
                 "1 cm and remains 2 N after force-confirmed contact. The force zone is "
                 "armed by confident XY alignment or contact, then held through temporary "
                 "vision-confidence loss; it releases after moving more than 5 cm laterally "
                 "from the last valid screw estimate. The calibrated nominal plane is not "
                 "simulator contact truth. "
                 if compliant_control else "")
              + ("When compensated axial contact force reaches 1 N during the approach, "
                 "the controller latches contact and suspends Cartesian motion. Then call "
                 "spiral_search exactly once. The search corrects residual lateral and "
                 "hex-phase misalignment. The tip searches in "
                 "small planar increments while the socket rotates 60 degrees per "
                 "increment, with axial compliance maintained. spiral_search_state gives "
                 "active, completed increment count, and force-guard status. spiral_tactile_state contains phase (0 waiting for axial-force drop, 1 waiting for sustained lateral-force rise, 2 calibrated seating, 3 uncalibrated candidate), confirmed-seated flag, axial-force drop from the search-start contact baseline in N, lateral force-vector change from its search-start baseline in N, consecutive confirmation samples, axial-drop event latch, and event age in physics samples. A candidate requires the 0.75 N axial-force drop and 0.75 N lateral force-vector rise to overlap, with both conditions holding for five consecutive physics samples; the axial-drop event expires after 25 samples. No extra downward displacement or axial-force recovery condition is used. Thresholds are not calibrated yet, so a candidate is not accepted as seated and cannot trigger turning or end the search. "
                 if spiral_search else "")
              + ("After spiral_tactile_state reports seated, call turn_socket to rotate the socket "
                 "counterclockwise by one revolution while maintaining axial contact. "
                 "This simplified phase allows the screw to rotate in place but keeps its "
                 "height fixed; it deliberately does not model thread pitch or extraction. "
                 "socket_turn_state contains active, signed socket angle in radians, and "
                 "completion flag. "
                 if twist_after_seating else "")
              + ("During spiral search, use the compensated force data: seating is "
                 "confirmed only after the axial force drops from its search-start contact "
                 "baseline and lateral force remains elevated relative to its own search-start "
                 "baseline for five consecutive physics samples. A one-sample lateral spike "
                 "does not count. No downward-motion or axial-force-recovery condition is "
                 "part of the seating criterion. Do not retract merely because force briefly "
                 "fluctuates; the controller continues its internal search until both force "
                 "conditions are confirmed. "
                 if include_force and spiral_search else "")
              + "Approach the external-hex head with visible clearance, align before "
              "descending, and do not rotate to extract the screw. If uncertain, inspect "
              "both labeled views before choosing the next motion."),
        )

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise EmbodimentFault("UR20 worker is not running")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise EmbodimentFault("UR20 worker exited")
        result = json.loads(line)
        if "error" in result:
            raise EmbodimentFault(result["error"])
        return result

    def _record_video_frame(self, data: dict[str, Any]) -> None:
        self.video_frames_dir.mkdir(parents=True, exist_ok=True)
        views = [
            np.asarray(
                Image.open(io.BytesIO(base64.b64decode(data["images"][name]))).convert("RGB"),
                dtype=np.uint8,
            )
            for name in ("agentview", "wristview")
        ]
        Image.fromarray(np.concatenate(views, axis=1)).save(
            self.video_frames_dir / f"ur20_video_{self.video_frame_index:06d}.jpg",
            format="JPEG", quality=82,
        )
        self.video_frame_index += 1

    @staticmethod
    def _observation(data: dict[str, Any], instruction: str, include_force: bool = False,
                     compliant_control: bool = False, spiral_search: bool = False,
                     twist_after_seating: bool = False) -> Observation:
        images = {name: np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"), dtype=np.uint8) for name, value in data["images"].items()}
        qpos = np.asarray(data["qpos"], dtype=float)
        qvel = np.asarray(data["qvel"], dtype=float)
        state = {"eef_pos": np.asarray(data["sites"]["tool_tip"], dtype=float),
                 "vision_screw_center_px": np.asarray(data["vision_screw_center_px"], dtype=float),
                 "vision_screw_xy_estimate_m": np.asarray(data["vision_screw_xy_estimate_m"], dtype=float),
                 "vision_screw_top_z_estimate_m": np.array(
                     [data["vision_screw_top_z_estimate_m"]], dtype=float),
                 "vision_xy_error_m": np.asarray(data["vision_xy_error_m"], dtype=float),
                 "vision_confidence": np.array([data["vision_confidence"]], dtype=float),
                 "joint_pos": qpos[:6].copy(), "joint_vel": qvel[:6].copy()}
        if include_force:
            state["force_torque"] = np.asarray(data["force_torque"], dtype=float)
            state["force_torque_compensated"] = np.asarray(data["force_torque_compensated"], dtype=float)
            state["force_torque_peak_compensated"] = np.asarray(data["force_torque_peak_compensated"], dtype=float)
        if compliant_control:
            state["force_control_state"] = np.array([
                float(data["force_control_active"]), data["force_control_target_n"],
                data["force_control_offset_m"],
            ], dtype=float)
        if include_force and spiral_search:
            state["spiral_axial_force_delta_n"] = np.array(
                [data.get("spiral_axial_force_delta_n", 0.0)], dtype=float)
        if spiral_search:
            state["spiral_search_state"] = np.asarray(data["spiral_search_state"], dtype=float)
        if include_force and spiral_search:
            state["spiral_tactile_state"] = np.asarray(
                data.get("spiral_tactile_state", [0.0] * 7), dtype=float
            )
        if twist_after_seating:
            state["socket_turn_state"] = np.asarray(data["socket_turn_state"], dtype=float)
        return Observation(images=images, state=state, instruction=instruction, state_time=float(data["time"]))

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.output.mkdir(parents=True, exist_ok=True)
        if self.process is None:
            env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
            env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
            self._worker_env = env
            self.process = subprocess.Popen([PYTHON, "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        self.instruction = scene.instruction
        self.task_success = False
        self.max_screw_lift = 0.0
        self.engaged_seen = False
        self.environment_steps = 0
        self.spiral_internal_increments = 0
        self.video_frame_index = 0
        if self.video_frames_dir.exists():
            for frame in self.video_frames_dir.glob("ur20_video_*.jpg"):
                frame.unlink()
        data = self._rpc({"op": "reset", "state_id": self.state_id,
                          "include_force": self.include_force,
                          "open_viewer": self.mujoco_viewer})
        self._record_video_frame(data)
        self.physics_steps = int(data["steps"])
        self._last_socket_engaged = bool(data["socket_engaged"])
        self._last_engaged_seen = bool(data["engaged_seen"])
        self._last_tool_tip = np.asarray(data["sites"]["tool_tip"], dtype=float)
        self._last_vision_confidence = float(data["vision_confidence"])
        self._last_vision_xy_error = np.asarray(data["vision_xy_error_m"], dtype=float)
        self._last_screw_top_z_estimate = float(data["vision_screw_top_z_estimate_m"])
        self._xy_alignment_locked = self._vision_xy_aligned(data)
        self._force_proximity_latched = self._xy_alignment_locked
        if self._last_vision_confidence >= VISION_XY_MOTION_MIN_CONFIDENCE:
            self._last_screw_xy_estimate = np.asarray(
                data["vision_screw_xy_estimate_m"], dtype=float
            )
        self._spiral_search_active = False
        self._spiral_search_guarded = False
        self._contact_seating_latched = False
        self._last_axial_contact = self._has_axial_contact(data)
        self.socket_seated = False
        self.socket_turn_completed = False
        self.socket_turn_screw_delta = 0.0
        self.socket_turn_stop_reason = None
        return self._observation(data, self.instruction, self.include_force,
                                 self.compliant_control, self.spiral_search,
                                 self.twist_after_seating)

    def step(self, action: Action) -> StepResult:
        xy_locked_before = self._xy_alignment_locked
        spiral_active_before = self._spiral_search_active
        force_guarded_before = self._spiral_search_guarded
        contact_latched_before = self._contact_seating_latched
        action_values = np.asarray(action.data, dtype=float)
        if self._last_tool_tip is None:
            raise EmbodimentFault("UR20 embodiment must be reset before stepping")
        start_spiral_search = bool(action.meta.get("ur20_spiral_search", False))
        start_socket_turn = bool(action.meta.get("ur20_turn_socket", False))
        if start_spiral_search and not self.spiral_search:
            raise EmbodimentFault("spiral_search was not enabled for this rollout")
        requested_contact_force = float(action.meta.get("ur20_axial_contact_force_n", 0.0))
        if start_spiral_search and abs(requested_contact_force) < CONTACT_FORCE_THRESHOLD_N:
            raise EmbodimentFault("spiral_search requires detected axial contact")
        if start_spiral_search:
            self._contact_seating_latched = True
        if start_socket_turn and not self.twist_after_seating:
            raise EmbodimentFault("turn_socket was not enabled for this rollout")
        if start_socket_turn and not self.socket_seated:
            raise EmbodimentFault("turn_socket requires stable socket seating")
        requested_target = action_values[:3].copy()
        policy_target = requested_target.copy()
        target_force_n = 0.0
        if self.spiral_search:
            requested_target = constrain_uncertain_lateral_target(
                self._last_tool_tip, requested_target, self._last_vision_confidence
            )
        if self.spiral_search and not self._xy_alignment_locked:
            requested_target = prevent_unaligned_descent(
                self._last_tool_tip, requested_target, self._last_vision_confidence,
                np.asarray(self._last_vision_xy_error, dtype=float),
            )
        if self.spiral_search and self._xy_alignment_locked:
            requested_target = constrain_locked_approach(
                self._last_tool_tip, requested_target,
                force_guarded=self._spiral_search_guarded,
            )
        amplification_active = (
            self.action_amplification > 1.0
            and self._last_vision_confidence >= VISION_XY_MOTION_MIN_CONFIDENCE
            and not self._last_axial_contact
            and not self._spiral_search_active
            and not self._spiral_search_guarded
        )
        if amplification_active:
            requested_target = amplify_cartesian_target(
                self._last_tool_tip, requested_target, self.action_amplification
            )
        bounded_policy_target = limit_cartesian_target(self._last_tool_tip, requested_target)
        if self._last_axial_contact or self._contact_seating_latched:
            self._force_proximity_latched = True
        elif self._force_proximity_latched and self._last_screw_xy_estimate is not None:
            lateral_gap = float(np.linalg.norm(
                self._last_tool_tip[:2] - self._last_screw_xy_estimate
            ))
            if lateral_gap > COMPLIANT_LATERAL_RELEASE_DISTANCE_M:
                self._force_proximity_latched = False
        target_force_n = force_setpoint_for_vertical_gap(
            self._last_tool_tip[2], self._last_screw_top_z_estimate,
            self.compliant_control and self._force_proximity_latched,
            contact_detected=self._last_axial_contact or self._contact_seating_latched,
        )
        safety_stop_reason = None
        socket_turn_iterations = 0
        if start_socket_turn:
            self.socket_turn_stop_reason = None
            while self.environment_steps < self.max_steps:
                data = self._rpc({"op": "socket_turn_step", "count": 25,
                                  "target_force_n": COMPLIANT_FORCE_SETPOINT_N})
                socket_turn_iterations += 1
                self._record_video_frame(data)
                self.environment_steps += 1
                self.physics_steps = int(data["steps"])
                self._last_tool_tip = np.asarray(data["sites"]["tool_tip"], dtype=float)
                self.socket_turn_completed = bool(data["socket_turn_state"][2])
                self.socket_turn_screw_delta = float(data.get("screw_rotation_delta", 0.0))
                self.socket_turn_stop_reason = data.get("socket_turn_stop_reason")
                if socket_turn_iterations % 20 == 0 or self.socket_turn_completed or self.socket_turn_stop_reason:
                    print(
                        "[UR20 socket-turn] angle=%+.1fdeg screw=%+.1fdeg Fz=%+.2fN %s" % (
                            np.rad2deg(float(data["socket_turn_state"][1])),
                            np.rad2deg(self.socket_turn_screw_delta),
                            float(np.asarray(data["force_torque_compensated"])[2]),
                            self.socket_turn_stop_reason or "running",
                        ), file=sys.stderr, flush=True,
                    )
                if (self.socket_turn_completed or self.socket_turn_stop_reason
                        or not data["socket_turn_state"][0]):
                    break
            if not self.socket_turn_completed and not self.socket_turn_stop_reason:
                data = self._rpc({"op": "socket_turn_stop"})
                safety_stop_reason = "socket_turn_step_limit"
                self.socket_turn_stop_reason = safety_stop_reason
            elif self.socket_turn_stop_reason:
                safety_stop_reason = f"socket_turn_{self.socket_turn_stop_reason}"
            applied_target = self._last_tool_tip.copy()
        elif start_spiral_search:
            data = self._run_spiral_increment(COMPLIANT_FORCE_SETPOINT_N)
            applied_target = self._last_tool_tip.copy()
        elif self._contact_seating_latched or (self.twist_after_seating and self.socket_seated):
            applied_target = self._last_tool_tip.copy()
            data = self._rpc({"op": "step", "target_position": applied_target.tolist(),
                              "socket_velocity": 0.0, "count": 25,
                              "include_force": self.include_force,
                              "target_force_n": COMPLIANT_FORCE_SETPOINT_N,
                              "spiral_search": False})
        else:
            applied_target = bounded_policy_target
            data = self._rpc({"op": "step", "target_position": applied_target.tolist(),
                              "socket_velocity": 0.0, "count": 25,
                              "include_force": self.include_force,
                              "target_force_n": target_force_n, "spiral_search": False})
        if not start_socket_turn:
            self._record_video_frame(data)
            self.environment_steps += 1
        self.physics_steps = int(data["steps"])
        self._last_tool_tip = np.asarray(data["sites"]["tool_tip"], dtype=float)
        self._last_vision_confidence = float(data["vision_confidence"])
        self._last_vision_xy_error = np.asarray(data["vision_xy_error_m"], dtype=float)
        self._last_axial_contact = self._has_axial_contact(data)
        self._xy_alignment_locked = self._vision_xy_aligned(data)
        if self._last_vision_confidence >= VISION_XY_MOTION_MIN_CONFIDENCE:
            self._last_screw_xy_estimate = np.asarray(
                data["vision_screw_xy_estimate_m"], dtype=float
            )
        if self._xy_alignment_locked or self._last_axial_contact:
            self._force_proximity_latched = True
        spiral_increments = int(start_spiral_search)
        self.spiral_internal_increments += spiral_increments
        if self.spiral_search:
            self._spiral_search_active = bool(data["spiral_search_state"][0])
            self._spiral_search_guarded = bool(data["spiral_search_state"][2])
            contact_detected = self._has_axial_contact(data)
            if contact_detected:
                self._contact_seating_latched = True
            while (start_spiral_search
                   and self._contact_seating_latched
                   and not self._spiral_search_guarded
                   and not data.get("spiral_tactile_seated", False)
                   and self.environment_steps < self.max_steps):
                data = self._run_spiral_increment(COMPLIANT_FORCE_SETPOINT_N)
                self._record_video_frame(data)
                spiral_increments += 1
                self.spiral_internal_increments += 1
                self.environment_steps += 1
                self.physics_steps = int(data["steps"])
                self._last_tool_tip = np.asarray(data["sites"]["tool_tip"], dtype=float)
                self._last_vision_confidence = float(data["vision_confidence"])
                self._last_vision_xy_error = np.asarray(data["vision_xy_error_m"], dtype=float)
                if self._last_vision_confidence >= VISION_XY_MOTION_MIN_CONFIDENCE:
                    self._last_screw_xy_estimate = np.asarray(
                        data["vision_screw_xy_estimate_m"], dtype=float
                    )
                self._xy_alignment_locked = self._vision_xy_aligned(data)
                self._spiral_search_active = bool(data["spiral_search_state"][0])
                self._spiral_search_guarded = bool(data["spiral_search_state"][2])
                self._last_axial_contact = self._has_axial_contact(data)
                self.max_screw_lift = max(self.max_screw_lift, float(data["screw_lift"]))
                self.engaged_seen = self.engaged_seen or bool(data["engaged_seen"])
                print(
                    "[UR20 force] internal spiral increment=%d Fz_comp=%+.2fN "
                    "dFz_comp=%+.2fN dFxy=%+.2fN confirm=%s tactile=%s seated=%s engaged=%s%s" % (
                        spiral_increments,
                        float(np.asarray(data["force_torque_compensated"])[2]),
                        float(data.get("spiral_axial_force_delta_n", 0.0)),
                        float(data.get("spiral_tactile_state", [0] * 7)[3]),
                        int(data.get("spiral_tactile_state", [0] * 7)[4]),
                        data.get("spiral_tactile_phase", "idle"),
                        data.get("spiral_tactile_seated", False),
                        data["socket_engaged"],
                        " FORCE_GUARD" if self._spiral_search_guarded else "",
                    ), file=sys.stderr, flush=True,
                )
            if (self._contact_seating_latched and not self._spiral_search_guarded
                    and not data.get("spiral_tactile_seated", False)
                    and self.environment_steps >= self.max_steps):
                safety_stop_reason = "spiral_search_step_limit"
            if self._spiral_search_guarded:
                safety_stop_reason = safety_stop_reason or "spiral_force_guard"
            if data.get("spiral_tactile_seated", False):
                self._contact_seating_latched = False
        if data.get("spiral_tactile_seated", False):
            self.socket_seated = True
        if self._xy_alignment_locked != xy_locked_before:
            if self._xy_alignment_locked:
                message = "vision XY error is within 2 mm; lateral policy motion locked"
            else:
                message = "vision XY error exceeded 2 mm or confidence fell; lateral correction resumed"
            print(f"[UR20 phase] {message}", file=sys.stderr, flush=True)
        if self._contact_seating_latched and not contact_latched_before:
            print("[UR20 phase] axial contact confirmed; residual XY/hex alignment "
                  "transferred to force-compliant spiral search",
                  file=sys.stderr, flush=True)
        if self._spiral_search_active and not spiral_active_before:
            print("[UR20 phase] force-compliant spiral search is active",
                  file=sys.stderr, flush=True)
        if self._contact_seating_latched and not contact_latched_before:
            print("[UR20 phase] CONTACT LATCH: Cartesian targets suspended; "
                  "LLM must call spiral_search",
                  file=sys.stderr, flush=True)
        if self._spiral_search_guarded and not force_guarded_before:
            print(f"[UR20 phase] SAFE STOP: {safety_stop_reason or 'spiral force guard'}; "
                  "no further LLM motion will be accepted",
                  file=sys.stderr, flush=True)
        self.max_screw_lift = max(self.max_screw_lift, float(data["screw_lift"]))
        self.engaged_seen = self.engaged_seen or bool(data["engaged_seen"])
        success = (self.socket_turn_completed if self.twist_after_seating
                   else bool(data.get("spiral_tactile_seated", False)))
        self.task_success = self.task_success or success
        socket_engaged = bool(data["socket_engaged"])
        engaged_seen = bool(data["engaged_seen"])
        if self.include_force:
            wrench = np.asarray(data["force_torque_compensated"], dtype=float)
            control_state = np.asarray(data.get("force_control_state", [0, 0, 0]), dtype=float)
            spiral_state = np.asarray(data.get("spiral_search_state", [0, 0, 0]), dtype=float)
            estimated_z_gap = abs(self._last_tool_tip[2] - self._last_screw_top_z_estimate)
            print(
                "[UR20 force] F_comp(sensor-frame)=(%.2f, %.2f, %.2f) N "
                "|F_comp|=%.2f N; z-gap-est=%.3fm; control=%s target_comp=%.1fN; "
                "spiral=%s increment=%.0f dFz=%+.2fN%s" % (
                    *wrench[:3], float(np.linalg.norm(wrench[:3])),
                    estimated_z_gap,
                    "active" if control_state[0] else "idle", control_state[1],
                    "active" if spiral_state[0] else "idle", spiral_state[1],
                    float(data.get("spiral_axial_force_delta_n", 0.0)),
                    " FORCE_GUARD" if spiral_state[2] else "",
                ), file=sys.stderr, flush=True,
            )
        if (self.environment_steps == 1 or self.environment_steps % 10 == 0 or success
                or socket_engaged != self._last_socket_engaged
                or engaged_seen != self._last_engaged_seen):
            print(
                f"[UR20 step {self.environment_steps}/{self.max_steps} | physics {self.physics_steps}] "
                f"socket_engaged={socket_engaged} stable={data['socket_engagement_stable']} "
                f"ever_engaged={engaged_seen} "
                f"screw_lift={data['screw_lift']:.4f}m "
                f"rotation={np.rad2deg(data['screw_rotation']):.1f}deg "
                f"socket_error(axial={data['socket_axial_error']:.4f}m, "
                f"radial={data['socket_radial_error']:.4f}m) success={success}",
                file=sys.stderr, flush=True)
        self._last_socket_engaged = socket_engaged
        self._last_engaged_seen = engaged_seen
        return StepResult(self._observation(data, self.instruction, self.include_force,
                                            self.compliant_control, self.spiral_search,
                                            self.twist_after_seating), reward=1.0 if success else 0.0,
                          terminated=success or safety_stop_reason is not None,
                          termination_reason=("success" if success else safety_stop_reason),
                          info={"steps": data["steps"], "screw_lift": data["screw_lift"],
                                "socket_engaged": data["socket_engaged"],
                                "socket_engagement_stable": data["socket_engagement_stable"],
                                "socket_seated": self.socket_seated,
                                "socket_turn_completed": self.socket_turn_completed,
                                "socket_turn_screw_rotation_delta": self.socket_turn_screw_delta,
                                "socket_turn_stop_reason": self.socket_turn_stop_reason,
                                "socket_turn_sim_steps": socket_turn_iterations,
                                "socket_contact": data["socket_contact"],
                                "socket_axial_error": data["socket_axial_error"],
                                "socket_radial_error": data["socket_radial_error"],
                                "tool_axis_alignment": data["tool_axis_alignment"],
                                "screw_rotation": data["screw_rotation"],
                                "engaged_seen": data["engaged_seen"], "success": success,
                                "force_control_active": data.get("force_control_active", False),
                                "force_torque_compensated": data.get("force_torque_compensated"),
                                "spiral_search_state": data.get("spiral_search_state"),
                                "spiral_tactile_state": data.get("spiral_tactile_state"),
                                "spiral_tactile_phase": data.get("spiral_tactile_phase"),
                                "spiral_tactile_seated": data.get("spiral_tactile_seated", False),
                                "spiral_internal_increments": spiral_increments,
                                "spiral_internal_increments_total": self.spiral_internal_increments,
                                "requested_cartesian_target": policy_target.tolist(),
                                "amplified_cartesian_target": requested_target.tolist(),
                                "applied_cartesian_target": applied_target.tolist(),
                                "action_amplification": self.action_amplification,
                                "action_amplification_active": amplification_active,
                                "xy_alignment_locked": self._xy_alignment_locked,
                                "spiral_search_active": self._spiral_search_active,
                                "spiral_search_force_guarded": self._spiral_search_guarded,
                                "safe_stop_reason": safety_stop_reason,
                                "cartesian_target_limited": not np.allclose(
                                    requested_target, applied_target)})

    @staticmethod
    def _vision_xy_aligned(data: dict[str, Any]) -> bool:
        error = np.asarray(data["vision_xy_error_m"], dtype=float)
        return (float(data["vision_confidence"]) >= VISION_XY_MOTION_MIN_CONFIDENCE
                and np.linalg.norm(error) <= XY_ALIGNMENT_TOLERANCE_M)

    @staticmethod
    def _has_axial_contact(data: dict[str, Any]) -> bool:
        force = np.asarray(data.get("force_torque_compensated", np.zeros(6)), dtype=float)
        return force.shape == (6,) and abs(float(force[2])) >= CONTACT_FORCE_THRESHOLD_N

    def _run_spiral_increment(self, target_force_n: float) -> dict[str, Any]:
        data = self._rpc({"op": "step", "target_position": self._last_tool_tip.tolist(),
                          "socket_velocity": 0.0, "count": 25, "include_force": True,
                          "target_force_n": target_force_n, "spiral_search": True})
        trace = data.get("spiral_force_trace", [])
        if trace:
            with self.spiral_force_trace_path.open("a", encoding="utf-8") as stream:
                for sample in trace:
                    stream.write(json.dumps({
                        "state_id": self.state_id,
                        "spiral_increment": self.spiral_internal_increments + 1,
                        **sample,
                    }, separators=(",", ":")) + "\n")
                stream.flush()
        return data

    def close(self) -> None:
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self._rpc({"op": "close"})
            except (EmbodimentFault, BrokenPipeError, OSError):
                pass
            finally:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                self.process = None
