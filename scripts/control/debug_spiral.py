"""Run UR20 spiral-search experiments without an LLM and log private sim truth."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

WORKER_PATH = Path(__file__).resolve().parents[2] / "src/ur20_screw/simulation/worker.py"
SPEC = importlib.util.spec_from_file_location("ur20_screw_worker", WORKER_PATH)
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WORKER_MODULE)
START_POSE_PATH = Path(__file__).resolve().parents[1] / "vision/view_socket_start.py"
START_POSE_SPEC = importlib.util.spec_from_file_location("ur20_socket_start", START_POSE_PATH)
START_POSE_MODULE = importlib.util.module_from_spec(START_POSE_SPEC)
assert START_POSE_SPEC.loader is not None
START_POSE_SPEC.loader.exec_module(START_POSE_MODULE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--start-offset-x-mm", type=float, default=1.0)
    parser.add_argument("--start-offset-y-mm", type=float, default=0.0)
    parser.add_argument("--screw-angle-deg", type=float, default=30.0,
                        help="initial screw-head hex angle for controlled simulation tests")
    parser.add_argument("--start-z-offset-mm", type=float, default=0.0,
                        help="tool-tip Z offset relative to the nominal screw-top plane")
    parser.add_argument(
        "--max-radius-mm",
        type=float,
        default=WORKER_MODULE.SPIRAL_SEARCH_MAX_RADIUS_M * 1000.0,
    )
    parser.add_argument("--max-points", type=int, default=WORKER_MODULE.SPIRAL_SEARCH_POINTS)
    parser.add_argument("--search-points", type=int, default=None,
                        help="override spiral resolution while preserving its configured turns")
    parser.add_argument("--mode", choices=("simultaneous", "phased"), default="simultaneous")
    parser.add_argument("--socket-speed-rad-s", type=float, default=3.0)
    parser.add_argument("--max-spin-steps-per-point", type=int, default=3000)
    parser.add_argument("--target-force-n", type=float, default=2.0,
                        help="axial force-admittance target during offline spiral search")
    parser.add_argument("--force-offset-limit-mm", type=float, default=None,
                        help="override the offline force-admittance offset limit")
    parser.add_argument("--approach-force-n", type=float, default=2.0,
                        help="axial force-admittance target while establishing contact")
    parser.add_argument("--lateral-threshold-n", type=float, default=None,
                        help="override the lateral tactile rise threshold for this offline run")
    parser.add_argument("--tactile-thresholds-calibrated", action="store_true",
                        help="enable tactile seating confirmation for this offline run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--camera-frame-stride", type=int, default=24,
                        help="capture all validation cameras every N search points")
    parser.add_argument("--compact-trace", action="store_true",
                        help="store only tactile deltas, engagement labels, and candidate contact pairs")
    parser.add_argument("--search-after-stable", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="disable real-time pacing; simulation dynamics are unchanged")
    parser.add_argument("--mujoco-viewer", action="store_true",
                        help="open a passive MuJoCo viewer during the experiment")
    parser.add_argument("--video-fps", type=int, default=0,
                        help="record a close screw-side video at this real-time FPS")
    return parser.parse_args()


def write_frame(encoded: str, path: Path) -> None:
    image = Image.open(__import__("io").BytesIO(base64.b64decode(encoded)))
    image.save(path, format="JPEG", quality=90)


def solve_tip_ik(worker, target: np.ndarray) -> None:
    import mujoco

    model, data = worker.model, worker.data
    target_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3).copy()
    for _ in range(500):
        mujoco.mj_forward(model, data)
        position_error = target - data.site_xpos[worker.tool_tip_id]
        current_rotation = data.site_xmat[worker.tool_tip_id].reshape(3, 3)
        rotation_error = 0.5 * sum(
            (np.cross(current_rotation[:, axis], target_rotation[:, axis])
             for axis in range(3)), np.zeros(3)
        )
        if np.linalg.norm(position_error) < 1e-6 and np.linalg.norm(rotation_error) < 1e-5:
            worker.osc_target_position = data.site_xpos[worker.tool_tip_id].copy()
            worker.osc_target_rotation = target_rotation
            return
        position_jacobian = np.zeros((3, model.nv))
        rotation_jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(
            model, data, position_jacobian, rotation_jacobian, worker.tool_tip_id
        )
        arm_jacobian = np.vstack((
            position_jacobian[:, :6], rotation_jacobian[:, :6] * 0.2
        ))
        pose_error = np.concatenate((position_error, rotation_error * 0.2))
        delta = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + 1e-5 * np.eye(6), pose_error
        )
        data.qpos[:6] += np.clip(delta, -0.08, 0.08)
    raise RuntimeError("could not solve the scripted spiral-search start pose")


def log_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    stream.flush()


def flush_socket_opening_trace(stream, worker) -> None:
    for sample in worker.socket_opening_trajectory:
        log_record(stream, {
            "event": "socket_opening_sample",
            "sim_time_s": sample["sim_time_s"],
            "socket_opening_center_world_m": sample["world_m"],
        })
    worker.socket_opening_trajectory.clear()


def wait_for_viewer_close(worker) -> None:
    if worker.viewer is None:
        return
    while worker.viewer.is_running():
        worker.viewer.sync()
        time.sleep(1.0 / 60.0)


def main() -> None:
    args = parse_args()
    if args.max_radius_mm <= 0 or args.max_points <= 0:
        raise SystemExit("max radius and max points must be positive")
    if args.search_points is not None:
        if args.search_points <= 0:
            raise SystemExit("search points must be positive")
        WORKER_MODULE.SPIRAL_SEARCH_POINTS = args.search_points
    output = args.output or Path("outputs") / (
        "ur20_spiral_debug_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    frames_dir = output / "frames"
    if not args.no_frames:
        frames_dir.mkdir()

    WORKER_MODULE.SPIRAL_SEARCH_MAX_RADIUS_M = args.max_radius_mm / 1000.0
    if args.lateral_threshold_n is not None:
        if args.lateral_threshold_n <= 0:
            raise SystemExit("lateral threshold must be positive")
        WORKER_MODULE.SPIRAL_LATERAL_FORCE_RISE_THRESHOLD_N = args.lateral_threshold_n
    if args.tactile_thresholds_calibrated:
        WORKER_MODULE.SPIRAL_TACTILE_THRESHOLDS_CALIBRATED = True
    if args.force_offset_limit_mm is not None:
        if args.force_offset_limit_mm <= 0:
            raise SystemExit("force offset limit must be positive")
        WORKER_MODULE.FORCE_ADMITTANCE_MAX_OFFSET_M = args.force_offset_limit_mm / 1000.0
    worker = WORKER_MODULE.Worker(render=not args.no_frames)
    worker.debug_tactile_diagnostics = True
    if args.fast:
        worker._sync_viewer_realtime = lambda *_args: None
    def observe_without_rendering(include_force: bool = False) -> dict:
        _, external = worker._contact_wrench()
        return {
            "force_torque_compensated": external.tolist(),
            "force_torque_peak_compensated": worker.ft_peak_external.tolist(),
            "force_control_offset_m": worker.force_control_offset_m,
        }

    worker.observe = observe_without_rendering
    physical_trajectory = []
    original_run_control_steps = worker._run_control_steps

    def recording_run_control_steps(count, include_force, target_force_n, socket_velocity, **kwargs):
        result = original_run_control_steps(
            count, include_force, target_force_n, socket_velocity, **kwargs
        )
        physical_trajectory.extend(result.get("tool_tip_trajectory", []))
        return result

    worker._run_control_steps = recording_run_control_steps
    video_frames = []
    video_period = 1.0 / args.video_fps if args.video_fps > 0 else 0.0
    next_video_time = 0.0
    video_started = time.perf_counter()

    def capture_closeup(sim_time: float) -> None:
        nonlocal next_video_time
        if not video_period or sim_time + 1e-9 < next_video_time:
            return
        image, _ = worker._render_rgb_depth(worker.camera_ids["screw_side"])
        video_frames.append((sim_time, image.copy()))
        next_video_time = sim_time + video_period

    if args.video_fps > 0:
        original_run_steps = worker._run_control_steps

        def recorded_run_steps(count, include_force, target_force_n, socket_velocity):
            result = original_run_steps(count, include_force, target_force_n, socket_velocity)
            capture_closeup(float(worker.data.time))
            WORKER_MODULE.wait_for_simulated_time(
                video_started, float(worker.data.time)
            )
            return result

        worker._run_control_steps = recorded_run_steps
    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "model": "MuJoCo UR20 screw station; no LLM calls",
        "state_id": args.state_id,
        "start_offset_mm": [args.start_offset_x_mm, args.start_offset_y_mm],
        "screw_angle_deg": args.screw_angle_deg,
        "max_radius_mm": args.max_radius_mm,
        "max_points": args.max_points,
        "mode": args.mode,
        "socket_speed_rad_s": args.socket_speed_rad_s,
        "target_force_n": args.target_force_n,
        "approach_force_n": args.approach_force_n,
        "seating_downward_travel_m": WORKER_MODULE.SPIRAL_SEATING_DOWNWARD_TRAVEL_M,
        "seated_downward_travel_m": WORKER_MODULE.SPIRAL_SEATED_DOWNWARD_TRAVEL_M,
        "socket_hex_engagement_depth_m": WORKER_MODULE.SOCKET_HEX_ENGAGEMENT_DEPTH_M,
        "minimum_socket_insertion_m": WORKER_MODULE.SPIRAL_MIN_SOCKET_INSERTION_M,
        "seated_min_axial_force_ratio": WORKER_MODULE.SPIRAL_SEATED_MIN_AXIAL_FORCE_RATIO,
        "force_offset_limit_mm": WORKER_MODULE.FORCE_ADMITTANCE_MAX_OFFSET_M * 1000.0,
        "search_points_per_turn": WORKER_MODULE.SPIRAL_SEARCH_POINTS
            / WORKER_MODULE.SPIRAL_SEARCH_TURNS,
        "socket_opening_wall_positions_m": worker.model.geom_pos[
            sorted(worker.socket_wall_ids)
        ].tolist(),
        "telemetry_is_private": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    trajectory_path = output / "trajectory.jsonl"
    result_path = output / "result.json"

    try:
        worker.reset(args.state_id, include_force=True, open_viewer=False)
        START_POSE_MODULE.set_socket_start_pose(
            worker, args.start_offset_x_mm / 1000.0, args.screw_angle_deg,
            args.start_offset_y_mm / 1000.0,
        )
        worker.record_socket_opening_trajectory = True
        if args.mujoco_viewer:
            worker._start_viewer()
        screw_center = worker.data.xpos[worker.screw_body_id, :2].copy()
        screw_top_z = float(START_POSE_MODULE.SCREW_HEAD_TOP_Z_M)
        mujoco_forward = __import__("mujoco").mj_forward
        mujoco_forward(worker.model, worker.data)
        actual_start = worker.data.site_xpos[worker.tool_tip_id].copy()
        start_error = actual_start[:2] - screw_center
        initial_raw, initial_external = worker._contact_wrench()
        with trajectory_path.open("w") as stream:
            log_record(stream, {
                "event": "start_pose",
                "sim_time_s": float(worker.data.time),
                "tool_tip_world_m": actual_start.tolist(),
                "socket_opening_center_world_m": actual_start.tolist(),
                "screw_center_world_m_private": screw_center.tolist(),
                "true_xy_error_mm_private": (start_error * 1000.0).tolist(),
                "true_xy_error_norm_mm_private": float(np.linalg.norm(start_error) * 1000.0),
                "screw_top_z_m": screw_top_z,
                "force_raw_n_nm": initial_raw.tolist(),
                "force_compensated_n_nm": initial_external.tolist(),
                "socket_angle_rad": float(worker.data.qpos[worker.socket_joint_qpos]),
                "screw_angle_rad": float(worker.data.qpos[worker.screw_spin_qpos]),
                "socket_engagement": worker._valid_socket_engagement(),
            })

            worker.osc_target_position = worker.data.site_xpos[worker.tool_tip_id].copy()
            contact_result = worker._run_control_steps(
                WORKER_MODULE.CONTROL_SUBSTEPS * 20, True, args.approach_force_n, 0.0
            )
            flush_socket_opening_trace(stream, worker)
            log_record(stream, {
                "event": "contact_preload",
                "sim_time_s": float(worker.data.time),
                "tool_tip_world_m": worker.data.site_xpos[worker.tool_tip_id].tolist(),
                "force_compensated_n_nm": contact_result["force_torque_compensated"],
                "force_peak_compensated_n_nm": contact_result["force_torque_peak_compensated"],
                "force_control_offset_m": contact_result["force_control_offset_m"],
                "socket_engagement": worker._valid_socket_engagement(),
            })
            initial_raw, initial_wrench = worker._contact_wrench()
            result = worker._begin_spiral_search(
                args.target_force_n, reference_tool_z_m=float(actual_start[2])
            )
            _, reference_wrench = worker._contact_wrench()
            log_record(stream, {
                "event": "spiral_force_baseline",
                "initial_force_torque_compensated_n_nm": initial_wrench.tolist(),
                "settled_force_torque_compensated_n_nm": reference_wrench.tolist(),
                "settle_steps": worker.spiral_force_baseline_settle_steps,
                "baseline_settled": worker.spiral_force_baseline_settled,
                "settle_diagnostic": result.get("spiral_baseline_diagnostic"),
                "force_control_offset_m": worker.force_control_offset_m,
                "search_center_world_m_private": (
                    worker.spiral_search_center.tolist()
                    if worker.spiral_search_center is not None else None
                ),
            })
            if worker.spiral_search_guarded:
                summary = {
                    "status": worker.spiral_force_baseline_guard_reason,
                    "search_points_executed": 0,
                    "baseline_settled": worker.spiral_force_baseline_settled,
                    "baseline_settle_steps": worker.spiral_force_baseline_settle_steps,
                    "baseline_diagnostic": result.get("spiral_baseline_diagnostic"),
                    "simulation_steps": worker.steps,
                    "simulation_time_s": float(worker.data.time),
                    "llm_calls": 0,
                }
                result_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps({"output": str(output), **summary}, indent=2))
                wait_for_viewer_close(worker)
                return
            center = worker.spiral_search_center.copy()
            rotation = worker.spiral_search_rotation.copy()
            sensor_rotation = worker.data.site_xmat[worker.ft_sensor_site_id].reshape(3, 3).copy()
            search_records = 0
            max_engagement_ticks = 0
            tactile_candidate_seen = False
            sensor_triggered_online = False
            sensor_settle_rotation_deg = 0.0
            force_guarded = False
            spin_timed_out = False
            started = time.monotonic()
            for point in range(args.max_points):
                worker.spiral_force_trace = []
                index = point % WORKER_MODULE.SPIRAL_SEARCH_POINTS
                radius = WORKER_MODULE.SPIRAL_SEARCH_MAX_RADIUS_M * (index + 1) / WORKER_MODULE.SPIRAL_SEARCH_POINTS
                angle = (worker.spiral_phase_offset + index * 2.0 * math.pi
                         * WORKER_MODULE.SPIRAL_SEARCH_TURNS / WORKER_MODULE.SPIRAL_SEARCH_POINTS)
                if args.mode == "simultaneous":
                    target_base = center + radius * (
                        math.cos(angle) * sensor_rotation[:, 0]
                        + math.sin(angle) * sensor_rotation[:, 1]
                    )
                    worker.osc_target_position = target_base.copy()
                    rotation_start = float(worker.data.qpos[worker.socket_joint_qpos])
                    rotation_steps = 0
                    rotation_target = math.pi / 3.0
                    point_steps = 0
                    max_point_steps = max(
                        500,
                        int(math.ceil(2.0 * math.pi / 3.0
                                     / (args.socket_speed_rad_s * worker.model.opt.timestep))),
                    )
                    while point_steps < max_point_steps:
                        angle_progress = -(float(worker.data.qpos[worker.socket_joint_qpos])
                                           - rotation_start)
                        point_reached = np.linalg.norm(
                            worker.data.site_xpos[worker.tool_tip_id] - target_base
                        ) <= WORKER_MODULE.SPIRAL_SEARCH_POSITION_TOLERANCE_M
                        socket_stepped = angle_progress >= rotation_target - (
                            args.socket_speed_rad_s * worker.model.opt.timestep
                        )
                        if socket_stepped:
                            break
                        batch = min(
                            5,
                            max_point_steps - point_steps,
                            max(1, int(math.ceil(
                                (rotation_target - angle_progress)
                                / (args.socket_speed_rad_s * worker.model.opt.timestep)
                            ))) if not socket_stepped else 5,
                        )
                        result = worker._run_control_steps(
                            batch, True, args.target_force_n,
                            -args.socket_speed_rad_s if not socket_stepped else 0.0,
                        )
                        point_steps += batch
                        rotation_steps += batch if not socket_stepped else 0
                        if np.linalg.norm(
                            np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
                        ) >= 50.0:
                            break
                    angle_progress = -(float(worker.data.qpos[worker.socket_joint_qpos])
                                       - rotation_start)
                    point_reached = np.linalg.norm(
                        worker.data.site_xpos[worker.tool_tip_id] - target_base
                    ) <= WORKER_MODULE.SPIRAL_SEARCH_POSITION_TOLERANCE_M
                    socket_stepped = angle_progress >= rotation_target - (
                        args.socket_speed_rad_s * worker.model.opt.timestep
                    )
                    rotation_timeout = not socket_stepped
                    point_timeout = not point_reached
                else:
                    target_base = center + np.array([
                        radius * math.cos(angle), radius * math.sin(angle), 0.0
                    ])
                    worker.osc_target_position = target_base.copy()
                    result = worker._run_control_steps(
                        WORKER_MODULE.CONTROL_SUBSTEPS, True, args.target_force_n, 0.0
                    )
                    rotation_start = float(worker.data.qpos[worker.socket_joint_qpos])
                    spin_direction = -1.0
                    spin_speed = args.socket_speed_rad_s
                    rotation_steps = 0
                    while (spin_direction * (float(worker.data.qpos[worker.socket_joint_qpos])
                                              - rotation_start) < math.pi / 3.0
                           and rotation_steps < args.max_spin_steps_per_point):
                        result = worker._run_control_steps(
                            10, True, args.target_force_n, spin_direction * spin_speed
                        )
                        rotation_steps += 10
                        result = worker._run_control_steps(10, True, args.target_force_n, 0.0)
                        if worker._confirmed_socket_seating():
                            rotation_timeout = False
                            break
                    worker.spiral_search_state = np.array([1.0, float(index + 1), 0.0])
                    rotation_timeout = (
                        spin_direction * (float(worker.data.qpos[worker.socket_joint_qpos])
                                          - rotation_start) < math.pi / 3.0
                    )
                flush_socket_opening_trace(stream, worker)
                target_tip = target_base
                actual_rotation = (
                    float(worker.data.qpos[worker.socket_joint_qpos]) - rotation_start
                    if rotation_start is not None else 0.0
                )
                tip = worker.data.site_xpos[worker.tool_tip_id].copy()
                tool_axis = worker.data.site_xmat[worker.tool_tip_id].reshape(3, 3)[:, 0]
                vertical_alignment = float(np.clip(-tool_axis[2], -1.0, 1.0))
                raw, external = worker._contact_wrench()
                engaged, contact, axial, radial, alignment = worker._valid_socket_engagement()
                sensor_tactile_trigger = worker._confirmed_socket_seating()
                engagement_settle_turns = 0
                tactile_candidate_seen = tactile_candidate_seen or bool(
                    worker.spiral_tactile_detector
                    and worker.spiral_tactile_detector.candidate_detected
                )
                max_engagement_ticks = max(max_engagement_ticks, worker.socket_engagement_ticks)
                search_records += 1
                log_record(stream, {
                    "event": "spiral_point",
                    "point": search_records,
                    "search_index": index,
                    "sim_time_s": float(worker.data.time),
                    "wall_elapsed_s": time.monotonic() - started,
                    "planned_radius_mm": radius * 1000.0,
                    "planned_angle_rad": angle,
                    "planned_tip_world_m": target_tip.tolist(),
                    "actual_tip_world_m": tip.tolist(),
                    "target_tracking_error_mm": float(np.linalg.norm(tip - target_tip) * 1000.0),
                    "target_tracking_error_xy_mm": float(
                        np.linalg.norm(tip[:2] - target_tip[:2]) * 1000.0
                    ),
                    "socket_rotation_this_point_deg": math.degrees(actual_rotation),
                    "socket_rotation_sim_steps": rotation_steps,
                    "engagement_settle_turns": engagement_settle_turns,
                    "socket_rotation_timeout": rotation_timeout if rotation_start is not None else False,
                    "screw_center_world_m_private": screw_center.tolist(),
                    "spiral_center_world_m": worker.spiral_search_center[:].tolist(),
                    "true_xy_error_mm_private": ((tip[:2] - screw_center) * 1000.0).tolist(),
                    "true_xy_error_norm_mm_private": float(np.linalg.norm(tip[:2] - screw_center) * 1000.0),
                    "force_raw_n_nm": raw.tolist(),
                    "force_compensated_n_nm": external.tolist(),
                    "force_peak_compensated_n_nm": result["force_torque_peak_compensated"],
                    "force_control_offset_m": result["force_control_offset_m"],
                    "socket_angle_rad": float(worker.data.qpos[worker.socket_joint_qpos]),
                    "socket_angle_delta_rad": float(worker.data.qpos[worker.socket_joint_qpos]
                                                      - (worker.socket_turn_start_angle)),
                    "screw_angle_rad": float(worker.data.qpos[worker.screw_spin_qpos]),
                    "socket_contact": contact,
                    "contact_pairs_private": [
                        {
                            "geom1": __import__("mujoco").mj_id2name(
                                worker.model, __import__("mujoco").mjtObj.mjOBJ_GEOM,
                                int(contact_pair.geom1),
                            ),
                            "geom2": __import__("mujoco").mj_id2name(
                                worker.model, __import__("mujoco").mjtObj.mjOBJ_GEOM,
                                int(contact_pair.geom2),
                            ),
                            "distance_m": float(contact_pair.dist),
                        }
                        for contact_pair in worker.data.contact[:worker.data.ncon]
                    ],
                    "socket_engaged": engaged,
                    "sensor_tactile_trigger": sensor_tactile_trigger,
                    "socket_engagement_stable": worker.socket_engagement_ticks >= 20,
                    "engaged_ticks": worker.socket_engagement_ticks,
                    "socket_axial_error_m": axial,
                    "socket_radial_error_m": radial,
                    "tool_axis_alignment": alignment,
                    "tool_tilt_from_vertical_deg": math.degrees(
                        math.acos(vertical_alignment)
                    ),
                    "tactile_phase": worker.spiral_tactile_phase if hasattr(worker, "spiral_tactile_phase") else (
                        worker.spiral_tactile_detector.phase if worker.spiral_tactile_detector else "idle"
                    ),
                    "tactile_seated": bool(worker.spiral_tactile_detector and
                                            worker.spiral_tactile_detector.seated),
                    "tactile_candidate": bool(worker.spiral_tactile_detector and
                                               worker.spiral_tactile_detector.candidate_detected),
                    "tactile_force_trace": (
                        [
                            {
                                "time": sample["time"],
                                "axial_drop_from_baseline_n": sample["axial_drop_from_baseline_n"],
                                "lateral_rise_from_baseline_n": sample["lateral_rise_from_baseline_n"],
                                "lateral_force_xy_n": sample["lateral_force_xy_n"],
                                "moment_xyz_nm_private": sample["moment_xyz_nm_private"],
                                "socket_motor_torque_nm_private": sample[
                                    "socket_motor_torque_nm_private"
                                ],
                                "socket_screw_normal_contact_force_n_private": sample[
                                    "socket_screw_normal_contact_force_n_private"
                                ],
                                "socket_screw_contact_count_private": sample[
                                    "socket_screw_contact_count_private"
                                ],
                                "tool_tip_xy_private_m": sample["tool_tip_xy_private_m"],
                                "socket_angle_private_rad": sample["socket_angle_private_rad"],
                                "socket_seated_diagnostic": sample["socket_seated_diagnostic"],
                                **({"contact_pairs": sample["contact_pairs"]}
                                   if "contact_pairs" in sample else {}),
                            }
                            for sample in worker.spiral_force_trace
                        ]
                        if args.compact_trace else worker.spiral_force_trace
                    ),
                    "spiral_search_state": worker.spiral_search_state.tolist(),
                })
                if not args.no_frames and (
                    search_records % args.camera_frame_stride == 0
                    or worker.spiral_tactile_detector.candidate_detected
                ):
                    frames = {
                        name: worker._encode_image(worker._render_rgb_depth(camera_id)[0])
                        for name, camera_id in worker.camera_ids.items()
                    }
                    for camera, encoded in frames.items():
                        write_frame(encoded, frames_dir / f"{search_records:04d}_{camera}.jpg")
                    if np.linalg.norm(
                        np.asarray(result["force_torque_peak_compensated"][:3], dtype=float)
                    ) >= 50.0:
                        force_guarded = True
                        break
                if (worker.spiral_tactile_detector
                        and worker.spiral_tactile_detector.candidate_detected):
                    if force_guarded:
                        break
                    sensor_triggered_online = True
                    if worker.spiral_tactile_detector.seated:
                        break
                    worker.spiral_tactile_detector.candidate_detected = False
                    worker.spiral_tactile_detector.candidate_latched = False
                    worker.spiral_tactile_detector.confirmation_steps = 0
                    worker.spiral_tactile_detector.force_drop_detected = False
                    worker.spiral_tactile_detector.force_drop_age_steps = None
                    worker.spiral_tactile_detector.axial_drop_condition_previous = False
                    worker.spiral_tactile_detector.candidate_cooldown_steps = 500
                    worker.spiral_tactile_detector.candidate_rearm_blocked = True
                if rotation_start is not None and rotation_timeout:
                    spin_timed_out = True
                    break

            engaged, contact, axial, radial, alignment = worker._valid_socket_engagement()
            posthoc_geometry_validation = bool(
                worker.socket_fit_ticks >= 20
                and -0.014 <= axial <= -WORKER_MODULE.SPIRAL_MIN_SOCKET_INSERTION_M
                and radial <= 0.0015 and alignment >= 0.995
            )
            tactile_triggered = sensor_triggered_online
            online_seating_confirmed = bool(
                worker.spiral_tactile_detector
                and worker.spiral_tactile_detector.seated
            )
            summary = {
                "status": (
                    "force_guard"
                    if force_guarded else
                    worker.spiral_force_baseline_guard_reason
                    if worker.spiral_force_baseline_guard_reason == "spiral_point_unreached" else
                    "socket_rotation_timeout"
                    if spin_timed_out else
                    "sensor_trigger_geometry_validated"
                    if online_seating_confirmed and posthoc_geometry_validation else
                    "sensor_trigger_not_geometry_validated"
                    if tactile_triggered else
                    "geometry_valid_without_sensor_trigger"
                    if posthoc_geometry_validation else "point_limit"
                ),
                "search_points_executed": search_records,
                "sensor_triggered_by_force_tactile": tactile_triggered,
                "online_sensor_triggered": sensor_triggered_online,
                "sensor_thresholds_calibrated": bool(
                    worker.spiral_tactile_detector
                    and worker.spiral_tactile_detector.thresholds_calibrated
                ),
                "sensor_settle_rotation_deg": sensor_settle_rotation_deg,
                "posthoc_geometry_validation": posthoc_geometry_validation,
                "validated_success": (
                    online_seating_confirmed and posthoc_geometry_validation
                ),
                "geometry_seated_posthoc": posthoc_geometry_validation,
                "online_seating_confirmed": online_seating_confirmed,
                "tactile_seated": bool(worker.spiral_tactile_detector and
                                        worker.spiral_tactile_detector.seated),
                "tactile_candidate": bool(worker.spiral_tactile_detector and
                                           worker.spiral_tactile_detector.candidate_detected),
                "tactile_candidate_seen": tactile_candidate_seen,
                "socket_engaged_at_end": engaged,
                "socket_contact_at_end": contact,
                "socket_axial_error_m_at_end": axial,
                "socket_radial_error_m_at_end": radial,
                "socket_engaged_seen": worker.engaged_seen,
                "socket_engagement_ticks_at_end": worker.socket_engagement_ticks,
                "max_socket_engagement_ticks": max_engagement_ticks,
                "socket_fit_geometry_ticks_at_end": worker.socket_fit_ticks,
                "max_socket_fit_geometry_ticks": worker.max_socket_fit_ticks,
                "tool_tip_world_m_at_end": worker.data.site_xpos[worker.tool_tip_id].tolist(),
                "true_screw_center_world_m_private": screw_center.tolist(),
                "true_xy_error_mm_private": ((worker.data.site_xpos[worker.tool_tip_id, :2]
                                              - screw_center) * 1000.0).tolist(),
                "force_compensated_n_nm_at_end": result["force_torque_compensated"],
                "spiral_phase_offset_rad": worker.spiral_phase_offset,
                "simulation_steps": worker.steps,
                "simulation_time_s": float(worker.data.time),
                "llm_calls": 0,
            }
            result_path.write_text(json.dumps(summary, indent=2) + "\n")
            if video_frames:
                from ur20_screw.evaluation.video import encode_jpeg_frames

                video_dir = output / "video_frames"
                video_dir.mkdir(exist_ok=True)
                for frame_index, (_, frame) in enumerate(video_frames):
                    Image.fromarray(frame).save(
                        video_dir / f"ur20_video_{frame_index:06d}.jpg", quality=92
                    )
                encode_jpeg_frames(video_dir, output / "screw_side_realtime.mp4",
                                   fps=args.video_fps)
                summary["video"] = str(output / "screw_side_realtime.mp4")
                summary["video_frames"] = len(video_frames)
                result_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"output": str(output), **summary}, indent=2))
            wait_for_viewer_close(worker)
    finally:
        trajectory_points_path = output / "tool_tip_trajectory.jsonl"
        with trajectory_points_path.open("w", encoding="utf-8") as stream:
            for point in physical_trajectory:
                stream.write(json.dumps(point, ensure_ascii=False) + "\n")
        worker.close()


if __name__ == "__main__":
    main()
