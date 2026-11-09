"""Run one UR20 screw-removal episode with the project-owned policy loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ur20_screw.evaluation.video import encode_jpeg_frames
from ur20_screw.policy.budget import BudgetTransport
from ur20_screw.policy.client import UR20Policy
from ur20_screw.control.actions import action_for
from ur20_screw.simulation.robot import UR20ScrewEmbodiment
from ur20_screw.contracts import EmbodimentFault, Scene
from ur20_screw.policy.tui import UR20Dashboard

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "luna": ("OPENAI_API_KEY_LUNA", "OPENAI_BASE_URL_LUNA"),
    "terra": ("OPENAI_API_KEY_TERRA", "OPENAI_BASE_URL_TERRA"),
    "deepseek": ("OPENAI_API_KEY_DEEPSEEK", "OPENAI_BASE_URL_DEEPSEEK"),
    "sol61": ("OPENAI_API_KEY_SOL61", "OPENAI_BASE_URL_SOL61"),
}
MODELS = ("openai/gpt-6-luna", "openai/gpt-6.1-terra",
          "openai/deepseek-v4.1-flash-1017", "openai/gpt-6.1-sol")


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def allocate_output_directory(requested: Path | None, profile: str, state_id: int) -> Path:
    if requested is None:
        requested = ROOT / "outputs" / (
            f"ur20_{profile}_state{state_id:02d}_{datetime.now():%Y%m%d_%H%M%S}"
        )
    candidate = requested
    suffix = 2
    while candidate.exists():
        candidate = requested.with_name(f"{requested.name}_run{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def read_dotenv(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return settings
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        settings[key.strip()] = value
    return settings


def task_instruction(args: argparse.Namespace) -> str:
    goal = ("Seat the socket securely on the M6 external-hex screw head, then call "
            "turn_socket once to unscrew and extract it through the modeled M6x1 thread. "
            if args.twist_after_seating else
            "Seat the socket securely on the M6 external-hex screw head.")
    guidance = (
        "Use camera-derived vision_screw_xy_estimate_m and vision_xy_error_m only when "
        "vision_confidence is at least 0.7. Below that, hold x/y fixed and do not treat a "
        "zero error as alignment. Once x/y has been aligned, low confidence is not a reason "
        "to stop: the controller keeps the alignment lock and continues descending, and you "
        "should keep requesting small downward steps while compensated axial force is zero. "
        "Do not call done merely because confidence dropped. Align x/y while holding z; the "
        "controller blocks downward motion while confident estimated XY error exceeds 2 mm. "
        "move_to z is an absolute world coordinate and smaller z means down. Approach with "
        "visible clearance and continue descending while there is clearance. Do not use "
        "move_to to rotate the socket. "
    )
    if args.spiral_search:
        guidance += (
            "In force-controlled spiral mode, re-check x/y error at every observation. "
            "When compensated axial force reaches 1 N, call spiral_search exactly once; "
            "do not issue move_to after contact. The operation blocks until seating, force "
            "protection, or its coverage limit. "
        )
    if args.twist_after_seating:
        guidance += (
            "Call turn_socket only after spiral_tactile_state reports seated AND the images "
            "show the socket centered on and covering the screw head. If the images show the "
            "socket beside the head, do not turn; keep searching/adjusting. The tactile flag "
            "alone is not sufficient. Success requires real geometric socket engagement and "
            "the complete 21 mm extraction. "
        )
    return goal + " " + guidance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=7500)
    parser.add_argument("--action-amplification", type=float, default=3.0)
    parser.add_argument("--state-id", type=int, choices=range(10), default=0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--prior-learnings", type=Path)
    parser.add_argument("--tui", action="store_true")
    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--mujoco-viewer", dest="mujoco_viewer", action="store_true",
                        help="open the MuJoCo viewer (default)")
    viewer.add_argument("--no-mujoco-viewer", dest="mujoco_viewer", action="store_false",
                        help="run without the MuJoCo viewer")
    parser.set_defaults(mujoco_viewer=True)
    controls = parser.add_argument_group("UR20 contact control")
    controls.add_argument("--force-feedback", dest="force_feedback", action="store_true",
                          help="enable force/torque observations (default)")
    controls.add_argument("--no-force-feedback", dest="force_feedback", action="store_false",
                          help="disable force/torque observations")
    controls.add_argument("--compliant-control", dest="compliant_control", action="store_true",
                          help="enable compliant approach control (default)")
    controls.add_argument("--no-compliant-control", dest="compliant_control", action="store_false",
                          help="disable compliant approach control")
    controls.add_argument("--spiral-search", dest="spiral_search", action="store_true",
                          help="enable force-controlled spiral seating (default)")
    controls.add_argument("--no-spiral-search", dest="spiral_search", action="store_false",
                          help="disable spiral seating")
    controls.add_argument("--twist-after-seating", dest="twist_after_seating", action="store_true",
                          help="extract the screw after seating (default)")
    controls.add_argument("--no-twist-after-seating", dest="twist_after_seating", action="store_false",
                          help="stop after socket seating")
    parser.set_defaults(force_feedback=True, compliant_control=True,
                        spiral_search=True, twist_after_seating=True)
    parser.add_argument("--output", type=Path,
                        help="output directory; defaults to a unique timestamped directory")
    parser.add_argument("--execute-paid-rollouts", action="store_true", required=True)
    args = parser.parse_args()
    if args.compliant_control and not args.force_feedback:
        parser.error("--compliant-control requires --force-feedback")
    if args.spiral_search and not args.compliant_control:
        parser.error("--spiral-search requires --compliant-control")
    if args.twist_after_seating and not args.spiral_search:
        parser.error("--twist-after-seating requires --spiral-search")
    if not 1.0 <= args.action_amplification <= 5.0:
        parser.error("--action-amplification must be between 1.0 and 5.0")

    settings = read_dotenv(ROOT / ".env")
    settings.update(os.environ)
    key_name, base_name = PROFILES[args.profile]
    if not settings.get(key_name) or not settings.get(base_name):
        parser.error(f"{key_name} and {base_name} must be set in .env")
    base_url = settings[base_name].rstrip("/")
    # Spacetime's Luna gateway exposes Responses at the origin root; the other
    # OpenAI-compatible profiles use the conventional /v1 prefix.
    endpoint_base = (
        base_url if args.profile == "luna" or base_url.endswith("/v1")
        else base_url + "/v1"
    )
    args.output = allocate_output_directory(args.output, args.profile, args.state_id)
    budget = BudgetTransport(
        args.output / "api", 100000, 100000, task_id=0, max_calls=args.max_calls,
        model=args.model, image_horizon=2, images_per_observation=2,
        max_output_tokens=args.max_output_tokens, base_url=base_url,
        endpoint_base_url=endpoint_base,
    )
    budget.begin_episode(0)
    robot = UR20ScrewEmbodiment(
        output_dir=args.output / "episode_00", state_id=args.state_id,
        max_steps=args.max_steps, include_force=args.force_feedback,
        compliant_control=args.compliant_control, spiral_search=args.spiral_search,
        mujoco_viewer=args.mujoco_viewer,
        action_amplification=args.action_amplification,
        twist_after_seating=args.twist_after_seating,
    )
    policy = UR20Policy(
        model=args.model.partition("/")[2], api_key=settings[key_name],
        endpoint=endpoint_base + "/responses", transport=budget,
        timeout=args.request_timeout, max_calls=args.max_calls,
        max_output_tokens=args.max_output_tokens,
        prior_learnings=args.prior_learnings.read_text(encoding="utf-8")
        if args.prior_learnings else None,
    )
    if args.spiral_search:
        policy.enable_spiral_search()
    if args.twist_after_seating:
        policy.enable_socket_turn()

    console = UR20Dashboard(sys.stderr) if args.tui else None
    transcript = args.output / "transcript.jsonl"
    actions_path = args.output / "actions.jsonl"
    stopped_by = "call_limit"
    try:
        instruction = task_instruction(args)
        observation = robot.reset(Scene(f"ur20-init-{args.state_id:02d}", instruction))
        policy.observe(observation, robot.info.docs)
        for call_number in range(1, args.max_calls + 1):
            name, arguments, call_id = policy.act()
            print(f"[agent] << tool_call {name}({json.dumps(arguments, ensure_ascii=False, default=json_default)})",
                  file=sys.stderr, flush=True)
            with transcript.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"role": "assistant", "tool": name,
                                         "arguments": arguments}, ensure_ascii=False,
                                       default=json_default) + "\n")
            if name == "done":
                stopped_by = "policy_done"
                break
            try:
                action = action_for(name, arguments, observation)
            except ValueError as error:
                policy.tool_result(call_id, str(error))
                with transcript.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"role": "tool", "tool": name,
                                             "error": str(error)}, ensure_ascii=False) + "\n")
                continue
            try:
                result = robot.step(action)
            except EmbodimentFault as error:
                policy.tool_result(call_id, json.dumps(
                    {"status": "error", "error": str(error)}))
                with transcript.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"role": "tool", "tool": name,
                                             "error": str(error)}, ensure_ascii=False) + "\n")
                continue
            with actions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"tool": name, "arguments": arguments,
                                         "state": {key: np.asarray(value).tolist()
                                                   for key, value in result.observation.state.items()},
                                         "info": result.info}, ensure_ascii=False,
                               default=json_default) + "\n")
            observation = result.observation
            policy.tool_result(call_id, json.dumps({"status": "ok", "info": result.info},
                                                   default=json_default))
            if robot.task_success:
                stopped_by = "environment_success"
                break
            if result.terminated:
                stopped_by = result.termination_reason or "environment_terminated"
                break
            policy.observe(observation, robot.info.docs)
        if console:
            console.finish(robot.task_success)

        video_path = args.output / "rollout.mp4"
        video_frames = encode_jpeg_frames(robot.video_frames_dir, video_path, fps=20)
        summary = {
            "status": "success" if robot.task_success else "task_failure",
            "runner_status": stopped_by,
            "environment_success": robot.task_success,
            "valid_socket_engagement_seen": robot.engaged_seen,
            "success_criterion": "threaded_screw_extraction_after_stable_seating"
            if args.twist_after_seating else "stable_socket_engagement",
            "required_stable_physics_steps": 20,
            "max_screw_lift_m": robot.max_screw_lift,
            "calls": budget.calls,
            "sim_steps": robot.environment_steps,
            "physics_steps": robot.physics_steps,
            "spiral_internal_increments": robot.spiral_internal_increments,
            "estimated_usd": budget.episode_estimated,
            "state_id": args.state_id,
            "force_feedback": args.force_feedback,
            "compliant_control": args.compliant_control,
            "spiral_search": args.spiral_search,
            "twist_after_seating": args.twist_after_seating,
            "socket_turn_completed": robot.socket_turn_completed,
            "socket_turn_stop_reason": robot.socket_turn_stop_reason,
            "socket_turn_screw_rotation_delta_rad": robot.socket_turn_screw_delta,
            "required_screw_extraction_m": 0.021 if args.twist_after_seating else None,
            "action_amplification": args.action_amplification,
            "video_frames": video_frames,
            "video_fps": 20,
            "simulation_realtime_pacing": True,
            "video": str(video_path) if video_path.is_file() else None,
            "output": str(args.output),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        robot.close()
        policy.close()
        budget.dispose()


if __name__ == "__main__":
    main()
