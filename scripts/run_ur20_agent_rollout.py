"""Run one real or dry-run agent-policy episode on the UR20 screw scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import httpx
import numpy as np
from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots_agent.policy import agent_policy

from encode_rollout_frames import encode_frames, encode_jpeg_frames
from inspect_robots_libero.budget import PRICING, BudgetTransport, wire_for
from inspect_robots_libero.ur20_embodiment import UR20ScrewEmbodiment
from ur20_tui import UR20Dashboard

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"luna": ("OPENAI_API_KEY_LUNA", "OPENAI_BASE_URL_LUNA"),
            "terra": ("OPENAI_API_KEY_TERRA", "OPENAI_BASE_URL_TERRA"),
            "deepseek": ("OPENAI_API_KEY_DEEPSEEK", "OPENAI_BASE_URL_DEEPSEEK"),
            "sol61": ("OPENAI_API_KEY_SOL61", "OPENAI_BASE_URL_SOL61")}


class LiveConsole:
    COLORS = {"reset": "\033[0m", "cyan": "\033[36m", "green": "\033[32m",
              "yellow": "\033[33m", "magenta": "\033[35m", "red": "\033[31m",
              "dim": "\033[2m", "bold": "\033[1m"}

    def __init__(self, stream):
        self.stream = stream
        self.tty = stream.isatty()
        self.pending = ""

    @property
    def encoding(self):
        return self.stream.encoding

    def isatty(self):
        return self.tty

    def fileno(self):
        return self.stream.fileno()

    def _paint(self, text, color):
        if not self.tty:
            return text
        return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"

    def _emit(self, line):
        if line.startswith("[agent] >> input: "):
            return
        if line.startswith("[agent] << tool_call "):
            payload = line[len("[agent] << tool_call "):]
            name, separator, raw_arguments = payload.partition("(")
            try:
                arguments = json.loads(raw_arguments[:-1]) if separator and raw_arguments.endswith(")") else {}
            except json.JSONDecodeError:
                arguments = {}
            targets = arguments.get("targets", arguments.get("deltas", {}))
            target_text = ", ".join(f"{key}={value}" for key, value in targets.items())
            self.stream.write(self._paint(f"  ACTION  {name}  {target_text}\n", "yellow"))
            note = arguments.get("note")
            if note:
                width = 112 if self.tty else 160
                wrapped = textwrap.fill(note, width=width, initial_indent="          ",
                                        subsequent_indent="          ")
                self.stream.write(self._paint(wrapped + "\n", "dim"))
            return
        if line.startswith("[agent] -- "):
            self.stream.write(self._paint(f"  EXEC    {line[len('[agent] -- '):]}\n", "magenta"))
            return
        if line.startswith("[agent] << "):
            self.stream.write(self._paint(f"  POLICY  {line[len('[agent] << '):]}\n", "magenta"))
            return
        if line.startswith("[episode "):
            color = "red" if any(word in line.lower() for word in ("error", "stopping", "exhausted")) else "cyan"
            self.stream.write(self._paint(f"  API     {line}\n", color))
            return
        if line.startswith("[UR20 step "):
            self.stream.write(self._paint(f"  SIM     {line}\n", "green"))
            return
        self.stream.write(line + "\n")
        self.stream.flush()

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self):
        if self.pending:
            self._emit(self.pending)
            self.pending = ""
        self.stream.flush()

    def close(self):
        self.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["openai/gpt-6-luna", "openai/gpt-6.1-terra",
                                             "openai/deepseek-v4.1-flash-1017", "openai/gpt-6.1-sol"],
                        required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-amplification", type=float, default=3.0,
                        help="multiply Cartesian target deltas before the 5 cm per-step cap (1-5; disabled after contact or low-confidence vision)")
    parser.add_argument("--state-id", type=int, choices=range(10), default=0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--prior-learnings", type=Path)
    parser.add_argument("--tui", action="store_true", help="use the experimental live dashboard")
    parser.add_argument("--mujoco-viewer", action="store_true",
                        help="open a real-time MuJoCo window using the agentview camera")
    parser.add_argument("--force-feedback", action="store_true",
                        help="include estimated MuJoCo socket contact forces in policy observations")
    parser.add_argument("--compliant-control", action="store_true",
                        help="enable optional 1D axial force-admittance control after contact")
    parser.add_argument("--spiral-search", action="store_true",
                        help="expose a bounded force-controlled spiral socket-seating primitive")
    parser.add_argument("--twist-after-seating", action="store_true",
                        help="after stable seating, allow one bounded socket revolution; screw height remains fixed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-paid-rollouts", action="store_true", required=True)
    args = parser.parse_args()
    if args.compliant_control and not args.force_feedback:
        parser.error("--compliant-control requires --force-feedback")
    if args.spiral_search and not args.compliant_control:
        parser.error("--spiral-search requires --compliant-control")
    if args.twist_after_seating and not args.spiral_search:
        parser.error("--twist-after-seating requires --spiral-search")
    if args.twist_after_seating and not args.force_feedback:
        parser.error("--twist-after-seating requires --force-feedback")
    if not np.isfinite(args.action_amplification) or not 1.0 <= args.action_amplification <= 5.0:
        parser.error("--action-amplification must be between 1.0 and 5.0")
    settings = {**read_dotenv(ROOT / ".env"), **os.environ}
    key_env, base_env = PROFILES[args.profile]
    if not settings.get(key_env) or not settings.get(base_env):
        parser.error(f"{key_env} and {base_env} must be configured in .env")
    base_url = settings[base_env].rstrip("/")
    wire = wire_for(args.model, base_url=base_url)
    policy_model = args.model.partition("/")[2]
    # Keep OpenAI-compatible Responses requests on the configured API version
    # path, matching the endpoint validated by BudgetTransport.
    policy_base = base_url if base_url.endswith("/v1") else base_url + "/v1"
    args.output.mkdir(parents=True, exist_ok=False)
    transport = BudgetTransport(args.output / "api", 100000, 100000, task_id=0,
                                max_calls=args.max_calls, model=args.model,
                                max_output_tokens=args.max_output_tokens, base_url=base_url,
                                endpoint_base_url=policy_base)
    transport.begin_episode(0)
    embodiment = UR20ScrewEmbodiment(output_dir=args.output / "episode_00", state_id=args.state_id,
                                     max_steps=args.max_steps, include_force=args.force_feedback,
                                     compliant_control=args.compliant_control,
                                     spiral_search=args.spiral_search,
                                     mujoco_viewer=args.mujoco_viewer,
                                     action_amplification=args.action_amplification,
                                     twist_after_seating=args.twist_after_seating)
    policy_options = dict(model=policy_model, base_url=policy_base, api_key_env=key_env,
                          env={key_env: settings[key_env]}, wire=wire,
                          max_llm_calls=args.max_calls, image_horizon=2,
                          prior_learnings=str(args.prior_learnings.resolve()) if args.prior_learnings else None,
                          allow_give_up=False, require_success_for_done=True,
                          transcript_echo=True,
                          transport=transport, wire_capture=True)
    if wire == "messages":
        policy_options["max_output_tokens"] = args.max_output_tokens
    policy = agent_policy(**policy_options)
    policy._client._http.timeout = httpx.Timeout(args.request_timeout, connect=10.0)
    task_instruction = (
        ("Seat the socket securely onto the M6 external-hex screw head, then turn the socket "
         "counterclockwise one revolution while maintaining contact; the screw may rotate "
         "in place but does not rise or extract. " if args.twist_after_seating else
         "Seat the socket securely onto the M6 external-hex screw head. ")
        + "At every observation, "
        "use wristview-derived vision_screw_xy_estimate_m and vision_xy_error_m only when "
        "vision_confidence is at least 0.7. Below 0.7, hold x/y fixed and do not treat "
        "a zero error as alignment. Align x/y while holding z; the controller "
        "blocks downward motion while confident estimated XY error exceeds 2 mm. "
        + ("In force-controlled spiral-search mode, reassess estimated x/y error at "
           "every observation: stop x/y corrections only while confident error is at "
           "most 2 mm; if it later exceeds 2 mm, resume x/y correction. While aligned, "
           "the controller enforces continued downward progress. "
           if args.spiral_search else "")
        + "move_to z is an absolute world coordinate and smaller z means downward. "
        "Descend toward the visible screw top: use the full 5 cm safety-limited target "
        "change while the visible gap is clearly larger than 5 cm; as the top gets near, "
        "continue down through the remaining visible clearance instead of hovering above "
        "it or commanding an upward retreat. "
        + ("After XY alignment, when the estimated Z gap to the calibrated nominal "
           "screw-top plane is at most "
           "5 cm, the controller enables a 2 N compliant axial setpoint. "
           if args.compliant_control else "")
        + ("Once compensated tool-axis force reaches 1 N, contact takes priority over "
           "policy motion regardless of residual XY alignment. The controller latches "
           "contact and suspends Cartesian motion. Call the spiral_search tool exactly "
           "once to start the bounded search; do not issue move_to after contact. The "
           "spiral_search call blocks until stable seating, force protection, or the "
           "configured step limit, without intermediate LLM calls. It resolves residual "
           "lateral and external-hex phase mismatch. Never command an upward/retracting "
           "move after contact. Each "
           "increment moves "
           "the tip a small amount in the plane while rotating the socket 60 degrees. "
           "Search stops on stable seating, its force guard, or the configured step limit. "
           if args.spiral_search else "")
        + "A useful visual cue is that, after the socket is properly seated over the "
        "external-hex screw head, the screw head will normally be occluded by the socket "
        "in agentview; treat this as supporting evidence, not proof by itself. Do not claim "
        "contact until supported by visual or force evidence. Success is "
        + ("completion of the one-revolution turn_socket operation after stable seating; "
           "the screw remains at fixed height and is not extracted. "
           if args.twist_after_seating else "stable physical socket engagement. ")
        + "Do not use move_to to rotate the socket."
    )
    task = Task(name=("ur20_hex_socket_turning" if args.twist_after_seating
                      else "ur20_hex_socket_seating"), scenes=[Scene(
        f"ur20-screw-init{args.state_id:02d}", task_instruction)],
        scorer="success_at_end", max_steps=args.max_steps)
    original_stderr = sys.stderr
    console = UR20Dashboard(original_stderr) if args.tui else LiveConsole(original_stderr)
    sys.stderr = console
    try:
        print(f"[UR20] model={args.model} state={args.state_id:02d} "
              f"limits={args.max_calls} calls/{args.max_steps} sim steps "
              f"action-gain={args.action_amplification:.1f}x (5 cm cap) "
              f"output={args.output}", file=sys.stderr, flush=True)
        log, = evaluate(task, policy, embodiment, log_dir=str(args.output / "inspect"),
                        store_frames=True, store_actions=True,
                        approver=ChainApprover(ClampApprover(embodiment.info.action_space),
                                               DeltaLimitApprover(embodiment.info.action_space)))
        if args.tui:
            console.finish(embodiment.task_success)
        frame_dirs = sorted((args.output / "inspect" / "frames").glob("*"))
        frame_dir = next((path for path in frame_dirs if list(path.glob("*_agentview_*.npy"))), None)
        video_path = args.output / "rollout.mp4"
        video_frame_count = 0
        video_fps = 20
        if list(embodiment.video_frames_dir.glob("ur20_video_*.jpg")):
            video_frame_count = encode_jpeg_frames(
                embodiment.video_frames_dir, video_path, fps=video_fps
            )
        elif frame_dir is not None:
            video_frame_count = encode_frames(frame_dir, video_path)
        eval_error = getattr(log, "error", None)
        status = "success" if embodiment.task_success else "error" if eval_error else "task_failure"
        summary = {"status": status, "runner_status": log.status,
                   "environment_success": embodiment.task_success,
                   "valid_socket_engagement_seen": embodiment.engaged_seen,
                   "success_criterion": ("one_socket_revolution_after_stable_seating"
                                         if args.twist_after_seating else "stable_socket_engagement"),
                   "required_stable_physics_steps": 20,
                   "max_screw_lift_m": embodiment.max_screw_lift,
                   "calls": transport.calls,
                   "sim_steps": embodiment.environment_steps,
                   "physics_steps": embodiment.physics_steps,
                   "spiral_internal_increments": embodiment.spiral_internal_increments,
                   "estimated_usd": transport.episode_estimated,
                   "state_id": args.state_id,
                   "force_feedback": args.force_feedback,
                   "compliant_control": args.compliant_control,
                   "spiral_search": args.spiral_search,
                   "twist_after_seating": args.twist_after_seating,
                   "socket_turn_completed": embodiment.socket_turn_completed,
                   "socket_turn_stop_reason": embodiment.socket_turn_stop_reason,
                   "socket_turn_screw_rotation_delta_rad": embodiment.socket_turn_screw_delta,
                   "action_amplification": args.action_amplification,
                   "video_frames": video_frame_count,
                   "video_fps": video_fps,
                   "simulation_realtime_pacing": True,
                   "video": str(video_path) if video_path.is_file() else None,
                   "output": str(args.output)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        status_color = "green" if embodiment.task_success else "red"
        print(f"\n{LiveConsole.COLORS[status_color] if sys.stdout.isatty() else ''}"
              f"{'SUCCESS' if embodiment.task_success else 'TASK FAILURE'}"
              f"{LiveConsole.COLORS['reset'] if sys.stdout.isatty() else ''}")
        print(json.dumps(summary, indent=2))
    finally:
        sys.stderr.flush()
        sys.stderr = original_stderr
        console.close()
        embodiment.close()
        policy._client.close()
        transport.dispose()


if __name__ == "__main__":
    main()
