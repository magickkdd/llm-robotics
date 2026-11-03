"""Run one real or dry-run agent-policy episode on the UR20 screw scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import httpx
from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots_agent.policy import agent_policy

from encode_rollout_frames import encode_frames
from inspect_robots_libero.budget import PRICING, BudgetTransport, wire_for
from inspect_robots_libero.ur20_embodiment import UR20ScrewEmbodiment

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"luna": ("OPENAI_API_KEY_LUNA", "OPENAI_BASE_URL_LUNA"),
            "terra": ("OPENAI_API_KEY_TERRA", "OPENAI_BASE_URL_TERRA"),
            "deepseek": ("OPENAI_API_KEY_DEEPSEEK", "OPENAI_BASE_URL_DEEPSEEK")}


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
        if line.startswith("[agent] << tool_call "):
            payload = line[len("[agent] << tool_call "):]
            name, separator, raw_arguments = payload.partition("(")
            try:
                arguments = json.loads(raw_arguments[:-1]) if separator and raw_arguments.endswith(")") else {}
            except json.JSONDecodeError:
                arguments = {}
            targets = arguments.get("targets", {})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["openai/gpt-6-luna", "openai/gpt-6.1-terra", "openai/deepseek-v4.1-flash-1017"], required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--state-id", type=int, choices=range(10), default=0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--prior-learnings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-paid-rollouts", action="store_true", required=True)
    args = parser.parse_args()
    settings = {**read_dotenv(ROOT / ".env"), **os.environ}
    key_env, base_env = PROFILES[args.profile]
    if not settings.get(key_env) or not settings.get(base_env):
        parser.error(f"{key_env} and {base_env} must be configured in .env")
    base_url = settings[base_env].rstrip("/")
    wire = wire_for(args.model, base_url=base_url)
    policy_model = args.model.partition("/")[2]
    # Luna's Spacetime Responses gateway uses the root URL; Discovery already
    # includes /v1 in its configured base URL.
    policy_base = base_url if args.profile == "luna" or base_url.endswith("/v1") else base_url + "/v1"
    args.output.mkdir(parents=True, exist_ok=False)
    transport = BudgetTransport(args.output / "api", 100000, 100000, task_id=0,
                                max_calls=args.max_calls, model=args.model,
                                max_output_tokens=args.max_output_tokens, base_url=base_url,
                                endpoint_base_url=policy_base)
    transport.begin_episode(0)
    embodiment = UR20ScrewEmbodiment(output_dir=args.output / "episode_00", state_id=args.state_id,
                                     max_steps=args.max_steps)
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
    task = Task(name="ur20_battery_disassembly", scenes=[Scene(
        f"ur20-screw-init{args.state_id:02d}",
        "Remove the M6 external-hex screw from the battery. Align and seat the socket on the hex head, then rotate counterclockwise when viewed from above until the screw is extracted.")],
        scorer="success_at_end", max_steps=args.max_steps)
    try:
        print(f"[UR20] model={args.model} state={args.state_id:02d} "
              f"limits={args.max_calls} calls/{args.max_steps} sim steps "
              f"output={args.output}", file=sys.stderr, flush=True)
        original_stderr = sys.stderr
        sys.stderr = LiveConsole(original_stderr)
        try:
            log, = evaluate(task, policy, embodiment, log_dir=str(args.output / "inspect"),
                            store_frames=True, store_actions=True,
                            approver=ChainApprover(ClampApprover(embodiment.info.action_space),
                                                   DeltaLimitApprover(embodiment.info.action_space)))
        finally:
            sys.stderr.flush()
            sys.stderr = original_stderr
        frame_dirs = sorted((args.output / "inspect" / "frames").glob("*"))
        frame_dir = next((path for path in frame_dirs if list(path.glob("*_agentview_*.npy"))), None)
        video_path = args.output / "rollout.mp4"
        if frame_dir is not None:
            encode_frames(frame_dir, video_path)
        eval_error = getattr(log, "error", None)
        status = "success" if embodiment.task_success else "error" if eval_error else "task_failure"
        summary = {"status": status, "runner_status": log.status,
                   "environment_success": embodiment.task_success,
                   "valid_socket_engagement_seen": embodiment.engaged_seen,
                   "max_screw_lift_m": embodiment.max_screw_lift,
                   "required_screw_lift_m": 0.020,
                   "calls": transport.calls,
                   "sim_steps": embodiment.environment_steps,
                   "physics_steps": embodiment.physics_steps,
                   "estimated_usd": transport.episode_estimated,
                   "state_id": args.state_id,
                   "video": str(video_path) if video_path.is_file() else None,
                   "output": str(args.output)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        status_color = "green" if embodiment.task_success else "red"
        print(f"\n{LiveConsole.COLORS[status_color] if sys.stdout.isatty() else ''}"
              f"{'SUCCESS' if embodiment.task_success else 'TASK FAILURE'}"
              f"{LiveConsole.COLORS['reset'] if sys.stdout.isatty() else ''}")
        print(json.dumps(summary, indent=2))
    finally:
        embodiment.close()
        policy._client.close()
        transport.dispose()


if __name__ == "__main__":
    main()
