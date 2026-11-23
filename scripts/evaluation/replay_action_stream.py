"""Paired simulator replay of archived UR20 decisions with guarded action streaming."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ur20_screw.contracts import EmbodimentFault, Scene
from ur20_screw.control.actions import action_for
from ur20_screw.simulation.robot import UR20ScrewEmbodiment

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / "archive" / "rollouts"


def load_trace(path: Path) -> tuple[list[dict], dict[int, float]]:
    transcript = [json.loads(line) for line in (path / "transcript.jsonl").read_text().splitlines()
                  if line.strip()]
    calls = [item for item in transcript
             if item.get("role") == "assistant" and item.get("tool")]
    usage = [json.loads(line) for line in (path / "api_usage.jsonl").read_text().splitlines()
             if line.strip()]
    latencies = {int(item["call"]): float(item["latency_s"])
                 for item in usage if item.get("event") == "response"
                 and item.get("latency_s") is not None}
    return calls, latencies


def run_trace(path: Path, fused: bool, output: Path) -> dict:
    calls, latencies = load_trace(path)
    robot = UR20ScrewEmbodiment(
        output_dir=output, state_id=0, max_steps=7500, include_force=True,
        compliant_control=True, spiral_search=True, action_amplification=3.0,
        twist_after_seating=True,
    )
    robot._record_video_frame = lambda _data: None
    observation = robot.reset(Scene(
        "ur20-replay-state00",
        "Seat the socket on the M6 screw, confirm tactile and visual seating, then "
        "extract at least 21 mm through the modeled thread.",
    ))
    started = time.perf_counter()
    turn_index = 0
    consumed_calls: list[int] = []
    executed_actions = 0
    try:
        while turn_index < len(calls) and not robot.task_success:
            current = calls[turn_index]
            name, arguments = current["tool"], current.get("arguments", {})
            if name == "done":
                break
            if fused and name == "move_to":
                end = turn_index + 1
                while (end < len(calls) and end - turn_index < 4
                       and calls[end].get("tool") == "move_to"):
                    end += 1
                if end - turn_index > 1:
                    robot.begin_action_stream()
                    if not robot.action_stream_should_commit():
                        executed = 0
                        for index in range(turn_index, end):
                            item = calls[index]
                            if robot.action_stream_should_commit():
                                break
                            action = action_for("move_to", item.get("arguments", {}), observation)
                            result = robot.step(action)
                            observation = result.observation
                            executed += 1
                            executed_actions += 1
                            if result.terminated or robot.task_success:
                                break
                            if robot.action_stream_should_commit():
                                break
                        consumed_calls.append(turn_index + 1)
                        turn_index += executed
                        continue
            try:
                action = action_for(name, arguments, observation)
            except ValueError:
                consumed_calls.append(turn_index + 1)
                turn_index += 1
                continue
            if action is not None:
                try:
                    result = robot.step(action)
                except EmbodimentFault:
                    turn_index += 1
                    continue
                observation = result.observation
                executed_actions += 1
            consumed_calls.append(turn_index + 1)
            turn_index += 1
        elapsed = time.perf_counter() - started
        policy_latency = sum(latencies.get(call, 0.0) for call in consumed_calls)
        return {
            "source_trace": path.name,
            "mode": "fused" if fused else "native",
            "policy_turns": len(consumed_calls),
            "source_policy_turns": len(calls),
            "executed_actions": executed_actions,
            "replay_wall_s": elapsed,
            "archived_policy_latency_s": policy_latency,
            "projected_total_s": elapsed + policy_latency,
            "success": robot.task_success,
            "engaged_seen": robot.engaged_seen,
            "socket_turn_completed": robot.socket_turn_completed,
            "socket_turn_extraction_m": robot.socket_turn_extraction_m,
            "max_screw_lift_m": robot.max_screw_lift,
            "environment_steps": robot.environment_steps,
            "physics_steps": robot.physics_steps,
            "spiral_internal_increments": robot.spiral_internal_increments,
            "source_call_ids_used": consumed_calls,
        }
    finally:
        robot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", nargs="*", type=Path,
                        default=sorted(ARCHIVES.glob("*/summary.json")))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for summary_path in args.traces:
        trace = summary_path.parent if summary_path.name == "summary.json" else summary_path
        for fused in (False, True):
            result = run_trace(trace, fused, args.output / f"{trace.name}_{'fused' if fused else 'native'}")
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    (args.output / "results.json").write_text(
        json.dumps({"protocol": "paired archived-decision replay; state_id=0", "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
