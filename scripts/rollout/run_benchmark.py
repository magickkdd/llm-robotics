"""Run all 10 deterministic UR20 screw-removal starts sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["openai/gpt-6-luna", "openai/gpt-6.1-terra", "openai/deepseek-v4.1-flash-1017", "openai/gpt-6.1-sol"], required=True)
    parser.add_argument("--profile", choices=["deepseek", "luna", "terra", "sol61"], required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--force-feedback", action="store_true")
    parser.add_argument("--compliant-control", action="store_true")
    parser.add_argument("--spiral-search", action="store_true")
    parser.add_argument("--execute-paid-rollouts", action="store_true", required=True)
    args = parser.parse_args()
    if args.compliant_control and not args.force_feedback:
        parser.error("--compliant-control requires --force-feedback")
    if args.spiral_search and not args.compliant_control:
        parser.error("--spiral-search requires --compliant-control")

    output_root = args.output_root or ROOT / "outputs" / f"ur20_screw_benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    output_root.mkdir(parents=True, exist_ok=False)
    runner = ROOT / "scripts" / "rollout" / "run_episode.py"
    print(f"Running 10 UR20 screw-removal starts under {output_root}", flush=True)
    failures = []
    for state_id in range(10):
        output = output_root / f"state{state_id:02d}"
        command = [
            sys.executable, str(runner), "--model", args.model, "--profile", args.profile,
            "--state-id", str(state_id), "--max-calls", str(args.max_calls),
            "--max-steps", str(args.max_steps), "--max-output-tokens", str(args.max_output_tokens),
            "--request-timeout", str(args.request_timeout), "--output", str(output),
            "--execute-paid-rollouts",
        ]
        if args.force_feedback:
            command.append("--force-feedback")
        if args.compliant_control:
            command.append("--compliant-control")
        if args.spiral_search:
            command.append("--spiral-search")
        print(f"\n[{state_id + 1}/10] state{state_id:02d}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            failures.append(state_id)
            print(f"state{state_id:02d} exited {result.returncode}; continuing with remaining starts", flush=True)
    print(f"Benchmark outputs: {output_root}")
    if failures:
        print("Failed states:", ", ".join(f"{state_id:02d}" for state_id in failures))


if __name__ == "__main__":
    main()
