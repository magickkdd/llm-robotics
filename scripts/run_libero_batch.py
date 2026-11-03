"""Run all standard LIBERO suites and a reproducible LIBERO-90 sample."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/deepseek-v4.1-flash-1017")
    parser.add_argument("--profile", default="deepseek")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-budget-per-task", type=float, default=100000.0)
    parser.add_argument("--episode-budget", type=float, default=100000.0)
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=24579)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--libero-90-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.libero_90_count <= 90:
        parser.error("--libero-90-count must be between 0 and 90")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    sample = sorted(random.Random(args.seed).sample(range(90), args.libero_90_count))
    plan = [(suite, task_id) for suite in ALL_SUITES for task_id in range(10)]
    plan += [("libero_90", task_id) for task_id in sample]
    print(f"Batch: {len(plan)} tasks; LIBERO-90 sample (seed={args.seed}): {sample}", flush=True)
    results = []
    for index, (suite, task_id) in enumerate(plan, start=1):
        task_output = output_root / suite / f"task{task_id:02d}"
        command = [
            sys.executable, str(ROOT / "scripts/run_budgeted_rollouts.py"),
            "--model", args.model, "--profile", args.profile,
            "--suite", suite, "--task-id", str(task_id),
            "--episodes", "1", "--start", "0",
            "--total-budget", str(args.total_budget_per_task),
            "--episode-budget", str(args.episode_budget),
            "--max-calls", str(args.max_calls), "--max-steps", str(args.max_steps),
            "--max-output-tokens", str(args.max_output_tokens),
            "--request-timeout", str(args.request_timeout),
            "--output", str(task_output), "--execute-paid-rollouts",
        ]
        print(f"\n[{index}/{len(plan)}] {suite} task {task_id:02d}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        summary_file = task_output / "summary.json"
        if not summary_file.is_file():
            results.append({"suite": suite, "task_id": task_id,
                            "status": "runner_failed", "exit_code": completed.returncode})
            break
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        row = (summary.get("results") or [{}])[0]
        results.append({"suite": suite, "task_id": task_id, **row})
        if row.get("status") in {"api_error_usage_unknown", "error", "not_run_budget_or_usage_unknown"}:
            print(f"Stopping batch safely after {row.get('status')}; completed task outputs are preserved.", flush=True)
            break
        if completed.returncode:
            print(f"Stopping batch after runner exit {completed.returncode}.", flush=True)
            break
    (output_root / "batch_summary.json").write_text(
        json.dumps({"seed": args.seed, "libero_90_task_ids": sample,
                    "planned_tasks": len(plan), "results": results}, indent=2),
        encoding="utf-8",
    )
    done = len(results)
    print(f"\nBatch stopped/finished after {done}/{len(plan)} tasks. Results: {output_root}", flush=True)
    return 0 if done == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
