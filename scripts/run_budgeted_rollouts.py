"""Paid LLM rollouts on one task from a native LIBERO suite with spending caps.

Same Task / Embodiment / agent policy as ``inspect-robots run``, plus:
  * a fail-closed per-episode and total USD budget (``BudgetTransport``),
  * a fresh policy per episode (no cross-episode memory),
  * optional demonstration storyboards (few-shot) and task notes,
  * a ``protocol.json`` + ``summary.json`` recording exactly what ran.

Keys are read from ``.env`` in the repo root (or the environment) and are only
handed to the policy; they are never written to disk or passed to LIBERO.
Nothing is sent unless ``--execute-paid-rollouts`` is given.

Example (one zero-shot episode of native task 2, at most $3):
    python scripts/run_budgeted_rollouts.py --model openai/gpt-6-sol --task-id 2 \
        --episodes 1 --total-budget 3 --output outputs/task2_sol_zeroshot \
        --execute-paid-rollouts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.task import Task
from inspect_robots_agent.policy import agent_policy

from inspect_robots_libero.budget import PRICING, BudgetTransport, wire_for
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import (
    NUM_INIT_STATES,
    libero_suite_task,
    load_suite_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}
OPENAI_PROFILES = {
    "luna": ("OPENAI_API_KEY_LUNA", "OPENAI_BASE_URL_LUNA"),
    "terra": ("OPENAI_API_KEY_TERRA", "OPENAI_BASE_URL_TERRA"),
    "deepseek": ("OPENAI_API_KEY_DEEPSEEK", "OPENAI_BASE_URL_DEEPSEEK"),
    "sol61": ("OPENAI_API_KEY_SOL61", "OPENAI_BASE_URL_SOL61"),
}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def load_settings() -> dict[str, str]:
    """Environment variables win over ``.env`` (same precedence as the upstream CLI)."""
    return {**read_dotenv(ROOT / ".env"), **os.environ}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(PRICING), required=True)
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    parser.add_argument("--suite", choices=suites, default="libero_10")
    parser.add_argument("--profile", choices=sorted(OPENAI_PROFILES),
                        help="select a named OpenAI-compatible credential/base-URL profile")
    parser.add_argument("--task-id", type=int, required=True,
                        help="native task id within the selected suite")
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes on init states start..start+N-1")
    parser.add_argument("--start", type=int, default=0, help="first LIBERO init state")
    parser.add_argument("--total-budget", type=float, required=True, help="USD cap for the run")
    parser.add_argument("--episode-budget", type=float, default=3.0, help="USD cap per episode")
    parser.add_argument("--max-calls", type=int, default=80, help="LLM requests per episode (paper limit)")
    parser.add_argument("--effort", choices=EFFORTS)
    parser.add_argument("--image-horizon", type=int, default=2,
                        help="how many past observations keep their images")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="output-token cap per OpenAI request, reasoning included")
    parser.add_argument("--request-timeout", type=float,
                        default=600.0,
                        help="seconds to wait for one LLM response (default: 600)")
    parser.add_argument("--demo-manifest", type=Path,
                        help="storyboards from scripts/prepare_demonstrations.py (few-shot)")
    parser.add_argument("--task-notes", action="store_true",
                        help="append demonstration-derived hints (not zero-shot)")
    parser.add_argument("--prior-learnings", type=Path,
                        help="optional rollout-context notes injected separately from base system docs")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--execute-paid-rollouts", action="store_true",
                        help="required: acknowledges that this sends paid API requests")
    args = parser.parse_args()
    if not args.execute_paid_rollouts:
        parser.error("paid requests require --execute-paid-rollouts")
    if args.total_budget <= 0 or args.episode_budget <= 0:
        parser.error("budgets must be positive")
    if not 0 <= args.start < args.start + args.episodes <= NUM_INIT_STATES:
        parser.error(f"init states must stay within 0..{NUM_INIT_STATES - 1}")

    settings = load_settings()
    default_libero_python = Path.home() / "miniconda3/envs/libero/bin/python"
    libero_python = settings.get("LIBERO_PYTHON") or (
        str(default_libero_python) if default_libero_python.is_file() else sys.executable
    )
    try:
        tasks = load_suite_tasks(args.suite, libero_python)
    except Exception as exc:
        parser.error(f"unable to load {args.suite} task catalog with {libero_python}: {exc}")
    if not 0 <= args.task_id < len(tasks):
        parser.error(f"--task-id must be in 0..{len(tasks) - 1} for {args.suite}")
    provider = args.model.partition("/")[0]
    key_env = KEY_ENV[provider]
    if args.profile:
        if provider != "openai":
            parser.error("--profile can only be used with openai/* models")
        key_env, profile_base_env = OPENAI_PROFILES[args.profile]
        if args.profile == "deepseek":
            if not settings.get(key_env):
                key_env = "OPENAI_API_KEY"
            if not settings.get(profile_base_env):
                profile_base_env = "OPENAI_BASE_URL"
        if not settings.get(key_env):
            parser.error(f"{key_env} is not set; configure the {args.profile} profile in {ROOT / '.env'}")
        if not settings.get(profile_base_env):
            parser.error(f"{profile_base_env} is not set; configure the {args.profile} profile in {ROOT / '.env'}")
        settings["OPENAI_BASE_URL"] = settings[profile_base_env]
    if not settings.get(key_env):
        parser.error(f"{key_env} is not set; put it in {ROOT / '.env'} (see .env.example)")

    base_url = settings.get("OPENAI_BASE_URL") if provider == "openai" else None
    wire = wire_for(args.model, base_url=base_url)
    effort = args.effort if args.effort is not None else (None if base_url else "low")
    policy_model = args.model.partition("/")[2] if base_url else args.model
    policy_base_url = (
        base_url.rstrip("/") + "/v1"
        if base_url and wire == "responses" and not base_url.rstrip("/").endswith("/v1")
        else base_url
    )

    # A fresh directory per run: never resume a paid run and risk double billing.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.umask(0o077)
    transport = BudgetTransport(
        output / "api", args.total_budget, args.episode_budget, task_id=args.task_id,
        demo_manifest=args.demo_manifest, max_calls=args.max_calls, model=args.model,
        image_horizon=args.image_horizon, max_output_tokens=args.max_output_tokens,
        base_url=base_url,
    )
    full_task = libero_suite_task(args.suite, args.task_id, tasks,
                                  num_scenes=args.episodes, start=args.start,
                                  max_steps=args.max_steps)
    protocol = {
        "model": args.model,
        "suite": args.suite,
        "wire": wire,
        "api_base_url": base_url,
        "task_id": args.task_id,
        "task_name": tasks[args.task_id][0],
        "instruction": tasks[args.task_id][1],
        "episodes": args.episodes,
        "initial_state_ids": [s.metadata["init_state"] for s in full_task.scenes],
        "env_seeds": [s.init_seed for s in full_task.scenes],
        "max_steps": args.max_steps,
        "max_api_attempts_per_episode": args.max_calls,
        "allow_model_give_up": False,
        "require_environment_success_for_done": args.suite.startswith("libero_"),
        "reasoning_effort": effort,
        "image_horizon": args.image_horizon,
        "max_output_tokens": args.max_output_tokens,
        "request_timeout_s": args.request_timeout,
        "total_budget": args.total_budget,
        "episode_budget": args.episode_budget,
        "zero_shot": args.demo_manifest is None and not args.task_notes,
        "task_notes": args.task_notes,
        "demonstrations": json.loads(args.demo_manifest.read_text()) if args.demo_manifest else False,
        "prior_learnings": str(args.prior_learnings.resolve()) if args.prior_learnings else None,
        "upstream_commit": (ROOT / "upstream_commit.txt").read_text().strip(),
        "budget_note": "Conservative local reservations; provider billing remains authoritative.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    results: list[dict[str, object]] = []

    def write_summary() -> None:
        (output / "summary.json").write_text(json.dumps(
            {"protocol": protocol, "results": results,
             "total_estimated_usd": transport.estimated,
             "total_budget_charge_usd": transport.used},
            indent=2))

    try:
        for ep, scene in enumerate(full_task.scenes):
            if transport.uncertain or transport.used >= args.total_budget:
                results.append({"episode": ep, "scene": scene.id,
                                "status": "not_run_budget_or_usage_unknown"})
                continue
            transport.begin_episode(ep)
            print(f"[episode {ep}] {scene.id}: starting simulator "
                  f"(up to {args.max_calls} LLM calls, {args.max_steps} steps)", file=sys.stderr, flush=True)
            folder = output / f"episode_{ep:02d}"
            embodiment = LiberoEmbodiment(task_id=args.task_id, suite=args.suite, output_dir=folder,
                                          libero_python=libero_python,
                                          task_notes=args.task_notes)
            policy_options = {
                "model": policy_model, "base_url": policy_base_url, "api_key_env": key_env,
                "env": {key_env: settings[key_env]}, "wire": wire,
                "max_llm_calls": args.max_calls,
                "image_horizon": args.image_horizon,
                "prior_learnings": str(args.prior_learnings.resolve()) if args.prior_learnings else None,
                "allow_give_up": False,
                "require_success_for_done": args.suite.startswith("libero_"),
                "transport": transport, "wire_capture": True,
            }
            if effort is not None:
                policy_options["effort"] = effort
            policy = agent_policy(
                **policy_options,
            )
            if args.request_timeout:
                # Only the wait changes; the request itself is identical. A response that takes
                # longer than the timeout is otherwise reported as a network error and stops the task.
                policy._client._http.timeout = httpx.Timeout(args.request_timeout, connect=60.0)
            task = Task(name=full_task.name, scenes=[scene], scorer=full_task.scorer,
                        max_steps=args.max_steps)
            # The CLI's default guardrails (bounds clamp + per-step delta limit);
            # the raw eval() API applies none unless asked.
            space = embodiment.info.action_space
            approver = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
            try:
                (log,) = evaluate(task, policy, embodiment, log_dir=str(folder / "inspect"),
                                  seed=42 + ep, store_frames=True, store_actions=True,
                                  grader=None, approver=approver)
                last = embodiment.last or {}
                success = bool(last.get("success"))
                if success:
                    status = "success"
                elif transport.uncertain:
                    status = "api_error_usage_unknown"
                elif transport.blocked or transport.calls >= args.max_calls:
                    status = "incomplete_budget"
                else:
                    status = "error" if log.status == "error" else "failure"
                row = {"episode": ep, "scene": scene.id, "status": status,
                       "environment_success": success, "framework_status": log.status,
                       "metrics": log.results.metrics if log.results else None,
                       "steps": last.get("steps", 0), "api_attempts": transport.calls,
                       "estimated_usd": transport.episode_estimated,
                       "budget_charge_usd": transport.episode_used,
                       "stop_detail": transport.blocked,
                       "video": str(embodiment.episode_dir / "rollout.mp4")
                       if embodiment.episode_dir else None}
            finally:
                embodiment.close()
                policy._client.close()
            results.append(row)
            write_summary()
            print(f"[episode {ep}] finished: {row.get('status')}, {row.get('steps')} steps, "
                  f"{row.get('api_attempts')} LLM calls", file=sys.stderr, flush=True)
            print(json.dumps(row), flush=True)
    finally:
        transport.dispose()
        write_summary()


if __name__ == "__main__":
    main()
