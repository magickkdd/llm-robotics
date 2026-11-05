"""Offline agent_policy tool-call check for the UR20 screw embodiment."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
from inspect_robots import eval as evaluate
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots.types import Action
from inspect_robots_agent.policy import agent_policy

from inspect_robots_libero.budget import BudgetTransport
from inspect_robots_libero.ur20_embodiment import UR20ScrewEmbodiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ur20_screw_agent_policy_check"
calls = []


def fake(request: httpx.Request) -> httpx.Response:
    calls.append(json.loads(request.content))
    index = len(calls)
    if index == 1:
        name = "move_to"
        arguments = {"targets": {"x": 0.25, "y": 0.0, "z": 0.4},
                     "note": "Offline mock: move the Cartesian tool target."}
    else:
        name = "give_up"
        arguments = {"reason": "Offline tool-call check complete", "hindsight": "worker and action schema verified"}
    return httpx.Response(200, headers={"x-request-id": f"offline-{index}"}, json={
        "id": f"offline-{index}", "status": "completed",
        "output": [{"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                    "name": name, "arguments": json.dumps(arguments)}],
        "usage": {"input_tokens": 100, "output_tokens": 100},
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    endpoint = "https://spacetimeai.cc"
    transport = BudgetTransport(OUT / "api", 15, 3, inner=httpx.MockTransport(fake),
                                model="openai/gpt-6-luna", max_calls=3,
                                base_url=endpoint, endpoint_base_url=endpoint)
    transport.begin_episode(0)
    policy = agent_policy(model="gpt-6-luna", base_url=endpoint,
                          env={"OPENAI_API_KEY": "offline"},
                          wire="responses",
                          max_llm_calls=3, transport=transport, wire_capture=False)
    embodiment = UR20ScrewEmbodiment(output_dir=OUT)
    task = Task(name="ur20_battery_disassembly", scenes=[Scene(
        "ur20-screw-init00", "Remove the M6 screw from the battery.")],
        scorer="success_at_end", max_steps=20)
    try:
        log, = evaluate(task, policy, embodiment, log_dir=str(OUT / "inspect"),
                        store_frames=True, store_actions=True,
                        approver=ChainApprover(ClampApprover(embodiment.info.action_space),
                                               DeltaLimitApprover(embodiment.info.action_space)))
        print(json.dumps({"framework_status": log.status, "mocked_requests": len(calls),
                          "steps": embodiment._rpc({"op": "observe"})["steps"]}))
    finally:
        embodiment.close()
        policy._client.close()
        transport.dispose()


if __name__ == "__main__":
    main()
