"""Exercise the UR20 policy tool contract with a mocked Responses API."""

from __future__ import annotations

import json
import tempfile

import httpx
import numpy as np

from ur20_screw.policy.budget import BudgetTransport
from ur20_screw.policy.client import UR20Policy
from ur20_screw.control.actions import action_for
from ur20_screw.contracts import Observation


def main() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-response",
            "status": "completed",
            "output": [{
                "type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "move_to",
                "arguments": json.dumps({"targets": {"x": 0.25, "z": 0.4}, "note": "Mock move."}),
            }],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        })

    with tempfile.TemporaryDirectory() as output:
        budget = BudgetTransport(
            output, 15, 3, inner=httpx.MockTransport(respond), model="openai/gpt-6-luna",
            max_calls=1, base_url="https://spacetimeai.cc",
            endpoint_base_url="https://spacetimeai.cc/v1",
        )
        budget.begin_episode(0)
        policy = UR20Policy(
            model="gpt-6-luna", api_key="offline",
            endpoint="https://spacetimeai.cc/v1/responses", transport=budget,
            timeout=10, max_calls=1, max_output_tokens=1024,
        )
        observation = Observation(
            images={"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
            state={"eef_pos": np.array([0.0, 0.0, 0.5])},
            instruction="Seat the socket.", state_time=0.0,
        )
        try:
            policy.observe(observation, "Use absolute world-frame Cartesian targets.")
            name, arguments, call_id = policy.act()
            action = action_for(name, arguments, observation)
            assert name == "move_to"
            np.testing.assert_allclose(action.data, [0.25, 0.0, 0.4])
            assert call_id == "call_1"
            assert len(requests) == 1
            print("UR20 policy contract passed")
        finally:
            policy.close()
            budget.dispose()


if __name__ == "__main__":
    main()
