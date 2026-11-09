"""Focused Responses API policy and tool surface for the UR20 screw task."""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from typing import Any

import httpx
import numpy as np
from PIL import Image

from ur20_screw.contracts import Observation

POLICY_CAMERAS = ("agentview", "wristview")
MAX_LIVE_POLICY_IMAGES = len(POLICY_CAMERAS) * 2


def _image_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object", "properties": properties, "required": required,
            "additionalProperties": False,
        },
        "strict": False,
    }


class UR20Policy:
    def __init__(self, *, model: str, api_key: str, endpoint: str,
                 transport: httpx.BaseTransport, timeout: float, max_calls: int,
                 max_output_tokens: int, prior_learnings: str | None = None,
                 effort: str | None = None):
        self.model = model
        self.endpoint = endpoint
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self.calls = 0
        self.max_response_retries = 2
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, transport=transport,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self.input: list[dict[str, Any]] = []
        self.system = (
            "You control the UR20 battery screw-removal simulation through one tool call per turn. "
            "Use the robot state and labeled RGB views, re-check after each motion, and do not claim "
            "success without physical evidence. The task instruction and robot operating notes follow.\n\n"
        )
        if prior_learnings:
            self.system += "Prior task notes:\n" + prior_learnings + "\n\n"
        self.tools = [
            _tool("move_to", "Move the tool tip to absolute world-frame x, y, z targets in meters. "
                  "Unnamed coordinates stay fixed. Changes are limited by the controller. Include a concise observation-based note.",
                  {"targets": {"type": "object", "properties": {
                      axis: {"type": "number"} for axis in ("x", "y", "z")},
                      "additionalProperties": False},
                   "note": {"type": "string"}}, ["targets", "note"]),
            _tool("done", "End the rollout when the task is complete.",
                  {"summary": {"type": "string"}}, ["summary"]),
        ]

    def enable_spiral_search(self) -> None:
        self.tools.append(_tool(
            "spiral_search", "After detected axial contact, start the blocking force-controlled socket seating search.",
            {"note": {"type": "string"}}, ["note"]))

    def enable_socket_turn(self) -> None:
        self.tools.append(_tool(
            "turn_socket", "After stable tactile seating, run the bounded threaded screw extraction operation.",
            {"note": {"type": "string"}}, ["note"]))

    def observe(self, observation: Observation, robot_docs: str) -> None:
        if not self.input:
            self.system += "Robot operating notes:\n" + robot_docs
            self.input.append({"role": "system", "content": self.system})
        parts: list[dict[str, Any]] = [{"type": "input_text", "text": (
            f"Goal: {observation.instruction}\nCurrent state (JSON): "
            + json.dumps({key: np.asarray(value).tolist() for key, value in observation.state.items()})
        )}]
        for name in POLICY_CAMERAS:
            image = observation.images.get(name)
            if image is None:
                continue
            parts.append({"type": "input_text", "text": f"Camera: {name}"})
            parts.append({"type": "input_image", "image_url": _image_url(image)})
        self.input.append({"role": "user", "content": parts})
        image_messages = [
            item for item in self.input
            if item.get("role") == "user" and isinstance(item.get("content"), list)
            and any(part.get("type") == "input_image" for part in item["content"])
        ]
        live_images = sum(
            part.get("type") == "input_image"
            for item in image_messages
            for part in item["content"]
        )
        for item in image_messages:
            if live_images <= MAX_LIVE_POLICY_IMAGES:
                break
            removed = sum(part.get("type") == "input_image" for part in item["content"])
            item["content"] = [
                part for part in item["content"] if part.get("type") != "input_image"
            ]
            item["content"].append({"type": "input_text", "text": "[Earlier camera images omitted]"})
            live_images -= removed

    def act(self) -> tuple[str, dict[str, Any], str]:
        if self.calls >= self.max_calls:
            raise RuntimeError("LLM call limit reached")
        payload: dict[str, Any] | None = None
        for attempt in range(self.max_response_retries + 1):
            response = self.http.post(
                self.endpoint,
                json={"model": self.model, "input": self.input, "tools": self.tools,
                      "max_output_tokens": self.max_output_tokens, "store": False,
                      "parallel_tool_calls": False,
                      "tool_choice": "required",
                      **({"reasoning": {"effort": self.effort}} if self.effort else {})},
            )
            response.raise_for_status()
            try:
                payload = response.json()
                break
            except ValueError:
                # 网关偶发返回 200 + 空/非 JSON 报文；重发同一请求，预算仍由
                # BudgetTransport 逐次记账。
                if attempt == self.max_response_retries:
                    raise
                time.sleep(2.0 * (attempt + 1))
        assert payload is not None
        output = payload.get("output", [])
        self.input.extend(output)
        calls = [item for item in output if item.get("type") == "function_call"]
        self.calls += 1
        if len(calls) != 1:
            raise RuntimeError(f"expected one tool call, received {len(calls)}")
        call = calls[0]
        arguments = json.loads(call.get("arguments") or "{}")
        return str(call["name"]), arguments, str(call["call_id"])

    def tool_result(self, call_id: str, result: str) -> None:
        self.input.append({"type": "function_call_output", "call_id": call_id, "output": result})

    def close(self) -> None:
        self.http.close()
