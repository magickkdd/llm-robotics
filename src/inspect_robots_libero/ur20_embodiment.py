"""Inspect Robots embodiment for the UR20 battery-disassembly scene."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "ur20_screw_worker.py"
PYTHON = os.environ.get("MUJOCO_PYTHON", sys.executable)


class UR20ScrewEmbodiment(EmbodimentBase):
    def __init__(self, output_dir: str | os.PathLike[str] = "outputs/ur20_screw", state_id: int = 0,
                 max_steps: int = 600) -> None:
        self.output = Path(output_dir)
        self.state_id = state_id
        self.max_steps = max_steps
        self.environment_steps = 0
        self.physics_steps = 0
        self._last_socket_engaged = False
        self._last_engaged_seen = False
        self.process: subprocess.Popen[str] | None = None
        self.instruction = ""
        self.task_success = False
        self.max_screw_lift = 0.0
        self.engaged_seen = False
        self.info = EmbodimentInfo(
            name="ur20_screw_station",
            is_simulated=True,
            control_hz=20.0,
            capabilities=frozenset({"resettable", "renderable"}),
            action_space=Box(
                shape=(7,), low=np.array([-6.3] * 6 + [-60.0]), high=np.array([6.3] * 6 + [60.0]),
                semantics=ActionSemantics(control_mode="joint_pos", frame="joint",
                                          dim_labels=("shoulder_pan", "shoulder_lift", "elbow",
                                                      "wrist_1", "wrist_2", "wrist_3",
                                                      "socket_velocity")),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 256, 256) for name in ("agentview", "wristview")),
                state=StateSpec((StateField("joint_pos", (7,)), StateField("qpos", (9,)),
                                 StateField("qvel", (9,)), StateField("sites", (9,)),
                                 StateField("task_state", (3,)))),
            ),
        docs=("UR20 with a rigidly wrist-mounted screwdriver; observations include "
              "agentview and wristview. Continue safe attempts until physical success or "
              "the configured call or environment-step limit; do not stop early because "
              "progress is uncertain. Inspect both views and task_state after every move; "
              "task_state is [screw_lift_m, socket_engaged_now, socket_ever_engaged]. "
              "If the screw or socket is occluded, make a small safe retreat or reposition "
              "to restore visibility before approaching again. Search laterally with small, "
              "controlled joint changes, one direction at a time; after two adjustments "
              "without improved alignment, change direction or return to a clear viewing "
              "pose instead of repeating the same motion. Align laterally before descending, "
              "and avoid pushing the tool into the battery. Rotate only after task_state "
              "confirms current physical socket engagement; use the next observation to "
              "verify that extraction is progressing."),
        )

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise EmbodimentFault("UR20 worker is not running")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise EmbodimentFault("UR20 worker exited")
        result = json.loads(line)
        if "error" in result:
            raise EmbodimentFault(result["error"])
        return result

    @staticmethod
    def _observation(data: dict[str, Any], instruction: str) -> Observation:
        images = {name: np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"), dtype=np.uint8) for name, value in data["images"].items()}
        sites = np.asarray([coordinate for name in ("attachment_site", "tool_tip", "socket_center") for coordinate in data["sites"][name]], dtype=float)
        qpos = np.asarray(data["qpos"], dtype=float)
        joint_pos = np.asarray([*qpos[:6], qpos[6]], dtype=float)
        task_state = np.asarray([data["screw_lift"], float(data["socket_engaged"]),
                                 float(data["engaged_seen"])], dtype=float)
        return Observation(images=images, state={"joint_pos": joint_pos, "qpos": qpos, "qvel": np.asarray(data["qvel"], dtype=float), "sites": sites, "task_state": task_state}, instruction=instruction, state_time=float(data["time"]))

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.output.mkdir(parents=True, exist_ok=True)
        if self.process is None:
            env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
            env.update(MUJOCO_GL=env.get("MUJOCO_GL", "egl"))
            self.process = subprocess.Popen([PYTHON, "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        self.instruction = scene.instruction
        self.task_success = False
        self.max_screw_lift = 0.0
        self.engaged_seen = False
        self.environment_steps = 0
        data = self._rpc({"op": "reset", "state_id": self.state_id})
        self.physics_steps = int(data["steps"])
        self._last_socket_engaged = bool(data["socket_engaged"])
        self._last_engaged_seen = bool(data["engaged_seen"])
        return self._observation(data, self.instruction)

    def step(self, action: Action) -> StepResult:
        data = self._rpc({"op": "step", "ctrl": np.asarray(action.data, dtype=float).tolist(), "count": 25})
        self.environment_steps += 1
        self.physics_steps = int(data["steps"])
        self.max_screw_lift = max(self.max_screw_lift, float(data["screw_lift"]))
        self.engaged_seen = self.engaged_seen or bool(data["engaged_seen"])
        success = self.engaged_seen and data["screw_lift"] >= 0.020 and data["screw_rotation"] >= 120.0
        self.task_success = self.task_success or success
        socket_engaged = bool(data["socket_engaged"])
        engaged_seen = bool(data["engaged_seen"])
        if (self.environment_steps == 1 or self.environment_steps % 10 == 0 or success
                or socket_engaged != self._last_socket_engaged
                or engaged_seen != self._last_engaged_seen):
            print(
                f"[UR20 step {self.environment_steps}/{self.max_steps} | physics {self.physics_steps}] "
                f"socket_engaged={socket_engaged} ever_engaged={engaged_seen} "
                f"screw_lift={data['screw_lift']:.4f}m "
                f"rotation={data['screw_rotation']:.1f}deg "
                f"socket_error(axial={data['socket_axial_error']:.4f}m, "
                f"radial={data['socket_radial_error']:.4f}m) success={success}",
                file=sys.stderr, flush=True)
        self._last_socket_engaged = socket_engaged
        self._last_engaged_seen = engaged_seen
        return StepResult(self._observation(data, self.instruction), reward=1.0 if success else 0.0,
                          terminated=success,
                          termination_reason="success" if success else None,
                          info={"steps": data["steps"], "screw_lift": data["screw_lift"],
                                "socket_engaged": data["socket_engaged"],
                                "socket_contact": data["socket_contact"],
                                "socket_axial_error": data["socket_axial_error"],
                                "socket_radial_error": data["socket_radial_error"],
                                "tool_axis_alignment": data["tool_axis_alignment"],
                                "screw_rotation": data["screw_rotation"],
                                "engaged_seen": data["engaged_seen"], "success": success})

    def close(self) -> None:
        if self.process is not None:
            try:
                self._rpc({"op": "close"})
            except EmbodimentFault:
                pass
            finally:
                self.process.wait(timeout=10)
                self.process = None
