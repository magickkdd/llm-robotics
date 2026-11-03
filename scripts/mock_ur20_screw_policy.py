"""Run a local, non-LLM smoke policy against the UR20 scene worker."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("MUJOCO_PYTHON", str(Path.home() / "miniconda3/envs/libero/bin/python"))


def request(process: subprocess.Popen, payload: dict) -> dict:
    assert process.stdin and process.stdout
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def main() -> None:
    process = subprocess.Popen(
        [PYTHON, str(ROOT / "scripts/ur20_screw_worker.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        initial = request(process, {"op": "reset"})
        if "error" in initial:
            raise RuntimeError(initial["error"])
        print("reset:", initial["sites"])
        # Keep the robot at its keyframe and rotate the socket to exercise the
        # screw-spin -> screw-lift equality, without calling an LLM.
        result = request(process, {"op": "step", "ctrl": [*initial["qpos"][:6], 0.0], "count": 200})
        if "error" in result:
            raise RuntimeError(result["error"])
        print("after socket rotation:", result["sites"])
        print("steps:", result["steps"], "time:", result["time"])
    finally:
        if process.stdin:
            process.stdin.write(json.dumps({"op": "close"}) + "\n")
            process.stdin.flush()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
