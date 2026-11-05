"""Capture policy-equivalent UR20 camera frames for offline vision checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ur20_screw_worker import Worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 64 <= args.resolution <= 720:
        parser.error("--resolution must be between 64 and 720 (MuJoCo framebuffer limit)")

    output = args.output or ROOT / "outputs" / (
        f"ur20_vision_pair_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    worker = Worker()
    renderer = None
    try:
        worker.reset(args.state_id)
        renderer = mujoco.Renderer(
            worker.model, height=args.resolution, width=args.resolution
        )
        for name, camera_id in worker.camera_ids.items():
            renderer.update_scene(worker.data, camera=camera_id)
            Image.fromarray(renderer.render()).save(output / f"{name}.png")
        manifest = {
            "state_id": args.state_id,
            "resolution": [args.resolution, args.resolution],
            "cameras": list(worker.camera_ids),
            "images": {name: f"{name}.png" for name in worker.camera_ids},
            "contains_simulator_coordinates": False,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(output)
    finally:
        if renderer is not None:
            renderer.close()
        worker.renderer.close()


if __name__ == "__main__":
    main()
