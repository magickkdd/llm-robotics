"""Offline-only accuracy and latency evaluation for the UR20 visual cues."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
from ur20_screw.simulation.worker import Worker
from ur20_screw.vision.estimation import estimate_pair, project_pixel_to_plane


def _project_world_point(worker: Worker, camera_id: int, point: np.ndarray,
                         width: int, height: int) -> np.ndarray | None:
    camera_position = worker.data.cam_xpos[camera_id]
    camera_rotation = worker.data.cam_xmat[camera_id].reshape(3, 3)
    camera_point = camera_rotation.T @ (point - camera_position)
    if camera_point[2] >= 0:
        return None
    focal = height / (2.0 * np.tan(np.deg2rad(worker.model.cam_fovy[camera_id]) / 2.0))
    return np.array([
        width / 2.0 + focal * camera_point[0] / -camera_point[2],
        height / 2.0 - focal * camera_point[1] / -camera_point[2],
    ])


def _decode_images(encoded: dict[str, str]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        for name, data in encoded.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    worker = Worker()
    report: dict[str, object] = {
        "evaluation_only": True,
        "simulator_truth_used_for_scoring_only": True,
        "estimator_inputs": ["agentview_rgb", "wristview_rgb"],
        "resolutions": {},
    }
    try:
        for resolution in args.resolutions:
            renderer = None
            if resolution != 256:
                renderer = mujoco.Renderer(worker.model, height=resolution, width=resolution)
            pair_latencies = []
            detections: dict[str, list[dict[str, object]]] = {
                "agentview": [], "wristview": []
            }
            world_estimates: list[dict[str, object]] = []
            try:
                for state_id in args.states:
                    reset = worker.reset(state_id)
                    if renderer is None:
                        images = _decode_images(reset["images"])
                    else:
                        images = {}
                        for name, camera_id in worker.camera_ids.items():
                            renderer.update_scene(worker.data, camera=camera_id)
                            images[name] = renderer.render().copy()

                    estimate = estimate_pair(images["agentview"], images["wristview"])
                    pair_latencies.append(estimate.processing_ms)
                    head_position = worker.data.geom_xpos[worker.screw_head_geom_id].copy()
                    wrist_id = worker.camera_ids["wristview"]
                    pixel = estimate.wristview_screw.center_px
                    world_estimate = None
                    if pixel is not None:
                        world_estimate = project_pixel_to_plane(
                            pixel,
                            worker.data.cam_xpos[wrist_id],
                            worker.data.cam_xmat[wrist_id].reshape(3, 3),
                            float(worker.model.cam_fovy[wrist_id]),
                            (resolution, resolution),
                        )
                    world_error = None if world_estimate is None else float(
                        np.linalg.norm(world_estimate[:2] - head_position[:2])
                    )
                    world_estimates.append({
                        "state_id": state_id,
                        "valid": world_estimate is not None,
                        "confidence_heuristic": estimate.wristview_screw.confidence,
                        "estimated_xy_m": None if world_estimate is None else world_estimate[:2].tolist(),
                        "offline_error_m": world_error,
                    })
                    for name, key in (("agentview", "agentview_screw"),
                                      ("wristview", "wristview_screw")):
                        detection = getattr(estimate, key)
                        projected = _project_world_point(
                            worker, worker.camera_ids[name], head_position,
                            resolution, resolution,
                        )
                        error_px = None
                        if detection.center_px is not None and projected is not None:
                            error_px = float(np.linalg.norm(
                                np.asarray(detection.center_px) - projected
                            ))
                        detections[name].append({
                            "state_id": state_id,
                            "visible": detection.visible,
                            "confidence_heuristic": detection.confidence,
                            "center_px": detection.center_px,
                            "offline_error_px": error_px,
                        })
            finally:
                if renderer is not None:
                    renderer.close()

            camera_summary = {}
            for name, records in detections.items():
                valid_errors = [record["offline_error_px"] for record in records
                                if record["offline_error_px"] is not None]
                camera_summary[name] = {
                    "detected": len(valid_errors),
                    "total": len(records),
                    "mean_offline_error_px": float(np.mean(valid_errors)) if valid_errors else None,
                    "max_offline_error_px": float(np.max(valid_errors)) if valid_errors else None,
                    "states": records,
                }
            report["resolutions"][str(resolution)] = {
                "mean_processing_ms": float(np.mean(pair_latencies)),
                "p95_processing_ms": float(np.percentile(pair_latencies, 95)),
                "cameras": camera_summary,
                "wristview_world_projection": {
                    "valid": sum(record["valid"] for record in world_estimates),
                    "total": len(world_estimates),
                    "mean_offline_error_m": float(np.mean([
                        record["offline_error_m"] for record in world_estimates
                        if record["offline_error_m"] is not None
                    ])) if any(record["offline_error_m"] is not None for record in world_estimates) else None,
                    "max_offline_error_m": float(np.max([
                        record["offline_error_m"] for record in world_estimates
                        if record["offline_error_m"] is not None
                    ])) if any(record["offline_error_m"] is not None for record in world_estimates) else None,
                    "states": world_estimates,
                },
            }
    finally:
        worker.renderer.close()

    result = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
