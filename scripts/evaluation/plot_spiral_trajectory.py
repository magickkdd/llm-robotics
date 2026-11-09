"""Plot measured UR20 spiral-search trajectories from a force trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.trace.open(encoding="utf-8")]
    spiral_rows = [row for row in rows if row.get("event") == "spiral_point"]
    if spiral_rows:
        measured = np.asarray([row["actual_tip_world_m"][:2] for row in spiral_rows], dtype=float)
        target = np.asarray([row["planned_tip_world_m"][:2] for row in spiral_rows], dtype=float)
        center = np.asarray(spiral_rows[0]["spiral_center_world_m"][:2], dtype=float)
    else:
        measured = np.asarray([row["tool_tip_xy_private_m"] for row in rows], dtype=float)
        target = np.asarray([row["spiral_target_world_m"][:2] for row in rows], dtype=float)
        center = np.asarray(rows[0]["spiral_center_xy_m"], dtype=float)

    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.plot(target[:, 0], target[:, 1], color="#f28e2b", linestyle="--", label="planned target")
    axis.plot(measured[:, 0], measured[:, 1], color="#1f77b4", linewidth=1.0, label="actual tool-tip")
    axis.scatter(*center, color="black", marker="+", s=100, label="spiral center")
    axis.scatter(*measured[0], color="green", s=55, label="start")
    axis.scatter(*measured[-1], color="red", marker="*", s=100, label="end")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title("Measured UR20 spiral-search trajectory")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
