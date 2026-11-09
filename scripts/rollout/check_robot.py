"""Exercise the standalone UR20 simulator adapter without an LLM."""

from __future__ import annotations

import numpy as np

from ur20_screw.simulation.robot import UR20ScrewEmbodiment
from ur20_screw.contracts import Action, Scene


def main() -> None:
    robot = UR20ScrewEmbodiment()
    try:
        observation = robot.reset(Scene("ur20-smoke", "Seat the socket on the screw."))
        print("reset cameras:", sorted(observation.images), "state:", sorted(observation.state))
        result = robot.step(Action(np.asarray(observation.state["eef_pos"], dtype=float)))
        print("step joint position:", result.observation.state["joint_pos"].tolist())
    finally:
        robot.close()


if __name__ == "__main__":
    main()
