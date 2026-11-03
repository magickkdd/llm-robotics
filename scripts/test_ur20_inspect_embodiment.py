"""Dry-run the UR20 embodiment through the Inspect Robots contract."""

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_libero.ur20_embodiment import UR20ScrewEmbodiment
import numpy as np


def main() -> None:
    embodiment = UR20ScrewEmbodiment()
    try:
        observation = embodiment.reset(Scene("ur20-screw", "Remove the M6 screw from the battery."))
        print("reset cameras:", sorted(observation.images), "state:", sorted(observation.state))
        action = np.array([*observation.state["joint_pos"][:6], 0.0])
        result = embodiment.step(Action(action))
        print("step:", result.observation.state["sites"].reshape(3, 3).tolist())
    finally:
        embodiment.close()


if __name__ == "__main__":
    main()
