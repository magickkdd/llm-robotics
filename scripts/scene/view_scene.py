"""Open the UR20 screw-disassembly MuJoCo scene in the interactive viewer."""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets/ur20_screw_station/scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
