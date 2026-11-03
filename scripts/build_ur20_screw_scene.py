"""Assemble the supplied screwdriver workcell with the UR20 MJCF model."""

from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATION = ROOT / "assets/ur20_screw_station"
UR20 = STATION / "ur20"
WORKCELL = STATION / "workcell"
SCENE = STATION / "scene.xml"
HOME_QPOS = "-2.31981851 -1.01306484 1.24914618 -0.26259425 -3.88367106 -1.57382793"


def _required(parent: ET.Element, path: str) -> ET.Element:
    found = parent.find(path)
    if found is None:
        raise ValueError(f"Missing XML element {path}")
    return found


def _copy_scene_assets() -> None:
    shared = STATION / "scene_assets"
    shared.mkdir(parents=True, exist_ok=True)
    for path in (UR20 / "assets").iterdir():
        if path.is_file():
            shutil.copy2(path, shared / path.name)
    for path in (WORKCELL / "meshes").iterdir():
        if path.is_file() and path.suffix.lower() == ".stl":
            shutil.copy2(path, shared / path.name)


def main() -> None:
    _copy_scene_assets()
    robot_root = ET.parse(UR20 / "ur20.xml").getroot()
    compiler = _required(robot_root, "compiler")
    compiler.set("meshdir", "scene_assets")
    compiler.set("texturedir", "scene_assets")

    robot_assets = _required(robot_root, "asset")
    for node in robot_assets:
        file_name = node.get("file")
        if file_name:
            node.set("file", file_name)

    source_root = ET.parse(WORKCELL / "scene_screw_out.xml").getroot()
    source_assets = _required(source_root, "asset")
    for node in source_assets:
        if node.tag not in ("mesh", "material"):
            continue
        clone = copy.deepcopy(node)
        if clone.get("file"):
            clone.set("file", Path(clone.get("file")).name)
        robot_assets.append(clone)

    robot_world = _required(robot_root, "worldbody")
    robot_world.append(ET.Element("geom", {
        "name": "floor", "type": "plane", "size": "3 3 0.1",
        "material": "floor", "contype": "2", "conaffinity": "3",
    }))
    robot = _required(robot_world, "body[@name='base']")
    robot.set("pos", "0 0 0.03")
    battery = copy.deepcopy(_required(source_root, "worldbody/body[@name='battery']"))
    battery.set("pos", "0.5 1.0 0")
    screw = copy.deepcopy(_required(source_root, "worldbody/body[@name='screw_m6']"))
    screw.set("pos", "0.8 1.0 0.132")
    for node in (battery, screw):
        for include in node.iter("include"):
            include.set("file", f"workcell/{include.get('file')}")
    robot_world.extend((battery, screw))
    robot_world.append(ET.Element("geom", {
        "name": "workbench", "type": "box", "pos": "0.5 1.0 -0.04",
        "size": "0.8 0.7 0.04", "material": "floor",
        "contype": "2", "conaffinity": "3",
    }))

    wrist = _required(robot_root, "worldbody/body[@name='base']/body/body/body/body/body/body[@name='wrist_3_link']")
    tool = copy.deepcopy(_required(source_root, "worldbody/body[@name='end_effector']"))
    tool.set("pos", "0 0.1543 0")
    tool.set("quat", "0.70710678 0 0 0.70710678")
    for joint in tool.findall("joint"):
        tool.remove(joint)
    for include in tool.iter("include"):
        include.set("file", f"workcell/{include.get('file')}")
    tool.append(ET.Element("camera", {
        "name": "wristview", "pos": "0.405 -0.055 0.045",
        "mode": "targetbody", "target": "screw_m6", "fovy": "52",
    }))
    wrist.append(tool)

    robot_world.extend((
        ET.Element("camera", {
            "name": "agentview", "pos": "0.75 -0.50 1.5",
            "mode": "targetbody", "target": "screw_m6", "fovy": "48",
        }),
    ))

    equality = _required(source_root, "equality")
    robot_root.append(copy.deepcopy(equality))
    contact = source_root.find("contact")
    if contact is not None:
        robot_root.append(copy.deepcopy(contact))
    actuator = _required(robot_root, "actuator")
    socket_actuator = _required(source_root, "actuator/velocity[@name='socket_velocity']")
    actuator.append(copy.deepcopy(socket_actuator))

    option = robot_root.find("option")
    if option is None:
        option = ET.SubElement(robot_root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    keyframe = _required(robot_root, "keyframe")
    home = _required(keyframe, "key[@name='home']")
    home.set("qpos", f"{HOME_QPOS} 0 0 0")
    home.set("ctrl", f"{HOME_QPOS} 0")

    visual = robot_root.find("visual")
    if visual is None:
        visual = ET.SubElement(robot_root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(robot_root, "statistic", {"center": "0.5 0.8 0.8", "extent": "2.0"})
    ET.indent(robot_root, space="  ")
    SCENE.write_text(ET.tostring(robot_root, encoding="unicode") + "\n", encoding="utf-8")
    print(f"Built {SCENE}")


if __name__ == "__main__":
    main()
