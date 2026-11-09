"""Assemble the supplied screwdriver workcell with the UR20 MJCF model.

The target screw names come from ``src/ur20_screw/target.py``; its hole
position and the camera poses are declared here.  Re-running this script
regenerates ``assets/ur20_screw_station/scene.xml`` in place.
"""

from __future__ import annotations

import copy
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ur20_screw.target import SCREW_TARGET  # noqa: E402

STATION = ROOT / "assets/ur20_screw_station"
UR20 = STATION / "ur20"
WORKCELL = STATION / "workcell"
SCENE = STATION / "scene.xml"
HOME_QPOS = "-2.31981851 -1.01306484 1.24914618 -0.26259425 -3.88367106 -1.57382793"
# 电池裙边法兰（孔口平面 z=0.058）前侧 09 号孔位，电池局部坐标。
BATTERY_WORLD_POS = np.array([0.5, 1.0, 0.0])
TARGET_SCREW_POS_VEC = np.array([0.2882, -0.4668, 0.043])
TARGET_SCREW_POS = " ".join(f"{value:.4f}" for value in TARGET_SCREW_POS_VEC)
AGENTVIEW_POS = np.array([1.02, 0.28, 0.40])
# M6 粗牙螺距 1 mm => 每弧度进给 1.59155e-4 m。
THREAD_LIFT_PER_RADIAN = "1.59155e-4"
EQUALITY_SOLREF = "0.002 1"
EQUALITY_SOLIMP = "0.9 0.95 0.001 0.5 2"
# worker 直接向关节写力矩，因此把上游的位置伺服 general 换成 motor。
# 腕部相机刚性固定在末端工具上，使套筒在图像中的位置恒定；不再使用
# targetbody 跟踪螺钉。四元数取“工具在螺钉正上方 8 cm”时旧跟踪相机的
# 工具系姿态：螺钉在画面中心、套筒固定出现在左上，视线不被工具本体遮挡。
WRISTVIEW_QUAT = "0.70407970 -0.06535961 -0.58632529 0.39524999"
MOTOR_LIMITS = {
    "shoulder_pan": "738 738",
    "shoulder_lift": "738 738",
    "elbow": "433 433",
    "wrist_1": "107 107",
    "wrist_2": "107 107",
    "wrist_3": "107 107",
}


def _required(parent: ET.Element, path: str) -> ET.Element:
    found = parent.find(path)
    if found is None:
        raise ValueError(f"Missing XML element {path}")
    return found


def _element(tag: str, **attributes: str) -> ET.Element:
    return ET.Element(tag, attributes)


def _comment(parent: ET.Element, text: str) -> None:
    parent.append(ET.Comment(text))


def _joint_equality(name: str, joint1: str, polycoef: str,
                    joint2: str | None = None, active: bool = True) -> ET.Element:
    attributes = {"name": name, "joint1": joint1}
    if joint2 is not None:
        attributes["joint2"] = joint2
    attributes["polycoef"] = polycoef
    if not active:
        attributes["active"] = "false"
    attributes["solref"] = EQUALITY_SOLREF
    attributes["solimp"] = EQUALITY_SOLIMP
    return _element("joint", **attributes)


def _camera(name: str, pos: str, fovy: str, target: str) -> ET.Element:
    return _element("camera", name=name, pos=pos, mode="targetbody",
                    target=target, fovy=fovy)


def _look_at_quat(position: np.ndarray, target: np.ndarray,
                  up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> str:
    """Return a MuJoCo quaternion (w x y z) for a camera at position aimed at target."""
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    z_axis = position - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.asarray(up, dtype=float), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
    quat /= np.linalg.norm(quat)
    return " ".join(f"{value:.8f}" for value in quat)


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
    robot_world.append(_element(
        "geom", name="floor", type="plane", size="3 3 0.1",
        material="floor", contype="2", conaffinity="3",
    ))
    robot = _required(robot_world, "body[@name='base']")
    robot.set("pos", "0 0 0.03")

    battery = _element("body", name="battery", pos="0.5 1.0 0")
    _comment(battery, "电池：裙边法兰孔口平面 z=0.058，本体顶面 z=0.148。")
    battery.append(_element("include", file="workcell/parts/battery_body.xml"))
    battery.append(_element("include", file="workcell/parts/installed_bolts.xml"))
    _comment(battery,
             "当前待旋出的螺钉：装在电池裙边法兰（z=0.058）前侧的 M6 通孔上。\n"
             "      目标名称由 src/ur20_screw/target.py 统一声明；孔位与本文件同步修改。")
    screw = _element("body", name=SCREW_TARGET.body, pos=TARGET_SCREW_POS)
    screw.append(_element(
        "joint", name=SCREW_TARGET.spin_joint, type="hinge", axis="0 0 1",
        damping="0.005", armature="0.01",
    ))
    screw.append(_element(
        "joint", name=SCREW_TARGET.lift_joint, type="slide", axis="0 0 1",
        damping="0.005", armature="0.05",
    ))
    screw.append(_element("include", file="workcell/parts/screw_m6_body.xml"))
    battery.append(screw)
    robot_world.append(battery)
    robot_world.append(_element(
        "geom", name="workbench", type="box", pos="0.5 1.0 -0.04",
        size="0.8 0.7 0.04", material="floor", contype="2", conaffinity="3",
    ))
    robot_world.append(_element(
        "camera", name="agentview", pos="1.02 0.28 0.40",
        quat=_look_at_quat(AGENTVIEW_POS,
                           BATTERY_WORLD_POS + TARGET_SCREW_POS_VEC),
        fovy="45",
    ))
    for camera in (
        _camera("screw_front", "0.55 0.30 0.22", "35", SCREW_TARGET.body),
        _camera("screw_side", "1.30 0.35 0.22", "35", SCREW_TARGET.body),
        _camera("screw_back", "0.80 0.15 0.55", "35", SCREW_TARGET.body),
    ):
        robot_world.append(camera)

    wrist = _required(
        robot_root,
        "worldbody/body[@name='base']/body/body/body/body/body/body[@name='wrist_3_link']",
    )
    tool = copy.deepcopy(_required(source_root, "worldbody/body[@name='end_effector']"))
    tool.set("pos", "0 0.1543 0")
    tool.set("quat", "0.70710678 0 0 0.70710678")
    for joint in tool.findall("joint"):
        tool.remove(joint)
    for include in tool.iter("include"):
        include.set("file", f"workcell/{include.get('file')}")
    tool.append(_element(
        "camera", name="wristview", pos="0.405 -0.055 0.045",
        quat=WRISTVIEW_QUAT, fovy="52",
    ))
    wrist.append(tool)

    equality = _element("equality")
    _comment(equality,
             "静置时螺钉被完全锁定；套筒转动阶段改用 socket 耦合 + 螺距耦合。\n"
             "      名字与 src/ur20_screw/target.py 的 ScrewTarget 保持一致。")
    equality.append(_joint_equality(
        SCREW_TARGET.rotation_lock, SCREW_TARGET.spin_joint, "0 0 0 0 0",
    ))
    equality.append(_joint_equality(
        SCREW_TARGET.socket_screw_coupling, SCREW_TARGET.spin_joint, "0 -1 0 0 0",
        joint2="socket_joint", active=False,
    ))
    equality.append(_joint_equality(
        SCREW_TARGET.height_lock, SCREW_TARGET.lift_joint, "0 0 0 0 0",
    ))
    equality.append(_joint_equality(
        SCREW_TARGET.thread_pitch_coupling, SCREW_TARGET.lift_joint,
        f"0 {THREAD_LIFT_PER_RADIAN} 0 0 0", joint2=SCREW_TARGET.spin_joint,
        active=False,
    ))
    robot_root.append(equality)

    contact = _element("contact")
    _comment(contact, "螺钉杆埋在裙边孔内，轴向往复由约束驱动，不需要与电池本体做接触求解。")
    contact.append(_element("exclude", body1="battery", body2=SCREW_TARGET.body))
    robot_root.append(contact)

    sensor = _element("sensor")
    sensor.append(_element("force", name="tool_ft_force", site="ft_sensor_site"))
    sensor.append(_element("torque", name="tool_ft_torque", site="ft_sensor_site"))
    robot_root.append(sensor)

    actuator = _required(robot_root, "actuator")
    for node in list(actuator):
        name = node.get("name")
        limits = MOTOR_LIMITS.get(name)
        if node.tag != "general" or limits is None:
            continue
        node.tag = "motor"
        node.attrib.clear()
        node.set("name", name)
        node.set("joint", f"{name}_joint")
        node.set("ctrlrange", f"-{limits}")
        node.set("forcerange", f"-{limits}")
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
    home.set("ctrl", "0 0 0 0 0 0 0")

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
