"""声明当前要旋出的螺钉，作为场景、worker、相机和视觉标定的唯一来源。

迁移到另一颗螺钉时只需两处同步修改：
1. ``scripts/scene/build_screw_scene.py`` 的目标孔位 ``TARGET_SCREW_POS``，
   以及 ``workcell/parts/installed_bolts.xml`` 里对应孔位的装饰螺钉，然后重新生成
   ``assets/ur20_screw_station/scene.xml``；
2. 本文件的 ``SCREW_TARGET``（名称、头顶高度、agentview ROI）。生成器会根据
   孔位自动重新对准固定的 agentview；若孔位变化较大，需重新标定该 ROI。
   wristview 刚性固定在工具上，不随孔位改变。

执行任务使用的观测不依赖螺钉的仿真真值，因此替换目标不会改变控制/视觉算法。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScrewTarget:
    """Name and calibration data of the screw the station currently works on."""

    body: str
    spin_joint: str
    lift_joint: str
    head_geom: str
    rotation_lock: str
    socket_screw_coupling: str
    height_lock: str
    thread_pitch_coupling: str
    head_top_z_m: float
    agentview_roi_256: tuple[int, int, int, int]


# 电池裙边法兰（电池局部 z=0.058）前侧 09 号孔位：局部 (0.2882, -0.4668)。
# 螺钉局部尖端 z=0、头顶 z=0.021，安装后 body 原点世界 z=0.043，头顶 z=0.064。
SCREW_TARGET = ScrewTarget(
    body="screw_m6",
    spin_joint="screw_spin",
    lift_joint="screw_lift",
    head_geom="screw_head",
    rotation_lock="screw_rotation_lock",
    socket_screw_coupling="socket_screw_rotation_coupling",
    height_lock="screw_height_lock",
    thread_pitch_coupling="screw_thread_pitch_coupling",
    head_top_z_m=0.064,
    agentview_roi_256=(112, 104, 32, 32),
)
