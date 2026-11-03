"""CAD(STEP) -> MuJoCo mesh pipeline.

用法（需要 gmsh / trimesh / numpy）：
    python build_assets.py

输出：
    meshes/end_effector_body.stl  末端执行器本体（不含套筒），原点在安装法兰中心，工具朝 +X
    meshes/socket_ns10la.stl      NS.10LA 套筒，原点在套筒中心，轴线沿 +X
    meshes/battery.stl            电池包，原点在底面中心
    meshes/screw_m6.stl           M6 电池螺钉，原点在螺钉尖端、头部朝 +Z
"""

import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import gmsh
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
SRC_ROOT = HERE.parent
MESH_DIR = HERE / "meshes"

# 工具坐标变换：把 moduan 装配的全局坐标 -> 安装法兰中心为原点、工具朝 +X、截面居中
# 安装法兰中心（moduan.STEP 全局坐标，mm）
MOUNT = np.array([84.66, -13.25, 410.75], dtype=float)
T_TOOL = np.array([
    [-1.0, 0.0, 0.0, MOUNT[0]],
    [0.0, -1.0, 0.0, MOUNT[1]],
    [0.0, 0.0, 1.0, -MOUNT[2]],
    [0.0, 0.0, 0.0, 1.0],
])


def gmsh_surface(src: Path, out: Path, max_size: float, min_size: float,
                 keep_length_x: float | None = None) -> trimesh.Trimesh:
    """用 gmsh 的 OCC 接口对 STEP 做表面网格化，返回 trimesh 对象。"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mesh")

    gmsh.model.occ.importShapes(str(src))
    gmsh.model.occ.synchronize()

    if keep_length_x is not None:
        volumes = gmsh.model.getEntities(3)
        keep = None
        for dim, tag in volumes:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            if abs((bbox[3] - bbox[0]) - keep_length_x) < 1.0:
                keep = tag
                break
        if keep is None:
            raise RuntimeError(f"在 {src.name} 中找不到长度 {keep_length_x} mm 的实体")
        gmsh.model.occ.remove([(3, tag) for dim, tag in volumes if tag != keep],
                              recursive=True)
        gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.Binary", 1)

    t0 = time.time()
    gmsh.model.mesh.generate(2)
    gmsh.write(str(out))
    gmsh.finalize()

    mesh = trimesh.load(str(out), force="mesh")
    mesh.merge_vertices()
    mesh.fix_normals()
    print(f"  {src.name}: {len(mesh.faces)} faces, {len(mesh.vertices)} verts, "
          f"{out.stat().st_size / 1024:.0f} KB, {time.time() - t0:.1f}s")
    return mesh


def write_binary_stl(path: Path, mesh: trimesh.Trimesh) -> None:
    """写二进制 STL，避免 trimesh 默认的 ASCII 体积。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.face_normals, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(b"Cad2MuJoCo".ljust(80, b" "))
        fp.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            fp.write(struct.pack("<3f", *normals[i]))
            for k in range(3):
                fp.write(struct.pack("<3f", *vertices[faces[i][k]]))
            fp.write(struct.pack("<H", 0))


def report(name: str, mesh: trimesh.Trimesh) -> None:
    bounds = mesh.bounds
    print(f"  {name}: bounds {np.round(bounds[0], 4)} .. {np.round(bounds[1], 4)}  "
          f"size {np.round(bounds[1] - bounds[0], 4)}  "
          f"faces {len(mesh.faces)}  volume {mesh.volume:.6f} m^3")


def main() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cad2mj_"))

    names = ["moduan.STEP", "moduan1.STEP", "battery.STEP", "M6batteryscrew.STEP"]
    for name in names:
        src = SRC_ROOT / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, tmp / name)

    print("== end effector ==")
    tool_base = gmsh_surface(tmp / "moduan.STEP", tmp / "tool_base.stl", 8, 1)
    socket = gmsh_surface(tmp / "moduan1.STEP", tmp / "socket.stl", 1.5, 0.2,
                          keep_length_x=78)

    for mesh in (tool_base, socket):
        mesh.apply_transform(T_TOOL)
        mesh.apply_scale(0.001)

    # 套筒单独作为可旋转子体：原点移到套筒中心（工具坐标系 x=0.4589 m）
    socket.apply_translation([-0.4589, 0, 0])

    report("end_effector_body", tool_base)
    write_binary_stl(MESH_DIR / "end_effector_body.stl", tool_base)
    report("socket_ns10la", socket)
    write_binary_stl(MESH_DIR / "socket_ns10la.stl", socket)

    print("== battery ==")
    battery = gmsh_surface(tmp / "battery.STEP", tmp / "battery.stl", 50, 10)
    battery.apply_scale(0.001)
    report("battery", battery)
    write_binary_stl(MESH_DIR / "battery.stl", battery)

    print("== M6 screw ==")
    screw = gmsh_surface(tmp / "M6batteryscrew.STEP", tmp / "screw_m6.stl", 0.5, 0.05)
    screw.apply_translation([0, 0, 21.0])  # STEP 头部在 z=0，平移到 z=21；尖端在 z=0
    screw.apply_scale(0.001)
    report("screw_m6", screw)
    write_binary_stl(MESH_DIR / "screw_m6.stl", screw)

    print(f"\n完成，网格输出到 {MESH_DIR}")


if __name__ == "__main__":
    main()
