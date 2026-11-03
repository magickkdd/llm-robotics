"""Generate the binary STL used for the M6 external-hex collision geometry."""

from __future__ import annotations

import math
import struct
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "assets/ur20_screw_station/workcell/meshes/screw_hex_head.stl"


def main() -> None:
    lower = [(0.005773503 * math.cos(i * math.pi / 3),
              0.005773503 * math.sin(i * math.pi / 3), 0.017) for i in range(6)]
    upper = [(x, y, 0.021) for x, y, _ in lower]
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    for index in range(1, 5):
        triangles.extend(((lower[0], lower[index + 1], lower[index]),
                          (upper[0], upper[index], upper[index + 1])))
    for index in range(6):
        following = (index + 1) % 6
        triangles.extend(((lower[index], lower[following], upper[following]),
                          (lower[index], upper[following], upper[index])))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as stream:
        stream.write(b"M6 10mm external hex head collision mesh".ljust(80, b" "))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            stream.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for vertex in triangle:
                stream.write(struct.pack("<3f", *vertex))
            stream.write(struct.pack("<H", 0))


if __name__ == "__main__":
    main()
