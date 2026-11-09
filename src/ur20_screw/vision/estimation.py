"""Fast, training-free visual cues for the UR20 screw-seating scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from ur20_screw.target import SCREW_TARGET

TOOL_TIP_UV_256 = (128.0, 87.7)
AGENTVIEW_SCREW_ROI_256 = SCREW_TARGET.agentview_roi_256
# wristview 刚性固定在工具上，套筒固定在画面左上、螺钉从画面中心接近。
# 中心 ROI 覆盖目标螺钉，同时避开套筒本体与裙边上的相邻螺钉。
WRISTVIEW_SCREW_ROI_256 = (96, 96, 64, 64)
SCREW_HEAD_TOP_Z_M = SCREW_TARGET.head_top_z_m


@dataclass(frozen=True)
class ScrewDetection:
    center_px: tuple[float, float] | None
    bbox_xywh: tuple[int, int, int, int] | None
    visible: bool
    confidence: float
    candidate_count: int


@dataclass(frozen=True)
class VisionEstimate:
    resolution: tuple[int, int]
    agentview_screw: ScrewDetection
    wristview_screw: ScrewDetection
    wristview_tool_tip_px: tuple[float, float]
    wristview_tool_tip_source: str
    wristview_tool_tip_visible: bool
    wristview_tool_tip_confidence: float
    wristview_screw_minus_tip_px: tuple[float, float] | None
    processing_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def detect_screw_head(
    rgb: np.ndarray, roi_xywh: tuple[int, int, int, int] | None = None
) -> ScrewDetection:
    """Find a compact, low-saturation metal component without scene-state inputs."""
    hsv = _rgb_to_hsv(rgb)
    full_height, full_width = hsv.shape[:2]
    offset_x = offset_y = 0
    if roi_xywh is not None:
        scale_x, scale_y = full_width / 256.0, full_height / 256.0
        roi_x, roi_y, roi_width, roi_height = roi_xywh
        offset_x, offset_y = int(round(roi_x * scale_x)), int(round(roi_y * scale_y))
        width, height = int(round(roi_width * scale_x)), int(round(roi_height * scale_y))
        hsv = hsv[offset_y:offset_y + height, offset_x:offset_x + width]
        if hsv.size == 0:
            return ScrewDetection(None, None, False, 0.0, 0)
    height, width = hsv.shape[:2]
    image_area = height * width
    mask = cv2.inRange(hsv, np.array([0, 0, 55 if roi_xywh is not None else 75], dtype=np.uint8),
                       np.array([179, 105, 255], dtype=np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(8, int(image_area * 0.00025))
    max_area = int(image_area * (0.12 if roi_xywh is not None else 0.015))
    candidates: list[tuple[float, int, tuple[int, int, int, int], tuple[float, float]]] = []

    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if area < min_area or area > max_area:
            continue
        if box_width < 3 or box_height < 3:
            continue
        if box_width > full_width * 0.14 or box_height > full_height * 0.14:
            continue
        if roi_xywh is not None and (
            x <= 0 or y <= 0 or x + box_width >= width or y + box_height >= height
        ):
            # ROI 边界裁断的连通域质心会落在裁切角上（例如被切掉的电池法兰），
            # 不是完整螺钉，必须丢弃。
            continue
        aspect = box_width / box_height
        if not 0.3 <= aspect <= 3.0:
            continue

        component = np.uint8(labels[y:y + box_height, x:x + box_width] == index) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        contour_area = cv2.contourArea(contour)
        if perimeter <= 0 or contour_area <= 0:
            continue
        compactness = min(1.0, 4.0 * np.pi * contour_area / (perimeter * perimeter))
        square_like = float(np.exp(-abs(np.log(aspect))))
        area_score = min(1.0, area / max(1.0, image_area * 0.001))
        full_x, full_y = x + offset_x, y + offset_y
        edge_penalty = 0.65 if (
            full_x == 0 or full_y == 0 or full_x + box_width >= full_width
            or full_y + box_height >= full_height
        ) else 1.0
        confidence = float(np.clip((0.48 * square_like + 0.27 * compactness +
                                    0.25 * area_score) * edge_penalty, 0.0, 1.0))
        candidates.append((confidence, area,
                           (x + offset_x, y + offset_y, box_width, box_height),
                           (float(centers[index][0] + offset_x),
                            float(centers[index][1] + offset_y))))

    if not candidates:
        return ScrewDetection(None, None, False, 0.0, 0)

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    confidence, _, bbox, center = candidates[0]
    visible = confidence >= 0.35
    return ScrewDetection(center if visible else None, bbox if visible else None,
                          visible, confidence, len(candidates))


def _tool_tip_visibility(rgb: np.ndarray) -> tuple[tuple[float, float], bool, float]:
    height, width = rgb.shape[:2]
    scale_x = width / 256.0
    scale_y = height / 256.0
    tip_x = TOOL_TIP_UV_256[0] * scale_x
    tip_y = TOOL_TIP_UV_256[1] * scale_y
    radius = max(3, int(round(9 * min(scale_x, scale_y))))
    x0, x1 = max(0, int(tip_x) - radius), min(width, int(tip_x) + radius + 1)
    y0, y1 = max(0, int(tip_y) - radius), min(height, int(tip_y) + radius + 1)
    hsv = _rgb_to_hsv(rgb)[y0:y1, x0:x1]
    dark_metal = (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 25) & (hsv[:, :, 2] < 150)
    fraction = float(np.mean(dark_metal)) if dark_metal.size else 0.0
    confidence = float(np.clip(fraction / 0.45, 0.0, 1.0))
    return (float(tip_x), float(tip_y)), confidence >= 0.35, confidence


def estimate_pair(agentview_rgb: np.ndarray, wristview_rgb: np.ndarray) -> VisionEstimate:
    """Estimate 2D cues from both RGB views; no simulator state is accepted."""
    import time

    started = time.perf_counter()
    if agentview_rgb.shape[:2] != wristview_rgb.shape[:2]:
        raise ValueError("agentview and wristview must have matching image dimensions")
    height, width = agentview_rgb.shape[:2]
    agent_screw = detect_screw_head(agentview_rgb, AGENTVIEW_SCREW_ROI_256)
    wrist_screw = detect_screw_head(wristview_rgb, WRISTVIEW_SCREW_ROI_256)
    tip_pixel, tip_visible, tip_confidence = _tool_tip_visibility(wristview_rgb)
    offset = None
    if wrist_screw.visible and tip_visible and wrist_screw.center_px is not None:
        offset = (wrist_screw.center_px[0] - tip_pixel[0],
                  wrist_screw.center_px[1] - tip_pixel[1])
    processing_ms = (time.perf_counter() - started) * 1000.0
    return VisionEstimate((width, height), agent_screw, wrist_screw, tip_pixel,
                          "rigid-camera/tool calibration", tip_visible,
                          tip_confidence, offset, processing_ms)


def project_pixel_to_plane(
    pixel_uv: tuple[float, float],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    vertical_fov_deg: float,
    image_size: tuple[int, int],
    plane_z_m: float = SCREW_HEAD_TOP_Z_M,
) -> np.ndarray | None:
    """Project a detected pixel through calibrated camera geometry onto a known plane."""
    width, height = image_size
    if width <= 0 or height <= 0 or not np.isfinite(vertical_fov_deg):
        return None
    if not 0.0 < vertical_fov_deg < 180.0:
        return None
    camera_position = np.asarray(camera_position, dtype=float)
    camera_rotation = np.asarray(camera_rotation, dtype=float).reshape(3, 3)
    if camera_position.shape != (3,) or not np.isfinite(camera_position).all():
        return None
    focal_length = height / (2.0 * np.tan(np.deg2rad(vertical_fov_deg) / 2.0))
    camera_ray = np.array([
        (pixel_uv[0] - width / 2.0) / focal_length,
        -(pixel_uv[1] - height / 2.0) / focal_length,
        -1.0,
    ])
    world_ray = camera_rotation @ camera_ray
    if abs(world_ray[2]) < 1e-8:
        return None
    distance = (plane_z_m - camera_position[2]) / world_ray[2]
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    point = camera_position + distance * world_ray
    return point if np.isfinite(point).all() else None


def project_pixel_with_depth(
    pixel_uv: tuple[float, float],
    depth_m: float,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    vertical_fov_deg: float,
    image_size: tuple[int, int],
) -> np.ndarray | None:
    width, height = image_size
    if width <= 0 or height <= 0 or not np.isfinite(depth_m) or depth_m <= 0:
        return None
    if not 0.0 < vertical_fov_deg < 180.0:
        return None
    camera_position = np.asarray(camera_position, dtype=float)
    camera_rotation = np.asarray(camera_rotation, dtype=float).reshape(3, 3)
    if camera_position.shape != (3,) or not np.isfinite(camera_position).all():
        return None
    focal_length = height / (2.0 * np.tan(np.deg2rad(vertical_fov_deg) / 2.0))
    camera_ray = np.array([
        (pixel_uv[0] - width / 2.0) / focal_length,
        -(pixel_uv[1] - height / 2.0) / focal_length,
        -1.0,
    ])
    world_ray = camera_rotation @ camera_ray
    if abs(world_ray[2]) < 1e-8:
        return None
    point = camera_position + world_ray * (float(depth_m) / -camera_ray[2])
    return point if np.isfinite(point).all() else None
