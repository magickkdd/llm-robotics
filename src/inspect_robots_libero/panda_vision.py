"""Fast, training-free image-space region proposals for LIBERO Panda views."""

from __future__ import annotations

import numpy as np


def propose_regions(rgb: np.ndarray, max_regions: int = 8) -> np.ndarray:
    """Return padded rows [cx, cy, width, height, score] normalized to [0, 1].

    These are color-contrast proposals, not object detections. They deliberately
    contain no simulator coordinates or semantic labels; the policy must match
    candidates against the task instruction and RGB image.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Panda vision proposals need OpenCV in the LIBERO worker environment"
        ) from exc

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    if max_regions < 1:
        raise ValueError("max_regions must be positive")

    height, width = image.shape[:2]
    image_area = height * width
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    background = cv2.GaussianBlur(lab, (31, 31), 0)
    residual = np.linalg.norm(lab.astype(np.float32) - background.astype(np.float32), axis=2)
    mask = np.asarray(residual >= 20.0, dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates: list[tuple[float, np.ndarray]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < max(20, int(image_area * 0.0005)) or area > image_area * 0.18:
            continue
        if box_width < 4 or box_height < 4:
            continue
        if box_width > width * 0.6 or box_height > height * 0.6:
            continue
        aspect = box_width / box_height
        if not 0.15 <= aspect <= 7.0:
            continue

        region = labels[y:y + box_height, x:x + box_width] == label
        mean_contrast = float(residual[y:y + box_height, x:x + box_width][region].mean())
        fill = area / (box_width * box_height)
        size_score = min(1.0, np.sqrt(area / (image_area * 0.004)))
        contrast_score = min(1.0, mean_contrast / 64.0)
        edge_factor = 0.7 if x == 0 or y == 0 or x + box_width >= width or y + box_height >= height else 1.0
        score = float(edge_factor * (0.55 * contrast_score + 0.30 * size_score + 0.15 * fill))
        row = np.array([
            centers[label, 0] / width,
            centers[label, 1] / height,
            box_width / width,
            box_height / height,
            score,
        ], dtype=np.float32)
        candidates.append((score, row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    rows = np.zeros((max_regions, 5), dtype=np.float32)
    for index, (_, row) in enumerate(candidates[:max_regions]):
        rows[index] = row
    return rows


def pixel_offset_to_camera_delta(
    rgb: np.ndarray,
    *,
    camera_fovy_deg: float,
    depth_m: float = 0.35,
    max_delta_m: float = 0.01,
    target_index: int | None = None,
    previous_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert the best image proposal's offset from image center to a camera-frame XY delta.

    The caller supplies the region index selected semantically by the LLM on
    the current observation. Later control frames can track the selected center
    using ``previous_center``. Without either signal this returns no correction;
    it never guesses a target from proposal score alone.
    """
    image = np.asarray(rgb)
    height, width = image.shape[:2]
    regions = propose_regions(image)
    valid = regions[:, 4] > 0.0
    if not np.any(valid):
        return np.zeros(2, dtype=float), np.zeros(2, dtype=float), 0.0
    candidates = np.flatnonzero(valid)
    if target_index is not None:
        if not 0 <= int(target_index) < len(regions):
            raise ValueError("target_index must be in the proposal range")
        index = int(target_index)
        if not valid[index]:
            return np.zeros(2, dtype=float), np.zeros(2, dtype=float), 0.0
    elif previous_center is not None:
        distances = [float(np.linalg.norm(regions[candidate, :2] - previous_center))
                     for candidate in candidates]
        index = int(candidates[int(np.argmin(distances))])
    else:
        return np.zeros(2, dtype=float), np.zeros(2, dtype=float), 0.0
    target_px = regions[index, :2] * np.array([width, height], dtype=float)
    offset_px = target_px - np.array([width / 2.0, height / 2.0])
    focal_px = (height / 2.0) / np.tan(np.deg2rad(float(camera_fovy_deg)) / 2.0)
    delta = np.array([
        offset_px[0] * float(depth_m) / focal_px,
        -offset_px[1] * float(depth_m) / focal_px,
    ], dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm > max_delta_m > 0.0:
        delta *= max_delta_m / norm
    return offset_px, delta, float(regions[index, 4])


def camera_delta_to_world(camera_rotation: np.ndarray, camera_delta: np.ndarray,
                          gain: float = 0.25) -> np.ndarray:
    """Map a target-directed camera-plane correction into world coordinates."""
    rotation = np.asarray(camera_rotation, dtype=float)
    delta = np.asarray(camera_delta, dtype=float)
    if rotation.shape != (3, 3) or delta.shape != (2,):
        raise ValueError("expected a 3x3 camera rotation and 2D camera delta")
    if not np.isfinite(rotation).all() or not np.isfinite(delta).all() or not np.isfinite(gain):
        raise ValueError("camera transform inputs must be finite")
    return float(gain) * (rotation[:, 0] * delta[0] + rotation[:, 1] * delta[1])
