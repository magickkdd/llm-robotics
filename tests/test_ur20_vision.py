import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ur20_vision import estimate_pair, project_pixel_to_plane


def test_fast_visual_cues_detect_hex_screw_and_calibrated_tool_tip():
    background = np.zeros((256, 256, 3), dtype=np.uint8)
    background[:] = (40, 80, 130)
    agentview = background.copy()
    wristview = background.copy()
    agentview[167:174, 124:133] = (190, 190, 190)
    wristview[114:124, 122:134] = (190, 190, 190)
    wristview[42:75, 79:91] = (65, 65, 65)

    estimate = estimate_pair(agentview, wristview)

    assert estimate.agentview_screw.visible
    assert estimate.wristview_screw.visible
    assert estimate.wristview_tool_tip_visible
    assert estimate.wristview_tool_tip_source == "rigid-camera/tool calibration"
    assert estimate.wristview_screw_minus_tip_px is not None
    assert estimate.processing_ms < 50.0


def test_blank_views_report_no_screw_or_tool_tip():
    blank = np.zeros((256, 256, 3), dtype=np.uint8)

    estimate = estimate_pair(blank, blank)

    assert not estimate.agentview_screw.visible
    assert not estimate.wristview_screw.visible
    assert not estimate.wristview_tool_tip_visible
    assert estimate.wristview_screw_minus_tip_px is None


def test_calibrated_pixel_ray_intersects_known_plane():
    camera_position = np.array([0.0, 0.0, 1.0])
    camera_rotation = np.eye(3)

    point = project_pixel_to_plane(
        (128.0, 128.0), camera_position, camera_rotation,
        vertical_fov_deg=60.0, image_size=(256, 256), plane_z_m=0.0,
    )

    assert point is not None
    np.testing.assert_allclose(point, [0.0, 0.0, 0.0], atol=1e-9)
