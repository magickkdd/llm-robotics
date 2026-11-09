import sys
import numpy as np

from ur20_screw.vision.estimation import (
    AGENTVIEW_SCREW_ROI_256,
    estimate_pair,
    project_pixel_to_plane,
)


def test_fast_visual_cues_detect_hex_screw_and_calibrated_tool_tip():
    background = np.zeros((256, 256, 3), dtype=np.uint8)
    background[:] = (40, 80, 130)
    agentview = background.copy()
    wristview = background.copy()
    roi_x, roi_y, roi_width, roi_height = AGENTVIEW_SCREW_ROI_256
    blob_x, blob_y = roi_x + roi_width // 2, roi_y + roi_height // 2
    agentview[blob_y - 3:blob_y + 4, blob_x - 4:blob_x + 5] = (190, 190, 190)
    wristview[114:124, 122:134] = (190, 190, 190)
    wristview[65:110, 121:135] = (65, 65, 65)

    estimate = estimate_pair(agentview, wristview)

    assert estimate.agentview_screw.visible
    assert estimate.wristview_screw.visible
    assert estimate.wristview_tool_tip_visible
    assert estimate.wristview_tool_tip_source == "rigid-camera/tool calibration"
    assert estimate.wristview_screw_minus_tip_px is not None
    assert estimate.processing_ms < 50.0


def test_wristview_ignores_decorative_bolt_outside_center_roi():
    background = np.zeros((256, 256, 3), dtype=np.uint8)
    background[:] = (40, 80, 130)
    agentview = background.copy()
    wristview = background.copy()
    roi_x, roi_y, roi_width, roi_height = AGENTVIEW_SCREW_ROI_256
    blob_x, blob_y = roi_x + roi_width // 2, roi_y + roi_height // 2
    agentview[blob_y - 3:blob_y + 4, blob_x - 4:blob_x + 5] = (190, 190, 190)
    wristview[114:124, 122:134] = (190, 190, 190)
    # 更大更亮的相邻装饰螺钉；全图搜索会选中它，中心 ROI 则必须忽略。
    wristview[50:70, 50:70] = (255, 255, 255)
    # 被 ROI 左上角裁断的电池法兰金属区；质心会落在裁切角上，不能当作螺钉。
    wristview[96:108, 96:108] = (255, 255, 255)

    estimate = estimate_pair(agentview, wristview)

    assert estimate.wristview_screw.visible
    center_x, center_y = estimate.wristview_screw.center_px
    assert abs(center_x - 128.0) < 16.0
    assert abs(center_y - 119.0) < 16.0


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
