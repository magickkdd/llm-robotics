import numpy as np
import pytest
from types import SimpleNamespace

from inspect_robots.types import Observation
from inspect_robots_agent._tools import build_toolset

from panda_vision import camera_delta_to_world, pixel_offset_to_camera_delta, propose_regions
from inspect_robots_libero.embodiment import LiberoEmbodiment


pytest.importorskip("cv2")


def test_panda_region_proposals_locate_contrasting_image_patch_without_semantics():
    image = np.full((256, 256, 3), (180, 160, 140), dtype=np.uint8)
    image[100:140, 70:115] = (20, 20, 20)

    regions = propose_regions(image)

    assert regions.shape == (8, 5)
    candidates = regions[regions[:, 4] > 0]
    assert len(candidates) > 0
    center = candidates[np.argmax(candidates[:, 4]), :2]
    np.testing.assert_allclose(center, [92.5 / 256, 120 / 256], atol=0.02)


def test_uniform_image_produces_zero_padded_no_candidate_result():
    image = np.full((256, 256, 3), 127, dtype=np.uint8)

    np.testing.assert_array_equal(propose_regions(image), np.zeros((8, 5), dtype=np.float32))


def test_pixel_offset_becomes_bounded_camera_delta():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[96:160, 160:224] = (255, 255, 255)
    offset, delta, score = pixel_offset_to_camera_delta(
        image, camera_fovy_deg=45.0, depth_m=0.35, max_delta_m=0.01
    )
    assert offset[0] > 0.0
    assert np.linalg.norm(delta) <= 0.010001
    assert score > 0.0


def test_camera_target_offset_maps_to_world_toward_target_in_both_image_axes():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[160:220, 180:240] = (255, 255, 255)
    _, delta, _ = pixel_offset_to_camera_delta(
        image, camera_fovy_deg=45.0, target_index=0, max_delta_m=0.01
    )
    correction = camera_delta_to_world(np.eye(3), delta, gain=1.0)
    assert correction[0] > 0.0
    assert correction[1] < 0.0


def test_tool_exposes_llm_visual_target_choice_and_attaches_it_to_move():
    embodiment = LiberoEmbodiment(suite="libero_spatial", task_id=0, vision_assist=True)
    toolset = build_toolset(
        embodiment.info.action_space,
        embodiment.info.observation_space,
        embodiment.info.control_hz,
        expose_give_up=False,
    )
    schema = next(item for item in toolset.schemas() if item["function"]["name"] == "move_to")
    assert schema["function"]["parameters"]["properties"]["visual_target_index"]["maximum"] == 7
    current = np.array([-0.2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    observation = Observation(state={"eef_state": current})
    call = SimpleNamespace(
        name="move_to",
        arguments='{"targets":{"x":-0.18},"visual_target_index":3,"note":"The bowl is region 3."}',
    )
    result = toolset.execute(call, observation)
    assert result.chunk is not None
    assert result.chunk.actions[0].meta["visual_target_index"] == 3
    assert all("visual_target_index" not in action.meta for action in result.chunk.actions[1:])
