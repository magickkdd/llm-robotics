import numpy as np
import pytest

from inspect_robots.errors import ConfigError
from inspect_robots_libero.embodiment import LiberoEmbodiment


def _worker_result(include_force: bool) -> dict:
    result = {
        "images": {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8).tolist(),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8).tolist(),
        },
        "eef_state": np.zeros(10).tolist(),
        "gripper_qpos": np.zeros(2).tolist(),
        "success": False,
        "steps": 0,
    }
    if include_force:
        result["wrist_ft_wrench"] = [1, 2, 3, 4, 5, 6]
    return result


def test_panda_force_feedback_is_opt_in_and_declared_in_observation_schema():
    baseline = LiberoEmbodiment()
    enabled = LiberoEmbodiment(force_feedback=True)

    assert "wrist_ft_wrench" not in [field.key for field in baseline.info.observation_space.state.fields]
    assert "wrist_ft_wrench" in [field.key for field in enabled.info.observation_space.state.fields]
    assert "raw simulated sensor reading" in enabled.info.docs

    baseline_observation = baseline._observation(_worker_result(include_force=False))
    enabled_observation = enabled._observation(_worker_result(include_force=True))
    assert "wrist_ft_wrench" not in baseline_observation.state
    np.testing.assert_array_equal(enabled_observation.state["wrist_ft_wrench"], [1, 2, 3, 4, 5, 6])


def test_panda_vision_assist_is_opt_in_and_declares_two_camera_proposal_arrays():
    baseline = LiberoEmbodiment()
    enabled = LiberoEmbodiment(vision_assist=True)

    baseline_keys = [field.key for field in baseline.info.observation_space.state.fields]
    enabled_keys = [field.key for field in enabled.info.observation_space.state.fields]
    assert "agentview_regions" not in baseline_keys
    assert "wrist_regions" not in baseline_keys
    assert "agentview_regions" in enabled_keys
    assert "wrist_regions" in enabled_keys
    assert "not semantic object detections" in enabled.info.docs


def test_panda_compliant_control_requires_explicit_force_feedback():
    with pytest.raises(ConfigError, match="requires force_feedback"):
        LiberoEmbodiment(compliant_control=True)

    enabled = LiberoEmbodiment(force_feedback=True, compliant_control=True)
    keys = [field.key for field in enabled.info.observation_space.state.fields]
    assert "wrist_force_delta_world" in keys
    assert "outer-loop pose correction" in enabled.info.docs


def test_libero_action_and_compliance_bounds_allow_reaching_below_042m():
    from inspect_robots_libero.panda_control import PANDA_POSITION_BOUNDS_M

    embodiment = LiberoEmbodiment()
    assert embodiment.info.action_space.low[2] == pytest.approx(0.05)
    assert PANDA_POSITION_BOUNDS_M[2, 0] == pytest.approx(0.05)


def test_compliant_axis_step_matches_agent_interpolation_limit():
    default = LiberoEmbodiment()
    compliant = LiberoEmbodiment(force_feedback=True, compliant_control=True)
    assert default.info.action_space.semantics.max_step[2] == pytest.approx(0.01)
    assert compliant.info.action_space.semantics.max_step[2] == pytest.approx(0.0025)
