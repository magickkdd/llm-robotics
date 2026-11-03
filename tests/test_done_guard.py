import unittest
from types import SimpleNamespace

import numpy as np
from inspect_robots.types import Observation
from inspect_robots_agent._tools import build_toolset

from inspect_robots_libero.embodiment import LiberoEmbodiment


class DoneGuardTests(unittest.TestCase):
    def setUp(self):
        info = LiberoEmbodiment(suite="libero_spatial", task_id=0).info
        self.toolset = build_toolset(
            info.action_space,
            info.observation_space,
            info.control_hz,
            expose_give_up=False,
            require_success_for_done=True,
        )
        self.state = np.array([-0.2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.call = SimpleNamespace(name="done", arguments='{"summary":"finished"}')

    def observation(self, success):
        return Observation(
            state={"eef_state": self.state},
            extra={"environment_success": success},
        )

    def test_rejects_done_before_environment_success_with_safe_hold(self):
        result = self.toolset.execute(self.call, self.observation(False))

        self.assertIsNotNone(result.chunk)
        action = result.chunk.actions[0]
        np.testing.assert_array_equal(action.data, self.state)
        self.assertFalse(action.meta.get("request_stop", False))
        self.assertIn("done rejected", result.note)

    def test_allows_done_after_environment_success(self):
        result = self.toolset.execute(self.call, self.observation(True))

        self.assertIsNotNone(result.chunk)
        self.assertTrue(result.chunk.actions[0].meta["request_stop"])

    def test_hides_give_up_from_model_tools(self):
        names = [tool["function"]["name"] for tool in self.toolset.schemas()]

        self.assertIn("done", names)
        self.assertNotIn("give_up", names)


if __name__ == "__main__":
    unittest.main()
