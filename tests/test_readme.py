"""README documents the standalone UR20 project and its supported entry points."""
import unittest
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


class ReadmeTests(unittest.TestCase):
    def test_readme_describes_ur20_install_and_run_commands(self):
        text = README.read_text()
        self.assertIn("UR20 Screw Removal", text)
        self.assertIn("scripts/rollout/run_episode.py", text)
        self.assertIn("scripts/rollout/run_mock.py", text)
        self.assertNotIn("LIBERO", text)
        self.assertNotIn("inspect-robots", text)


if __name__ == "__main__":
    unittest.main()
