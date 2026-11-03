"""Task factory and entry-point wiring; no LIBERO or API needed."""
import os
import sys
import unittest
from unittest import mock

from inspect_robots.errors import ConfigError
from inspect_robots.registry import registered

from inspect_robots_libero import LIBERO_10_TASKS, LiberoEmbodiment, libero_10
from inspect_robots_libero.tasks import (
    LIBERO_OBJECT_TASKS,
    libero_object_task,
    libero_suite_task,
    load_suite_tasks,
)


class TaskTests(unittest.TestCase):
    def test_scenes_carry_task_and_init_state(self):
        task = libero_10(task_id=2, num_scenes=3, start=5)
        self.assertEqual([s.metadata['init_state'] for s in task.scenes], [5, 6, 7])
        self.assertEqual({s.metadata['libero_task_id'] for s in task.scenes}, {2})
        self.assertEqual([s.init_seed for s in task.scenes], [147, 148, 149])
        self.assertEqual(task.scenes[0].instruction, LIBERO_10_TASKS[2][1])

    def test_rejects_bad_ids(self):
        with self.assertRaises(ConfigError):
            libero_10(task_id=10)
        with self.assertRaises(ConfigError):
            libero_10(task_id=0, num_scenes=2, start=49)

    def test_libero_object_scenes_carry_suite_and_instruction(self):
        task = libero_object_task(task_id=0, num_scenes=1, start=0)
        scene = task.scenes[0]
        self.assertEqual(len(LIBERO_OBJECT_TASKS), 10)
        self.assertEqual(scene.metadata['libero_suite'], 'libero_object')
        self.assertEqual(scene.metadata['libero_task_id'], 0)
        self.assertEqual(scene.instruction, LIBERO_OBJECT_TASKS[0][1])
        self.assertEqual(scene.metadata['init_state'], 0)

    def test_libero_object_rejects_bad_ids(self):
        with self.assertRaises(ConfigError):
            libero_object_task(task_id=10)

    def test_generic_suite_task_carries_suite_metadata(self):
        catalog = (("sample_task", "do the sample task"),)
        task = libero_suite_task("libero_spatial", 0, catalog)
        scene = task.scenes[0]
        self.assertEqual(scene.metadata['libero_suite'], 'libero_spatial')
        self.assertEqual(scene.metadata['libero_task_id'], 0)
        self.assertEqual(scene.instruction, 'do the sample task')

    def test_builtin_suite_catalogs(self):
        self.assertEqual(load_suite_tasks('libero_10'), LIBERO_10_TASKS)
        self.assertEqual(load_suite_tasks('libero_object'), LIBERO_OBJECT_TASKS)

    def test_entry_points_registered(self):
        self.assertIn('libero', registered('embodiment'))
        self.assertIn('libero_10', registered('task'))
        self.assertIn('agent', registered('policy'))

    def test_embodiment_defaults_to_current_interpreter(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LIBERO_PYTHON', None)
            self.assertEqual(LiberoEmbodiment().libero_python, sys.executable)

    def test_task_notes_need_task_id(self):
        with self.assertRaises(ConfigError):
            LiberoEmbodiment(libero_python='/bin/true', task_notes=True)


if __name__ == '__main__':
    unittest.main()
