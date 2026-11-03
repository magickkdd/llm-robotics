"""LIBERO-10 as an Inspect Robots task: one scene per LIBERO initial state.

Importing LIBERO is slow and can prompt interactively on first use, so task
names and instructions are a static copy of
``benchmark.get_benchmark_dict()["libero_10"]()`` (native task order).
"""

from __future__ import annotations

import json
import subprocess
import sys

from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

# Native LIBERO-10 task id -> (BDDL task name, language instruction).
LIBERO_10_TASKS: tuple[tuple[str, str], ...] = (
    (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
    (
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "put both the cream cheese box and the butter in the basket",
    ),
    (
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "turn on the stove and put the moka pot on it",
    ),
    (
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
    ),
    (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
    ),
    (
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "pick up the book and place it in the back compartment of the caddy",
    ),
    (
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    ),
    (
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "put both the alphabet soup and the cream cheese box in the basket",
    ),
    (
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "put both moka pots on the stove",
    ),
    (
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "put the yellow and white mug in the microwave and close it",
    ),
)

LIBERO_OBJECT_TASKS: tuple[tuple[str, str], ...] = (
    ("pick_up_the_alphabet_soup_and_place_it_in_the_basket", "pick up the alphabet soup and place it in the basket"),
    ("pick_up_the_cream_cheese_and_place_it_in_the_basket", "pick up the cream cheese and place it in the basket"),
    ("pick_up_the_salad_dressing_and_place_it_in_the_basket", "pick up the salad dressing and place it in the basket"),
    ("pick_up_the_bbq_sauce_and_place_it_in_the_basket", "pick up the bbq sauce and place it in the basket"),
    ("pick_up_the_ketchup_and_place_it_in_the_basket", "pick up the ketchup and place it in the basket"),
    ("pick_up_the_tomato_sauce_and_place_it_in_the_basket", "pick up the tomato sauce and place it in the basket"),
    ("pick_up_the_butter_and_place_it_in_the_basket", "pick up the butter and place it in the basket"),
    ("pick_up_the_milk_and_place_it_in_the_basket", "pick up the milk and place it in the basket"),
    ("pick_up_the_chocolate_pudding_and_place_it_in_the_basket", "pick up the chocolate pudding and place it in the basket"),
    ("pick_up_the_orange_juice_and_place_it_in_the_basket", "pick up the orange juice and place it in the basket"),
)

NUM_INIT_STATES = 50
MAX_STEPS = 900
# LIBERO env seed for init state i is ENV_SEED_BASE + i (matches the X-VLA
# LIBERO client: 42 + episode + 100).
ENV_SEED_BASE = 142


def check_task_id(task_id: int) -> int:
    """Return ``task_id`` if it names a LIBERO-10 task, else raise ``ConfigError``."""
    if not isinstance(task_id, int) or not 0 <= task_id < len(LIBERO_10_TASKS):
        raise ConfigError(
            f"task_id must be an integer in 0..{len(LIBERO_10_TASKS) - 1}, got {task_id!r}.\n"
            "fix: pass -T task_id=N (native LIBERO-10 order, see README)"
        )
    return task_id


def check_object_task_id(task_id: int) -> int:
    if not isinstance(task_id, int) or not 0 <= task_id < len(LIBERO_OBJECT_TASKS):
        raise ConfigError(
            f"task_id must be an integer in 0..{len(LIBERO_OBJECT_TASKS) - 1}, got {task_id!r}.\n"
            "fix: pass a LIBERO-Object task_id from 0 to 9"
        )
    return task_id


def load_suite_tasks(suite_name: str, libero_python: str | None = None) -> tuple[tuple[str, str], ...]:
    """Read task names and language instructions from an installed LIBERO suite."""
    if suite_name == "libero_10":
        return LIBERO_10_TASKS
    if suite_name == "libero_object":
        return LIBERO_OBJECT_TASKS
    interpreter = libero_python or sys.executable
    script = (
        "import json; from libero.libero import benchmark; "
        f"s=benchmark.get_benchmark_dict()[{suite_name!r}](); "
        "print(json.dumps([(t.name,t.language) for t in s.tasks]))"
    )
    completed = subprocess.run(
        [interpreter, "-c", script], check=True, capture_output=True, text=True
    )
    return tuple(tuple(item) for item in json.loads(completed.stdout.strip().splitlines()[-1]))


def libero_suite_task(
    suite_name: str,
    task_id: int,
    suite_tasks: tuple[tuple[str, str], ...],
    num_scenes: int = 1,
    start: int = 0,
    max_steps: int = MAX_STEPS,
) -> Task:
    if not 0 <= task_id < len(suite_tasks):
        raise ConfigError(f"task_id must be in 0..{len(suite_tasks) - 1} for {suite_name}")
    if not 0 <= start < start + num_scenes <= NUM_INIT_STATES:
        raise ConfigError(f"init states must stay within 0..{NUM_INIT_STATES - 1}")
    name, instruction = suite_tasks[task_id]
    scenes = [
        Scene(
            id=f"{suite_name}_task{task_id:02d}_init{i:02d}",
            instruction=instruction,
            init_seed=ENV_SEED_BASE + i,
            metadata={"libero_suite": suite_name, "libero_task_id": task_id,
                      "init_state": i, "libero_task_name": name},
        )
        for i in range(start, start + num_scenes)
    ]
    return Task(name=f"{suite_name}_task{task_id:02d}", scenes=scenes,
                scorer=success_at_end(), max_steps=max_steps)


def libero_10_task(
    task_id: int = 0,
    num_scenes: int = 1,
    start: int = 0,
    max_steps: int = MAX_STEPS,
) -> Task:
    """Build one LIBERO-10 task over init states ``start .. start+num_scenes-1``.

    Each scene carries ``libero_task_id`` and ``init_state`` in its metadata; the
    ``libero`` embodiment reads them on reset, so ``-E task_id`` is optional.
    Success is the LIBERO environment predicate at the end of the trial.
    """
    check_task_id(task_id)
    if not 0 <= start < start + num_scenes <= NUM_INIT_STATES:
        raise ConfigError(
            f"init states {start}..{start + num_scenes - 1} are outside 0..{NUM_INIT_STATES - 1}.\n"
            "fix: choose -T start and -T num_scenes within LIBERO's 50 initial states"
        )
    name, instruction = LIBERO_10_TASKS[task_id]
    scenes = [
        Scene(
            id=f"task{task_id:02d}_init{i:02d}",
            instruction=instruction,
            init_seed=ENV_SEED_BASE + i,
            metadata={"libero_task_id": task_id, "init_state": i, "libero_task_name": name},
        )
        for i in range(start, start + num_scenes)
    ]
    return Task(
        name=f"libero_10_task{task_id:02d}",
        scenes=scenes,
        scorer=success_at_end(),
        max_steps=max_steps,
    )


def libero_object_task(
    task_id: int = 0,
    num_scenes: int = 1,
    start: int = 0,
    max_steps: int = MAX_STEPS,
) -> Task:
    check_object_task_id(task_id)
    if not 0 <= start < start + num_scenes <= NUM_INIT_STATES:
        raise ConfigError(
            f"init states {start}..{start + num_scenes - 1} are outside 0..{NUM_INIT_STATES - 1}.\n"
            "fix: choose start and num_scenes within LIBERO's 50 initial states"
        )
    name, instruction = LIBERO_OBJECT_TASKS[task_id]
    scenes = [
        Scene(
            id=f"object_task{task_id:02d}_init{i:02d}",
            instruction=instruction,
            init_seed=ENV_SEED_BASE + i,
            metadata={"libero_suite": "libero_object", "libero_task_id": task_id,
                      "init_state": i, "libero_task_name": name},
        )
        for i in range(start, start + num_scenes)
    ]
    return Task(
        name=f"libero_object_task{task_id:02d}",
        scenes=scenes,
        scorer=success_at_end(),
        max_steps=max_steps,
    )
