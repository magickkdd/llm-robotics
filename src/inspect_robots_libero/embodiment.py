"""LIBERO-10 embodiment: a Panda arm in MuJoCo, driven over a private stdio worker.

The simulator runs in a subprocess (``_worker.py``, by default under the same
interpreter) and this class speaks line-delimited JSON with it. This keeps API
credentials out of the simulator process and isolates MuJoCo/EGL crashes from
the eval loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, Any

import numpy as np

from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_libero.notes import TASK_NOTES
from inspect_robots_libero.panda_control import (
    PANDA_COMPLIANT_AXIS_STEP_M,
    PANDA_POSITION_BOUNDS_M,
)
from inspect_robots_libero.spiral import SPIRAL_POINTS, SPIRAL_MAX_RADIUS_M
from inspect_robots_libero.tasks import ENV_SEED_BASE, check_object_task_id, check_task_id

CONTROL_HZ = 20
WORKER = Path(__file__).with_name("_worker.py")

BASE_DOCS = (
    "Single Panda arm. move_to uses world-frame x,y,z in meters and rot6d "
    "orientation (first two rotation-matrix columns). z points up. Gripper 0=open, 1=close. "
    "The gripper is open/close only: any value above 0.5 keeps closing and any value below "
    "0.5 keeps opening, so intermediate values do not give a partial opening. The gripper "
    "entry of eef_state is the current gripper command, not the finger position: it reads "
    "1 while closed on an object even though the fingers are stopped apart. Measured finger "
    "positions are in gripper_qpos (meters, about +/-0.04 each when fully open, near 0 when "
    "closed on nothing); fingers stopped clearly apart after closing indicate something is "
    "between them. Omitting gripper from a move_to call keeps the current command. Keep "
    "orientation unchanged unless necessary. Images are upright agentview and wrist camera "
    "views. Only robot proprioception and RGB are available; no object coordinates. "
    "Intermediate target arrival must be checked from the next observation. Verify the "
    "object actually followed the lift in the next observation before assuming the grasp "
    "is secure.\n\n"
    "General manipulation guidance:\n"
    "1. Avoid collisions: before a lateral move that could intersect nearby objects or "
"furniture, lift to a visibly safe clearance. Do not sweep through or push objects "
"unless the task requires it. Use large moves in free space and small moves only in "
"the last few centimeters near an object.\n"
    "2. Identify the target object carefully; similar-looking distractors may be present. "
    "Cross-check both camera views.\n"
    "3. A closed gripper does not mean a successful grasp. After closing, lift 3-5 cm and "
    "confirm in both cameras that the object rose with the gripper.\n"
    "4. Handle objects the way a person would: grasp a stable, graspable part without "
    "pushing or tipping the object, keep it in the same orientation (usually upright) "
    "from grasp to placement, and lower it close to the support surface before "
    "releasing.\n"
    "5. If the same approach fails twice, change it (grasp point, approach direction, or "
    "a small adjustment); do not repeat the same motion.\n"
    "6. After each move, confirm from the next observation that it succeeded before "
    "starting the next step; do not assume success.\n"
    "7. Prefer fewer, purposeful moves over many tiny corrections; your call budget is "
    "limited.\n"
    "8. Declare success only when the latest observation clearly shows that the requested "
    "final state has been achieved."
)


def _action_space(compliant_axis: str | None = None) -> Box:
    """Absolute end-effector pose (xyz + rot6d) plus a continuous gripper command."""
    max_step = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1])
    if compliant_axis is not None:
        max_step[("x", "y", "z").index(compliant_axis)] = PANDA_COMPLIANT_AXIS_STEP_M
    return Box(
        shape=(10,),
        low=np.concatenate((PANDA_POSITION_BOUNDS_M[:, 0], np.full(6, -1.0), [0.0])),
        high=np.concatenate((PANDA_POSITION_BOUNDS_M[:, 1], np.ones(6), [1.0])),
        semantics=ActionSemantics(
            control_mode="eef_abs_pose",
            rotation_repr="rot6d",
            gripper="continuous",
            frame="world",
            dim_labels=("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"),
            max_step=tuple(max_step.tolist()),
        ),
    )


class LiberoEmbodiment(EmbodimentBase):
    """One LIBERO-10 simulator instance; each reset opens a fresh episode.

    Args:
        task_id: native LIBERO-10 task id. Optional when scenes come from the
            ``libero_10`` task (their metadata names the task); required for
            ``--instruction`` runs and for ``task_notes``.
        init_state: initial-state index used when a scene carries none.
        output_dir: where ``worker.log`` and one ``rollout.mp4`` per episode go.
        libero_python: interpreter that runs the worker. Defaults to
            ``$LIBERO_PYTHON``, else the current interpreter.
        task_notes: append the demonstration-derived hints in ``notes.py``
            (not zero-shot; off by default).
    """

    def __init__(
        self,
        task_id: int | None = None,
        init_state: int = 0,
        output_dir: str | os.PathLike[str] = "outputs/libero",
        libero_python: str | None = None,
        task_notes: bool = False,
        suite: str = "libero_10",
        force_feedback: bool = False,
        vision_assist: bool = False,
        compliant_control: bool = False,
        compliant_axis: str = "z",
        compliant_target_force_n: float = -2.0,
        compliant_contact_threshold_n: float = 1.0,
        compliant_gain_m_per_n: float = 0.0003,
        spiral_search: bool = False,
    ) -> None:
        if suite not in ("libero_10", "libero_object", "libero_spatial", "libero_goal", "libero_90"):
            raise ConfigError(f"unsupported LIBERO suite {suite!r}")
        if task_notes and suite != "libero_10":
            raise ConfigError("task_notes are only defined for libero_10")
        self.suite = suite
        if suite == "libero_10":
            self.task_id = None if task_id is None else check_task_id(task_id)
        elif suite == "libero_object":
            self.task_id = None if task_id is None else check_object_task_id(task_id)
        else:
            self.task_id = task_id
        self.init_state = init_state
        self.output = Path(output_dir)
        self.libero_python = libero_python or os.environ.get("LIBERO_PYTHON") or sys.executable
        self.force_feedback = force_feedback
        self.vision_assist = vision_assist
        if compliant_control and not force_feedback:
            raise ConfigError("compliant_control=True requires force_feedback=True")
        if spiral_search and (not force_feedback or not compliant_control):
            raise ConfigError("spiral_search requires force_feedback and compliant_control")
        if compliant_axis not in ("x", "y", "z"):
            raise ConfigError("compliant_axis must be x, y, or z")
        self.compliant_control = compliant_control
        self.compliant_axis = compliant_axis
        self.compliant_target_force_n = compliant_target_force_n
        self.compliant_contact_threshold_n = compliant_contact_threshold_n
        self.compliant_gain_m_per_n = compliant_gain_m_per_n
        self.spiral_search = spiral_search
        self.spiral_search_started = False
        if task_notes and self.task_id is None:
            raise ConfigError("task_notes=True needs an explicit task_id.\nfix: add -E task_id=N")
        docs = BASE_DOCS
        if force_feedback:
            docs += (
                "\n\nOptional Panda wrist force/torque sensing is enabled. "
                "wrist_ft_wrench is the six-axis reading at the Panda gripper's "
                "MuJoCo FT site, expressed in that sensor frame: Fx,Fy,Fz in N, "
                "then Tx,Ty,Tz in Nm. It is a raw simulated sensor reading and "
                "can include gravity and motion effects; use changes and contact "
                "context rather than treating a nonzero value alone as contact."
            )
        if vision_assist:
            docs += (
                "\n\nTraining-free Panda image-region proposals are enabled. "
                "agentview_regions and wrist_regions contain rows [center_x, "
                "center_y, width, height, score], normalized to [0,1]. Nonzero "
                "rows are color-contrast regions, not semantic object detections "
                "or simulator coordinates. Match them to the task using the RGB "
                "images; score is a heuristic, not a probability. All-zero rows "
                "mean no candidate was proposed. Select the task object from the "
                "wrist image and pass its row index as visual_target_index in move_to. "
                "The worker tracks only that selected proposal, converts its pixel "
                "offset to a small world-frame Cartesian assist, and does not infer "
                "object identity or read simulator coordinates."
            )
        if compliant_control:
            docs += (
                "\n\nPanda Cartesian admittance is enabled on world "
                f"{compliant_axis.upper()} with a signed force target of "
                f"{compliant_target_force_n:g} N. Bias-subtracted force in the "
                "target direction reaching "
                f"{compliant_contact_threshold_n:g} N engages a latched correction; "
                "the target ramps from 0 N to the configured value over control cycles "
                "and temporary image-confidence loss does not disable it. "
                "The wrench baseline is captured at reset; this is a "
                "Panda-specific outer-loop pose correction, not direct torque "
                "control."
            )
        if spiral_search:
            docs += (
                "\n\nAn atomic force-compliant spiral_search tool is enabled. Call it only "
                "after visual contact with the target. It runs one bounded spiral "
                f"({SPIRAL_POINTS} small XY increments over {SPIRAL_MAX_RADIUS_M * 1000:g} mm, "
                "repeated four times) "
                "without intermediate model calls, holding the current orientation "
                "and using the configured Z-axis force compliance. Seating is judged "
                "only from wrist force and measured end-effector travel; no object "
                "geometry or simulator target coordinates are read. The LIBERO Panda "
                "has no independently rotating socket, so hex-phase rotation is not "
                "actuated."
            )
        if task_notes and self.task_id in TASK_NOTES:
            docs += "\n\n" + TASK_NOTES[self.task_id]
        name = suite if self.task_id is None else f"{suite}_task{self.task_id:02d}"
        self.info = EmbodimentInfo(
            name=name,
            is_simulated=True,
            control_hz=CONTROL_HZ,
            capabilities=frozenset(
                {"seedable", "resettable", "privileged_success", "renderable"}
                | ({"atomic_spiral_search"} if spiral_search else set())
            ),
            action_space=_action_space(compliant_axis if compliant_control else None),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(k, 256, 256) for k in ("agentview", "wrist")),
                state=StateSpec(
                    tuple(
                        [StateField("eef_state", (10,)), StateField("gripper_qpos", (2,), "m")]
                        + ([StateField("wrist_ft_wrench", (6,), "N,Nm")] if force_feedback else [])
                        + ([StateField("agentview_regions", (8, 5), "normalized image coords"),
                            StateField("wrist_regions", (8, 5), "normalized image coords")]
                           if vision_assist else [])
                        + ([StateField("vision_pixel_offset", (2,), "pixels"),
                            StateField("vision_cartesian_delta_world", (3,), "m"),
                            StateField("vision_confidence", (1,))]
                           if vision_assist else [])
                        + ([StateField("wrist_force_delta_world", (3,), "N"),
                            StateField("compliance_target_force_n", (1,), "N"),
                            StateField("compliance_state", (2,))]
                           if compliant_control else [])
                        + ([StateField("spiral_search_state", (3,)),
                            StateField("spiral_tactile_state", (2,)),
                            StateField("spiral_search_exhausted", (1,))]
                           if spiral_search else [])
                    )
                ),
            ),
            docs=docs,
        )
        self.process: subprocess.Popen[str] | None = None
        self.worker_log: IO[str] | None = None
        self.last: dict[str, Any] | None = None
        self.instruction = ""
        self.episode_dir: Path | None = None

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the worker's reply; worker errors raise."""
        assert self.process is not None and self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise EmbodimentFault(f"LIBERO worker exited; see {self.output / 'worker.log'}")
        result: dict[str, Any] = json.loads(line)
        if "error" in result:
            raise EmbodimentFault(f"LIBERO worker: {result['error']}")
        self.last = result
        return result

    def _observation(self, data: dict[str, Any]) -> Observation:
        images = {
            name: np.asarray(data["images"][key], dtype=np.uint8)
            for name, key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image"))
        }
        state = {key: np.asarray(data[key], dtype=float) for key in ("eef_state", "gripper_qpos")}
        if self.force_feedback:
            state["wrist_ft_wrench"] = np.asarray(data["wrist_ft_wrench"], dtype=float)
        if self.vision_assist:
            state["agentview_regions"] = np.asarray(data["agentview_regions"], dtype=float)
            state["wrist_regions"] = np.asarray(data["wrist_regions"], dtype=float)
            state["vision_pixel_offset"] = np.asarray(data["vision_pixel_offset"], dtype=float)
            state["vision_cartesian_delta_world"] = np.asarray(
                data["vision_cartesian_delta_world"], dtype=float
            )
            state["vision_confidence"] = np.asarray([data["vision_confidence"]], dtype=float)
        if self.compliant_control:
            state["wrist_force_delta_world"] = np.asarray(data["wrist_force_delta_world"], dtype=float)
            state["compliance_target_force_n"] = np.asarray(
                [data.get("compliance_target_force_n", 0.0)], dtype=float
            )
            state["compliance_state"] = np.asarray([
                float(data.get("compliance_engaged", False)),
                float(data.get("compliance_ramp_step", 0)),
            ], dtype=float)
        if self.spiral_search:
            state["spiral_search_state"] = np.asarray(data["spiral_search_state"], dtype=float)
            state["spiral_tactile_state"] = np.asarray(data["spiral_tactile_state"], dtype=float)
            state["spiral_search_exhausted"] = np.asarray(
                [float(data.get("spiral_search_exhausted", False))], dtype=float
            )
        return Observation(
            images=images,
            state=state,
            instruction=self.instruction,
            state_time=data["steps"] / CONTROL_HZ,
            extra={"environment_success": bool(data.get("success", False))},
        )

    def _start_worker(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        env = {
            key: value
            for key, value in os.environ.items()
            if "API_KEY" not in key and "TOKEN" not in key and "SECRET" not in key
        }
        env.update(
            MUJOCO_GL=env.get("MUJOCO_GL", "egl"),
            PYOPENGL_PLATFORM=env.get("PYOPENGL_PLATFORM", "egl"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMBA_CACHE_DIR=env.get(
                "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "llm_libero_numba")
            ),
        )
        self.worker_log = (self.output / "worker.log").open("a")
        self.process = subprocess.Popen(
            [str(self.libero_python), "-u", str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.worker_log,
            text=True,
            env=env,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Load the scene's LIBERO task and init state, settle 10 steps, start a video."""
        self.spiral_search_started = False
        meta = scene.metadata or {}
        suite = meta.get("libero_suite", self.suite)
        if suite != self.suite:
            raise ConfigError(f"scene belongs to {suite}, but embodiment uses {self.suite}")
        task_id = meta.get("libero_task_id", self.task_id)
        if task_id is None:
            raise ConfigError(
                "this scene does not name a LIBERO task.\n"
                "fix: use --task libero_10 -T task_id=N, or pass -E task_id=N"
            )
        if self.task_id is not None and task_id != self.task_id:
            raise ConfigError(
                f"scene is LIBERO task {task_id} but the embodiment was built for task "
                f"{self.task_id}.\nfix: make -T task_id and -E task_id agree"
            )
        init_state = int(meta.get("init_state", self.init_state))
        env_seed = scene.init_seed if scene.init_seed is not None else ENV_SEED_BASE + init_state
        if self.process is None:
            self._start_worker()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.episode_dir = self.output / f"{stamp}_task{task_id:02d}_init{init_state:02d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        data = self._rpc(
            {
                "op": "reset",
                "suite": suite,
                "task_id": int(task_id),
                "init_state": init_state,
                "env_seed": int(env_seed),
                "video": str(self.episode_dir / "rollout.mp4"),
                "force_feedback": self.force_feedback,
                "vision_assist": self.vision_assist,
                "compliant_control": self.compliant_control,
                "compliant_axis": self.compliant_axis,
                "compliant_target_force_n": self.compliant_target_force_n,
                "compliant_contact_threshold_n": self.compliant_contact_threshold_n,
                "compliant_gain_m_per_n": self.compliant_gain_m_per_n,
                "spiral_search": self.spiral_search,
            }
        )
        self.instruction = data["instruction"]
        return self._observation(data)

    def step(self, action: Action) -> StepResult:
        """Execute one 20 Hz absolute-pose action; terminate on LIBERO success."""
        if action.meta.get("libero_spiral_search"):
            if not self.spiral_search:
                raise EmbodimentFault("spiral_search was not enabled for this rollout")
            if self.spiral_search_started:
                raise EmbodimentFault("spiral_search can only be called once per episode")
            if self.last is None or not self.last.get("compliance_engaged", False):
                raise EmbodimentFault("spiral_search requires force-detected axial contact")
            self.spiral_search_started = True
            data = self._rpc({"op": "spiral_search"})
        else:
            data = self._rpc({
                "op": "step",
                "action": np.asarray(action.data).tolist(),
                "visual_target_index": action.meta.get("visual_target_index"),
            })
        success = bool(data["success"])
        return StepResult(
            self._observation(data),
            reward=float(success),
            terminated=success,
            termination_reason="success" if success else None,
            info={"success": success},
        )

    def close(self) -> None:
        """Stop the worker (flushing the last video); safe to call twice."""
        if self.process is None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=10)
        finally:
            if self.worker_log is not None:
                self.worker_log.close()
            self.process = None
            self.worker_log = None
