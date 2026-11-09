"""Small data contracts shared by the UR20 simulator and policy runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class EmbodimentFault(RuntimeError):
    """The simulator worker or its control contract failed."""


@dataclass(frozen=True)
class Scene:
    name: str
    instruction: str


@dataclass
class Action:
    data: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str
    state_time: float


@dataclass
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    termination_reason: str | None
    info: dict[str, Any]


@dataclass(frozen=True)
class ActionSemantics:
    control_mode: str
    frame: str
    dim_labels: tuple[str, ...]
    max_step: tuple[float, ...]


@dataclass(frozen=True)
class Box:
    shape: tuple[int, ...]
    low: np.ndarray
    high: np.ndarray
    semantics: ActionSemantics


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class StateField:
    key: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class StateSpec:
    fields: tuple[StateField, ...]


@dataclass(frozen=True)
class ObservationSpace:
    cameras: tuple[CameraSpec, ...]
    state: StateSpec


@dataclass(frozen=True)
class EmbodimentInfo:
    name: str
    is_simulated: bool
    control_hz: float
    capabilities: frozenset[str]
    action_space: Box
    observation_space: ObservationSpace
    docs: str
