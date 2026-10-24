"""Agent state (spec §2.2). Stage 1 holds body and task state only; the
social and language state is added in later stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Task(str, Enum):
    IDLE = "idle"
    REST = "rest"
    COLLECT_FOOD = "collect_food"
    COLLECT_MATERIAL = "collect_material"
    MAINTAIN = "maintain"
    BUILD = "build"


WORK_TASKS = (Task.COLLECT_FOOD, Task.COLLECT_MATERIAL, Task.MAINTAIN, Task.BUILD)
OUTSIDE_TASKS = (Task.COLLECT_FOOD, Task.COLLECT_MATERIAL)


class Phase(str, Enum):
    """Sub-states of a task."""

    NONE = "none"
    OUTBOUND = "outbound"  # walking to a patch
    HARVEST = "harvest"  # working at a patch
    RETURN = "return"  # walking home with the load
    WORK = "work"  # maintaining / building inside


@dataclass
class Agent:
    id: int
    born: int
    thresholds: dict[Task, float]  # θ_iτ
    strength: float  # hidden k_i
    hunger: float = 0.3  # H
    fatigue: float = 0.0  # X
    exposure: float = 0.2  # Z
    health: float = 1.0  # Q
    rations: int = 1  # r
    load: float = 0.0
    load_kind: str | None = None
    task: Task = Task.IDLE
    phase: Phase = Phase.NONE
    patch: int | None = None  # target patch index
    bout_left: int = 0
    inside: bool = False
    spot: np.ndarray | None = None  # where to stand inside the shelter
    last_damage: str | None = None
    alive: bool = True
    waiting_at_door: bool = False

    @property
    def working(self) -> bool:
        if self.task in OUTSIDE_TASKS:
            return True
        return self.task in (Task.MAINTAIN, Task.BUILD) and self.phase == Phase.WORK

    @property
    def resting(self) -> bool:
        return self.inside and self.task in (Task.IDLE, Task.REST)
