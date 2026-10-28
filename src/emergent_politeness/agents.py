"""Agent state (spec §2.2): body, task and (from stage 2) social state.
Language state is added in stage 3."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # Social state (stage 2+)
    self_status: float = 0.0  # Ŝ
    acceptance: float = 0.0  # Â
    memory: dict = field(default_factory=dict)  # other id -> social.Impression
    expect: dict = field(default_factory=dict)  # learned status expectations, see social.Opinions
    qmodels: dict = field(default_factory=dict)  # learned returns of acts, see learning.ConsequenceLearner
    welfare: float = 0.0  # running total of material returns (store food, gifts, refusals, damage, work, talk)
    welfare_parts: dict = field(default_factory=dict)  # the same total broken down by source (for analysis)
    paused_until: int = -1  # talking: stands still until this step
    task_for: int | None = None  # set while doing a task someone asked for (its work is a social cost)
    lexicon: object = None  # language.Lexicon (stage 4: markers and affect)

    def gain(self, part: str, amount: float, counted: bool = True) -> None:
        """Book a material return. `counted=False` records it for analysis only (not in welfare)."""
        self.welfare_parts[part] = self.welfare_parts.get(part, 0.0) + amount
        if counted:
            self.welfare += amount
    refused_until: int = -1  # cannot ask to enter again before this step
    refused_by: int | None = None
    refused_at: int = -10**9

    @property
    def working(self) -> bool:
        if self.task in OUTSIDE_TASKS:
            return True
        return self.task in (Task.MAINTAIN, Task.BUILD) and self.phase == Phase.WORK

    @property
    def resting(self) -> bool:
        return self.inside and self.task in (Task.IDLE, Task.REST)
