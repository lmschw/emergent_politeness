"""Shelter, patches and climate (spec §2.1, §5.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import WorldParams

FOOD = "food"
MATERIAL = "material"


@dataclass
class Patch:
    idx: int
    kind: str  # FOOD or MATERIAL
    pos: np.ndarray
    danger: float  # d_p: injury probability per step while working here
    stock: float
    max_stock: float
    regrowth: float

    def update(self) -> None:
        self.stock = min(self.max_stock, self.stock + self.regrowth)


class Shelter:
    def __init__(self, p: WorldParams) -> None:
        self.p = p
        self.centre = np.array([p.arena_size / 2, p.arena_size / 2])
        self.radius = p.shelter_radius
        self.food = p.initial_food  # F_s
        self.material = p.initial_material  # M_s
        self.integrity = p.initial_integrity  # I
        self.level = 0  # ℓ
        self.build_progress = 0.0
        d = np.array([math.cos(p.door_angle), math.sin(p.door_angle)])
        self.door_dir = d
        self.door_out = self.centre + d * (self.radius + 1.0)  # waypoint just outside the door
        self.door_in = self.centre + d * (self.radius - 1.0)  # waypoint just inside the door

    def queue_spot(self, agent_id: int) -> np.ndarray:
        """Where an agent waits when the door is closed: in front of the door, spread sideways."""
        perp = np.array([-self.door_dir[1], self.door_dir[0]])
        return self.centre + self.door_dir * (self.radius + 2.2) + perp * ((agent_id % 5) - 2) * 0.7

    @property
    def capacity(self) -> int:
        return self.p.capacity_base + self.p.capacity_per_level * self.level

    def contains(self, pos: np.ndarray) -> bool:
        return math.hypot(pos[0] - self.centre[0], pos[1] - self.centre[1]) < self.radius

    def in_doorway(self, pos: np.ndarray, tol: float = 0.9) -> bool:
        """True near the door opening, on either side of the wall."""
        rx, ry = pos[0] - self.centre[0], pos[1] - self.centre[1]
        dx, dy = self.door_dir
        r = math.hypot(rx, ry)
        along = rx * dx + ry * dy
        lateral = abs(rx * dy - ry * dx)
        return abs(r - self.radius) < tol and along > 0 and lateral < self.p.door_width / 2

    def update(self) -> None:
        self.food *= 1.0 - self.p.food_spoilage
        # Larger shelters wear faster, so size settles where maintenance labour can sustain it.
        size = self.capacity / self.p.capacity_base if self.p.capacity_base > 0 else 1.0
        self.integrity = max(0.0, self.integrity - self.p.integrity_decay * size)

    def required_progress(self) -> float:
        return self.p.build_required * (self.level + 1)


class Environment:
    def __init__(self, p: WorldParams, rng: np.random.Generator) -> None:
        self.p = p
        self.shelter = Shelter(p)
        self.patches: list[Patch] = []
        kinds = [FOOD] * p.n_food_patches + [MATERIAL] * p.n_material_patches
        n = len(kinds)
        angles = rng.permutation(np.linspace(0, 2 * math.pi, n, endpoint=False))
        angles = angles + rng.uniform(-0.2, 0.2, n)
        for i, (kind, a) in enumerate(zip(kinds, angles)):
            r = rng.uniform(*p.patch_ring)
            pos = self.shelter.centre + r * np.array([math.cos(a), math.sin(a)])
            regrowth = p.food_regrowth if kind == FOOD else p.material_regrowth
            self.patches.append(
                Patch(
                    idx=i,
                    kind=kind,
                    pos=pos,
                    danger=float(rng.choice(p.patch_danger_levels)),
                    stock=p.patch_max_stock,
                    max_stock=p.patch_max_stock,
                    regrowth=regrowth,
                )
            )

    def temperature(self, step: int) -> float:
        return self.p.temp_mean + self.p.temp_amplitude * math.sin(2 * math.pi * step / self.p.day_length)

    def cold(self, step: int) -> float:
        """0 when warm, 1 at the coldest point of the night."""
        return max(0.0, self.p.temp_comfort - self.temperature(step)) / self.p.temp_scale

    def update(self) -> None:
        self.shelter.update()
        for patch in self.patches:
            patch.update()
