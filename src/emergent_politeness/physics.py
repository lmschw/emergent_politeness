"""Thin wrapper around pymunk (2D rigid bodies).

Only movement and collisions live here. The model talks to this class through
add/remove/set_velocity/positions, so the backend can be swapped (e.g. for a
pure-NumPy kinematic version) without touching model code.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pymunk

from .environment import Shelter

AGENT = 1
WALL = 2
DOOR = 3
WALL_THICKNESS = 0.2  # segment radius; thin walls let crowds squeeze through


class PhysicsWorld:
    def __init__(
        self,
        arena_size: float,
        shelter: Shelter,
        door_width: float,
        agent_radius: float,
        may_pass_door: Callable[[int], bool],
    ) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.agent_radius = agent_radius
        self.bodies: dict[int, pymunk.Body] = {}
        self._shape_owner: dict[pymunk.Shape, int] = {}
        self._may_pass_door = may_pass_door

        self._add_arena_walls(arena_size)
        self._add_shelter_walls(shelter, door_width)
        self.space.on_collision(AGENT, DOOR, pre_solve=self._door_pre_solve)

    # --- construction -------------------------------------------------
    def _static_segment(self, a, b, collision_type: int) -> pymunk.Segment:
        seg = pymunk.Segment(self.space.static_body, tuple(a), tuple(b), WALL_THICKNESS)
        seg.collision_type = collision_type
        seg.friction = 0.0
        self.space.add(seg)
        return seg

    def _add_arena_walls(self, size: float) -> None:
        corners = [(0, 0), (size, 0), (size, size), (0, size)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            self._static_segment(a, b, WALL)

    def _add_shelter_walls(self, shelter: Shelter, door_width: float, n_segments: int = 32) -> None:
        c, r = shelter.centre, shelter.radius
        half_gap = math.asin(min(1.0, (door_width / 2) / r))
        door_a = math.atan2(shelter.door_dir[1], shelter.door_dir[0])
        start, end = door_a + half_gap, door_a + 2 * math.pi - half_gap
        angles = np.linspace(start, end, n_segments + 1)
        pts = [c + r * np.array([math.cos(a), math.sin(a)]) for a in angles]
        for a, b in zip(pts, pts[1:]):
            self._static_segment(a, b, WALL)
        # Door: a segment that blocks only agents not currently allowed through.
        self._static_segment(pts[-1], pts[0], DOOR)

    def _door_pre_solve(self, arbiter: pymunk.Arbiter, space, data) -> None:
        agent_shape = arbiter.shapes[0]
        agent_id = self._shape_owner.get(agent_shape)
        if agent_id is not None and self._may_pass_door(agent_id):
            arbiter.process_collision = False

    # --- agents -------------------------------------------------------
    def add_agent(self, agent_id: int, pos: np.ndarray) -> None:
        body = pymunk.Body(1.0, float("inf"))
        body.position = (float(pos[0]), float(pos[1]))
        shape = pymunk.Circle(body, self.agent_radius)
        shape.collision_type = AGENT
        shape.friction = 0.0
        self.space.add(body, shape)
        self.bodies[agent_id] = body
        self._shape_owner[shape] = agent_id

    def remove_agent(self, agent_id: int) -> None:
        body = self.bodies.pop(agent_id)
        for shape in list(body.shapes):
            self._shape_owner.pop(shape, None)
            self.space.remove(shape)
        self.space.remove(body)

    def position(self, agent_id: int) -> pymunk.Vec2d:
        return self.bodies[agent_id].position  # Vec2d, indexable like (x, y)

    def set_position(self, agent_id: int, pos) -> None:
        body = self.bodies[agent_id]
        body.position = (float(pos[0]), float(pos[1]))
        body.velocity = (0.0, 0.0)
        self.space.reindex_shapes_for_body(body)

    def set_velocity(self, agent_id: int, vx: float, vy: float) -> None:
        self.bodies[agent_id].velocity = (vx, vy)

    def step(self, dt: float) -> None:
        self.space.step(dt)
