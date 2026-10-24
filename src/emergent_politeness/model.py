"""Simulation loop (spec §3). Stage 1: survival world only (condition 0)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .agents import OUTSIDE_TASKS, WORK_TASKS, Agent, Phase, Task
from .config import Config
from .environment import FOOD, MATERIAL, Environment, Patch
from .events import EventLog
from .physics import PhysicsWorld

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


GRANT_TIMEOUT = 30  # decision steps before an unused door grant expires
TASK_RESOURCE = {Task.COLLECT_FOOD: FOOD, Task.COLLECT_MATERIAL: MATERIAL}


class Simulation:
    def __init__(self, config: Config) -> None:
        c = config.condition
        if c.physical_acts or c.language or c.affect:
            raise NotImplementedError(
                f"Condition {c.name!r} needs stage 2+; only condition 0 (survival world) is implemented."
            )
        self.cfg = config
        self.rng = np.random.default_rng(config.sim.seed)
        self.env = Environment(config.world, self.rng)
        self.log = EventLog()
        self.agents: dict[int, Agent] = {}
        self.dead: list[Agent] = []
        self.k = 0  # current decision step
        self._next_id = 0
        self.n_inside = 0
        # Door grants: agent id -> step granted. Entering grants count against capacity.
        self.entering: dict[int, int] = {}
        self.exiting: dict[int, int] = {}
        self.physics = PhysicsWorld(
            arena_size=config.world.arena_size,
            shelter=self.env.shelter,
            door_width=config.world.door_width,
            agent_radius=config.body.radius,
            may_pass_door=self._may_pass_door,
        )
        self.series: dict[str, list] = defaultdict(list)
        self.task_steps: Counter[tuple[int, Task]] = Counter()  # for division of labour
        self._spawn_initial()

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def _random_spot(self) -> np.ndarray:
        sh = self.env.shelter
        r = (sh.radius - 0.8) * math.sqrt(self.rng.random())
        a = self.rng.uniform(0, 2 * math.pi)
        return sh.centre + r * np.array([math.cos(a), math.sin(a)])

    def _add_agent(self, pos: np.ndarray, hunger: float) -> Agent:
        tp = self.cfg.tasks
        a = Agent(
            id=self._next_id,
            born=self.k,
            thresholds={t: float(self.rng.uniform(tp.theta_min, tp.theta_max)) for t in WORK_TASKS},
            strength=float(self.rng.normal(1.0, self.cfg.body.strength_sd)),
            hunger=hunger,
        )
        self._next_id += 1
        self.agents[a.id] = a
        self.physics.add_agent(a.id, pos)
        a.inside = self.env.shelter.contains(pos)
        if a.inside:
            a.spot = pos.copy()
            self.n_inside += 1
        self.log.log(self.k, "spawn", agent=a.id, inside=a.inside, pos=[round(float(x), 2) for x in pos])
        return a

    def _spawn_initial(self) -> None:
        sh = self.env.shelter
        for i in range(self.cfg.population.n_initial):
            if i < sh.capacity:
                pos = self._random_spot()
            else:  # the rest start just outside the shelter
                ang = self.rng.uniform(0, 2 * math.pi)
                pos = sh.centre + self.rng.uniform(sh.radius + 1.5, sh.radius + 4) * np.array(
                    [math.cos(ang), math.sin(ang)]
                )
            self._add_agent(pos, hunger=float(self.rng.uniform(0.1, 0.4)))

    def _spawn_newcomers(self) -> None:
        L = self.cfg.world.arena_size
        for _ in range(self.rng.poisson(self.cfg.population.arrival_rate)):
            side, u = self.rng.integers(4), self.rng.uniform(1, L - 1)
            pos = np.array([[u, 1.0], [L - 1.0, u], [u, L - 1.0], [1.0, u]][side])
            self._add_agent(pos, hunger=0.3)

    def _remove_dead(self) -> None:
        for a in [a for a in self.agents.values() if a.health <= 0]:
            a.alive = False
            self.log.log(
                self.k, "death", agent=a.id, cause=a.last_damage, age=self.k - a.born, inside=a.inside
            )
            if a.inside:
                self.n_inside -= 1
            self.entering.pop(a.id, None)
            self.exiting.pop(a.id, None)
            self.physics.remove_agent(a.id)
            del self.agents[a.id]
            self.dead.append(a)

    # ------------------------------------------------------------------
    # Movement and the door
    # ------------------------------------------------------------------
    def _may_pass_door(self, agent_id: int) -> bool:
        return agent_id in self.entering or agent_id in self.exiting

    def _goal(self, a: Agent) -> np.ndarray:
        if a.task in OUTSIDE_TASKS and a.phase in (Phase.OUTBOUND, Phase.HARVEST):
            return self.env.patches[a.patch].pos
        if a.spot is None:
            a.spot = self._random_spot()
        return a.spot

    def _detour(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Route around the shelter wall if the straight line would hit it."""
        sh = self.env.shelter
        c, clearance = sh.centre, sh.radius + 0.8
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        dd = dx * dx + dy * dy
        t = 0.0 if dd == 0 else clip((c[0] - pos[0]) * dx + (c[1] - pos[1]) * dy, 0.0, dd) / dd
        if math.hypot(pos[0] + t * dx - c[0], pos[1] + t * dy - c[1]) >= clearance:
            return target
        ap = math.atan2(pos[1] - c[1], pos[0] - c[0])
        at = math.atan2(target[1] - c[1], target[0] - c[0])
        delta = (at - ap + math.pi) % (2 * math.pi) - math.pi
        ang = ap + math.copysign(min(abs(delta), 0.8), delta)
        return c + (sh.radius + 1.6) * np.array([math.cos(ang), math.sin(ang)])

    def _waypoint(self, a: Agent, pos: np.ndarray) -> np.ndarray:
        sh = self.env.shelter
        goal = self._goal(a)
        goal_inside = sh.contains(goal)
        if a.inside:
            if goal_inside:
                return goal
            if sh.in_doorway(pos) or dist(pos, sh.door_in) < 0.7:
                self.exiting.setdefault(a.id, self.k)
                return sh.door_out
            return sh.door_in
        if not goal_inside:
            return self._detour(pos, goal)
        self.exiting.pop(a.id, None)  # permission to leave never lets an agent back in
        at_door = self._at_door(a, pos)
        if a.id in self.entering:
            return sh.door_in if sh.in_doorway(pos) or dist(pos, sh.door_out) < 0.6 else sh.door_out
        if not at_door:
            return self._detour(pos, sh.queue_spot(a.id))
        # At the door: ask to be let in (capacity check; REFUSE_ENTRY plugs in here in stage 2).
        if self.n_inside + len(self.entering) < sh.capacity:
            self.entering[a.id] = self.k
            a.waiting_at_door = False
            return sh.door_out
        if not a.waiting_at_door:
            a.waiting_at_door = True
            self.log.log(self.k, "door_blocked", agent=a.id, reason="capacity")
        return sh.queue_spot(a.id)

    def _at_door(self, a: Agent, pos) -> bool:
        sh = self.env.shelter
        return sh.in_doorway(pos) or dist(pos, sh.door_out) < 1.0 or dist(pos, sh.queue_spot(a.id)) < 1.0

    def _move(self) -> None:
        sp = self.cfg.sim
        for _ in range(sp.physics_ticks):
            for a in self.agents.values():
                pos = self.physics.position(a.id)
                w = self._waypoint(a, pos)
                dx, dy = w[0] - pos[0], w[1] - pos[1]
                d = math.hypot(dx, dy)
                if d < 0.05:
                    self.physics.set_velocity(a.id, 0.0, 0.0)
                    continue
                f = self.cfg.body.speed * (1.0 - 0.5 * a.fatigue) * min(1.0, d / 0.5) / d
                self.physics.set_velocity(a.id, dx * f, dy * f)
            self.physics.step(sp.physics_dt)

    def _update_locations(self) -> None:
        sh = self.env.shelter
        for a in self.agents.values():
            pos = self.physics.position(a.id)
            now_inside = sh.contains(pos)
            if now_inside and not a.inside and not self._entry_allowed(a):
                # Safety net: physics glitches (crowd pressure) must not break the capacity rule.
                self.physics.set_position(a.id, sh.queue_spot(a.id))
                self.log.log(self.k, "ejected", agent=a.id)
                continue
            if now_inside != a.inside:
                a.inside = now_inside
                self.n_inside += 1 if now_inside else -1
                self.log.log(self.k, "enter" if now_inside else "exit", agent=a.id)
                if not now_inside:
                    a.spot = None
            if a.id in self.entering and (a.inside or self.k - self.entering[a.id] > GRANT_TIMEOUT):
                del self.entering[a.id]
            if a.id in self.exiting and (
                (not a.inside and not sh.in_doorway(pos, tol=1.2)) or self.k - self.exiting[a.id] > GRANT_TIMEOUT
            ):
                del self.exiting[a.id]

    def _entry_allowed(self, a: Agent) -> bool:
        if a.id in self.entering:
            return True
        # Pushed back in while leaving: harmless as long as there is room.
        return a.id in self.exiting and self.n_inside < self.env.shelter.capacity

    # ------------------------------------------------------------------
    # Bodies (spec §5.1)
    # ------------------------------------------------------------------
    def _eat(self, a: Agent) -> None:
        b, sh = self.cfg.body, self.env.shelter
        if a.inside and sh.food >= 1:
            sh.food -= 1
            source = "store"
        elif a.rations > 0:
            a.rations -= 1
            source = "ration"
        elif a.load_kind == FOOD and a.load >= 1:
            a.load -= 1
            source = "load"
        else:
            return
        a.hunger = max(0.0, a.hunger - b.eat_amount)
        self.log.log(self.k, "eat", agent=a.id, source=source)

    def _update_body(self, a: Agent, cold: float) -> None:
        b, sh = self.cfg.body, self.env.shelter
        a.hunger = min(1.0, a.hunger + b.hunger_rate * (1 + b.hunger_work_mult * a.working))
        if a.hunger > b.eat_threshold:
            self._eat(a)
        if a.working:
            a.fatigue = min(1.0, a.fatigue + b.fatigue_work)
        elif a.resting:
            a.fatigue = max(0.0, a.fatigue - b.fatigue_rest)
        if a.inside:  # the shelter protects in proportion to its integrity
            dz = b.exposure_out * cold * (1 - sh.integrity) - b.exposure_in * sh.integrity
        else:
            dz = b.exposure_out * cold
        a.exposure = clip(a.exposure + dz, 0.0, 1.0)
        if a.hunger >= 1.0:
            a.health -= b.starve_damage
            a.last_damage = "starvation"
        if a.exposure >= 1.0:
            a.health -= b.freeze_damage
            a.last_damage = "cold"
        if a.resting and a.hunger < 0.5 and a.exposure < 0.5:
            a.health = min(1.0, a.health + b.regen)

    # ------------------------------------------------------------------
    # Tasks (spec §5.2)
    # ------------------------------------------------------------------
    def _choose_patch(self, kind: str) -> Patch:
        tp = self.cfg.tasks
        patches = [p for p in self.env.patches if p.kind == kind]
        u = np.array([p.stock / max(p.max_stock, 1e-9) - tp.patch_danger_aversion * p.danger for p in patches])
        w = np.exp((u - u.max()) / tp.patch_choice_temp)
        return patches[self.rng.choice(len(patches), p=w / w.sum())]

    def _stimuli(self) -> dict[Task, float]:
        """Task stimuli s_τ (spec §5.2); global, so computed once per step."""
        sh, tp, wp = self.env.shelter, self.cfg.tasks, self.cfg.world
        n = max(1, len(self.agents))
        build = clip((n - sh.capacity) / max(1, sh.capacity), 0.0, 1.0)
        material_need = max(1 - sh.integrity, build)  # material is only worth collecting if it will be used
        return {
            Task.COLLECT_FOOD: clip(1 - sh.food / (tp.food_per_capita * n), 0.0, 1.0),
            Task.COLLECT_MATERIAL: clip(1 - sh.material / tp.material_target, 0.0, 1.0) * material_need,
            Task.MAINTAIN: 1 - sh.integrity if sh.material >= wp.material_per_maintain_step else 0.0,
            Task.BUILD: build if sh.material >= wp.material_per_build_step else 0.0,
        }

    @staticmethod
    def response_probability(s: float, theta: float) -> float:
        """Fixed-threshold response function (Bonabeau et al. 1996)."""
        return s * s / (s * s + theta * theta) if s > 0 else 0.0

    def _start(self, a: Agent, task: Task, patch: Patch | None = None) -> None:
        sh = self.env.shelter
        a.task, a.patch = task, (patch.idx if patch else None)
        if task in OUTSIDE_TASKS:
            a.phase = Phase.OUTBOUND
            if a.inside and a.rations < self.cfg.body.max_rations and sh.food >= 1:
                sh.food -= 1
                a.rations += 1
                self.log.log(self.k, "take_ration", agent=a.id)
        elif task in (Task.MAINTAIN, Task.BUILD):
            a.phase = Phase.WORK if a.inside else Phase.OUTBOUND
            a.bout_left = self.cfg.tasks.bout_length
        else:
            a.phase = Phase.NONE
        self.log.log(
            self.k, "task_start", agent=a.id, task=task.value,
            patch=a.patch, danger=(patch.danger if patch else 0.0),
        )

    def _end(self, a: Agent, reason: str) -> None:
        self.log.log(self.k, "task_end", agent=a.id, task=a.task.value, reason=reason)
        a.task, a.phase, a.patch = Task.IDLE, Phase.NONE, None

    def _execute_task(self, a: Agent) -> None:
        sh, b, wp = self.env.shelter, self.cfg.body, self.cfg.world
        if a.task in OUTSIDE_TASKS:
            patch = self.env.patches[a.patch]
            pos = self.physics.position(a.id)
            if a.phase == Phase.OUTBOUND and dist(pos, patch.pos) < wp.patch_radius:
                a.phase = Phase.HARVEST
            if a.phase == Phase.HARVEST:
                take = min(b.harvest_rate, patch.stock, b.carry_capacity - a.load)
                patch.stock -= take
                a.load += take
                a.load_kind = patch.kind
                if self.rng.random() < patch.danger * (1 + b.injury_fatigue_mult * a.fatigue):
                    a.health -= b.injury_damage
                    a.last_damage = "injury"
                    self.log.log(self.k, "injury", agent=a.id, patch=patch.idx)
                if a.load >= b.carry_capacity - 1e-9 or patch.stock < 1:
                    a.phase = Phase.RETURN
            if a.phase == Phase.RETURN and a.inside:  # must get inside to deposit (no drop-off at the door)
                if a.load > 0:
                    if a.load_kind == FOOD:
                        sh.food += a.load
                    else:
                        sh.material += a.load
                    self.log.log(self.k, "deposit", agent=a.id, resource=a.load_kind, amount=a.load)
                a.load, a.load_kind = 0.0, None
                self._end(a, "done")
        elif a.task in (Task.MAINTAIN, Task.BUILD):
            if a.phase == Phase.OUTBOUND and a.inside:
                a.phase = Phase.WORK
            if a.phase != Phase.WORK:
                return
            if a.task == Task.MAINTAIN:
                if sh.material < wp.material_per_maintain_step or sh.integrity >= 1.0:
                    return self._end(a, "no_need")
                sh.material -= wp.material_per_maintain_step
                sh.integrity = min(1.0, sh.integrity + wp.maintain_rate)
            else:
                if sh.material < wp.material_per_build_step:
                    return self._end(a, "no_material")
                sh.material -= wp.material_per_build_step
                sh.build_progress += 1
                if sh.build_progress >= sh.required_progress():
                    sh.level += 1
                    sh.build_progress = 0.0
                    self.log.log(self.k, "level_up", level=sh.level, capacity=sh.capacity)
            a.bout_left -= 1
            if a.bout_left <= 0:
                self._end(a, "bout")
        elif a.task == Task.REST and a.inside and a.fatigue <= self.cfg.tasks.fatigue_rested:
            self._end(a, "rested")

    def _overrides(self, a: Agent) -> None:
        tp = self.cfg.tasks
        too_cold = not a.inside and a.exposure > tp.exposure_critical
        spent = a.fatigue > tp.fatigue_critical or a.health < tp.health_critical
        if a.task in OUTSIDE_TASKS and a.phase != Phase.RETURN and (too_cold or spent):
            a.phase = Phase.RETURN
            self.log.log(self.k, "override", agent=a.id, reason="cold" if too_cold else "spent")
        elif a.task in (Task.MAINTAIN, Task.BUILD) and spent:
            self._end(a, "spent")

    def _decide(self, a: Agent, stimuli: dict[Task, float], cold: float) -> None:
        tp = self.cfg.tasks
        if a.fatigue > tp.fatigue_critical or a.health < tp.health_critical:
            return self._start(a, Task.REST)
        if not a.inside and a.exposure > tp.exposure_critical:
            return  # stays idle, i.e. heads home
        for i in self.rng.permutation(len(WORK_TASKS)):
            task = WORK_TASKS[i]
            s = stimuli[task]
            if s <= 0:
                continue
            patch = self._choose_patch(TASK_RESOURCE[task]) if task in OUTSIDE_TASKS else None
            danger = patch.danger if patch else 0.0
            theta = a.thresholds[task] * (1 + tp.danger_weight * danger) * (1 + tp.fatigue_weight * a.fatigue)
            if task in OUTSIDE_TASKS:
                theta *= 1 + tp.cold_weight * cold
            if self.rng.random() < self.response_probability(s, theta):
                return self._start(a, task, patch)

    def _reinforce_thresholds(self) -> None:
        """Threshold reinforcement (Theraulaz et al. 1998)."""
        tp = self.cfg.tasks
        for a in self.agents.values():
            for task in WORK_TASKS:
                if a.task == task:
                    a.thresholds[task] = max(tp.theta_min, a.thresholds[task] - tp.theta_learn)
                    self.task_steps[(a.id, task)] += 1
                else:
                    a.thresholds[task] = min(tp.theta_max, a.thresholds[task] + tp.theta_forget)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _record(self, cold: float) -> None:
        if self.k % self.cfg.sim.record_every:
            return
        sh, s = self.env.shelter, self.series
        agents = list(self.agents.values())
        s["step"].append(self.k)
        s["n"].append(len(agents))
        s["n_inside"].append(self.n_inside)
        s["capacity"].append(sh.capacity)
        s["food"].append(sh.food)
        s["material"].append(sh.material)
        s["integrity"].append(sh.integrity)
        s["level"].append(sh.level)
        s["cold"].append(cold)
        s["waiting"].append(sum(a.waiting_at_door for a in agents))
        for key in ("hunger", "fatigue", "exposure", "health"):
            s[f"mean_{key}"].append(float(np.mean([getattr(a, key) for a in agents])) if agents else np.nan)
        counts = Counter(a.task for a in agents)
        for task in Task:
            s[f"task_{task.value}"].append(counts.get(task, 0))

    def step(self) -> None:
        self._move()
        self._update_locations()
        self.env.update()
        cold = self.env.cold(self.k)
        for a in self.agents.values():
            self._update_body(a, cold)
            self._execute_task(a)
        self._remove_dead()
        self._spawn_newcomers()
        # Talk phase: stage 3+.
        for a in self.agents.values():
            self._overrides(a)
        stimuli = self._stimuli()
        for aid in self.rng.permutation(list(self.agents)):
            a = self.agents[int(aid)]
            if a.task == Task.IDLE:
                self._decide(a, stimuli, cold)
        self._reinforce_thresholds()
        self._record(cold)
        self.k += 1

    def run(self, steps: int | None = None) -> "Simulation":
        for _ in range(self.cfg.sim.steps if steps is None else steps):
            self.step()
        return self
