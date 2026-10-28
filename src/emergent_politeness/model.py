"""Simulation loop (spec §3).

Stage 1: survival world (condition 0). Stage 2: agent-to-agent physical acts
(GIVE, REFUSE_ENTRY, ATTACK) with ledger opinions (condition 1) or affective
opinions (condition 4)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .agents import OUTSIDE_TASKS, WORK_TASKS, Agent, Phase, Task
from .config import Config
from .environment import FOOD, MATERIAL, Environment, Patch
from .events import EventLog
from .physics import PhysicsWorld
from .language import Language
from .learning import ConsequenceLearner
from .social import Opinions, sigmoid


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


GRANT_TIMEOUT = 30  # decision steps before an unused door grant expires
QUEUE_JOIN_RADIUS = 6.0  # an agent wanting in joins the entry queue within this distance of the door
TASK_RESOURCE = {Task.COLLECT_FOOD: FOOD, Task.COLLECT_MATERIAL: MATERIAL}
# Return parts counted when return_mode == "social": what others do to me, plus the direct cost of my own act
SOCIAL_PARTS = {"gift_given", "gift_received", "refused", "fight", "talk", "work_for_others", "help_received"}


class Simulation:
    def __init__(self, config: Config) -> None:
        c = config.condition
        if config.language.motive != "intrinsic":
            raise NotImplementedError("motive='learned' is not implemented yet; use 'intrinsic' (spec §5.5).")
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
        # Turn-taking at the door: first come, first served (avoids hidden id-order bias).
        self.entry_queue: list[int] = []
        self.exit_queue: list[int] = []
        self.queue_side: dict[int, int] = {}  # side of the door an agent queues on, fixed when it joins
        self.physics = PhysicsWorld(
            arena_size=config.world.arena_size,
            shelter=self.env.shelter,
            door_width=config.world.door_width,
            agent_radius=config.body.radius,
            may_pass_door=self._may_pass_door,
        )
        self.series: dict[str, list] = defaultdict(list)
        self.task_steps: Counter[tuple[int, Task]] = Counter()  # for division of labour
        # Stage 2: social acts and opinions. Nothing here touches the RNG in condition 0.
        self.acts_on = c.physical_acts
        self.opinions = Opinions(config.social, c.affect, self.rng) if (c.physical_acts or c.language) else None
        self.learner = ConsequenceLearner(config.social, self.opinions) if (c.physical_acts or c.language) else None
        self.talk_on = c.language
        # Stage 4: affect markers in language (conditions 5 and 6)
        self.lang = Language(config.language, config.social, self.opinions, self.rng) if (c.language and c.affect) else None
        self.traces: dict[tuple[int, int], tuple[int, tuple, list]] = {}  # (speaker, addressee) -> request markers
        self._last_bundle: tuple = ()
        if self.learner is not None and config.sim.log_learning:
            self.learner.on_resolve = lambda d, r, parts: self.log.log(
                self.k, "learn", agent=d.agent.id, act=d.act, chosen=d.chosen, decided=d.due - config.social.return_window,
                ret=round(r, 4), parts={key: round(v, 4) for key, v in parts.items()},
                f=[round(float(x), 4) for x in d.f])
        self.snapshots: list[dict] = []  # daily opinion snapshots for analysis
        # Refusals awaiting a possible retaliation: (arriver, guard) -> (deadline step, [(observer, bin)])
        self._pending_retaliation: dict[tuple[int, int], tuple[int, list]] = {}
        self._exits_active = self._entries_active = False
        self._stimuli_now: dict[Task, float] = {}
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
        if self.opinions is not None:
            a.expect = self.opinions.new_expectations()
        if self.learner is not None:
            a.qmodels = self.learner.new_models()
            if self.cfg.social.innate_variation:  # option 2: small random weight on liking, mean 0
                for w_adv, _ in a.qmodels.values():
                    w_adv[4] = self.rng.normal(0.0, self.cfg.social.innate_sd)
        if self.lang is not None:
            a.lexicon = self.lang.new_lexicon()
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
            for q in (self.entry_queue, self.exit_queue):
                if a.id in q:
                    q.remove(a.id)
            self.physics.remove_agent(a.id)
            del self.agents[a.id]
            self.dead.append(a)
            if self.opinions is not None:  # idealised: everyone learns of a death
                for other in self.agents.values():
                    other.memory.pop(a.id, None)

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
        flow = self.cfg.world.door_flow
        if a.inside:
            if goal_inside:
                return goal
            if a.id in self.exiting:
                return sh.door_out if sh.in_doorway(pos) or dist(pos, sh.door_in) < 0.7 else sh.door_in
            # Ask to leave: wait for your turn where you are.
            if a.id not in self.exit_queue:
                self.exit_queue.append(a.id)
            if self.exit_queue[0] == a.id and len(self.exiting) < flow and not self._entries_active:
                self.exit_queue.pop(0)
                self.exiting[a.id] = self.k
                return sh.door_in
            return pos
        if not goal_inside:
            return self._detour(pos, goal)
        self.exiting.pop(a.id, None)  # permission to leave never lets an agent back in
        if a.id in self.entering:
            if sh.in_doorway(pos) or dist(pos, sh.door_out) < 0.6:
                return sh.door_in
            if self._exits_active:  # one direction at a time; leaving agents go first
                return pos
            return self._detour(pos, sh.door_out)
        if a.id not in self.entry_queue:
            if dist(pos, sh.door_out) > QUEUE_JOIN_RADIUS:
                return self._detour(pos, sh.door_out)
            self.entry_queue.append(a.id)  # join at the back, on the side the agent arrived from
            self.queue_side[a.id] = sh.door_side(pos)
        side = self.queue_side[a.id]
        ahead = sum(1 for i in self.entry_queue[: self.entry_queue.index(a.id)] if self.queue_side.get(i) == side)
        queue = self._detour(pos, sh.queue_spot(ahead, side))
        if self.acts_on and self.k < a.refused_until:
            return queue
        if self.n_inside + len(self.entering) >= sh.capacity:
            if not a.waiting_at_door:
                a.waiting_at_door = True
                self.log.log(self.k, "door_blocked", agent=a.id, reason="capacity")
            return queue
        head = next((i for i in self.entry_queue
                     if not (self.acts_on and self.k < self.agents[i].refused_until)), None)
        in_transit = sum(1 for i in self.entering if not self.agents[i].inside)
        if head != a.id or in_transit >= flow:
            return queue
        if self.acts_on and self._refused_by_guard(a):
            self.entry_queue.remove(a.id)  # a refused agent goes to the back of the queue
            return queue
        self.entry_queue.remove(a.id)
        self.entering[a.id] = self.k
        a.waiting_at_door = False
        return sh.door_out

    def _at_door(self, a: Agent, pos) -> bool:
        sh = self.env.shelter
        return a.id in self.entry_queue or sh.in_doorway(pos) or dist(pos, sh.door_out) < 1.0

    def _door_traffic(self) -> None:
        """Who is using the door right now (spec §10, door rules): agents leaving that are in or
        next to the doorway, and agents entering that are in the doorway."""
        sh = self.env.shelter
        self._exits_active = self._entries_active = False
        for aid in self.exiting:
            p = self.physics.position(aid)
            if sh.in_doorway(p, tol=1.2) or dist(p, sh.door_in) < 1.2:
                self._exits_active = True
                break
        for aid in self.entering:
            if sh.in_doorway(self.physics.position(aid), tol=1.2):
                self._entries_active = True
                break

    def _move(self) -> None:
        sp = self.cfg.sim
        for _ in range(sp.physics_ticks):
            self._door_traffic()
            for a in self.agents.values():
                if a.paused_until > self.k:  # talking: stands still
                    self.physics.set_velocity(a.id, 0.0, 0.0)
                    continue
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
                self.physics.set_position(a.id, sh.queue_spot(len(self.entry_queue), sh.door_side(pos)))
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

    def _clean_door_queues(self) -> None:
        sh, ag = self.env.shelter, self.agents
        self.entry_queue = [i for i in self.entry_queue if i in ag and not ag[i].inside and i not in self.entering
                            and sh.contains(self._goal(ag[i]))]
        self.queue_side = {i: s for i, s in self.queue_side.items() if i in self.entry_queue}
        self.exit_queue = [i for i in self.exit_queue if i in ag and ag[i].inside and i not in self.exiting
                           and not sh.contains(self._goal(ag[i]))]

    def _entry_allowed(self, a: Agent) -> bool:
        if a.id in self.entering:
            return True
        # Pushed back in while leaving: harmless as long as there is room.
        return a.id in self.exiting and self.n_inside < self.env.shelter.capacity

    def _book(self, a: Agent, part: str, amount: float) -> None:
        """Book a material event; whether it counts as a return depends on return_mode (spec §5.3d)."""
        a.gain(part, amount, counted=self.cfg.social.return_mode == "all" or part in SOCIAL_PARTS)

    # ------------------------------------------------------------------
    # Bodies (spec §5.1)
    # ------------------------------------------------------------------
    def _eat(self, a: Agent) -> None:
        b, sh = self.cfg.body, self.env.shelter
        if a.inside and sh.food >= 1:
            sh.food -= 1
            self._book(a, "store_food", self.cfg.social.welfare_store_food)
            source = "store"
        elif a.rations > 0:
            a.rations -= 1
            a.gain("eat_ration", 1.0, counted=False)
            source = "ration"
        elif a.load_kind == FOOD and a.load >= 1:
            a.load -= 1
            a.gain("eat_load", 1.0, counted=False)
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
            self._book(a, "work_for_others" if a.task_for is not None else "work", self.cfg.social.welfare_work)
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
        build = clip((n - sh.capacity) / max(1, sh.capacity), 0.0, 1.0) if sh.level < wp.max_level else 0.0
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
                self._book(a, "store_food", self.cfg.social.welfare_store_food)
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
        a.task_for = None

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
                    self._book(a, "injury", self.cfg.social.welfare_damage * b.injury_damage / self.cfg.social.fight_damage)
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
                    if self.opinions is not None:
                        self._contribution_seen(a)
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
                if sh.material < wp.material_per_build_step or sh.level >= wp.max_level:
                    return self._end(a, "no_material" if sh.level < wp.max_level else "max_level")
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
            if self.opinions is not None and self.opinions.affect:  # approval seeking (Roberts 1998)
                theta *= 1 - tp.approval_weight * max(0.0, -a.acceptance)
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
    # Agent-to-agent physical acts (spec §5.3)
    # ------------------------------------------------------------------
    def _observers(self, *parties: Agent) -> list[Agent]:
        """Agents within r_see of any party (parties included)."""
        r = self.cfg.social.r_see
        pts = [self.physics.position(x.id) for x in parties]
        out = []
        for o in self.agents.values():
            po = self.physics.position(o.id)
            if any(dist(po, p) < r for p in pts):
                out.append(o)
        return out

    def _act(self, actor: Agent, target: Agent, kind: str, value, ledger: float, appraise: str | None = None,
             observers: list[Agent] | None = None, **extra) -> list[Agent]:
        """Log an act and let everyone nearby update their opinions. `appraise` names the
        expectation table whose surprise scales the affective impact (spec §5.3a)."""
        self.log.log(self.k, "act", act=kind, actor=actor.id, target=target.id, **extra)
        if self.lang is not None:  # spec §5.7: markers predict what the speaker then does to the addressee
            trace = self.traces.get((actor.id, target.id))
            if trace is not None and self.k - trace[0] <= self.cfg.language.trace_window:
                self.lang.condition([o for o in trace[2] if o.alive], trace[1], value)
                del self.traces[(actor.id, target.id)]
        if observers is None:
            observers = self._observers(actor, target)
        if target not in observers:
            observers.append(target)
        for o in observers:
            self.opinions.observe_act(o, actor, target, value, ledger, self.k, act=appraise)
        return observers

    def _resolve_retaliation(self, avenged: tuple[int, int] | None = None) -> None:
        """Observers of a refusal learn whether it was avenged (attack within one step)."""
        op = self.opinions
        if avenged is not None and avenged in self._pending_retaliation:
            _, seen = self._pending_retaliation.pop(avenged)
            for o, b in seen:
                if o.alive:
                    op.observe_retaliation(o, b, True)
        for key in [key for key, (deadline, _) in self._pending_retaliation.items() if deadline < self.k]:
            _, seen = self._pending_retaliation.pop(key)
            for o, b in seen:
                if o.alive:
                    op.observe_retaliation(o, b, False)

    def _refused_by_guard(self, a: Agent) -> bool:
        """Idle insiders near the door may refuse an arriving agent (REFUSE_ENTRY)."""
        sh, sp, op = self.env.shelter, self.cfg.social, self.opinions
        guards = [
            g for g in self.agents.values()
            if g.inside and g.task == Task.IDLE and dist(self.physics.position(g.id), sh.door_in) < sp.r_door
        ]
        crowding = self.n_inside / max(1, sh.capacity)
        for i in self.rng.permutation(len(guards)):
            g = guards[i]
            carries = 1.0 if (a.load_kind == FOOD and a.load >= 1) else 0.0
            f = self.learner.features(g, a, "refuse", (carries, a.hunger, crowding))
            adv = self.learner.advantage(g, "refuse", f)
            refused = self.rng.random() < sigmoid(sp.exploration_bias + sp.choice_gain * adv)
            self.learner.record(g, "refuse", f, refused, self.k)
            self.log.log(self.k, "entry_check", guard=g.id, agent=a.id, liking=round(op.liking(g, a.id), 4),
                         status_gap=round(g.self_status - op.status(g, a.id), 4),
                         risk=round(op.refusal_risk(g, a.id), 4), crowding=round(crowding, 4),
                         carries=carries, advantage=round(adv, 4), refused=refused)
            observers = self._observers(g, a)
            # Status configurations as each observer saw them *before* the decision (spec §5.3a).
            bins_refuse = [op.bin(o, g.id, a.id) for o in observers]
            bins_avenge = [op.bin(o, a.id, g.id) for o in observers]
            if refused:
                self._book(a, "refused", sp.welfare_refused)
                a.refused_until = self.k + sp.refusal_cooldown
                a.refused_by, a.refused_at = g.id, self.k
                if not a.waiting_at_door:
                    a.waiting_at_door = True
                    self.log.log(self.k, "door_blocked", agent=a.id, reason="refused")
                self._act(g, a, "refuse_entry", sp.v_refuse_entry, sp.ledger_refuse_entry, appraise="refuse",
                          observers=observers)
                # Will the refused agent retaliate? Observers learn from the answer (see _resolve_retaliation).
                self._pending_retaliation[(a.id, g.id)] = (
                    self.k + 1, [(o, b) for o, b in zip(observers, bins_avenge) if o is not g])
            for o, b in zip(observers, bins_refuse):
                op.observe_entry_decision(o, g, a, refused, b)
            self._vicarious(observers, g, a, "refuse", (carries, a.hunger, crowding), refused, exclude=(a,))
            if refused:
                return True
        return False

    def _social_acts(self) -> None:
        """Each agent may ATTACK or GIVE once per step (attack triggers first)."""
        ids = list(self.agents)
        if len(ids) < 2:
            return
        sp = self.cfg.social
        pos = np.array([tuple(self.physics.position(i)) for i in ids])
        d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
        harvesters: dict[int, list[Agent]] = defaultdict(list)
        for a in self.agents.values():
            if a.task in OUTSIDE_TASKS and a.phase == Phase.HARVEST:
                harvesters[a.patch].append(a)
        acted: set[int] = set()
        for n in self.rng.permutation(len(ids)):
            a = self.agents[ids[n]]
            if a.id in acted or a.health <= 0:
                continue
            near = [self.agents[ids[m]] for m in np.flatnonzero(d[n] < sp.r_act)
                    if m != n and ids[m] not in acted and self.agents[ids[m]].health > 0]
            choice = self._attack_choice(a, near, harvesters, acted)
            if choice is not None:
                target, reason = choice
                acted.update((a.id, target.id))
                self._fight(a, target, reason)
                continue
            choice = self._give_choice(a, near)
            if choice is None:
                continue
            target, gave = choice
            context = self._gift_context(a, target) if self.cfg.social.gift_status else None
            if gave:
                acted.add(a.id)
                self._give(a, target, context)
            if context is not None:  # learn after appraising, from expectations held before the gift
                for o, b in zip(*context):
                    self.opinions.observe_gift_opportunity(o, a, target, gave, b)

    def _attack_choice(self, a: Agent, near: list[Agent], harvesters, acted) -> tuple[Agent, str] | None:
        sp, op = self.cfg.social, self.opinions
        candidates: list[tuple[Agent, str]] = []
        if a.refused_by is not None and self.k - a.refused_at <= 1 and not a.inside:
            g = self.agents.get(a.refused_by)
            if g is not None and g.id not in acted and g.inside and self._at_door(a, self.physics.position(a.id)):
                candidates.append((g, "refused"))  # reach across the doorway (idealised)
        if a.task in OUTSIDE_TASKS and a.phase == Phase.HARVEST:
            here = harvesters[a.patch]
            rivals = [r for r in here if r is not a and r.id not in acted and r.health > 0]
            if rivals and self.env.patches[a.patch].stock < self.cfg.body.harvest_rate * len(here):
                candidates.append((max(rivals, key=lambda r: op.expected_win(a, a.id, r.id)), "patch"))
        if op.affect:
            for t in near:
                if op.transient_e(a, t.id) < -sp.attack_transient_trigger:
                    candidates.append((t, "grudge"))
                    break
        for t, reason in candidates:
            f = self.learner.features(a, t, "attack", (2 * op.expected_win(a, a.id, t.id) - 1, a.hunger))
            adv = self.learner.advantage(a, "attack", f)
            chosen = self.rng.random() < sigmoid(sp.exploration_bias + sp.choice_gain * adv)
            self.learner.record(a, "attack", f, chosen, self.k)
            if reason == "refused":  # attacking or walking away after a refusal is visible either way
                self._vicarious(self._observers(a, t), a, t, "attack",
                                lambda o: (2 * op.expected_win(o, o.id, t.id) - 1, o.hunger), chosen, exclude=(t,))
            if chosen:
                return t, reason
        return None

    def _fight(self, a: Agent, t: Agent, reason: str) -> None:
        sp = self.cfg.social

        def strength(x: Agent) -> float:
            return x.health * (1 - 0.5 * x.fatigue) * x.strength

        a_wins = self.rng.random() < sigmoid(sp.fight_sharpness * (strength(a) - strength(t)))
        winner, loser = (a, t) if a_wins else (t, a)
        loser.health -= sp.fight_damage
        winner.health -= sp.winner_damage_frac * sp.fight_damage
        self._book(loser, "fight", sp.welfare_damage)
        self._book(winner, "fight", sp.welfare_damage * sp.winner_damage_frac)
        loser.last_damage = winner.last_damage = "fight"
        value = sp.v_attack_win if a_wins else sp.v_attack_lose
        observers = self._observers(a, t)
        op = self.opinions  # status configurations as seen before the fight (spec §5.3a)
        bins = {o.id: (op.bin(o, winner.id, loser.id), op.bin(o, loser.id, winner.id)) for o in observers}
        observers = self._act(a, t, "attack", value, sp.ledger_attack, appraise="attack", observers=observers,
                              reason=reason, actor_won=a_wins)
        if reason == "refused":
            self._resolve_retaliation(avenged=(a.id, t.id))
        for o in observers:
            self.opinions.observe_fight(o, winner, loser, self.k, bins.get(o.id))
        if loser.task in OUTSIDE_TASKS and loser.phase == Phase.HARVEST:
            loser.phase = Phase.RETURN  # the loser retreats from the patch
        if reason == "refused":  # winning the fight settles whether the door opens
            a.refused_until = self.k if a_wins else self.k + sp.refusal_cooldown

    def _gift_context(self, a: Agent, t: Agent) -> tuple[list[Agent], list[int]]:
        """Observers of a gift opportunity and their status configuration (giver vs target), taken before."""
        observers = self._observers(a, t)
        return observers, [self.opinions.bin(o, a.id, t.id) for o in observers]

    def _give_choice(self, a: Agent, near: list[Agent]) -> tuple[Agent, bool] | None:
        """A visibly hungry neighbour is a gift opportunity: return (target, whether a gift is made)."""
        sp = self.cfg.social
        food = a.rations + (int(a.load) if a.load_kind == FOOD else 0)
        hungry = [t for t in near if t.hunger > sp.hungry_visible]
        if food < 1 or not hungry:
            return None
        t = max(hungry, key=lambda x: x.hunger)
        f = self.learner.features(a, t, "give", (t.hunger,))
        adv = self.learner.advantage(a, "give", f)
        gave = self.rng.random() < sigmoid(sp.exploration_bias - sp.give_own_hunger * a.hunger + sp.choice_gain * adv)
        self.learner.record(a, "give", f, gave, self.k)
        return t, gave

    def _give(self, a: Agent, t: Agent, context: tuple[list[Agent], list[int]] | None = None) -> None:
        sp = self.cfg.social
        if a.load_kind == FOOD and a.load >= 1:
            a.load -= 1
            source = "load"
        else:
            a.rations -= 1
            source = "ration"
        t.rations += 1
        self._book(a, "gift_given", -sp.welfare_gift)
        self._book(t, "gift_received", sp.welfare_gift)
        if sp.gift_status:  # option B: the gift asserts standing; its weight depends on its surprise
            value = (sp.v_give[0], sp.gift_status_value)
            self._act(a, t, "give", value, sp.ledger_give, appraise="give", observers=list(context[0]), source=source)
        else:
            self._act(a, t, "give", sp.v_give, sp.ledger_give, source=source)

    # ------------------------------------------------------------------
    # Language, stage 3 (spec §5.4): requests and responses, content only
    # ------------------------------------------------------------------
    def _say(self, speaker: Agent, addressee: Agent, act: str, content: str, **extra) -> list[Agent]:
        """Log an utterance; return everyone in earshot (speaker and addressee included).
        Talking costs the speaker time (it stands still) and effort (spec §5.4, talk cost).
        With markers (stage 4) the speaker chooses a bundle and every hearer appraises it (spec §5.5–5.6)."""
        sp = self.cfg.social
        bundle, expected = (), None
        if self.lang is not None:
            speech = f"request_{'food' if content == 'food' else 'task'}" if act == "request" else f"respond_{content}"
            bundle, expected = self.lang.choose(speaker, addressee, speech)
        self.log.log(self.k, "utterance", speaker=speaker.id, addressee=addressee.id, act=act, content=content,
                     markers=list(bundle), **extra)
        record = self.log.records[-1]
        pause = sp.talk_pause_answer if act == "respond" else sp.talk_pause_speaker
        speaker.paused_until = max(speaker.paused_until, self.k + pause)
        self._book(speaker, "talk", sp.welfare_talk * (1 + self.cfg.language.marker_talk * len(bundle)))
        ps = self.physics.position(speaker.id)
        hearers = [o for o in self.agents.values() if dist(self.physics.position(o.id), ps) < self.cfg.social.r_hear]
        if addressee not in hearers:
            hearers.append(addressee)
        for o in hearers:  # hearing someone talk makes them familiar
            if o is not speaker:
                self.opinions.recall(o, speaker.id, self.k).F += 1
        if self.lang is not None:
            record.update(exp_e=round(float(expected[0]), 4), exp_p=round(float(expected[1]), 4),
                          s_cfg=self.opinions.bin(speaker, speaker.id, addressee.id),
                          s_liking=round(self.opinions.liking(speaker, addressee.id), 4))
            d = self.lang.appraise(hearers, speaker, addressee, speech, bundle, self.k)
            if d is not None:
                record.update(app_e=round(float(d[0]), 4), app_p=round(float(d[1]), 4),
                              accidental_rudeness=bool(expected[0] > 0 and d[0] < 0))
        self._last_bundle = bundle
        return hearers

    def _choose_addressee(self, a: Agent, act: str, candidates: list[Agent], situation) -> tuple[Agent, np.ndarray]:
        """Ask whoever the agent has learned is most worth asking (ties broken at random)."""
        scored = []
        for t in candidates:
            f = self.learner.features(a, t, act, situation)
            scored.append((self.learner.advantage(a, act, f), self.rng.random(), t, f))
        _, _, t, f = max(scored, key=lambda x: (x[0], x[1]))
        return t, f

    def _talk(self) -> None:
        """Each agent not at work may make one request per step; the addressee answers at once."""
        sp, tp = self.cfg.social, self.cfg.tasks
        ids = list(self.agents)
        if len(ids) < 2:
            return
        pos = np.array([tuple(self.physics.position(i)) for i in ids])
        d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
        stimuli = self._stimuli()
        busy: set[int] = set()
        for n in self.rng.permutation(len(ids)):
            a = self.agents[ids[n]]
            if a.id in busy or a.phase in (Phase.HARVEST, Phase.WORK):  # hands busy; walking agents can talk
                continue
            near = [self.agents[ids[m]] for m in np.flatnonzero(d[n] < sp.r_hear)
                    if m != n and ids[m] not in busy]
            has_food = a.rations > 0 or (a.load_kind == FOOD and a.load >= 1)
            if self.acts_on and a.hunger > sp.ask_food_hunger and not has_food:
                holders = [t for t in near if t.rations > 0 or (t.load_kind == FOOD and t.load >= 1)]
                if holders:
                    t, f = self._choose_addressee(a, "ask_food", holders, (a.hunger,))
                    if self._decide_learned(a, "ask_food", f):
                        busy.update((a.id, t.id))
                        heard = self._say(a, t, "request", "food")
                        self._remember_request(a, t, heard)
                        self._answer_food_request(t, a, self._last_bundle, heard)
                    continue
            task = max(WORK_TASKS, key=lambda x: stimuli[x])
            if a.fatigue > sp.ask_task_fatigue and stimuli[task] > sp.ask_task_stimulus:
                idle = [t for t in near if t.task == Task.IDLE]
                if idle:
                    t, f = self._choose_addressee(a, "ask_task", idle, (a.fatigue, stimuli[task]))
                    if self._decide_learned(a, "ask_task", f):
                        busy.update((a.id, t.id))
                        heard = self._say(a, t, "request", task.value)
                        self._remember_request(a, t, heard)
                        self._answer_task_request(t, a, task, self._last_bundle, heard)

    def _decide_learned(self, a: Agent, act: str, f: np.ndarray, grounding: float = 0.0) -> bool:
        sp = self.cfg.social
        yes = self.rng.random() < sigmoid(sp.exploration_bias + grounding + sp.choice_gain * self.learner.advantage(a, act, f))
        self.learner.record(a, act, f, yes, self.k)
        return yes

    def _remember_request(self, asker: Agent, t: Agent, hearers: list[Agent]) -> None:
        if self.lang is not None and self._last_bundle:
            self.traces[(asker.id, t.id)] = (self.k, self._last_bundle, hearers)

    def _markers_learned(self, req_bundle: tuple, req_hearers: list[Agent], resp_hearers: list[Agent],
                         t: Agent, asker: Agent, act: str, yes: bool, value) -> None:
        """Stage 4: the answer's markers are conditioned on what the answerer does (spec §5.7), and everyone
        who heard the request learns which markers get a yes (spec §5.8)."""
        if self.lang is None:
            return
        att = self.cfg.social.attention
        learners = [o for o in resp_hearers if o is t or o is asker or self.rng.random() < att]
        self.lang.condition(learners, self._last_bundle, value)
        knowers = [o for o in req_hearers if o.alive and (o is asker or self.rng.random() < att)]
        self.lang.learn_compliance(knowers, asker, t, act, req_bundle, yes)

    def _answer_food_request(self, t: Agent, asker: Agent, req_bundle: tuple = (), req_hearers=()) -> None:
        sp = self.cfg.social
        f = self.learner.features(t, asker, "comply_food", (asker.hunger, t.hunger))
        adv = self.learner.advantage(t, "comply_food", f)
        yes = self._decide_learned(t, "comply_food", f, grounding=-sp.give_own_hunger * t.hunger)
        hearers = self._say(t, asker, "respond", "yes" if yes else "no", request="food",
                            own_hunger=round(t.hunger, 3), asker_hunger=round(asker.hunger, 3),
                            cfg=int(f[:4].argmax()), liking=round(float(f[4]), 4), advantage=round(adv, 4))
        self._vicarious(hearers, t, asker, "comply_food", lambda o: (asker.hunger, o.hunger), yes)
        value = ((sp.v_give[0], sp.gift_status_value) if sp.gift_status else sp.v_give) if yes else sp.v_decline
        self._markers_learned(req_bundle, list(req_hearers), hearers, t, asker, "request_food", yes, value)
        if yes:
            self._give(t, asker, self._gift_context(t, asker) if sp.gift_status else None)
        else:
            self._respond_seen(t, asker, hearers, declined=True)

    def _answer_task_request(self, t: Agent, asker: Agent, task: Task, req_bundle: tuple = (),
                             req_hearers=()) -> None:
        danger = 0.0
        patch = None
        if task in OUTSIDE_TASKS:
            patch = self._choose_patch(TASK_RESOURCE[task])
            danger = patch.danger
        f = self.learner.features(t, asker, "comply_task", (t.fatigue, danger))
        adv = self.learner.advantage(t, "comply_task", f)
        yes = self._decide_learned(t, "comply_task", f)
        hearers = self._say(t, asker, "respond", "yes" if yes else "no", request=task.value,
                            own_fatigue=round(t.fatigue, 3), cfg=int(f[:4].argmax()),
                            liking=round(float(f[4]), 4), advantage=round(adv, 4))
        self._vicarious(hearers, t, asker, "comply_task", lambda o: (o.fatigue, danger), yes)
        sp = self.cfg.social
        self._markers_learned(req_bundle, list(req_hearers), hearers, t, asker, "request_task", yes,
                              sp.v_comply if yes else sp.v_decline)
        if yes:
            self._start(t, task, patch)
            t.task_for = asker.id  # its work is now a cost of complying
            self._book(asker, "help_received", self.cfg.social.welfare_help)
        self._respond_seen(t, asker, hearers, declined=not yes)

    def _respond_seen(self, t: Agent, asker: Agent, hearers: list[Agent], declined: bool) -> None:
        """Complying with or declining a request is an act that hearers judge (spec §5.3 table)."""
        sp = self.cfg.social
        kind = "decline" if declined else "comply"
        self.log.log(self.k, "act", act=kind, actor=t.id, target=asker.id)
        value, ledger = (sp.v_decline, sp.ledger_decline) if declined else (sp.v_comply, sp.ledger_comply)
        for o in hearers:
            self.opinions.observe_act(o, t, asker, value, ledger, self.k)

    def _vicarious(self, observers: list[Agent], actor: Agent, target: Agent, act: str, situation,
                   chosen: bool, exclude: tuple = ()) -> None:
        """Learning from others' experience (spec §5.3d; Bandura 1977): observers of a decision whose
        outcome is visible either way learn from what then happens to the actor, as if they had been in
        its place (their own view of the target). `situation` is a tuple or a function of the observer."""
        sp = self.cfg.social
        if not sp.vicarious_learning:
            return
        for o in observers:
            if o is actor or o in exclude or not o.qmodels or self.rng.random() >= sp.attention:
                continue
            sit = situation(o) if callable(situation) else situation
            f = self.learner.features(o, target, act, sit)
            self.learner.record(o, act, f, chosen, self.k, subject=actor, weight=sp.vicarious_weight)

    def _social_learning(self) -> None:
        """Payoff-biased copying of learned weights (spec §5.3e): now and then an agent looks at a random
        agent in view and imitates it if it is visibly doing better."""
        sp = self.cfg.social
        ids = list(self.agents)
        for n in self.rng.permutation(len(ids)):
            a = self.agents[ids[n]]
            if self.rng.random() >= sp.copy_rate:
                continue
            pa = self.physics.position(a.id)
            in_view = [o for o in self.agents.values()
                       if o is not a and dist(self.physics.position(o.id), pa) < sp.r_see]
            if not in_view:
                continue
            model = in_view[int(self.rng.integers(len(in_view)))]
            if self.learner.imitate(a, model, self.rng):
                self.log.log(self.k, "imitate", agent=a.id, model=model.id)

    def _contribution_seen(self, a: Agent) -> None:
        """Delivering to the store is visible work (spec §5.2): u = s_τ·(1 + d_τ)."""
        danger = self.env.patches[a.patch].danger if a.patch is not None else 0.0
        u = self._stimuli_now.get(a.task, 0.0) * (1 + danger)
        for o in self._observers(a):
            self.opinions.observe_contribution(o, a, u, danger, self.k)

    def _snapshot(self) -> None:
        op = self.opinions
        self.snapshots.append({
            "step": self.k,
            "agents": {
                a.id: {
                    "self_status": a.self_status,
                    "acceptance": a.acceptance,
                    "strength": a.strength,
                    "expect": {name: list(v) for name, v in a.expect.items()},
                    "advantage": {act: list(w[0]) for act, w in a.qmodels.items()},
                    "connotation": a.lexicon.c.tolist() if a.lexicon is not None else None,
                    "norm": a.lexicon.norm.tolist() if a.lexicon is not None else None,
                    "memory": {j: (op.liking(a, j), imp.P, imp.F) for j, imp in a.memory.items()},
                }
                for a in self.agents.values()
            },
        })

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
        self._clean_door_queues()
        self.env.update()
        cold = self.env.cold(self.k)
        for a in self.agents.values():
            self._update_body(a, cold)
            self._execute_task(a)
        self._remove_dead()
        self._spawn_newcomers()
        for a in self.agents.values():
            self._overrides(a)
        if self.talk_on:  # talk phase (spec §3, §5.4)
            self._talk()
        if self.acts_on:
            self._social_acts()
            self._resolve_retaliation()
        if self.learner is not None:
            self.learner.resolve(self.k)
            if self.cfg.social.social_learning:
                self._social_learning()
            self._remove_dead()
        stimuli = self._stimuli_now = self._stimuli()
        for aid in self.rng.permutation(list(self.agents)):
            a = self.agents[int(aid)]
            if a.task == Task.IDLE:
                self._decide(a, stimuli, cold)
        self._reinforce_thresholds()
        if self.opinions is not None:
            for a in self.agents.values():
                self.opinions.decay(a)
                if self.cfg.social.relative_status:
                    self.opinions.recenter(a)
            if self.k % self.cfg.world.day_length == 0:
                self._snapshot()
        self._record(cold)
        self.k += 1

    def run(self, steps: int | None = None) -> "Simulation":
        for _ in range(self.cfg.sim.steps if steps is None else steps):
            self.step()
        return self
