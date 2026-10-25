"""Social memory, learned status expectations and opinion updates
(spec §2.2, §5.3, §5.3a, §5.10, §5.11).

Two opinion regimes share one data structure:
- affect on: liking E and perceived status P move with the built-in affective
  values of acts, scaled by how unexpected the act was for the status
  configuration, with transient impressions and self-image updates;
- affect off: liking is a material ledger, E = tanh(L / L0); acts have fixed
  material values whatever the status of those involved.
In both regimes perceived status moves through fights, Elo-style: by how
unexpected the outcome was.

Nothing here encodes how status *should* matter. Agents learn, per status
configuration (actor above / level with / below the target, or unknown when the
viewer has no impression of one of them), how often fights
are won, entry is refused and refusals are avenged. Those expectations then
drive decisions and scale appraisals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .agents import Agent
from .config import SocialParams

BELOW, LEVEL, ABOVE, UNKNOWN = 0, 1, 2, 3  # actor's status relative to the target, as someone perceives it
BIN_LABELS = ("below", "level", "above", "unknown")
EXPECTATIONS = ("win", "refuse", "attack", "give")


@dataclass(slots=True)
class Impression:
    E: float = 0.0  # liking (affect on)
    P: float = 0.0  # perceived status
    F: float = 0.0  # familiarity
    te: float = 0.0  # transient impression, e
    tp: float = 0.0  # transient impression, p
    L: float = 0.0  # material ledger (affect off)
    last_seen: int = 0


def clip1(x: float) -> float:
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else x


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Opinions:
    def __init__(self, p: SocialParams, affect: bool, rng: np.random.Generator) -> None:
        self.p = p
        self.affect = affect
        self.rng = rng

    # --- memory (spec §5.11) -------------------------------------------
    def recall(self, owner: Agent, other: int, k: int) -> Impression:
        """Return owner's impression of `other`, creating it (and evicting if needed)."""
        imp = owner.memory.get(other)
        if imp is None:
            if len(owner.memory) >= self.p.memory_size:
                T = self.p.memory_timescale
                victim = min(owner.memory, key=lambda j: owner.memory[j].F * math.exp(-(k - owner.memory[j].last_seen) / T))
                del owner.memory[victim]
            imp = owner.memory[other] = Impression(last_seen=k)
        imp.last_seen = k
        return imp

    # --- what decisions read --------------------------------------------
    def liking(self, owner: Agent, other: int) -> float:
        """Ẽ: effective liking (E + transient) or ledger liking when affect is off."""
        imp = owner.memory.get(other)
        if imp is None:
            return 0.0
        if self.affect:
            return imp.E + imp.te
        return math.tanh(imp.L / self.p.ledger_scale)

    def status(self, owner: Agent, other: int) -> float:
        """P̃: perceived status of `other` (owner's own self-status if other is owner)."""
        if other == owner.id:
            return owner.self_status
        imp = owner.memory.get(other)
        if imp is None:
            return 0.0
        return imp.P + imp.tp if self.affect else imp.P

    def transient_e(self, owner: Agent, other: int) -> float:
        imp = owner.memory.get(other)
        return imp.te if (imp is not None and self.affect) else 0.0

    # --- learned status expectations (spec §5.3a) -----------------------
    def new_expectations(self) -> dict[str, list[float]]:
        return {name: [self.p.expectation_prior] * len(BIN_LABELS) for name in EXPECTATIONS}

    def bin(self, viewer: Agent, actor: int, target: int) -> int:
        """Actor's status relative to the target, as the viewer perceives it. UNKNOWN if the
        viewer has no impression of one of them (a stranger), so that 'level' never means 'stranger'."""
        for x in (actor, target):
            if x != viewer.id and x not in viewer.memory:
                return UNKNOWN
        d = self.status(viewer, actor) - self.status(viewer, target)
        w = self.p.status_bin_width
        return ABOVE if d > w else BELOW if d < -w else LEVEL

    def expected(self, viewer: Agent, name: str, actor: int, target: int) -> float:
        return viewer.expect[name][self.bin(viewer, actor, target)]

    def learn(self, viewer: Agent, name: str, b: int, outcome: float) -> None:
        table = viewer.expect[name]
        table[b] += self.p.expectation_rate * (outcome - table[b])

    def expected_win(self, viewer: Agent, x: int, y: int) -> float:
        return self.expected(viewer, "win", x, y)

    def refusal_risk(self, guard: Agent, arriver: int) -> float:
        """Expected chance that refusing `arriver` provokes a fight the guard loses."""
        return self.expected(guard, "attack", arriver, guard.id) * (1 - self.expected_win(guard, guard.id, arriver))

    # --- observing acts (spec §5.3) --------------------------------------
    def observe_act(
        self, observer: Agent, actor: Agent, target: Agent, value: tuple[float, float], ledger: float, k: int,
        act: str | None = None,
    ) -> None:
        """`act` names the expectation table ('refuse' or 'attack') whose surprise scales the
        affective impact; None means the act is appraised at full value (e.g. GIVE)."""
        p = self.p
        if observer is actor:
            return
        if observer is not target and self.rng.random() >= p.attention:
            return
        surprise = 1.0
        if self.affect and act is not None:
            surprise = 1.0 - self.expected(observer, act, actor.id, target.id)
        about_actor = self.recall(observer, actor.id, k)
        about_actor.F += 1
        ve, vp = value[0] * surprise, value[1] * surprise
        if observer is target:
            if self.affect:
                about_actor.E = clip1(about_actor.E + p.alpha_E_act * ve)
                about_actor.P = clip1(about_actor.P + p.alpha_P_act * vp)
                about_actor.te += ve
                about_actor.tp += vp
                target.self_status = clip1(target.self_status - p.self_status_rate * vp)
                target.acceptance = (1 - p.acceptance_rate) * target.acceptance + p.acceptance_rate * ve
            else:
                about_actor.L += ledger
            return
        about_target = self.recall(observer, target.id, k)
        about_target.F += 1
        if self.affect:
            g = (1 - p.balance_mix) + p.balance_mix * math.tanh(p.balance_gain * about_target.E)
            about_actor.E = clip1(about_actor.E + p.bystander_weight * p.alpha_E_act * ve * g)
            about_actor.P = clip1(about_actor.P + p.bystander_weight * p.alpha_P_act * vp)
            about_target.P = clip1(about_target.P - p.bystander_weight * p.alpha_P_act * vp)
        else:
            about_actor.L += p.bystander_weight * ledger

    def observe_entry_decision(self, observer: Agent, guard: Agent, arriver: Agent, refused: bool, b: int) -> None:
        """Learn how often guards in status configuration b (taken before the decision) refuse."""
        if observer is not guard and observer is not arriver and self.rng.random() >= self.p.attention:
            return
        self.learn(observer, "refuse", b, float(refused))

    def observe_gift_opportunity(self, observer: Agent, giver: Agent, target: Agent, gave: bool, b: int) -> None:
        """Learn how often someone in configuration b (giver vs hungry neighbour) gives (spec §5.3c)."""
        if observer is not giver and observer is not target and self.rng.random() >= self.p.attention:
            return
        self.learn(observer, "give", b, float(gave))

    def observe_retaliation(self, observer: Agent, b: int, attacked: bool) -> None:
        """Learn how often a refused agent in configuration b (refused vs guard) attacks."""
        self.learn(observer, "attack", b, float(attacked))

    def observe_fight(self, observer: Agent, winner: Agent, loser: Agent, k: int,
                      bins: tuple[int, int] | None = None) -> None:
        """Elo-style status update (Albers & de Vries 2001): the more unexpected the win,
        the more status moves. Then learn the win expectation for both positions.
        `bins` = (winner vs loser, loser vs winner) as the observer saw them before the fight."""
        p = self.p
        if observer is not winner and observer is not loser and self.rng.random() >= p.attention:
            return
        if bins is None:
            bins = (self.bin(observer, winner.id, loser.id), self.bin(observer, loser.id, winner.id))
        b_w, b_l = bins
        upset = 2 * (1 - observer.expect["win"][b_w])  # 1 when the outcome was a coin flip
        if observer is winner:
            winner.self_status = clip1(winner.self_status + p.winner_loser_self * upset)
            imp = self.recall(winner, loser.id, k)
            imp.P = clip1(imp.P - p.winner_loser_observer * upset)
        elif observer is loser:
            loser.self_status = clip1(loser.self_status - p.winner_loser_self * upset)
            imp = self.recall(loser, winner.id, k)
            imp.P = clip1(imp.P + p.winner_loser_observer * upset)
        else:
            w = self.recall(observer, winner.id, k)
            w.P = clip1(w.P + p.winner_loser_observer * upset)
            lo = self.recall(observer, loser.id, k)
            lo.P = clip1(lo.P - p.winner_loser_observer * upset)
        self.learn(observer, "win", b_w, 1.0)
        self.learn(observer, "win", b_l, 0.0)

    def observe_contribution(self, observer: Agent, worker: Agent, value: float, danger: float, k: int) -> None:
        """Seeing someone deliver to the shelter (spec §5.2, 'Announcements')."""
        if observer is worker or self.rng.random() >= self.p.attention:
            return
        imp = self.recall(observer, worker.id, k)
        imp.F += 1
        if self.affect:
            imp.E = clip1(imp.E + self.p.contribution_E * value)
            imp.P = clip1(imp.P + self.p.contribution_P * danger)
        else:
            imp.L += self.p.ledger_contribution * value

    # --- status as a positional good (spec §5.3b) ------------------------
    def recenter(self, agent: Agent) -> None:
        """Shift the agent's perceived statuses of everyone it knows, and its self-status, so that
        they average zero: someone rising in its eyes means everyone else, itself included, falling
        (Frank 1985; social comparison, Festinger 1954)."""
        if not agent.memory:
            return
        total = agent.self_status + sum(imp.P for imp in agent.memory.values())
        shift = total / (len(agent.memory) + 1)
        if abs(shift) < 1e-12:
            return
        agent.self_status = clip1(agent.self_status - shift)
        for imp in agent.memory.values():
            imp.P = clip1(imp.P - shift)

    # --- per-step decay ------------------------------------------------
    def decay(self, agent: Agent) -> None:
        p = self.p
        agent.self_status *= 1 - p.self_status_decay
        if self.affect:
            f = 1 - p.transient_decay
            for imp in agent.memory.values():
                if imp.te or imp.tp:
                    imp.te *= f
                    imp.tp *= f
                    if abs(imp.te) < 1e-4 and abs(imp.tp) < 1e-4:
                        imp.te = imp.tp = 0.0
