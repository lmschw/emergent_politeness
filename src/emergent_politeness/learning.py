"""Learned consequences of social acts (spec §5.3d).

No fixed-sign social dispositions: whether giving to, refusing or attacking someone
pays is learned. At each opportunity an agent uses a learned linear
*advantage*: how much more material return follows acting than not acting, over
the next `return_window` steps. The return is modelled as baseline(f) + [acted]·advantage(f),
and every decision updates both parts. A shared baseline absorbs drift in returns over
time; learning the two options separately produced a spurious advantage for refusing
(the rarely chosen option's estimate lagged behind a drifting baseline; 2026-10-27). The return counts everything that happens to the agent in
that window, from anyone (direct and indirect reciprocity, Trivers 1971; Nowak &
Sigmund 1998): food from the shared store, gifts received or given away, being
refused, damage taken. Learning is a normalised delta rule, i.e. a contextual
bandit with linear features. Learning dynamics of this kind can produce reciprocity
without built-in cooperativeness (Macy & Flache 2002), but nothing guarantees it.

Features (all learnable, starting at weight 0):
- status configuration of the agent vs the other: below / level / above / unknown;
- liking of the other (affective E, or ledger liking when affect is off);
- anger toward the other (transient e; affect conditions only);
- act-specific situation features (see FEATURES).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .agents import Agent
from .config import SocialParams
from .social import Opinions

COMMON = ("cfg_below", "cfg_level", "cfg_above", "cfg_unknown", "liking", "anger")
FEATURES = {
    "give": COMMON + ("other_hunger",),
    "refuse": COMMON + ("other_carries_food", "other_hunger", "crowding"),
    "attack": COMMON + ("expected_win", "own_hunger"),
    # Language, stage 3 (spec §5.4): asking and complying are learned the same way
    "ask_food": COMMON + ("own_hunger",),
    "comply_food": COMMON + ("other_hunger", "own_hunger"),
    "ask_task": COMMON + ("own_fatigue", "stimulus"),
    "comply_task": COMMON + ("own_fatigue", "danger"),
}


@dataclass(slots=True)
class Pending:
    due: int
    agent: Agent
    act: str
    chosen: bool
    f: np.ndarray
    welfare0: float
    parts0: dict
    subject: Agent | None = None  # vicarious: whose return is observed (None = the learner's own)
    weight: float = 1.0


class ConsequenceLearner:
    def __init__(self, p: SocialParams, opinions: Opinions) -> None:
        self.p = p
        self.op = opinions
        self.pending: deque[Pending] = deque()
        self.on_resolve = None  # optional callback(pending, return, parts delta) for analysis logging

    @staticmethod
    def new_models() -> dict[str, list[np.ndarray]]:
        """Per act: [advantage weights, baseline weights]."""
        return {act: [np.zeros(len(names)), np.zeros(len(names))] for act, names in FEATURES.items()}

    def features(self, i: Agent, j: Agent, act: str, situation: tuple[float, ...]) -> np.ndarray:
        f = np.zeros(len(FEATURES[act]))
        f[self.op.bin(i, i.id, j.id)] = 1.0  # one-hot status configuration (acts as intercepts)
        f[4] = self.op.liking(i, j.id) - self.op.transient_e(i, j.id)  # fundamental liking / ledger liking
        f[5] = -self.op.transient_e(i, j.id)  # anger: a negative transient toward j (0 without affect)
        f[6:] = situation
        return f

    def advantage(self, i: Agent, act: str, f: np.ndarray) -> float:
        return float(i.qmodels[act][0] @ f)

    def record(self, i: Agent, act: str, f: np.ndarray, chosen: bool, k: int,
               subject: Agent | None = None, weight: float = 1.0) -> None:
        """Remember a decision. With `subject`, i learns vicariously from the decision of `subject`
        (f is i's own view of the situation; the return is what then happens to the subject)."""
        who = subject or i
        self.pending.append(Pending(k + self.p.return_window, i, act, chosen, f, who.welfare,
                                    dict(who.welfare_parts), subject, weight))

    @staticmethod
    def success(a: Agent) -> float:
        """What others can see of how well an agent is doing: healthy and well fed. Material only,
        so that no idea of politeness enters through the back door."""
        return a.health * (1.0 - a.hunger)

    def imitate(self, learner: Agent, model: Agent, rng) -> bool:
        """Proportional imitation (Schlag 1998): copy with probability equal to how much better the model
        is doing; copying moves learned weights and status expectations a fraction μ toward the model's."""
        gap = self.success(model) - self.success(learner)
        if gap <= 0 or rng.random() >= gap:
            return False
        mu = self.p.copy_strength
        for act, (w_adv, w_base) in learner.qmodels.items():
            m_adv, m_base = model.qmodels[act]
            w_adv += mu * (m_adv - w_adv)
            w_base += mu * (m_base - w_base)
        for name, table in learner.expect.items():
            other = model.expect.get(name)
            if other is not None:
                for b in range(len(table)):
                    table[b] += mu * (other[b] - table[b])
        return True

    def resolve(self, k: int) -> None:
        """Learn from opportunities whose return window has closed (agents that died are skipped)."""
        p = self.p
        while self.pending and self.pending[0].due <= k:
            d = self.pending.popleft()
            who = d.subject or d.agent
            if not d.agent.alive or not who.alive:
                continue
            r = who.welfare - d.welfare0
            if self.on_resolve is not None:
                parts = {key: v - d.parts0.get(key, 0.0) for key, v in who.welfare_parts.items()
                         if v != d.parts0.get(key, 0.0)}
                self.on_resolve(d, r, parts)
            w_adv, w_base = d.agent.qmodels[d.act]
            c = 1.0 if d.chosen else 0.0
            err = r - float(w_base @ d.f) - c * float(w_adv @ d.f)
            step = d.weight * p.return_rate * err / ((1.0 + c) * float(d.f @ d.f) + 1e-6)  # normalised over [f, c·f]
            w_base += step * d.f
            if c:
                w_adv += step * d.f
