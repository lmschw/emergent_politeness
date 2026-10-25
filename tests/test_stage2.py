import math
from collections import Counter

import numpy as np
import pytest

from emergent_politeness.agents import Agent
from emergent_politeness.config import CONDITIONS, Config, SocialParams
from emergent_politeness.model import Simulation
from emergent_politeness.social import ABOVE, BELOW, LEVEL, UNKNOWN, Opinions


def agent(i: int) -> Agent:
    a = Agent(id=i, born=0, thresholds={}, strength=1.0)
    a.expect = Opinions(SocialParams(), True, np.random.default_rng(0)).new_expectations()
    return a


def opinions(affect: bool, **params) -> Opinions:
    # attention = 1 so that bystander updates are deterministic
    return Opinions(SocialParams(attention=1.0, **params), affect, np.random.default_rng(0))


def test_condition0_fingerprint():
    """Regression fingerprint of condition 0. Social acts must not leak into it.
    Updated deliberately on 2026-10-25 after the door turn-taking fix and the switch to constant shelter wear."""
    sim = Simulation(Config()).run(1500)
    s = sim.series
    assert len(sim.log.records) == 1783
    assert sum(s["n"]) == 24682
    assert round(sum(s["food"]), 6) == 55976.216252
    assert sim.log.count("door_blocked") == 44


@pytest.mark.parametrize("cond", ["2", "3", "5", "6"])
def test_language_conditions_not_yet_implemented(cond):
    with pytest.raises(NotImplementedError):
        Simulation(Config(condition=CONDITIONS[cond]))


def test_ledger_opinions_without_affect():
    op = opinions(affect=False)
    giver, receiver, bystander = agent(0), agent(1), agent(2)
    for o in (receiver, bystander):
        op.observe_act(o, giver, receiver, (0.6, 0.0), ledger=1.0, k=0)
    assert op.liking(receiver, 0) == pytest.approx(math.tanh(1.0 / 5.0))
    assert op.liking(bystander, 0) == pytest.approx(math.tanh(0.5 / 5.0))  # β_by = 0.5
    assert receiver.self_status == 0 and receiver.acceptance == 0  # no self-image updates without affect
    assert op.transient_e(receiver, 0) == 0


def test_affective_update_of_target():
    op = opinions(affect=True)
    attacker, victim = agent(0), agent(1)
    op.observe_act(victim, attacker, victim, (-0.8, 0.7), ledger=-1.0, k=0)
    imp = victim.memory[0]
    assert imp.E == pytest.approx(-0.12) and imp.P == pytest.approx(0.105)
    assert op.transient_e(victim, 0) == pytest.approx(-0.8)
    assert victim.self_status == pytest.approx(-0.03 * 0.7)
    assert victim.acceptance == pytest.approx(0.03 * -0.8)


def test_balance_theory_bystander():
    """With λ_b = 1, harming someone the observer dislikes raises liking for the attacker."""
    op = opinions(affect=True, balance_mix=1.0)
    attacker, victim, friend_of_victim, enemy_of_victim = agent(0), agent(1), agent(2), agent(3)
    op.recall(friend_of_victim, 1, 0).E = 0.8
    op.recall(enemy_of_victim, 1, 0).E = -0.8
    for o in (friend_of_victim, enemy_of_victim):
        op.observe_act(o, attacker, victim, (-0.8, 0.7), ledger=-1.0, k=0)
    assert friend_of_victim.memory[0].E < 0 < enemy_of_victim.memory[0].E


def test_winner_loser_effect():
    op = opinions(affect=False)
    w, l, o = agent(0), agent(1), agent(2)
    for x in (w, l, o):
        op.observe_fight(x, w, l, k=0)
    assert w.self_status > 0 > l.self_status
    assert o.memory[0].P > 0 > o.memory[1].P
    assert l.memory[0].P > 0 and w.memory[1].P < 0


def test_memory_evicts_least_familiar_when_full():
    op = opinions(affect=True, memory_size=3)
    me = agent(0)
    for j, fam in ((1, 5.0), (2, 0.0), (3, 2.0)):
        op.recall(me, j, k=0).F = fam
    op.recall(me, 4, k=0)
    assert set(me.memory) == {1, 3, 4}


def test_transients_decay():
    op = opinions(affect=True)
    me = agent(0)
    op.recall(me, 1, 0).te = -1.0
    for _ in range(30):
        op.decay(me)
    assert abs(me.memory[1].te) < 1e-3


def test_hostile_guards_keep_capacity_and_log_refusals():
    cfg = Config(condition=CONDITIONS["1"]).with_(social={"exploration_bias": 5.0})
    sim = Simulation(cfg).run(1500)
    assert np.array(sim.series["n_inside"]).max() <= max(sim.series["capacity"])
    refusals = [r for r in sim.log.of_kind("act") if r["act"] == "refuse_entry"]
    assert refusals
    assert Counter(r["reason"] for r in sim.log.of_kind("door_blocked"))["refused"] > 0


@pytest.mark.parametrize("cond", ["1", "4"])
def test_social_conditions_run_and_are_deterministic(cond):
    a = Simulation(Config(condition=CONDITIONS[cond])).run(1300)
    b = Simulation(Config(condition=CONDITIONS[cond])).run(1300)
    assert a.log.records == b.log.records
    assert a.snapshots and a.snapshots[-1]["agents"]


def test_elo_upset_moves_status_more_than_expected_win():
    op = opinions(affect=False)
    o, x, y = agent(0), agent(1), agent(2)
    o.expect["win"] = [0.1, 0.5, 0.9, 0.5]  # o has learned that the higher-rated one usually wins
    op.recall(o, 1, 0).P = 0.5  # o rates x above y
    op.recall(o, 2, 0).P = 0.0
    op.observe_fight(o, x, y, k=0)  # expected outcome
    gain_expected = o.memory[1].P - 0.5
    op.recall(o, 1, 0).P, op.recall(o, 2, 0).P = 0.5, 0.0
    op.observe_fight(o, y, x, k=0)  # upset
    gain_upset = o.memory[2].P
    assert gain_expected == pytest.approx(0.1 * 2 * (1 - 0.9))
    assert gain_upset > 5 * gain_expected


def test_win_expectation_is_learned_from_observed_fights():
    op = opinions(affect=False)
    o, strong, weak = agent(0), agent(1), agent(2)
    op.recall(o, 1, 0).P = 0.6
    op.recall(o, 2, 0).P = -0.6
    for _ in range(40):
        op.observe_fight(o, strong, weak, k=0)
        o.memory[1].P, o.memory[2].P = 0.6, -0.6  # keep the configuration fixed
    assert o.expect["win"][ABOVE] > 0.8 and o.expect["win"][BELOW] < 0.2
    assert o.expect["win"][LEVEL] == pytest.approx(0.5)


def test_expected_refusal_is_appraised_more_mildly():
    op = opinions(affect=True)
    guard, used_to_it, surprised = agent(0), agent(1), agent(2)
    used_to_it.expect["refuse"] = [0.9] * 4
    surprised.expect["refuse"] = [0.1] * 4
    for t in (used_to_it, surprised):
        op.observe_act(t, guard, t, (-0.5, 0.4), ledger=-0.5, k=0, act="refuse")
    assert surprised.memory[0].E < used_to_it.memory[0].E < 0
    assert surprised.memory[0].E == pytest.approx(9 * used_to_it.memory[0].E)


def test_ledger_ignores_surprise():
    op = opinions(affect=False)
    guard, t = agent(0), agent(1)
    t.expect["refuse"] = [0.9] * 4
    op.observe_act(t, guard, t, (-0.5, 0.4), ledger=-0.5, k=0, act="refuse")
    assert t.memory[0].L == pytest.approx(-0.5)


def test_refusal_risk_combines_retaliation_and_losing():
    op = opinions(affect=True)
    guard = agent(0)
    guard.expect["attack"] = [0.2, 0.4, 0.8, 0.5]
    guard.expect["win"] = [0.3, 0.5, 0.7, 0.5]
    op.recall(guard, 1, 0).P = 0.6  # arriver rated well above the guard (self-status 0)
    # arriver above guard -> attack[ABOVE]; guard below arriver -> win[BELOW]
    assert op.refusal_risk(guard, 1) == pytest.approx(0.8 * (1 - 0.3))


def test_strangers_are_unknown_not_level():
    op = opinions(affect=True)
    viewer, x, y = agent(0), agent(1), agent(2)
    assert op.bin(viewer, 1, 2) == UNKNOWN  # never met either
    op.recall(viewer, 1, 0)
    assert op.bin(viewer, 1, 2) == UNKNOWN  # still a stranger to y
    op.recall(viewer, 2, 0)
    assert op.bin(viewer, 1, 2) == LEVEL  # both known, no perceived difference
    assert op.bin(viewer, 0, 1) == LEVEL  # the viewer always knows itself
    assert op.bin(viewer, 0, 7) == UNKNOWN


def test_relative_status_is_zero_sum_including_self():
    op = opinions(affect=False)
    me, x, y = agent(0), agent(1), agent(2)
    op.recall(me, 1, 0).P = 0.6  # x just rose in my eyes
    op.recall(me, 2, 0).P = 0.0
    op.recenter(me)
    total = me.self_status + me.memory[1].P + me.memory[2].P
    assert total == pytest.approx(0.0)
    assert me.memory[2].P < 0 and me.self_status < 0  # everyone else, me included, fell
    assert me.memory[1].P - me.memory[2].P == pytest.approx(0.6)  # differences are kept


def test_gift_status_toggle():
    """Off: gifts are never appraised by status configuration, so the 'give' expectations stay at the prior."""
    off = Simulation(Config(condition=CONDITIONS["4"]).with_(social={"gift_status": False})).run(9000)
    on = Simulation(Config(condition=CONDITIONS["4"]).with_(social={"gift_status": True})).run(9000)
    assert any(r["act"] == "give" for r in off.log.of_kind("act"))
    assert all(v == 0.5 for a in off.agents.values() for v in a.expect["give"])
    assert any(v != 0.5 for a in on.agents.values() for v in a.expect["give"])


def test_gift_raises_givers_standing_when_unexpected():
    op = opinions(affect=True)
    giver, receiver = agent(0), agent(1)
    receiver.expect["give"] = [0.0] * 4  # gifts are never expected
    op.observe_act(receiver, giver, receiver, (0.6, 0.2), ledger=1.0, k=0, act="give")
    assert receiver.memory[0].P > 0 and receiver.self_status < 0  # indebted to the giver (Mauss)
    jaded = agent(2)
    jaded.expect["give"] = [1.0] * 4  # gifts always expected: no effect
    op.observe_act(jaded, giver, jaded, (0.6, 0.2), ledger=1.0, k=0, act="give")
    assert jaded.memory[0].P == 0 and jaded.memory[0].E == 0


def test_gift_opportunities_are_learned():
    op = opinions(affect=True)
    o, giver, hungry = agent(0), agent(1), agent(2)
    for _ in range(50):
        op.observe_gift_opportunity(o, giver, hungry, gave=False, b=LEVEL)
    assert o.expect["give"][LEVEL] < 0.05 and o.expect["give"][ABOVE] == 0.5


def test_learner_moves_chosen_prediction_toward_return():
    from emergent_politeness.learning import ConsequenceLearner
    p = SocialParams(return_window=10, return_rate=0.5)
    op = Opinions(p, True, np.random.default_rng(0))
    learner = ConsequenceLearner(p, op)
    giver, hungry = agent(0), agent(1)
    giver.alive = True
    giver.qmodels = learner.new_models()
    f = learner.features(giver, hungry, "give", (0.9,))
    assert f[UNKNOWN] == 1.0 and f[:4].sum() == 1.0  # one-hot status configuration: a stranger
    for _ in range(30):  # giving is always followed by a net gain of 2 food units
        learner.record(giver, "give", f, True, k=0)
        giver.welfare += 2.0
        learner.resolve(k=10)
    assert learner.advantage(giver, "give", f) > 1.5  # acting predicted ~2 better than not acting (0)


def test_no_fixed_sign_social_terms_left():
    """Audit guard: the removed disposition weights must not come back (docs/assumptions_audit.md)."""
    removed = ("give_liking", "give_need", "refuse_liking", "refuse_crowding", "refuse_risk",
               "attack_liking", "attack_expected_win", "attack_hunger", "attack_transient",
               "give_bias", "refuse_bias", "attack_bias")  # act-specific propensities removed too
    params = SocialParams.__dataclass_fields__
    assert not [name for name in removed if name in params]


def test_imitation_only_copies_from_better_off_agents():
    from emergent_politeness.learning import ConsequenceLearner
    p = SocialParams(copy_strength=0.5)
    learner = ConsequenceLearner(p, Opinions(p, True, np.random.default_rng(0)))
    me, better, worse = agent(0), agent(1), agent(2)
    for a in (me, better, worse):
        a.qmodels = learner.new_models()
    better.qmodels["give"][0][4] = 1.0  # the better-off agent has learned that giving to liked agents pays
    worse.qmodels["give"][0][4] = -1.0
    me.health, me.hunger = 0.5, 0.5
    better.health, better.hunger = 1.0, 0.0
    worse.health, worse.hunger = 0.2, 0.9
    rng = np.random.default_rng(1)
    assert not learner.imitate(me, worse, rng)  # never copy someone doing worse
    assert me.qmodels["give"][0][4] == 0.0
    assert learner.imitate(me, better, rng)  # gap 0.75: copied with probability 0.75 (this draw copies)
    assert me.qmodels["give"][0][4] == pytest.approx(0.5)
