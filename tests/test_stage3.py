from collections import Counter

import pytest

from emergent_politeness.config import CONDITIONS, Config
from emergent_politeness.model import Simulation


@pytest.mark.parametrize("cond", ["2", "3"])
def test_language_conditions_run_and_are_deterministic(cond):
    a = Simulation(Config(condition=CONDITIONS[cond])).run(9000)
    b = Simulation(Config(condition=CONDITIONS[cond])).run(9000)
    assert a.log.records == b.log.records
    utter = list(a.log.of_kind("utterance"))
    assert any(r["act"] == "request" for r in utter)
    # every request is answered immediately by its addressee
    requests = [r for r in utter if r["act"] == "request"]
    answers = [r for r in utter if r["act"] == "respond"]
    assert len(requests) == len(answers)
    assert all(q["addressee"] == r["speaker"] and q["speaker"] == r["addressee"] for q, r in zip(requests, answers))


def test_language_only_has_no_physical_acts_or_food_requests():
    sim = Simulation(Config(condition=CONDITIONS["2"])).run(9000)
    acts = Counter(r["act"] for r in sim.log.of_kind("act"))
    assert acts["give"] == acts["refuse_entry"] == acts["attack"] == 0
    assert not any(r["content"] == "food" for r in sim.log.of_kind("utterance"))


def test_yes_to_a_food_request_is_a_gift():
    sim = Simulation(Config(condition=CONDITIONS["3"])).run(12000)
    log = sim.log.records
    for n, r in enumerate(log):
        if r["kind"] == "utterance" and r["act"] == "respond" and r["content"] == "yes":
            q = log[n - 1]
            if q["kind"] == "utterance" and q["content"] == "food":
                gift = log[n + 1]
                assert gift["kind"] == "act" and gift["act"] == "give"
                assert gift["actor"] == r["speaker"] and gift["target"] == r["addressee"]
                return
    pytest.fail("no granted food request in 20 days")


def test_social_returns_ignore_store_food_and_routine_work():
    sim = Simulation(Config(condition=CONDITIONS["3"]))
    a = next(iter(sim.agents.values()))
    sim._book(a, "store_food", 1.0)
    sim._book(a, "work", -0.01)
    assert a.welfare == 0.0  # recorded for analysis, not counted
    sim._book(a, "gift_received", 1.0)
    sim._book(a, "work_for_others", -0.01)
    assert a.welfare == 0.99
    allmode = Simulation(Config(condition=CONDITIONS["3"]).with_(social={"return_mode": "all"}))
    b = next(iter(allmode.agents.values()))
    allmode._book(b, "store_food", 1.0)
    assert b.welfare == 1.0


def test_vicarious_learning_uses_the_subjects_return():
    import numpy as np
    from emergent_politeness.agents import Agent
    from emergent_politeness.config import SocialParams
    from emergent_politeness.learning import ConsequenceLearner
    from emergent_politeness.social import Opinions
    p = SocialParams(return_window=10, return_rate=0.5)
    op = Opinions(p, False, np.random.default_rng(0))
    learner = ConsequenceLearner(p, op)
    watcher, doer, target = (Agent(id=i, born=0, thresholds={}, strength=1.0) for i in range(3))
    for x in (watcher, doer, target):
        x.expect = op.new_expectations()
        x.qmodels = learner.new_models()
    f = learner.features(watcher, target, "refuse", (0.0, 0.5, 0.5))
    for n in range(200):  # the doer refuses every 4th time and then always loses 1 food unit; the watcher sees it
        refused = n % 4 == 0
        learner.record(watcher, "refuse", f, refused, k=0, subject=doer, weight=1.0)
        doer.welfare += -1.0 if refused else 0.0
        learner.resolve(k=10)
    assert learner.advantage(watcher, "refuse", f) < -0.5
    assert watcher.welfare == 0.0  # the watcher's own return is untouched
