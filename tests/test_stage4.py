import numpy as np
import pytest

from emergent_politeness.agents import Agent
from emergent_politeness.config import CONDITIONS, Config, LanguageParams, SocialParams
from emergent_politeness.language import SPEECH_ACTS, Language, all_bundles
from emergent_politeness.model import Simulation
from emergent_politeness.social import Opinions


def setup(**lang):
    lp = LanguageParams(**lang)
    sp = SocialParams(attention=1.0)
    rng = np.random.default_rng(0)
    op = Opinions(sp, True, rng)
    language = Language(lp, sp, op, rng)
    agents = []
    for i in range(3):
        a = Agent(id=i, born=0, thresholds={}, strength=1.0)
        a.expect = op.new_expectations()
        a.lexicon = language.new_lexicon()
        agents.append(a)
    return language, op, agents


def test_bundles_are_all_multisets_up_to_two():
    assert len(all_bundles(6, 2)) == 1 + 6 + 21


def test_conditioning_gives_markers_the_meaning_of_what_follows():
    language, _, (s, t, o) = setup()
    for _ in range(200):  # marker 0 always comes with a decline, marker 1 with a gift
        language.condition([o], (0,), (-0.3, 0.2))
        language.condition([o], (1,), (0.6, 0.0))
    assert o.lexicon.c[0] == pytest.approx([-0.3, 0.2], abs=0.01)
    assert o.lexicon.c[1] == pytest.approx([0.6, 0.0], abs=0.01)


def test_appraisal_is_relative_to_the_norm_and_moves_it():
    language, op, (s, t, o) = setup()
    t.lexicon.c[0] = [0.8, 0.0]  # to the addressee, marker 0 is warm
    t.lexicon.norm[SPEECH_ACTS.index("request_task")] = [[0.8, 0.0]] * 4  # ...and warmth is expected here
    d = language.appraise([s, t], s, t, "request_task", (0,), k=0)
    assert abs(d[0]) < 0.2  # politic, not polite: barely registers
    t.lexicon.norm[:] = 0.0
    d = language.appraise([s, t], s, t, "request_task", (0,), k=0)
    assert d[0] > 0.5 and t.memory[s.id].E > 0  # polite: above the norm, liked more


def test_speaker_wanting_to_be_liked_prefers_warmer_markers():
    language, op, (s, t, o) = setup(w_expr=0.0, w_inst=0.0, marker_cost=0.0, temperature=0.05)
    s.lexicon.c[:] = 0.0
    s.lexicon.c[3] = [0.9, 0.0]  # the speaker believes marker 3 is warm
    picks = [language.choose(s, t, "request_task")[0] for _ in range(200)]
    assert np.mean([3 in b for b in picks]) > 0.9


@pytest.mark.parametrize("cond", ["5", "6"])
def test_marker_conditions_run_and_are_deterministic(cond):
    a = Simulation(Config(condition=CONDITIONS[cond])).run(6000)
    b = Simulation(Config(condition=CONDITIONS[cond])).run(6000)
    assert a.log.records == b.log.records
    utter = [r for r in a.log.of_kind("utterance")]
    assert any(r["markers"] for r in utter)
    assert all("exp_e" in r for r in utter)


def test_markers_do_not_leak_into_conditions_without_affect():
    sim = Simulation(Config(condition=CONDITIONS["3"])).run(6000)
    assert all(r["markers"] == [] for r in sim.log.of_kind("utterance"))
    assert all(a.lexicon is None for a in sim.agents.values())


def test_innate_variation_toggle():
    off = Simulation(Config(condition=CONDITIONS["6"]))
    on = Simulation(Config(condition=CONDITIONS["6"]).with_(social={"innate_variation": True}))
    assert all(w[0][4] == 0.0 for a in off.agents.values() for w in a.qmodels.values())
    liking_w = [w[0][4] for a in on.agents.values() for w in a.qmodels.values()]
    assert np.std(liking_w) > 0.1 and abs(np.mean(liking_w)) < 0.1


def test_learned_motive_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        Simulation(Config(condition=CONDITIONS["6"]).with_(language={"motive": "learned"}))
