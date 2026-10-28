from collections import Counter

import numpy as np
import pytest

from emergent_politeness.agents import Task
from emergent_politeness.config import CONDITIONS, Config
from emergent_politeness.metrics import division_of_labour, summary
from emergent_politeness.model import Simulation


def test_config_with_changes_only_named_fields():
    cfg = Config()
    new = cfg.with_(world={"food_regrowth": 0.01}, sim={"seed": 7})
    assert new.world.food_regrowth == 0.01 and new.sim.seed == 7
    assert cfg.world.food_regrowth == 0.05  # original untouched
    assert new.world.shelter_radius == cfg.world.shelter_radius


def test_same_seed_same_run():
    a = Simulation(Config()).run(400)
    b = Simulation(Config()).run(400)
    assert a.series == b.series
    assert a.log.records == b.log.records


def test_response_probability():
    f = Simulation.response_probability
    assert f(0.0, 0.5) == 0.0
    assert f(0.5, 0.5) == pytest.approx(0.5)
    assert f(0.9, 0.1) > f(0.1, 0.9)


def test_shelter_capacity_never_exceeded_and_agents_enter():
    cfg = Config().with_(
        population={"n_initial": 12, "arrival_rate": 0.05},
        world={"build_required": 1e9},  # no expansion
    )
    sim = Simulation(cfg).run(1500)
    n_inside = np.array(sim.series["n_inside"])
    assert n_inside.max() <= cfg.world.capacity_base
    assert sim.log.count("enter") > 0
    assert sim.log.count("door_blocked") > 0


def test_starvation_kills():
    cfg = Config().with_(
        population={"n_initial": 4, "arrival_rate": 0.0},
        world={"initial_food": 0.0, "food_regrowth": 0.0, "patch_max_stock": 0.0},
    )
    sim = Simulation(cfg).run(1500)
    causes = Counter(r["cause"] for r in sim.log.of_kind("death"))
    assert not sim.agents
    assert causes["starvation"] >= 3


def test_cold_kills_when_locked_out():
    cfg = Config().with_(
        population={"n_initial": 3, "arrival_rate": 0.0},
        world={"capacity_base": 0, "build_required": 1e9},
        body={"hunger_rate": 0.0},
    )
    sim = Simulation(cfg).run(3000)
    causes = Counter(r["cause"] for r in sim.log.of_kind("death"))
    assert causes == Counter({"cold": 3})


def test_division_of_labour_index():
    specialists = Counter({(0, Task.COLLECT_FOOD): 10, (1, Task.BUILD): 10})
    generalists = Counter({(0, Task.COLLECT_FOOD): 5, (0, Task.BUILD): 5, (1, Task.COLLECT_FOOD): 5, (1, Task.BUILD): 5})
    assert division_of_labour(specialists) == pytest.approx(1.0)
    assert division_of_labour(generalists) == pytest.approx(0.0)


def test_summary_runs():
    sim = Simulation(Config()).run(1300)
    out = summary(sim, burn_in_days=1)
    assert out["n_final"] >= 0
    assert 0 <= out["frac_steps_shelter_full"] <= 1
