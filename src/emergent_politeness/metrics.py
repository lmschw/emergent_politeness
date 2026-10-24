"""Post-hoc measures computed from the event log and time series (spec §7)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .agents import WORK_TASKS, Task


def division_of_labour(task_steps: Counter[tuple[int, Task]]) -> float:
    """Normalised mutual information between individual and task, I(ind; task) / H(task).

    0 = everyone does the same mix of tasks, 1 = every individual does a single
    task. Follows the mutual-entropy approach of Gorelick et al. (2004).
    """
    total = sum(task_steps.values())
    if total == 0:
        return float("nan")
    p_ind: Counter[int] = Counter()
    p_task: Counter[Task] = Counter()
    for (ind, task), n in task_steps.items():
        p_ind[ind] += n
        p_task[task] += n
    h_task = -sum((n / total) * math.log(n / total) for n in p_task.values())
    if h_task == 0:
        return float("nan")
    mi = sum(
        (n / total) * math.log((n / total) / ((p_ind[ind] / total) * (p_task[task] / total)))
        for (ind, task), n in task_steps.items()
        if n > 0
    )
    return mi / h_task


def summary(sim, burn_in_days: float = 2.0) -> dict:
    """Key stage-1 calibration measures. Time averages skip the burn-in period."""
    s = {k: np.asarray(v) for k, v in sim.series.items()}
    day = sim.cfg.world.day_length
    days = sim.k / day
    keep = s["step"] >= burn_in_days * day
    n = s["n"][keep]
    deaths = list(sim.log.of_kind("death"))
    causes = Counter(d["cause"] for d in deaths)
    night = keep & (s["cold"] > 0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        outside_frac = np.where(s["n"] > 0, 1 - s["n_inside"] / s["n"], np.nan)
    work = Counter()
    for (_, task), k in sim.task_steps.items():
        work[task] += k
    total_work = sum(work.values()) or 1
    agent_days = float(np.sum(s["n"])) * sim.cfg.sim.record_every / day
    return {
        "days": round(days, 1),
        "n_final": int(s["n"][-1]),
        "n_mean": float(n.mean()) if n.size else float("nan"),
        "n_min": int(n.min()) if n.size else 0,
        "extinct": bool(s["n"][-1] == 0),
        "arrivals": sim.log.count("spawn") - sim.cfg.population.n_initial,
        "deaths": len(deaths),
        "deaths_starvation": causes.get("starvation", 0),
        "deaths_cold": causes.get("cold", 0),
        "deaths_injury": causes.get("injury", 0),
        "deaths_per_100_agent_days": 100 * len(deaths) / agent_days if agent_days else float("nan"),
        "mean_age_at_death_days": float(np.mean([d["age"] for d in deaths]) / day) if deaths else float("nan"),
        "frac_steps_shelter_full": float(np.mean(s["n_inside"][keep] >= s["capacity"][keep])),
        "frac_outside_at_night": float(np.nanmean(outside_frac[night])) if night.any() else float("nan"),
        "door_blocked_per_day": sim.log.count("door_blocked") / days if days else float("nan"),
        "food_per_capita": float(np.nanmean(s["food"][keep] / np.maximum(n, 1))),
        "level_final": int(s["level"][-1]),
        "integrity_mean": float(s["integrity"][keep].mean()),
        "injuries_per_day": sim.log.count("injury") / days if days else float("nan"),
        **{f"work_share_{t.value}": work[t] / total_work for t in WORK_TASKS},
        "division_of_labour": division_of_labour(sim.task_steps),
    }
