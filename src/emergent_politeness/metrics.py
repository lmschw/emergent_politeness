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


def linearity(wins: Counter[tuple[int, int]]) -> tuple[float, int]:
    """de Vries (1995) improved Landau linearity h' from a win matrix.

    wins[(i, j)] = fights i won against j. Dyads with no fights or equal wins
    count as unknown/tied and give half a dominance to each. Under random
    dominance the expected value is 3/(n+1). Returns (h', number of individuals).
    """
    ids = sorted({i for pair in wins for i in pair})
    n = len(ids)
    if n < 3:
        return float("nan"), n
    dominates: Counter[int] = Counter()
    unknown = 0
    for x in range(n):
        for y in range(x + 1, n):
            i, j = ids[x], ids[y]
            wij, wji = wins.get((i, j), 0), wins.get((j, i), 0)
            if wij > wji:
                dominates[i] += 1
            elif wji > wij:
                dominates[j] += 1
            else:  # unknown or tied: half a dominance each
                dominates[i] += 0.5
                dominates[j] += 0.5
                unknown += 1
    h = 12 / (n**3 - n) * sum((dominates[i] - (n - 1) / 2) ** 2 for i in ids)
    return h + 6 * unknown / (n**3 - n), n


def status_agreement(snapshot: dict) -> float:
    """Share of the variance in perceived status P explained by who is being rated
    (1 = observers agree perfectly, 0 = no agreement). Targets rated by ≥ 3 observers."""
    ratings: dict[int, list[float]] = {}
    for obs in snapshot["agents"].values():
        for j, (_, P, _) in obs["memory"].items():
            ratings.setdefault(j, []).append(P)
    groups = [np.array(v) for v in ratings.values() if len(v) >= 3]
    if len(groups) < 3:
        return float("nan")
    allv = np.concatenate(groups)
    total = allv.var()
    if total == 0:
        return float("nan")
    between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / len(allv)
    return float(between / total)


def social_summary(sim, burn_in_days: float = 2.0) -> dict:
    """Stage-2 measures: act rates, fight deaths, hierarchy (spec §7 'Social structure')."""
    day = sim.cfg.world.day_length
    start = burn_in_days * day
    days_kept = max(1e-9, (sim.k - start) / day)
    acts = [r for r in sim.log.of_kind("act") if r["step"] >= start]
    count = Counter(r["act"] for r in acts)
    reasons = Counter(r.get("reason") for r in acts if r["act"] == "attack")
    attacks = [r for r in acts if r["act"] == "attack"]
    wins: Counter[tuple[int, int]] = Counter()
    for r in attacks:
        w, l = (r["actor"], r["target"]) if r["actor_won"] else (r["target"], r["actor"])
        wins[(w, l)] += 1
    h, n_fighters = linearity(wins)
    deaths = [r for r in sim.log.of_kind("death") if r["step"] >= start]
    out = {
        "gives_per_day": count["give"] / days_kept,
        "refusals_per_day": count["refuse_entry"] / days_kept,
        "attacks_per_day": count["attack"] / days_kept,
        "attacks_refused_per_day": reasons["refused"] / days_kept,
        "attacks_patch_per_day": reasons["patch"] / days_kept,
        "attacks_grudge_per_day": reasons["grudge"] / days_kept,
        "attacker_win_rate": float(np.mean([r["actor_won"] for r in attacks])) if attacks else float("nan"),
        "fight_death_share": (sum(d["cause"] == "fight" for d in deaths) / len(deaths)) if deaths else float("nan"),
        "imitations_per_day": sum(1 for r in sim.log.of_kind("imitate") if r["step"] >= start) / days_kept,
        "linearity_h": h,
        "linearity_chance": 3 / (n_fighters + 1) if n_fighters >= 3 else float("nan"),
        "n_fighters": n_fighters,
    }
    snaps = [s for s in sim.snapshots if s["step"] >= start]
    if snaps:
        last = snaps[-1]
        S = np.array([a["self_status"] for a in last["agents"].values()])
        k = np.array([a["strength"] for a in last["agents"].values()])
        out["corr_selfstatus_strength"] = float(np.corrcoef(S, k)[0, 1]) if len(S) > 2 and S.std() > 0 else float("nan")
        out["status_agreement"] = status_agreement(last)
        perceived: dict[int, list[float]] = {}
        for obs in last["agents"].values():
            for j, (_, P, _) in obs["memory"].items():
                perceived.setdefault(j, []).append(P)
        pairs = [(np.mean(perceived[j]), last["agents"][j]["strength"]) for j in perceived if j in last["agents"]]
        if len(pairs) > 2:
            pm, st = np.array(pairs).T
            out["corr_perceived_status_strength"] = float(np.corrcoef(pm, st)[0, 1]) if pm.std() > 0 else float("nan")
        out["mean_memory_size"] = float(np.mean([len(a["memory"]) for a in last["agents"].values()]))
        # What agents have learned about consequences (spec §5.3d): mean advantage weight per feature,
        # and how much agents differ (mean SD across agents; low = a shared norm)
        from .learning import FEATURES
        spreads = []
        for act, names in FEATURES.items():
            ws = np.array([a["advantage"][act] for a in last["agents"].values() if a.get("advantage")])
            if len(ws):
                for n_, name in enumerate(names):
                    out[f"w_{act}_{name}"] = float(ws[:, n_].mean())
                spreads.append(ws.std(axis=0).mean())
        if spreads:
            out["weight_spread_between_agents"] = float(np.mean(spreads))
        # What agents have learned about status (spec §5.3a): mean over living agents
        for name in ("win", "refuse", "attack", "give"):
            tables = np.array([a["expect"][name] for a in last["agents"].values() if a.get("expect")])
            if len(tables):
                for b, label in enumerate(("below", "level", "above", "unknown")):
                    out[f"expect_{name}_{label}"] = float(tables[:, b].mean())
    return out
