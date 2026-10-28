"""Run one condition over several seeds and print calibration measures.

Examples:
    uv run python scripts/run.py --seeds 8 --days 20                 # condition 0
    uv run python scripts/run.py --condition 1 --days 50 --burn-in 30
    uv run python scripts/run.py --set world.food_regrowth=0.03 --set tasks.food_per_capita=4
    uv run python scripts/run.py --condition 4 --sweep social.exploration_bias=-2,-3,-4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from emergent_politeness.config import CONDITIONS, Config
from emergent_politeness.metrics import social_summary, summary
from emergent_politeness.model import Simulation

SHOW = [
    "n_mean", "n_min", "extinct", "deaths_per_100_agent_days", "deaths_starvation", "deaths_cold",
    "deaths_injury", "frac_steps_shelter_full", "frac_outside_at_night", "door_blocked_per_day",
    "food_per_capita", "level_final", "integrity_mean", "work_share_collect_food", "work_share_collect_material",
    "work_share_maintain", "work_share_build", "division_of_labour",
]
SHOW_SOCIAL = [
    "food_requests_per_day", "task_requests_per_day", "compliance_rate",
    "gives_per_day", "refusals_per_day", "attacks_per_day", "attacks_refused_per_day", "attacks_patch_per_day",
    "attacks_grudge_per_day", "attacker_win_rate", "fight_death_share", "linearity_h", "linearity_chance", "n_fighters",
    "corr_selfstatus_strength", "corr_perceived_status_strength", "status_agreement", "mean_memory_size",
    "weight_spread_between_agents", "imitations_per_day",
    *[f"expect_{n}_{b}" for n in ("win", "refuse", "attack", "give") for b in ("below", "level", "above", "unknown")],
]


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_config(sets: list[str], days: float, seed: int, condition: str = "0") -> Config:
    groups: dict[str, dict] = {}
    for item in sets:
        key, value = item.split("=", 1)
        group, name = key.split(".", 1)
        groups.setdefault(group, {})[name] = parse_value(value)
    cfg = Config(condition=CONDITIONS[condition])
    cfg = cfg.with_(**groups) if groups else cfg
    return cfg.with_(sim={"steps": int(days * cfg.world.day_length), "seed": seed})


def run_one(args: tuple[list[str], float, int, float, str]) -> dict:
    sets, days, seed, burn_in, condition = args
    sim = Simulation(build_config(sets, days, seed, condition)).run()
    out = summary(sim, burn_in_days=burn_in)
    if sim.opinions is not None:
        out.update(social_summary(sim, burn_in_days=burn_in))
    return out


def report(label: str, results: list[dict]) -> None:
    print(f"\n== {label}  ({len(results)} seeds)")
    learned = sorted(k for k in results[0] if k.startswith("w_"))
    for key in SHOW + (SHOW_SOCIAL + learned if "gives_per_day" in results[0] else []):
        vals = np.array([r.get(key, np.nan) for r in results], dtype=float)
        print(f"  {key:28s} {np.nanmean(vals):9.3f}  ± {np.nanstd(vals):7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="0", choices=sorted(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--days", type=float, default=20)
    ap.add_argument("--burn-in", type=float, default=3, help="days excluded from time averages")
    ap.add_argument("--set", action="append", default=[], metavar="group.param=value")
    ap.add_argument("--sweep", default=None, metavar="group.param=v1,v2,...")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    variants = [("base", args.set)]
    if args.sweep:
        key, values = args.sweep.split("=", 1)
        variants = [(f"{key}={v}", args.set + [f"{key}={v}"]) for v in values.split(",")]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for label, sets in variants:
            jobs = [(sets, args.days, seed, args.burn_in, args.condition) for seed in range(args.seeds)]
            report(f"condition {args.condition}: {label}", list(pool.map(run_one, jobs)))


if __name__ == "__main__":
    main()
